// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// ARC task I/O + Kaggle submission writer. See include/arc/taskio.hpp.

#include "arc/taskio.hpp"

#include <algorithm>
#include <cstdio>
#include <dirent.h>

#include "arc/json.hpp"

namespace arc {

namespace {

// Read a whole file into a string. Returns false on open failure.
bool read_file(const std::string& path, std::string* out) {
  FILE* fh = std::fopen(path.c_str(), "rb");
  if (!fh) return false;
  out->clear();
  char buf[8192];
  size_t got;
  while ((got = std::fread(buf, 1, sizeof(buf), fh)) > 0) out->append(buf, got);
  std::fclose(fh);
  return true;
}

// Parse a JSON grid value into a Grid. Returns false on shape error.
bool grid_from_jsonval(const JsonValue& v, Grid* out) {
  std::vector<std::vector<int>> rows;
  if (!grid_from_json(v, &rows)) return false;
  *out = Grid::from_rows(rows);
  return true;
}

// Parse a list of {input,output?} entries into TaskPairs.
bool parse_pairs(const JsonValue& arr, bool require_output, std::vector<TaskPair>* out) {
  if (!arr.is_array()) return false;
  for (const JsonValue& entry : arr.items()) {
    if (!entry.is_object()) return false;
    const JsonValue* inp = entry.find("input");
    if (!inp) return false;
    TaskPair p;
    if (!grid_from_jsonval(*inp, &p.input)) return false;
    const JsonValue* outp = entry.find("output");
    if (outp) {
      if (!grid_from_jsonval(*outp, &p.output)) return false;
      p.has_output = true;
    } else if (require_output) {
      return false;
    }
    out->push_back(std::move(p));
  }
  return true;
}

// Parse a challenges-style task object into a Task (test outputs optional).
bool parse_task_object(const JsonValue& obj, const std::string& task_id, Task* out) {
  if (!obj.is_object()) return false;
  out->task_id = task_id;
  out->train.clear();
  out->test.clear();
  const JsonValue* train = obj.find("train");
  if (train) {
    if (!parse_pairs(*train, /*require_output=*/true, &out->train)) return false;
  }
  const JsonValue* test = obj.find("test");
  if (test) {
    if (!parse_pairs(*test, /*require_output=*/false, &out->test)) return false;
  }
  return true;
}

}  // namespace

bool task_from_json_text(const std::string& text, const std::string& task_id, Task* out) {
  bool ok = false;
  JsonValue v = json_parse(text, &ok);
  if (!ok) return false;
  return parse_task_object(v, task_id, out);
}

bool load_task_file(const std::string& path, Task* out) {
  std::string text;
  if (!read_file(path, &text)) return false;
  // task_id = basename without ".json".
  std::string base = path;
  size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  if (base.size() >= 5 && base.substr(base.size() - 5) == ".json")
    base = base.substr(0, base.size() - 5);
  return task_from_json_text(text, base, out);
}

bool load_task_dir(const std::string& dir, std::vector<Task>* out) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) return false;
  std::vector<std::string> names;
  struct dirent* ent;
  while ((ent = ::readdir(d)) != nullptr) {
    std::string name = ent->d_name;
    if (name.size() >= 5 && name.substr(name.size() - 5) == ".json") names.push_back(name);
  }
  ::closedir(d);
  // ASCENDING task_id order (stable): the filenames ARE "<id>.json".
  std::stable_sort(names.begin(), names.end());
  for (const std::string& name : names) {
    Task t;
    if (load_task_file(dir + "/" + name, &t)) out->push_back(std::move(t));
  }
  return true;
}

bool load_kaggle_split(const std::string& kaggle_dir, const std::string& split,
                       std::vector<Task>* out) {
  std::string challenges_path = kaggle_dir + "/arc-agi_" + split + "_challenges.json";
  std::string ctext;
  if (!read_file(challenges_path, &ctext)) return false;
  bool ok = false;
  JsonValue challenges = json_parse(ctext, &ok);
  if (!ok || !challenges.is_object()) return false;

  // Optional solutions file.
  std::string sol_path = kaggle_dir + "/arc-agi_" + split + "_solutions.json";
  std::string stext;
  JsonValue solutions = JsonValue::make_object();
  bool have_solutions = false;
  if (read_file(sol_path, &stext)) {
    bool sok = false;
    JsonValue s = json_parse(stext, &sok);
    if (sok && s.is_object()) {
      solutions = s;
      have_solutions = true;
    }
  }

  // Parse every task, merging solutions into the matching test entries.
  std::vector<Task> tasks;
  for (const auto& kv : challenges.members()) {
    const std::string& task_id = kv.first;
    Task t;
    if (!parse_task_object(kv.second, task_id, &t)) continue;
    if (have_solutions) {
      const JsonValue* sol = solutions.find(task_id);
      if (sol && sol->is_array()) {
        for (size_t i = 0; i < t.test.size() && i < sol->size(); ++i) {
          Grid g;
          if (grid_from_jsonval((*sol)[i], &g)) {
            t.test[i].output = std::move(g);
            t.test[i].has_output = true;
          }
        }
      }
    }
    tasks.push_back(std::move(t));
  }
  // ASCENDING task_id order (stable), independent of challenges-file order, so
  // the loaded set is deterministic.
  std::stable_sort(tasks.begin(), tasks.end(),
                   [](const Task& a, const Task& b) { return a.task_id < b.task_id; });
  for (Task& t : tasks) out->push_back(std::move(t));
  return true;
}

void Submission::set(const std::string& task_id, std::vector<SubmissionAttempt> attempts) {
  for (auto& e : entries) {
    if (e.first == task_id) {
      e.second = std::move(attempts);
      return;
    }
  }
  entries.emplace_back(task_id, std::move(attempts));
}

Grid submission_placeholder_grid() {
  Grid g(2, 2);  // [[0,0],[0,0]]
  return g;
}

std::string submission_to_json(const Submission& sub) {
  // STABLE-sort task records by task_id, for a deterministic output.
  std::vector<const std::pair<std::string, std::vector<SubmissionAttempt>>*> ordered;
  ordered.reserve(sub.entries.size());
  for (const auto& e : sub.entries) ordered.push_back(&e);
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto* a, const auto* b) { return a->first < b->first; });

  JsonValue root = JsonValue::make_object();
  for (const auto* e : ordered) {
    JsonValue attempts = JsonValue::make_array();
    for (const SubmissionAttempt& a : e->second) {
      JsonValue obj = JsonValue::make_object();
      // Build grid arrays directly from the Grid cells (row-major).
      auto grid_to_json = [](const Grid& g) {
        JsonValue arr = JsonValue::make_array();
        for (int y = 0; y < g.height(); ++y) {
          JsonValue row = JsonValue::make_array();
          for (int x = 0; x < g.width(); ++x) row.push_back(JsonValue::make_int(g.at(y, x)));
          arr.push_back(std::move(row));
        }
        return arr;
      };
      obj.set("attempt_1", grid_to_json(a.attempt_1));
      obj.set("attempt_2", grid_to_json(a.attempt_2));
      attempts.push_back(std::move(obj));
    }
    root.set(e->first, std::move(attempts));
  }
  return json_dump(root);
}

}  // namespace arc
