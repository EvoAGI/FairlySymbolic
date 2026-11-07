// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_TASKIO_HPP
#define ARC_TASKIO_HPP

// ARC task I/O + Kaggle submission writer.
//
// Loads per-file tasks (data/training/<id>.json: {train:[{input,output}],
// test:[{input[,output]}]}) and Kaggle combined-file splits
// (arc-agi_<split>_challenges.json + optional _solutions.json), then writes a
// canonical submission.json ({task_id:[{attempt_1,attempt_2},...]}) with the
// task records STABLE-SORTED by task_id, so the file is deterministic. Uses the
// arc/json.hpp parser/writer; no exceptions.

#include <string>
#include <vector>

#include "arc/grid.hpp"

namespace arc {

// One (input, output) pair. For a test entry without a visible output,
// has_output is false and output is empty.
struct TaskPair {
  Grid input;
  Grid output;
  bool has_output = false;
};

// A fully-parsed ARC task: id + train pairs + test entries (each an input, with
// output present only on splits that ship solutions).
struct Task {
  std::string task_id;
  std::vector<TaskPair> train;   // always has_output == true
  std::vector<TaskPair> test;    // has_output true iff the split has solutions
};

// Parse an in-memory ARC task JSON object (already parsed) into a Task. The
// task_id is supplied by the caller (file basename or Kaggle key). Returns false
// if the shape is malformed (missing/non-array grids).
bool task_from_json_text(const std::string& text, const std::string& task_id,
                         Task* out);

// Load a per-file task from data/<...>/<id>.json. Returns false on read/parse
// failure. task_id defaults to the file basename (without ".json").
bool load_task_file(const std::string& path, Task* out);

// Load an entire per-file dataset directory (every *.json under dir). Tasks are
// returned in ASCENDING task_id order (stable). Returns false if the directory
// cannot be listed.
bool load_task_dir(const std::string& dir, std::vector<Task>* out);

// Load one split of a Kaggle dataset folder (challenges + optional solutions),
// merging solutions into the matching test entries. Tasks are returned in
// ASCENDING task_id order (stable). Returns false if the challenges file is
// absent or malformed.
bool load_kaggle_split(const std::string& kaggle_dir, const std::string& split,
                       std::vector<Task>* out);

// One test input's two attempts (each a grid). Both default to the sample
// placeholder [[0,0],[0,0]] so a failed/absent task still yields a complete
// submission entry.
struct SubmissionAttempt {
  Grid attempt_1;
  Grid attempt_2;
};

// The whole submission: task_id -> per-test-input attempts.
struct Submission {
  // Insertion order is not load-bearing: serialization STABLE-sorts by task_id.
  std::vector<std::pair<std::string, std::vector<SubmissionAttempt>>> entries;

  // Add / replace a task's attempts.
  void set(const std::string& task_id, std::vector<SubmissionAttempt> attempts);
};

// The Kaggle sample placeholder grid ([[0,0],[0,0]]).
Grid submission_placeholder_grid();

// Serialize a Submission to canonical submission.json text. Task records are
// STABLE-sorted by task_id; within each task the attempts are in test-input
// order; each attempt is {"attempt_1":grid,"attempt_2":grid}.
std::string submission_to_json(const Submission& sub);

}  // namespace arc

#endif  // ARC_TASKIO_HPP
