// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// arc_solver -- the standalone C++ ARC solver binary.
//
// Pipeline per task:
//   1. per-task seed = task_seed(cfg.seed, task_id)
//   2. color prior   = sorted set of train-output colors
//   3. run_task_full (the search: multi-restart + occam/view continuations +
//      finalist gather + LOO/MDL promotion) -> primary genome + ordered
//      finalists + view solvers
//   4. attempt_1 = primary genome executed on each test input (predict_grid)
//      attempt_2 = the best DISTINCT alternative from the view solvers, else a
//      distinct finalist, else a copy of attempt_1
//   5. write submission.json = {task_id:[{attempt_1,attempt_2},..]}, task records
//      STABLE-sorted by task_id (determinism), placeholders pre-seeded so every
//      task has a complete entry even if it errors.
//
// DEPLOYMENT vs WALL-CLOCK: by default (and always under --verify) the search
// runs the CONFIGURED `generations` to completion -- fixed-generations, so each
// task's result depends ONLY on its seed + config and the submission is
// n_workers-invariant and byte-identical from run to run. Passing an explicit
// `--max-task-seconds S` (non-verify) OPTS IN to a per-task wall-clock cap:
// run_task_full stamps a steady_clock deadline and the generation + restart
// /occam/view loops break to best-so-far once it passes. That path is NOT
// reproducible (wall-clock is machine-speed dependent, so which tasks finish /
// how deep the search goes varies across hosts).
//
// TASK-LEVEL PARALLELISM: a std::thread pool sized by n_workers processes tasks; each
// task is independent + deterministic from its own seed, so results are pooled by
// task index and the submission is STABLE-sorted by task_id -> byte-identical across
// worker counts.
//
// --verify pins a fixed configuration: occam+view continuations ON,
// tabu restart OFF, macro libraries OFF.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

#include "arc/config_load.hpp"
#include "arc/deploy.hpp"
#include "arc/fitness.hpp"
#include "arc/grid.hpp"
#include "arc/macro.hpp"   // macro library (Phase-A macro seeds)
#include "arc/codec.hpp"   // g_force_connectivity (ARC_ENS_ENC)
#include "arc/concepts.hpp"  // g_task_colors (ARC_TASK_OTHER)
#include "arc/search.hpp"
#include "arc/taskio.hpp"
#include "arc/json.hpp"    // --full-results per-task record
#include "arc/config.hpp"  // config_to_dict (encoder_config JSON)
#include "arc/dsl.hpp"     // DslProgram / DslValue / DecoderConfig
#include "arc/ast.hpp"     // to_string(SelectorKind/ActionKind/...)
#include "arc/scene_cache.hpp"

namespace {

// invert_view: map a view-space grid back to task space.
// transpose/fliplr/flipud/identity are self-inverse; rot90 <-> rot270 swap.
// (Provided by arc/deploy.hpp; see build_attempts below.)

// The DEFAULT decimal seed string (any bit-width) for run_task_full.
std::string to_decimal(long long v) { return std::to_string(v); }

// Build the SearchConfig from the resolved ArcConfig. No deadline here (fixed
// generations); main() sets max_task_seconds when a wall-clock cap is asked for.
arc::SearchConfig make_search_config(const arc::ArcConfig& cfg) {
  arc::SearchConfig sc;
  sc.pop_size = cfg.population;
  sc.generations = cfg.generations;
  sc.max_rules = cfg.max_rules;
  sc.selection_mode = cfg.selection;
  sc.lexicase_case_mode = cfg.lexicase_cases;
  sc.distance_mode = cfg.fitness_distance;
  sc.stagnation_inject_threshold = cfg.stagnation_inject;
  sc.stagnation_restart_threshold = cfg.stagnation_restart;
  sc.diversity_gate = cfg.diversity_gate;
  sc.lex_champions = cfg.lex_champions;
  sc.lex_complement = cfg.lex_complement;
  sc.solved_refine_stagnation = 8;  // default
  // ARC_SOLVED_REFINE=N: after a fit, keep evolving until N generations bring
  // no better (simpler) fitting program.
  if (const char* e = std::getenv("ARC_SOLVED_REFINE")) sc.solved_refine_stagnation = std::max(1, std::atoi(e));
  sc.rank_specificity = cfg.rank_specificity != 0;
  sc.phase_a_shortcut = cfg.phase_a_shortcut;
  sc.enable_seed_enumerator = true;   // Phase-A seed enumeration ON: seed programs are
                                      // scored at generation 0, before evolution.
  sc.run_refinement = false;          // run_task_full's attempts drive their own refinement
  // ARC_SEED_ENUM=0 switches the Phase-A seed enumerator off (evolution alone).
  if (const char* e = std::getenv("ARC_SEED_ENUM")) sc.enable_seed_enumerator = std::string(e) != "0";
  // Encoder scene cache: run_task_full builds a per-task cache when
  // encoder_cache is on (the default). --no-encoder-cache / encoder_cache=false
  // disables it; the results are byte-identical either way.
  sc.encoder_cache = cfg.encoder_cache;
  sc.encoder_cache_mb = cfg.encoder_cache_mb;

  // Macro libraries, built from the config: macro_seed_library feeds the top-N
  // mined macros to the seed enumerator as gen-0 Phase-A SEEDS; macro_library
  // adds macros as genome genes. DEFAULT OFF: when a library path is empty its
  // pointer stays null and the search runs without macros (--verify clears
  // cfg.macro_* before this call). The stores are function-local statics:
  // make_search_config runs once, before any worker thread, and const
  // MacroLibrary::sample() is safe under concurrent reads, so the pointers stay
  // valid for the whole run.
  static arc::MacroLibrary macro_seed_store;
  static arc::MacroLibrary macro_gene_store;
  if (!cfg.macro_seed_library.empty() && cfg.macro_seed_topn > 0) {
    macro_seed_store.macros = arc::load_macros(cfg.macro_seed_library);
    macro_seed_store.seed_topn = cfg.macro_seed_topn;
    if (!macro_seed_store.macros.empty()) {
      sc.macro_seed_lib = &macro_seed_store;
      std::fprintf(stderr, "arc_solver: macro seed lane ON (%zu macros, top-%d) from %s\n",
                   macro_seed_store.macros.size(), cfg.macro_seed_topn,
                   cfg.macro_seed_library.c_str());
    }
  }
  if (!cfg.macro_library.empty() && cfg.macro_rate > 0.0) {
    macro_gene_store.macros = arc::load_macros(cfg.macro_library);
    macro_gene_store.rate = cfg.macro_rate;
    macro_gene_store.mutate_rate = cfg.macro_mutate_rate;
    if (!macro_gene_store.macros.empty()) {
      sc.macro_lib = &macro_gene_store;
      std::fprintf(stderr, "arc_solver: macro genome lane ON (%zu macros, rate=%.3f) from %s\n",
                   macro_gene_store.macros.size(), cfg.macro_rate,
                   cfg.macro_library.c_str());
    }
  }
  return sc;
}

struct Args {
  std::string config_path;   // --config
  std::map<std::string, std::string> cli;  // field-name overrides
  std::string out_path;      // --out (submission.json), default runs/submission.json
  std::string progress_dir;  // --progress-dir DIR: live/crash-safe run folder (off by default)
  bool verify = false;       // --verify: pin the fixed config knobs
  bool full_results = false; // --full-results: write full per-task JSON (program, configs, fitness) to progress-dir/results (needs --progress-dir)
};

void usage() {
  std::fprintf(stderr,
      "usage: arc_solver [--config PATH] [--data-dir DIR | --kaggle DIR --split S]\n"
      "                  [--pop N] [--generations N] [--restarts N] [--max-rules N]\n"
      "                  [--seed N] [--n-workers N] [--max-tasks N] [--task-offset N]\n"
      "                  [--max-task-seconds S] [--selection MODE] [--lexicase-cases MODE]\n"
      "                  [--fitness-distance D]\n"
      "                  [--occam-continue 0|1] [--tabu-restart 0|1] [--view-continue V]\n"
      "                  [--phase-a-shortcut 0|1] [--verify] [--out submission.json]\n"
      "                  [--progress-dir DIR]\n");
}

// Map a --flag to its ArcConfig field name.
bool set_cli(Args& args, const std::string& flag, const std::string& val) {
  static const std::map<std::string, std::string> kFlagToField = {
      {"--data-dir", "data_dir"}, {"--kaggle", "kaggle"}, {"--split", "split"},
      {"--output-root", "output_root"}, {"--max-tasks", "max_tasks"},
      {"--task-offset", "task_offset"}, {"--pop", "population"},
      {"--population", "population"}, {"--generations", "generations"},
      {"--restarts", "restarts"}, {"--max-rules", "max_rules"}, {"--seed", "seed"},
      {"--max-task-seconds", "max_task_seconds"}, {"--n-workers", "n_workers"},
      {"--fitness-target", "fitness_target"}, {"--fitness-distance", "fitness_distance"},
      {"--phase-a-shortcut", "phase_a_shortcut"}, {"--selection", "selection"},
      {"--lexicase-cases", "lexicase_cases"}, {"--occam-continue", "occam_continue"},
      {"--tabu-restart", "tabu_restart"}, {"--view-continue", "view_continue"},
  };
  auto it = kFlagToField.find(flag);
  if (it == kFlagToField.end()) return false;
  args.cli[it->second] = val;
  return true;
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    }
    if (a == "--verify") {
      args.verify = true;
      continue;
    }
    if (a == "--full-results") {
      args.full_results = true;
      continue;
    }
    if (a == "--config") {
      if (++i >= argc) return false;
      args.config_path = argv[i];
      continue;
    }
    if (a == "--out") {
      if (++i >= argc) return false;
      args.out_path = argv[i];
      continue;
    }
    if (a == "--progress-dir") {
      if (++i >= argc) return false;
      args.progress_dir = argv[i];
      continue;
    }
    // value flags
    if (++i >= argc) {
      std::fprintf(stderr, "arc_solver: missing value for %s\n", a.c_str());
      return false;
    }
    if (!set_cli(args, a, argv[i])) {
      std::fprintf(stderr, "arc_solver: unknown flag %s\n", a.c_str());
      return false;
    }
  }
  return true;
}

// Resolve the config-file path: --config, then $ARC_CONFIG, then arc_config.conf
// in the current directory, else none (pure defaults).
std::string resolve_conf_path(const std::string& explicit_path) {
  if (!explicit_path.empty()) return explicit_path;
  const char* env = std::getenv("ARC_CONFIG");
  if (env && env[0]) return env;
  // arc_config.conf in the current directory.
  FILE* fh = std::fopen("arc_config.conf", "rb");
  if (fh) {
    std::fclose(fh);
    return "arc_config.conf";
  }
  return "";  // pure defaults
}

// Build the ARC_* env layer from the process environment (config_env_map subset).
std::map<std::string, std::string> read_env_layer() {
  std::map<std::string, std::string> env;
  for (const auto& fe : arc::config_env_map()) {
    const char* v = std::getenv(fe.second.c_str());
    if (v) env[fe.second] = v;
  }
  return env;
}

// --- Progress run folder (--progress-dir): incremental observability -------
// Written ONLY when --progress-dir is set -- a pure side-channel that NEVER
// affects the final submission. Enables live monitoring and crash-safe partial
// reconstruction (results/<task_id>.json per task).
void ensure_dir(const std::string& path) {
  ::mkdir(path.c_str(), 0755);  // best-effort; EEXIST is fine.
}

// One task's attempts as a single-entry submission JSON, so a killed run can be
// reconstructed from results/<task_id>.json.
void write_task_result(const std::string& dir, const std::string& task_id,
                       const std::vector<arc::Attempts>& atts) {
  arc::Submission one;
  std::vector<arc::SubmissionAttempt> attempts;
  attempts.reserve(atts.size());
  for (const arc::Attempts& at : atts) {
    arc::SubmissionAttempt sa;
    sa.attempt_1 = at.attempt_1;
    sa.attempt_2 = at.attempt_2;
    attempts.push_back(std::move(sa));
  }
  one.set(task_id, std::move(attempts));
  std::string js = arc::submission_to_json(one);
  std::string path = dir + "/results/" + task_id + ".json";
  FILE* fh = std::fopen(path.c_str(), "wb");
  if (fh) {
    std::fwrite(js.data(), 1, js.size(), fh);
    std::fclose(fh);
  }
}

// ---- --full-results: rich per-task record ----------------------------------
arc::JsonValue grid_to_json(const arc::Grid& g) {
  std::vector<std::vector<int>> rows(static_cast<size_t>(g.height()),
                                     std::vector<int>(static_cast<size_t>(g.width())));
  for (int y = 0; y < g.height(); ++y)
    for (int x = 0; x < g.width(); ++x) rows[static_cast<size_t>(y)][static_cast<size_t>(x)] = g.at(y, x);
  return arc::json_from_grid(rows);
}

arc::JsonValue dslargs_to_json(const arc::DslArgs& args);  // fwd (mutual recursion via Dict)

arc::JsonValue dslvalue_to_json(const arc::DslValue& v) {
  using T = arc::DslValue::Tag;
  switch (v.tag) {
    case T::Int: return arc::JsonValue::make_int(v.i);
    case T::Float: return arc::JsonValue::make_double(v.f);
    case T::Str: return arc::JsonValue::make_string(v.s);
    case T::Bool: return arc::JsonValue::make_bool(v.b);
    case T::Dict: return v.dict ? dslargs_to_json(*v.dict) : arc::JsonValue::make_object();
    case T::Null: default: return arc::JsonValue::make_null();
  }
}

arc::JsonValue dslargs_to_json(const arc::DslArgs& args) {
  arc::JsonValue o = arc::JsonValue::make_object();
  for (const auto& kv : args) o.set(kv.first, dslvalue_to_json(kv.second));
  return o;
}

arc::JsonValue program_to_json(const arc::DslProgram& p) {
  arc::JsonValue prog = arc::JsonValue::make_object();
  arc::JsonValue rules = arc::JsonValue::make_array();
  for (const arc::DslRule& r : p.rules) {
    arc::JsonValue rj = arc::JsonValue::make_object();
    arc::JsonValue sel = arc::JsonValue::make_object();
    sel.set("kind", arc::JsonValue::make_string(arc::to_string(r.selector.kind)));
    sel.set("args", dslargs_to_json(r.selector.args));
    rj.set("selector", std::move(sel));
    arc::JsonValue act = arc::JsonValue::make_object();
    act.set("kind", arc::JsonValue::make_string(arc::to_string(r.action.kind)));
    act.set("args", dslargs_to_json(r.action.args));
    rj.set("action", std::move(act));
    arc::JsonValue guards = arc::JsonValue::make_array();
    for (const arc::DslGuard& g : r.guards) {
      arc::JsonValue gj = arc::JsonValue::make_object();
      gj.set("kind", arc::JsonValue::make_string(arc::to_string(g.kind)));
      gj.set("args", dslargs_to_json(g.args));
      guards.push_back(std::move(gj));
    }
    rj.set("guards", std::move(guards));
    rules.push_back(std::move(rj));
  }
  prog.set("rules", std::move(rules));
  prog.set("global_transform", arc::JsonValue::make_string(arc::to_string(p.global_transform)));
  return prog;
}

// Full per-task record: winning program + encoder/decoder config + fitness +
// generations + solved flag + the submitted attempts, for debugging one task in
// depth.
void write_task_result_full(const std::string& dir, const std::string& task_id,
                            const arc::TaskResultFull& res, int n_train,
                            const std::vector<arc::Attempts>& atts) {
  arc::JsonValue root = arc::JsonValue::make_object();
  root.set("task_id", arc::JsonValue::make_string(task_id));
  root.set("solved", arc::JsonValue::make_bool(res.solved));
  root.set("solved_at_idx", arc::JsonValue::make_int(res.solved_at_idx));
  root.set("generations_used", arc::JsonValue::make_int(res.generations_used));
  root.set("n_train_pairs", arc::JsonValue::make_int(n_train));
  if (!arc::fitness_curve().empty()) {  // ARC_FITLOG=1: [gen, best, mean], at most 2000 points
    const auto& c = arc::fitness_curve();
    size_t step = std::max<size_t>(1, (c.size() + 1999) / 2000);
    arc::JsonValue curve = arc::JsonValue::make_array();
    for (size_t i = 0; i < c.size(); i += step) {
      arc::JsonValue pt = arc::JsonValue::make_array();
      pt.push_back(arc::JsonValue::make_int(static_cast<long long>(c[i][0])));
      pt.push_back(arc::JsonValue::make_double(c[i][1]));
      pt.push_back(arc::JsonValue::make_double(c[i][2]));
      curve.push_back(std::move(pt));
    }
    root.set("fit_curve", std::move(curve));
  }
  root.set("n_test_inputs", arc::JsonValue::make_int(static_cast<long long>(atts.size())));

  const arc::TransformFitnessFull& f = res.primary_fit;
  arc::JsonValue fit = arc::JsonValue::make_object();
  fit.set("solved", arc::JsonValue::make_bool(f.solved));
  fit.set("score", arc::JsonValue::make_double(f.score));
  fit.set("exact_ratio", arc::JsonValue::make_double(f.exact_ratio));
  fit.set("max_pair_mismatch", arc::JsonValue::make_double(f.max_pair_mismatch));
  fit.set("pixel_mismatch_mean", arc::JsonValue::make_double(f.pixel_mismatch_mean));
  fit.set("complexity", arc::JsonValue::make_double(f.complexity));
  fit.set("loo_fail", arc::JsonValue::make_int(f.loo_fail));
  fit.set("spec", arc::JsonValue::make_int(f.spec));
  root.set("fitness", std::move(fit));

  root.set("dsl_program", program_to_json(res.primary_genome.program));

  bool ok = false;
  arc::JsonValue enc = arc::json_parse(arc::config_to_dict(res.primary_genome.encoder_config), &ok);
  root.set("encoder_config", ok ? std::move(enc) : arc::JsonValue::make_null());

  const arc::DecoderConfig& d = res.primary_genome.decoder;
  arc::JsonValue dec = arc::JsonValue::make_object();
  dec.set("background_policy", arc::JsonValue::make_string(d.background_policy));
  dec.set("paint_order", arc::JsonValue::make_string(d.paint_order));
  dec.set("overlap_policy", arc::JsonValue::make_string(d.overlap_policy));
  dec.set("postprocess", arc::JsonValue::make_string(d.postprocess));
  root.set("decoder_config", std::move(dec));

  arc::JsonValue sub = arc::JsonValue::make_array();
  for (const arc::Attempts& at : atts) {
    arc::JsonValue o = arc::JsonValue::make_object();
    o.set("attempt_1", grid_to_json(at.attempt_1));
    o.set("attempt_2", grid_to_json(at.attempt_2));
    if (!at.candidates.empty()) {
      arc::JsonValue cs = arc::JsonValue::make_array();
      for (const auto& lc : at.candidates) {
        arc::JsonValue c = arc::JsonValue::make_object();
        c.set("label", arc::JsonValue::make_string(lc.first));
        c.set("grid", grid_to_json(lc.second));
        cs.push_back(std::move(c));
      }
      o.set("candidates", std::move(cs));
    }
    sub.push_back(std::move(o));
  }
  root.set("submission", std::move(sub));

  std::string js = arc::json_dump(root);
  std::string path = dir + "/results/" + task_id + ".json";
  FILE* fh = std::fopen(path.c_str(), "wb");
  if (fh) {
    std::fwrite(js.data(), 1, js.size(), fh);
    std::fclose(fh);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    usage();
    return 2;
  }

  std::string conf_path = resolve_conf_path(args.config_path);
  std::map<std::string, std::string> env = read_env_layer();
  std::vector<std::string> warnings;
  arc::ArcConfig cfg = arc::config_load(conf_path, env, args.cli, &warnings);
  for (const std::string& w : warnings) std::fprintf(stderr, "[config] %s\n", w.c_str());

  // --verify pins these knobs regardless of config: occam+view ON, tabu
  // OFF, macro libraries OFF, unbounded Phase A.
  if (args.verify) {
    cfg.occam_continue = 1;
    cfg.tabu_restart = 0;
    if (cfg.view_continue == "0" || cfg.view_continue.empty()) cfg.view_continue = "1";
    cfg.macro_library.clear();
    cfg.macro_seed_library.clear();
    cfg.verify_unbounded_phase_a = true;
  }

  // Load tasks (Kaggle combined-file split, else per-file data_dir).
  std::vector<arc::Task> tasks;
  if (!cfg.kaggle.empty()) {
    if (!arc::load_kaggle_split(cfg.kaggle, cfg.split, &tasks)) {
      std::fprintf(stderr, "arc_solver: failed to load kaggle split %s/%s\n",
                   cfg.kaggle.c_str(), cfg.split.c_str());
      return 1;
    }
  } else {
    if (!arc::load_task_dir(cfg.data_dir, &tasks)) {
      std::fprintf(stderr, "arc_solver: failed to load task dir %s\n", cfg.data_dir.c_str());
      return 1;
    }
  }

  // task_offset + max_tasks slicing (tasks are already ascending-id sorted).
  if (cfg.task_offset > 0 && cfg.task_offset < static_cast<int>(tasks.size()))
    tasks.erase(tasks.begin(), tasks.begin() + cfg.task_offset);
  else if (cfg.task_offset >= static_cast<int>(tasks.size()))
    tasks.clear();
  if (cfg.max_tasks > 0 && static_cast<int>(tasks.size()) > cfg.max_tasks)
    tasks.resize(cfg.max_tasks);

  const int n_tasks = static_cast<int>(tasks.size());
  // DEPLOY per-task wall-clock cap: opt-in via an explicit --max-task-seconds and
  // never under --verify, so a fresh / verification run stays fixed-generations
  // and byte-identical. When on, run_task_full bounds each task to this budget.
  double task_wall_cap =
      (!args.verify && args.cli.count("max_task_seconds") && cfg.max_task_seconds > 0.0)
          ? cfg.max_task_seconds
          : 0.0;
  if (task_wall_cap > 0.0) {
    std::fprintf(stderr, "arc_solver: %d task(s), pop=%d gen=%d restarts=%d max_rules=%d "
                 "seed=%d n_workers=%d (per-task wall cap=%.1fs)\n",
                 n_tasks, cfg.population, cfg.generations, cfg.restarts, cfg.max_rules,
                 cfg.seed, cfg.n_workers, task_wall_cap);
  } else {
    std::fprintf(stderr, "arc_solver: %d task(s), pop=%d gen=%d restarts=%d max_rules=%d "
                 "seed=%d n_workers=%d (fixed-generations; wall-clock cap not applied)\n",
                 n_tasks, cfg.population, cfg.generations, cfg.restarts, cfg.max_rules,
                 cfg.seed, cfg.n_workers);
  }

  arc::SearchConfig sc = make_search_config(cfg);
  sc.max_task_seconds = task_wall_cap;  // 0 -> deadline dormant (fixed-generations)

  // --progress-dir: initialise the run folder (meta.json + results/) up front so
  // a watcher can attach immediately. Pure side-channel; off by default.
  if (!args.progress_dir.empty()) {
    ensure_dir(args.progress_dir);
    ensure_dir(args.progress_dir + "/results");
    std::string mp = args.progress_dir + "/meta.json";
    FILE* mf = std::fopen(mp.c_str(), "wb");
    if (mf) {
      std::fprintf(mf,
          "{\"n_tasks\":%d,\"t_start_epoch\":%lld,\"pop\":%d,\"generations\":%d,"
          "\"restarts\":%d,\"max_rules\":%d,\"seed\":%d,\"n_workers\":%d,"
          "\"max_task_seconds\":%.1f}\n",
          n_tasks, static_cast<long long>(std::time(nullptr)), cfg.population,
          cfg.generations, cfg.restarts, cfg.max_rules, cfg.seed, cfg.n_workers,
          task_wall_cap);
      std::fclose(mf);
    }
  }
  std::mutex progress_mu;          // guards the shared progress.jsonl append
  std::atomic<int> progress_seq{0};
  arc::TaskTailConfig tail;
  tail.occam_continue = cfg.occam_continue != 0;
  tail.tabu_restart = cfg.tabu_restart != 0;
  tail.views = arc::parse_view_continue(cfg.view_continue);

  // Pre-seed the submission with placeholders so every task has a complete
  // entry even if solving errors or yields no test inputs.
  std::vector<std::vector<arc::Attempts>> results(static_cast<size_t>(n_tasks));
  for (int ti = 0; ti < n_tasks; ++ti) {
    results[static_cast<size_t>(ti)].resize(tasks[static_cast<size_t>(ti)].test.size());
    for (auto& at : results[static_cast<size_t>(ti)]) {
      at.attempt_1 = arc::submission_placeholder_grid();
      at.attempt_2 = arc::submission_placeholder_grid();
    }
  }

  // Per-task train outcome, for the end-of-run summary (log line + progress-dir).
  // Each index is written by exactly one worker, so no synchronization is needed.
  std::vector<char> task_solved(static_cast<size_t>(n_tasks), 0);      // all train pairs exact
  std::vector<double> task_exact_ratio(static_cast<size_t>(n_tasks), 0.0);  // fraction of train pairs exact
  std::vector<int> task_n_train(static_cast<size_t>(n_tasks), 0);
  std::vector<int> task_gens(static_cast<size_t>(n_tasks), 0);         // GA generations run (search effort)

  // Aggregate encoder scene-cache telemetry across all tasks (for the SUMMARY).
  // Written under progress_mu (below); each task adds its per-task totals.
  long long total_cache_hits = 0;
  long long total_cache_misses = 0;
  long long total_cache_evictions = 0;
  long long peak_cache_bytes = 0;  // max per-task peak observed

  // Crash-safe submission: rebuild + ATOMICALLY replace submission.json as each
  // task finishes, so a mid-run crash (e.g. a deep-search segfault) still leaves
  // every COMPLETED task's real prediction on disk (the rest stay the pre-seeded
  // placeholders). MUST be called under progress_mu (it reads the shared results).
  std::string out_path = args.out_path.empty() ? "submission.json" : args.out_path;
  auto flush_submission = [&]() -> size_t {
    arc::Submission sub;
    for (int j = 0; j < n_tasks; ++j) {
      std::vector<arc::SubmissionAttempt> attempts;
      attempts.reserve(results[static_cast<size_t>(j)].size());
      for (const arc::Attempts& at : results[static_cast<size_t>(j)]) {
        arc::SubmissionAttempt sa;
        sa.attempt_1 = at.attempt_1;
        sa.attempt_2 = at.attempt_2;
        attempts.push_back(std::move(sa));
      }
      sub.set(tasks[static_cast<size_t>(j)].task_id, std::move(attempts));
    }
    std::string json = arc::submission_to_json(sub);
    std::string tmp = out_path + ".tmp";
    FILE* fh = std::fopen(tmp.c_str(), "wb");
    if (fh) {
      std::fwrite(json.data(), 1, json.size(), fh);
      std::fclose(fh);
      std::rename(tmp.c_str(), out_path.c_str());  // atomic replace
    }
    return json.size();
  };

  // Task-level parallelism: a fixed pool of n_workers threads pulls task indices
  // off a shared atomic counter. Each task is fully independent + deterministic
  // from its own seed, so pooling by index is worker-count-invariant.
  std::atomic<int> next_task{0};
  int n_threads = cfg.n_workers > 0 ? cfg.n_workers : 1;
  if (n_threads > n_tasks && n_tasks > 0) n_threads = n_tasks;
  if (n_threads < 1) n_threads = 1;

  auto worker = [&]() {
    for (;;) {
      int ti = next_task.fetch_add(1);
      if (ti >= n_tasks) return;
      const arc::Task& task = tasks[static_cast<size_t>(ti)];

      std::vector<arc::TrainPair> train_pairs;
      train_pairs.reserve(task.train.size());
      for (const arc::TaskPair& p : task.train)
        train_pairs.push_back(arc::TrainPair{p.input, p.output});

      long long seed = arc::task_seed(cfg.seed, task.task_id);
      std::vector<int> prior = arc::observed_output_colors(train_pairs);
      arc::SearchConfig task_sc = sc;
      task_sc.color_prior = prior;
      // ARC_TASK_OTHER=1: the train inputs' colours, most frequent first.
      arc::g_task_colors.clear();
      if (std::getenv("ARC_TASK_OTHER")) {
        std::map<int, int> n;
        for (const auto& tp : train_pairs)
          for (int v : tp.input.cells()) n[v]++;
        std::vector<std::pair<int, int>> byc;
        for (const auto& kv : n) byc.push_back({-kv.second, kv.first});
        std::sort(byc.begin(), byc.end());
        for (const auto& p : byc) arc::g_task_colors.push_back(p.second);
      }

      arc::fitness_curve().clear();
      std::vector<arc::Grid> test_inputs;
      test_inputs.reserve(task.test.size());
      for (const arc::TaskPair& tp : task.test) test_inputs.push_back(tp.input);

      // ARC_ENSEMBLE=K: K independent searches (seeds apart, each on 1/K of the
      // task budget); their answers are voted per test input, attempt_1 counting
      // twice and attempt_2 once (ties: the earlier search). The first search's
      // result stands for the task record; it is marked solved if any search fit.
      static const int ensemble = [] {
        const char* e = std::getenv("ARC_ENSEMBLE");
        return e ? std::max(1, std::atoi(e)) : 1;
      }();
      arc::SearchConfig member_sc = task_sc;
      if (ensemble > 1 && member_sc.max_task_seconds > 0.0) member_sc.max_task_seconds /= ensemble;
      // One scene cache for the task: the ensemble members encode the same train
      // inputs with the same encoder configs, and build_attempts the same test
      // inputs. A hit is a copy of a fresh encode, so only the time changes.
      std::optional<arc::SceneCache> task_cache;
      if (member_sc.encoder_cache) {
        task_cache.emplace(member_sc.encoder_cache_mb > 0
                               ? static_cast<std::size_t>(member_sc.encoder_cache_mb) * 1024ULL * 1024ULL
                               : 1500ULL * 1024ULL * 1024ULL);
        member_sc.scene_cache = &*task_cache;
      }
      arc::TaskResultFull res;
      std::vector<std::vector<arc::Attempts>> member_atts;
      std::vector<arc::TransformFitnessFull> member_fit;
      for (int m = 0; m < ensemble; ++m) {
        long long ms = m == 0 ? seed : arc::task_seed(cfg.seed + 7919LL * m, task.task_id);
        // ARC_ENS_ENC=1: members 1, 5, 9, ... pin connectivity 4 and members
        // 3, 7, 11, ... pin 8, so the ensemble cannot settle on one parse.
        static const bool ens_enc = [] {
          const char* e = std::getenv("ARC_ENS_ENC");
          return e && std::string(e) == "1";
        }();
        arc::g_force_connectivity = ens_enc && m % 4 == 1 ? 4 : ens_enc && m % 4 == 3 ? 8 : 0;
        arc::TaskResultFull r = arc::run_task_full(
            train_pairs, to_decimal(ms), /*seed_is_u64=*/0, /*base_seed_u64=*/0,
            cfg.restarts, member_sc, tail);
        member_atts.push_back(arc::build_attempts(r, train_pairs, test_inputs, member_sc.scene_cache));
        member_fit.push_back(r.primary_fit);
        arc::g_force_connectivity = 0;
        if (m == 0) {
          res = std::move(r);
        } else {
          res.generations_used += r.generations_used;
          res.solved = res.solved || r.solved;
        }
      }
      if (task_cache) {
        res.scene_cache_hits = static_cast<long long>(task_cache->hits());
        res.scene_cache_misses = static_cast<long long>(task_cache->misses());
        res.scene_cache_evictions = static_cast<long long>(task_cache->evictions());
        res.scene_cache_peak_bytes = static_cast<long long>(task_cache->peak_bytes());
      }
      // ARC_MEMBER_DUMP=1: every member's fit and answers go to
      // <progress-dir>/members/<task>.json, for replaying vote policies offline.
      if (std::getenv("ARC_MEMBER_DUMP") && !args.progress_dir.empty()) {
        auto grid = [](const arc::Grid& g) {
          std::string o = "[";
          for (int y = 0; y < g.height(); ++y) {
            o += y ? ",[" : "[";
            for (int x = 0; x < g.width(); ++x) o += (x ? "," : "") + std::to_string(g.cells()[y * g.width() + x]);
            o += "]";
          }
          return o + "]";
        };
        std::string o = "[";
        for (size_t m = 0; m < member_atts.size(); ++m) {
          const arc::TransformFitnessFull& f = member_fit[m];
          o += (m ? ",\n" : "") + std::string("{\"solved\":") + (f.solved ? "true" : "false") +
               ",\"exact_ratio\":" + std::to_string(f.exact_ratio) + ",\"complexity\":" + std::to_string(f.complexity) +
               ",\"attempts\":[";
          for (size_t i = 0; i < member_atts[m].size(); ++i)
            o += (i ? "," : "") + std::string("[") + grid(member_atts[m][i].attempt_1) + "," +
                 grid(member_atts[m][i].attempt_2) + "]";
          o += "]}";
        }
        std::string dir = args.progress_dir + "/members";
        mkdir(dir.c_str(), 0755);
        if (FILE* fp = std::fopen((dir + "/" + task.task_id + ".json").c_str(), "w")) {
          std::fputs((o + "]\n").c_str(), fp);
          std::fclose(fp);
        }
      }
      std::vector<arc::Attempts> attempts_ti = member_atts[0];
      if (ensemble > 1) {
        for (size_t i = 0; i < test_inputs.size(); ++i) {
          std::vector<std::pair<arc::Grid, int>> votes;  // (grid, weight), first-seen order
          auto add = [&](const arc::Grid& g, int w) {
            for (auto& v : votes)
              if (v.first.height() == g.height() && v.first.width() == g.width() && v.first.cells() == g.cells()) {
                v.second += w;
                return;
              }
            votes.push_back({g, w});
          };
          // ARC_VOTE_TYPES=1: when every train pair keeps its size, an answer that
          // makes a kind of change (add / delete / recolour, against the test
          // input's most frequent colour) no train pair makes counts half.
          static const bool vote_types = [] {
            const char* e = std::getenv("ARC_VOTE_TYPES");
            return e && std::string(e) == "1";
          }();
          auto change_types = [](const arc::Grid& a, const arc::Grid& b) {
            std::map<int, int> n;
            for (int v : a.cells()) n[v]++;
            int bg = 0, best = -1;
            for (const auto& kv : n)
              if (kv.second > best) { bg = kv.first; best = kv.second; }
            int t = 0;
            for (size_t k = 0; k < a.cells().size(); ++k) {
              int x = a.cells()[k], y = b.cells()[k];
              if (x != y) t |= x == bg ? 1 : y == bg ? 2 : 4;
            }
            return t;
          };
          auto same_size = [](const arc::Grid& a, const arc::Grid& b) {
            return a.height() == b.height() && a.width() == b.width();
          };
          int train_types = -1;  // -1: some train pair changes size
          if (vote_types) {
            train_types = 0;
            for (const auto& p : train_pairs) {
              if (!same_size(p.input, p.target)) { train_types = -1; break; }
              train_types |= change_types(p.input, p.target);
            }
          }
          auto weight = [&](const arc::Grid& g, int w) {
            if (train_types < 0 || !same_size(test_inputs[i], g)) return 2 * w;
            return (change_types(test_inputs[i], g) & ~train_types) ? w : 2 * w;
          };
          // ARC_VOTE_FIT=K: the votes of a search that fits every train pair count
          // K times (in an offline replay of 81 ConceptARC runs, K=2 got 66 more
          // test inputs right).
          static const int vote_fit = [] {
            const char* e = std::getenv("ARC_VOTE_FIT");
            return e ? std::max(1, std::atoi(e)) : 1;
          }();
          for (size_t m = 0; m < member_atts.size(); ++m) {
            const auto& ma = member_atts[m];
            if (i >= ma.size()) continue;
            int k = member_fit[m].solved ? vote_fit : 1;
            add(ma[i].attempt_1, weight(ma[i].attempt_1, 2) * k);
            add(ma[i].attempt_2, weight(ma[i].attempt_2, 1) * k);
          }
          std::stable_sort(votes.begin(), votes.end(),
                           [](const std::pair<arc::Grid, int>& a, const std::pair<arc::Grid, int>& b) {
                             return a.second > b.second;
                           });
          if (!votes.empty() && i < attempts_ti.size()) {
            attempts_ti[i].attempt_1 = votes[0].first;
            attempts_ti[i].attempt_2 = votes.size() > 1 ? votes[1].first : votes[0].first;
          }
        }
      }

      // Store + persist under the lock so flush_submission() reads a consistent
      // results vector. submission.json is atomically rewritten every task, so a
      // crash mid-run still leaves every completed task's real prediction on disk.
      {
        std::lock_guard<std::mutex> lk(progress_mu);
        results[static_cast<size_t>(ti)] = std::move(attempts_ti);
        task_solved[static_cast<size_t>(ti)] = res.solved ? 1 : 0;
        task_exact_ratio[static_cast<size_t>(ti)] = res.primary_fit.exact_ratio;
        task_n_train[static_cast<size_t>(ti)] = static_cast<int>(train_pairs.size());
        task_gens[static_cast<size_t>(ti)] = res.generations_used;
        total_cache_hits += res.scene_cache_hits;
        total_cache_misses += res.scene_cache_misses;
        total_cache_evictions += res.scene_cache_evictions;
        if (res.scene_cache_peak_bytes > peak_cache_bytes)
          peak_cache_bytes = res.scene_cache_peak_bytes;
        flush_submission();
        if (!args.progress_dir.empty()) {
          if (args.full_results)
            write_task_result_full(args.progress_dir, task.task_id, res,
                                   static_cast<int>(train_pairs.size()),
                                   results[static_cast<size_t>(ti)]);
          else
            write_task_result(args.progress_dir, task.task_id, results[static_cast<size_t>(ti)]);
          int seq = progress_seq.fetch_add(1) + 1;
          std::string pp = args.progress_dir + "/progress.jsonl";
          FILE* pf = std::fopen(pp.c_str(), "ab");
          if (pf) {
            std::fprintf(pf,
                "{\"seq\":%d,\"task_id\":\"%s\",\"solved\":%s,\"n_test\":%zu}\n",
                seq, task.task_id.c_str(), res.solved ? "true" : "false",
                task.test.size());
            std::fclose(pf);
          }
        }
      }
    }
  };

  if (n_threads == 1 || n_tasks == 0) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n_threads));
    for (int i = 0; i < n_threads; ++i) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
  }

  // Final submission write (each task already flushed incrementally under the
  // lock; this also covers the no-tasks case and reports the final size).
  size_t json_size;
  {
    std::lock_guard<std::mutex> lk(progress_mu);
    json_size = flush_submission();
  }
  std::fprintf(stderr, "arc_solver: wrote %s (%zu bytes, %d tasks)\n",
               out_path.c_str(), json_size, n_tasks);

  // End-of-run summary -> stderr (lands in the Kaggle notebook/submission log) and,
  // when --progress-dir is set, summary.json. "train_solved" = tasks whose ALL
  // train pairs are reproduced exactly; "train_pairs_exact" = total exact train
  // pairs across all tasks; "partial_train" = tasks matching some-but-not-all pairs.
  int n_solved = 0, n_partial = 0, seed_solves = 0, evolved_solves = 0;
  long long total_train = 0, total_train_exact = 0;
  double sum_ratio = 0.0;
  for (int ti = 0; ti < n_tasks; ++ti) {
    double r = task_exact_ratio[static_cast<size_t>(ti)];
    int nt = task_n_train[static_cast<size_t>(ti)];
    sum_ratio += r;
    total_train += nt;
    total_train_exact += static_cast<long long>(r * nt + 0.5);
    if (task_solved[static_cast<size_t>(ti)]) {
      n_solved++;
      // gens==0 -> solved by a Phase-A seed; >0 -> found by evolution.
      if (task_gens[static_cast<size_t>(ti)] == 0) seed_solves++;
      else evolved_solves++;
    } else if (r > 0.0) {
      n_partial++;
    }
  }
  double mean_ratio = n_tasks > 0 ? sum_ratio / n_tasks : 0.0;
  double pct = n_tasks > 0 ? 100.0 * n_solved / n_tasks : 0.0;
  // Generation stats: total GA generations run per task (search effort across
  // restarts). median/mean/p90/max show how deep search went + where budget went.
  std::vector<int> gens(task_gens.begin(), task_gens.end());
  std::sort(gens.begin(), gens.end());
  long long gsum = 0;
  for (int g : gens) gsum += g;
  int g_min = gens.empty() ? 0 : gens.front();
  int g_max = gens.empty() ? 0 : gens.back();
  int g_med = gens.empty() ? 0 : gens[gens.size() / 2];
  int g_p90 = gens.empty() ? 0 : gens[(gens.size() * 9) / 10];
  double g_mean = gens.empty() ? 0.0 : static_cast<double>(gsum) / static_cast<double>(gens.size());
  // Encoder scene-cache stats (0 when the cache is off).
  long long cache_total = total_cache_hits + total_cache_misses;
  double cache_hit_rate = cache_total > 0
                              ? static_cast<double>(total_cache_hits) / static_cast<double>(cache_total)
                              : 0.0;
  double cache_peak_mib = static_cast<double>(peak_cache_bytes) / (1024.0 * 1024.0);
  std::fprintf(stderr,
      "arc_solver: SUMMARY tasks=%d train_solved=%d (%.1f%%) [seeds=%d evolved=%d] "
      "partial_train=%d train_pairs_exact=%lld/%lld mean_train_exact_ratio=%.3f "
      "gens[median=%d mean=%.1f p90=%d min=%d max=%d total=%lld] "
      "scene_cache[hits=%lld misses=%lld hit_rate=%.3f evict=%lld peak_mib=%.1f] "
      "seed_enumerator=%s\n",
      n_tasks, n_solved, pct, seed_solves, evolved_solves, n_partial,
      total_train_exact, total_train, mean_ratio,
      g_med, g_mean, g_p90, g_min, g_max, gsum,
      total_cache_hits, total_cache_misses, cache_hit_rate, total_cache_evictions, cache_peak_mib,
      sc.enable_seed_enumerator ? "on" : "off");
  if (!args.progress_dir.empty()) {
    std::string sp = args.progress_dir + "/summary.json";
    FILE* sf = std::fopen(sp.c_str(), "wb");
    if (sf) {
      std::fprintf(sf,
          "{\"tasks\":%d,\"train_solved\":%d,\"seed_solves\":%d,\"evolved_solves\":%d,"
          "\"partial_train\":%d,\"train_pairs_exact\":%lld,\"train_pairs_total\":%lld,"
          "\"mean_train_exact_ratio\":%.6f,\"gens_median\":%d,\"gens_mean\":%.3f,"
          "\"gens_p90\":%d,\"gens_min\":%d,\"gens_max\":%d,\"gens_total\":%lld,"
          "\"scene_cache_hits\":%lld,\"scene_cache_misses\":%lld,\"scene_cache_hit_rate\":%.6f,"
          "\"scene_cache_evictions\":%lld,\"scene_cache_peak_mib\":%.3f,"
          "\"seed_enumerator\":%s}\n",
          n_tasks, n_solved, seed_solves, evolved_solves, n_partial,
          total_train_exact, total_train, mean_ratio,
          g_med, g_mean, g_p90, g_min, g_max, gsum,
          total_cache_hits, total_cache_misses, cache_hit_rate, total_cache_evictions, cache_peak_mib,
          sc.enable_seed_enumerator ? "true" : "false");
      std::fclose(sf);
    }
  }
  return 0;
}
