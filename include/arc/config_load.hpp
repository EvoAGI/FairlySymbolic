// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_CONFIG_LOAD_HPP
#define ARC_CONFIG_LOAD_HPP

// The RUNTIME config (ArcConfig), its layered resolution and the per-task seed
// derivation.
//
// This is DISTINCT from the encoder GraphConfig (arc/config.hpp). ArcConfig holds
// the run-level knobs the solver reads: data source, GA params
// (population/generations/restarts/max_rules/seed/max_task_seconds/n_workers),
// selection/lexicase settings, continuation toggles, and the macro/* fields.
//
// Resolution precedence (low -> high):
//     struct defaults  <  arc_config.conf  <  ARC_* env vars  <  CLI overrides
//
// The conf file is a simple commented `key = value` format (`#` starts a
// comment; `key: value` also accepted; unknown keys warned + kept out). Only
// the fields in config_env_map() (behaviour/encoder/macro flags) are read from
// ARC_* env vars.

#include <map>
#include <string>
#include <vector>

namespace arc {

struct ArcConfig {
  // --- data / run source ---
  std::string data_dir = "data/training";
  std::string kaggle = "";
  std::string split = "evaluation";
  std::string output_root = "runs";
  int max_tasks = 0;  // 0 = all tasks in the split
  int task_offset = 0;

  // --- search / GA run params ---
  int population = 40;
  int generations = 60;
  int restarts = 5;
  int max_rules = 5;
  int seed = 47;
  double max_task_seconds = 3840.0;
  int n_workers = 6;
  std::string fitness_target = "train";
  std::string fitness_distance = "graph";
  bool phase_a_shortcut = true;

  // --- verification mode (set by --verify; default off) ---
  bool verify_unbounded_phase_a = false;

  // --- selection / GA behaviour ---
  std::string selection = "lexicase";       // tournament | lexicase | lexicase_inject
  std::string lexicase_cases = "fine2";     // coarse | fine | fine2
  int lex_complement = 0;
  int lex_champions = 0;
  int rank_specificity = 0;
  int stagnation_inject = 2;
  int stagnation_restart = 4;
  double diversity_gate = 0.0;

  // --- post-solve continuations ---
  int occam_continue = 1;
  int tabu_restart = 0;
  std::string view_continue = "1";          // "0" | "1" | comma list of view names

  // --- encoder scene cache ---
  bool encoder_cache = true;
  int encoder_cache_mb = 1500;

  // --- macro library (default off) ---
  std::string macro_library = "";
  double macro_rate = 0.0;
  int macro_seed_topn = 0;
  std::string macro_seed_library = "";
  double macro_mutate_rate = -1.0;
};

// Resolve an ArcConfig honouring precedence default<conf<env<cli.
//   conf_path : path to arc_config.conf (or "" to skip the file layer).
//   env       : the process environment, as a name->value map (only the
//               ARC_* vars in config_env_map() are consulted); pass an empty
//               map to skip the env layer.
//   cli       : field-name -> raw string overrides (highest precedence). Keys
//               use the ArcConfig field names (e.g. "population", "seed").
// Unknown conf/cli keys are ignored (a note is appended to *warnings if given).
ArcConfig config_load(const std::string& conf_path,
                      const std::map<std::string, std::string>& env,
                      const std::map<std::string, std::string>& cli,
                      std::vector<std::string>* warnings = nullptr);

// Parse a `key = value` conf file body into raw string values (`#` comment,
// blank-line skip, `key: value` accepted, keys lowercased with '-' -> '_').
// Values are stripped of surrounding quotes.
std::map<std::string, std::string> parse_conf(const std::string& text);

// The config-field -> ARC_* env var map. Exposed so the binary can
// build the env layer.
const std::vector<std::pair<std::string, std::string>>& config_env_map();

// Serialize a resolved ArcConfig to canonical JSON (field-name keys in
// declaration order; bools as JSON true/false; floats via repr).
std::string config_to_json(const ArcConfig& cfg);

// Per-task seed: h = the first 4 bytes of sha1(task_id as UTF-8), read as a
// big-endian integer; returns (base_seed ^ h) & 0x7fffffff.
long long task_seed(long long base_seed, const std::string& task_id);

}  // namespace arc

#endif  // ARC_CONFIG_LOAD_HPP
