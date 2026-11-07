// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Runtime ArcConfig layered resolution (file, then environment, then CLI) +
// per-task seed. See include/arc/config_load.hpp.

#include "arc/config_load.hpp"

#include <cctype>
#include <cstdlib>

#include "arc/pyrepr.hpp"
#include "arc/sha1.hpp"

namespace arc {

namespace {

// Boolean spellings, case-insensitive: 1/true/yes/on and 0/false/no/off/empty.
bool parse_bool(const std::string& raw, bool* ok) {
  std::string s;
  for (char c : raw) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  // strip surrounding whitespace
  size_t a = s.find_first_not_of(" \t");
  size_t b = s.find_last_not_of(" \t");
  s = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
  if (s == "1" || s == "true" || s == "yes" || s == "on") {
    *ok = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "no" || s == "off" || s.empty()) {
    *ok = true;
    return false;
  }
  *ok = false;  // unparseable: the caller keeps the current value and warns
  return false;
}

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t");
  return s.substr(a, b - a + 1);
}

// Apply one raw (string) value to the named field, coerced to the field's type
// (int, float, bool spelling, or the string as is). Unknown field names return
// false (caller warns).
bool apply_field(ArcConfig& cfg, const std::string& name, const std::string& raw,
                 std::vector<std::string>* warnings) {
  std::string v = strip(raw);
  auto warn_bool = [&](const std::string& n) {
    if (warnings) warnings->push_back("config " + n + ": cannot parse bool from " + raw);
  };

  // --- str fields ---
  if (name == "data_dir") { cfg.data_dir = v; return true; }
  if (name == "kaggle") { cfg.kaggle = v; return true; }
  if (name == "split") { cfg.split = v; return true; }
  if (name == "output_root") { cfg.output_root = v; return true; }
  if (name == "fitness_target") { cfg.fitness_target = v; return true; }
  if (name == "fitness_distance") { cfg.fitness_distance = v; return true; }
  if (name == "selection") { cfg.selection = v; return true; }
  if (name == "lexicase_cases") { cfg.lexicase_cases = v; return true; }
  if (name == "view_continue") { cfg.view_continue = v; return true; }
  if (name == "macro_library") { cfg.macro_library = v; return true; }
  if (name == "macro_seed_library") { cfg.macro_seed_library = v; return true; }

  // --- int fields ---
  if (name == "max_tasks") { cfg.max_tasks = std::atoi(v.c_str()); return true; }
  if (name == "task_offset") { cfg.task_offset = std::atoi(v.c_str()); return true; }
  if (name == "population") { cfg.population = std::atoi(v.c_str()); return true; }
  if (name == "generations") { cfg.generations = std::atoi(v.c_str()); return true; }
  if (name == "restarts") { cfg.restarts = std::atoi(v.c_str()); return true; }
  if (name == "max_rules") { cfg.max_rules = std::atoi(v.c_str()); return true; }
  if (name == "seed") { cfg.seed = std::atoi(v.c_str()); return true; }
  if (name == "n_workers") { cfg.n_workers = std::atoi(v.c_str()); return true; }
  if (name == "lex_complement") { cfg.lex_complement = std::atoi(v.c_str()); return true; }
  if (name == "lex_champions") { cfg.lex_champions = std::atoi(v.c_str()); return true; }
  if (name == "rank_specificity") { cfg.rank_specificity = std::atoi(v.c_str()); return true; }
  if (name == "stagnation_inject") { cfg.stagnation_inject = std::atoi(v.c_str()); return true; }
  if (name == "stagnation_restart") { cfg.stagnation_restart = std::atoi(v.c_str()); return true; }
  if (name == "occam_continue") { cfg.occam_continue = std::atoi(v.c_str()); return true; }
  if (name == "tabu_restart") { cfg.tabu_restart = std::atoi(v.c_str()); return true; }
  if (name == "encoder_cache_mb") { cfg.encoder_cache_mb = std::atoi(v.c_str()); return true; }
  if (name == "macro_seed_topn") { cfg.macro_seed_topn = std::atoi(v.c_str()); return true; }

  // --- float fields ---
  if (name == "max_task_seconds") { cfg.max_task_seconds = std::strtod(v.c_str(), nullptr); return true; }
  if (name == "diversity_gate") { cfg.diversity_gate = std::strtod(v.c_str(), nullptr); return true; }
  if (name == "macro_rate") { cfg.macro_rate = std::strtod(v.c_str(), nullptr); return true; }
  if (name == "macro_mutate_rate") { cfg.macro_mutate_rate = std::strtod(v.c_str(), nullptr); return true; }

  // --- bool fields ---
  if (name == "phase_a_shortcut") {
    bool ok = false, b = parse_bool(v, &ok);
    if (ok) cfg.phase_a_shortcut = b; else warn_bool(name);
    return true;
  }
  if (name == "verify_unbounded_phase_a") {
    bool ok = false, b = parse_bool(v, &ok);
    if (ok) cfg.verify_unbounded_phase_a = b; else warn_bool(name);
    return true;
  }
  if (name == "encoder_cache") {
    bool ok = false, b = parse_bool(v, &ok);
    if (ok) cfg.encoder_cache = b; else warn_bool(name);
    return true;
  }

  return false;  // unknown field
}

}  // namespace

std::map<std::string, std::string> parse_conf(const std::string& text) {
  std::map<std::string, std::string> out;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t nl = text.find('\n', pos);
    std::string line = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;

    // strip inline comment
    size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = strip(line);
    if (line.empty()) continue;

    size_t sep = line.find('=');
    char sepc = '=';
    if (sep == std::string::npos) {
      sep = line.find(':');
      sepc = ':';
    }
    if (sep == std::string::npos) continue;
    (void)sepc;
    std::string key = strip(line.substr(0, sep));
    std::string val = strip(line.substr(sep + 1));
    // lowercase + '-' -> '_'
    for (char& c : key) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (c == '-') c = '_';
    }
    // strip surrounding quotes
    if (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') ||
                            (val.front() == '\'' && val.back() == '\''))) {
      val = val.substr(1, val.size() - 2);
    }
    out[key] = val;
  }
  return out;
}

const std::vector<std::pair<std::string, std::string>>& config_env_map() {
  // Config field -> environment variable, in field declaration order.
  static const std::vector<std::pair<std::string, std::string>> kMap = {
      {"selection", "ARC_SELECTION"},
      {"lexicase_cases", "ARC_LEXICASE_CASES"},
      {"lex_complement", "ARC_LEX_COMPLEMENT"},
      {"lex_champions", "ARC_LEX_CHAMPIONS"},
      {"rank_specificity", "ARC_RANK_SPECIFICITY"},
      {"stagnation_inject", "ARC_STAGNATION_INJECT"},
      {"stagnation_restart", "ARC_STAGNATION_RESTART"},
      {"diversity_gate", "ARC_DIVERSITY_GATE"},
      {"occam_continue", "ARC_OCCAM_CONTINUE"},
      {"tabu_restart", "ARC_TABU_RESTART"},
      {"view_continue", "ARC_VIEW_CONTINUE"},
      {"encoder_cache", "ARC_ENCODER_CACHE"},
      {"encoder_cache_mb", "ARC_ENCODER_CACHE_MB"},
      {"macro_library", "ARC_MACRO_LIBRARY"},
      {"macro_rate", "ARC_MACRO_RATE"},
      {"macro_seed_topn", "ARC_MACRO_SEED_TOPN"},
      {"macro_seed_library", "ARC_MACRO_SEED_LIBRARY"},
      {"macro_mutate_rate", "ARC_MACRO_MUTATE_RATE"},
      {"verify_unbounded_phase_a", "ARC_VERIFY_UNBOUNDED_PHASE_A"},
  };
  return kMap;
}

ArcConfig config_load(const std::string& conf_path,
                      const std::map<std::string, std::string>& env,
                      const std::map<std::string, std::string>& cli,
                      std::vector<std::string>* warnings) {
  ArcConfig cfg;

  // 1. file
  if (!conf_path.empty()) {
    FILE* fh = std::fopen(conf_path.c_str(), "rb");
    if (fh) {
      std::string text;
      char buf[4096];
      size_t got;
      while ((got = std::fread(buf, 1, sizeof(buf), fh)) > 0)
        text.append(buf, got);
      std::fclose(fh);
      std::map<std::string, std::string> raw = parse_conf(text);
      for (const auto& kv : raw) {
        if (!apply_field(cfg, kv.first, kv.second, warnings)) {
          if (warnings) warnings->push_back("config: ignoring unknown key " + kv.first);
        }
      }
    } else if (warnings) {
      warnings->push_back("config: failed to read " + conf_path);
    }
  }

  // 2. environment (only the ENV_MAP-bridged fields)
  for (const auto& fe : config_env_map()) {
    auto it = env.find(fe.second);
    if (it != env.end()) apply_field(cfg, fe.first, it->second, warnings);
  }

  // 3. CLI overrides (field names, highest precedence)
  for (const auto& kv : cli) {
    if (!apply_field(cfg, kv.first, kv.second, warnings)) {
      if (warnings) warnings->push_back("config: ignoring unknown CLI key " + kv.first);
    }
  }

  return cfg;
}

std::string config_to_json(const ArcConfig& cfg) {
  auto q = [](const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') out.push_back('\\');
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  };
  std::string o = "{";
  o += q("data_dir") + ":" + q(cfg.data_dir) + ",";
  o += q("kaggle") + ":" + q(cfg.kaggle) + ",";
  o += q("split") + ":" + q(cfg.split) + ",";
  o += q("output_root") + ":" + q(cfg.output_root) + ",";
  o += q("max_tasks") + ":" + std::to_string(cfg.max_tasks) + ",";
  o += q("task_offset") + ":" + std::to_string(cfg.task_offset) + ",";
  o += q("population") + ":" + std::to_string(cfg.population) + ",";
  o += q("generations") + ":" + std::to_string(cfg.generations) + ",";
  o += q("restarts") + ":" + std::to_string(cfg.restarts) + ",";
  o += q("max_rules") + ":" + std::to_string(cfg.max_rules) + ",";
  o += q("seed") + ":" + std::to_string(cfg.seed) + ",";
  o += q("max_task_seconds") + ":" + py_repr_float(cfg.max_task_seconds) + ",";
  o += q("n_workers") + ":" + std::to_string(cfg.n_workers) + ",";
  o += q("fitness_target") + ":" + q(cfg.fitness_target) + ",";
  o += q("fitness_distance") + ":" + q(cfg.fitness_distance) + ",";
  o += q("phase_a_shortcut") + ":" + (cfg.phase_a_shortcut ? "true" : "false") + ",";
  o += q("verify_unbounded_phase_a") + ":" + (cfg.verify_unbounded_phase_a ? "true" : "false") + ",";
  o += q("selection") + ":" + q(cfg.selection) + ",";
  o += q("lexicase_cases") + ":" + q(cfg.lexicase_cases) + ",";
  o += q("lex_complement") + ":" + std::to_string(cfg.lex_complement) + ",";
  o += q("lex_champions") + ":" + std::to_string(cfg.lex_champions) + ",";
  o += q("rank_specificity") + ":" + std::to_string(cfg.rank_specificity) + ",";
  o += q("stagnation_inject") + ":" + std::to_string(cfg.stagnation_inject) + ",";
  o += q("stagnation_restart") + ":" + std::to_string(cfg.stagnation_restart) + ",";
  o += q("diversity_gate") + ":" + py_repr_float(cfg.diversity_gate) + ",";
  o += q("occam_continue") + ":" + std::to_string(cfg.occam_continue) + ",";
  o += q("tabu_restart") + ":" + std::to_string(cfg.tabu_restart) + ",";
  o += q("view_continue") + ":" + q(cfg.view_continue) + ",";
  o += q("encoder_cache") + ":" + (cfg.encoder_cache ? "true" : "false") + ",";
  o += q("encoder_cache_mb") + ":" + std::to_string(cfg.encoder_cache_mb) + ",";
  o += q("macro_library") + ":" + q(cfg.macro_library) + ",";
  o += q("macro_rate") + ":" + py_repr_float(cfg.macro_rate) + ",";
  o += q("macro_seed_topn") + ":" + std::to_string(cfg.macro_seed_topn) + ",";
  o += q("macro_seed_library") + ":" + q(cfg.macro_seed_library) + ",";
  o += q("macro_mutate_rate") + ":" + py_repr_float(cfg.macro_mutate_rate);
  o += "}";
  return o;
}

long long task_seed(long long base_seed, const std::string& task_id) {
  uint8_t digest[20];
  sha1(reinterpret_cast<const uint8_t*>(task_id.data()), task_id.size(), digest);
  // The first 4 digest bytes as a big-endian integer.
  uint32_t h = (static_cast<uint32_t>(digest[0]) << 24) |
               (static_cast<uint32_t>(digest[1]) << 16) |
               (static_cast<uint32_t>(digest[2]) << 8) |
               (static_cast<uint32_t>(digest[3]));
  return (base_seed ^ static_cast<long long>(h)) & 0x7FFFFFFFLL;
}

}  // namespace arc
