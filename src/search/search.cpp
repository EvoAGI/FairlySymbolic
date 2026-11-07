// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// GA generation loop and its helpers. See search.hpp for the contract.
//
// The RNG draw order is fixed, so a seed always gives the same run.

#include "arc/search.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <list>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

#include "arc/fitness.hpp"
#include "arc/pyrepr.hpp"
#include "arc/scene_cache.hpp"
#include "arc/seed_enum.hpp"
#include "arc/delta.hpp"
#include "arc/concepts.hpp"
#include "arc/encode.hpp"
#include "arc/object_attrs.hpp"

namespace arc {

// Per-thread fitness curve (ARC_FITLOG=1), see search.hpp.
std::vector<std::array<double, 3>>& fitness_curve() {
  static thread_local std::vector<std::array<double, 3>> curve;
  return curve;
}

// ---------------------------------------------------------------------------
// evaluate_genome. Multi-pair aggregation over the per-pair render, the
// pixel-exact solved gate, mdl_complexity / rule count, specificity, and the
// lexicase case vector.
// ---------------------------------------------------------------------------

namespace {

// ARC_MDL_PRIOR=1: among programs with as many rules, a rule whose action is
// not a concept op costs 0.1 more, and so does a turn or flip of the whole
// output on top of rules (in ConceptARC runs, one-rule train solvers got 92 % of
// their tests right with a concept op, 73 % with another action, 43 % with a
// turn or flip; the turn or flip mostly made a mirror action look like a
// completion). A train solver that carries this cost does
// not end the search early: the Phase-A shortcut is skipped and the solved
// refinement waits four times longer for a cheaper solver.
bool mdl_prior() {
  static const bool on = [] {
    const char* e = std::getenv("ARC_MDL_PRIOR");
    return e && std::string(e) == "1";
  }();
  return on;
}

double mdl_prior_cost(const TransformGenome& g) {
  if (!mdl_prior()) return 0.0;
  double c = 0.0;
  for (const auto& rule : g.program.rules)
    if (!find_concept_op(rule.action.kind)) c += 0.1;
  GlobalTransform t = g.program.global_transform;
  if (!g.program.rules.empty() && t >= GlobalTransform::ROT90 && t <= GlobalTransform::ANTI_TRANSPOSE) c += 0.1;
  return c;
}

// Selector args the MDL counts: a merged kind's variant arg is part of its
// name, so a merged selector costs what the folded kind cost.
int counted_selector_args(const DslSelector& s) {
  int n = static_cast<int>(s.args.size());
  const std::string& key = selector_variant_key(s.kind);
  if (!key.empty() && args_get(s.args, key)) --n;
  return n;
}

// The same for a merged core action.
int counted_action_args(const DslAction& a) {
  int n = static_cast<int>(a.args.size());
  const std::string& key = action_variant_key(a.kind);
  if (!key.empty() && args_get(a.args, key)) --n;
  return n;
}

// mdl_complexity: n_rules + 0.02 * total arg count (selector + action + guard).
double mdl_complexity(const TransformGenome& g) {
  constexpr double MDL_ARG_PENALTY = 0.02;
  int n_rules = static_cast<int>(g.program.rules.size());
  int arg_count = 0;
  for (const auto& rule : g.program.rules) {
    arg_count += counted_selector_args(rule.selector);
    arg_count += counted_action_args(rule.action);
    for (const auto& guard : rule.guards) arg_count += static_cast<int>(guard.args.size());
  }
  double c = static_cast<double>(n_rules) + MDL_ARG_PENALTY * static_cast<double>(arg_count);
  // ARC_MDL_STRICT=1: a guard and a repeated action each cost half a rule (the
  // programs that carry them fit the train pairs and miss the test far more often).
  static const bool strict = [] {
    const char* e = std::getenv("ARC_MDL_STRICT");
    return e && std::string(e) == "1";
  }();
  if (strict) {
    // A merged core action's variants count as the kinds they replace.
    std::set<std::string> kinds;
    int guards = 0;
    for (const auto& rule : g.program.rules) {
      guards += static_cast<int>(rule.guards.size());
      kinds.insert(to_string(rule.action.kind) + ":" + action_variant(rule.action));
    }
    c += 0.5 * guards + 0.5 * (n_rules - static_cast<int>(kinds.size()));
  }
  return c + mdl_prior_cost(g);
}

// specificity: selector + guard arg counts (action args excluded).
int specificity(const TransformGenome& g) {
  int n = 0;
  for (const auto& rule : g.program.rules) {
    n += counted_selector_args(rule.selector);
    for (const auto& guard : rule.guards) n += static_cast<int>(guard.args.size());
  }
  return n;
}

}  // namespace

TransformFitnessFull evaluate_genome(const TransformGenome& genome,
                                     const std::vector<TrainPair>& train_pairs,
                                     const std::string& distance_mode,
                                     bool collect_cases,
                                     const std::string& case_mode,
                                     SceneCache* scene_cache) {
  std::vector<double> soft_scores;
  std::vector<bool> exact_flags;
  std::vector<double> case_errors;
  soft_scores.reserve(train_pairs.size());
  exact_flags.reserve(train_pairs.size());

  for (const auto& tp : train_pairs) {
    PairEval pe = fitness_pair_eval(tp.input, tp.target, genome.encoder_config,
                                    genome.decoder, genome.program, distance_mode,
                                    collect_cases, case_mode, scene_cache);
    soft_scores.push_back(pe.soft);
    exact_flags.push_back(pe.exact);
    if (collect_cases) {
      for (double v : pe.cases) case_errors.push_back(v);
    }
  }

  TransformFitnessFull out;
  // n = number of exact flags, or 1 for an empty train set.
  int n = static_cast<int>(exact_flags.size());
  if (n == 0) n = 1;
  // mean_soft / max_soft: the mean (a plain left-to-right sum / n) and the max
  // of the soft scores; 1.0 when there are none.
  double mean_soft = 1.0;
  double max_soft = 1.0;
  if (!soft_scores.empty()) {
    double s = 0.0;
    for (double v : soft_scores) s += v;
    mean_soft = s / static_cast<double>(soft_scores.size());
    max_soft = soft_scores[0];
    for (size_t i = 1; i < soft_scores.size(); ++i)
      if (soft_scores[i] > max_soft) max_soft = soft_scores[i];
  }
  int n_exact = 0;
  for (bool e : exact_flags) if (e) ++n_exact;
  double exact_ratio = static_cast<double>(n_exact) / static_cast<double>(n);
  bool solved = !exact_flags.empty();
  for (bool e : exact_flags) if (!e) solved = false;

  out.pixel_mismatch_mean = mean_soft;
  out.max_pair_mismatch = max_soft;
  out.complexity = solved ? mdl_complexity(genome)
                          : static_cast<double>(genome.program.rules.size());
  out.score = mean_soft;
  out.exact_ratio = exact_ratio;
  out.solved = solved;
  out.spec = specificity(genome);
  out.loo_fail = 0;
  out.case_errors = std::move(case_errors);
  return out;
}

// ---------------------------------------------------------------------------
// The generation loop internals.
// ---------------------------------------------------------------------------

namespace {

// score_population: score each genome (fitness cache keyed by the ga-dedup
// tuple (encoder_sig, hash, bg, po, ov, pp)), then STABLE-sort
// rank-best-first. Without a score deadline every genome is scored.
std::vector<ScoredGenome> score_population(
    const std::vector<TransformGenome>& pop, const std::vector<TrainPair>& train_pairs,
    const SearchConfig& cfg, std::unordered_map<std::string, TransformFitnessFull>& cache,
    std::optional<std::chrono::steady_clock::time_point> score_deadline = std::nullopt) {
  bool collect_cases = (cfg.selection_mode == "lexicase" || cfg.selection_mode == "lexicase_inject");
  std::vector<ScoredGenome> scored;
  scored.reserve(pop.size());
  for (const auto& g : pop) {
    // Phase-A prefix break: once score_deadline passes, score only the prefix
    // ranked so far. score_deadline is set ONLY for the deadline-bounded
    // Phase-A scoring call; nullopt (every other call) scores the whole
    // population.
    if (score_deadline.has_value() && std::chrono::steady_clock::now() >= score_deadline.value())
      break;
    std::string key = dedup_ga_key(g);
    auto it = cache.find(key);
    TransformFitnessFull fit;
    if (it != cache.end()) {
      fit = it->second;
    } else {
      fit = evaluate_genome(g, train_pairs, cfg.distance_mode, collect_cases,
                            cfg.lexicase_case_mode, cfg.scene_cache);
      cache.emplace(key, fit);
    }
    scored.push_back(ScoredGenome{g, std::move(fit)});
  }
  // Stable sort by fitness rank: ties keep their population order.
  std::stable_sort(scored.begin(), scored.end(),
                   [&](const ScoredGenome& a, const ScoredGenome& b) {
                     return rank_less(fitness_rank(a.fit.scalars(), cfg.rank_specificity),
                                      fitness_rank(b.fit.scalars(), cfg.rank_specificity));
                   });
  return scored;
}

bool is_better_fit(const TransformFitnessFull& cand, const TransformFitnessFull& best,
                   bool rank_specificity) {
  return rank_less(fitness_rank(cand.scalars(), rank_specificity),
                   fitness_rank(best.scalars(), rank_specificity));
}

// decoder_sweep: try every (bg x overlap x postprocess) decoder combo
// (paint_order fixed at the genome's current value), keep the rank-best.
// Deterministic -- NO RNG. Phase A runs it on its elites before they seed the
// loop. The score cache is keyed by the ga-dedup tuple, so a re-scored variant
// reuses its cached fitness.
const char* const kBgPolicies[] = {
    "scene_background_or_zero", "zero", "most_frequent_input",
    "most_frequent_predicted", "input_copy"};
const char* const kOverlapPolicies[] = {"overwrite", "keep_first"};
const char* const kPostprocessPolicies[] = {"none", "crop_foreground", "keep_lines"};

ScoredGenome decoder_sweep(const ScoredGenome& start, const std::vector<TrainPair>& train_pairs,
                           const SearchConfig& cfg,
                           std::unordered_map<std::string, TransformFitnessFull>& cache) {
  bool collect_cases =
      (cfg.selection_mode == "lexicase" || cfg.selection_mode == "lexicase_inject");
  ScoredGenome best = start;
  const DecoderConfig& cur = start.genome.decoder;
  for (const char* bg : kBgPolicies) {
    for (const char* overlap : kOverlapPolicies) {
      for (const char* pp : kPostprocessPolicies) {
        if (bg == cur.background_policy && overlap == cur.overlap_policy &&
            pp == cur.postprocess)
          continue;
        if (past_deadline(cfg) || past_post_deadline(cfg)) return best;  // DEPLOY: a slow grid can use up the time budget
        TransformGenome cand = start.genome;  // same encoder + program
        cand.decoder.background_policy = bg;
        cand.decoder.paint_order = cur.paint_order;
        cand.decoder.overlap_policy = overlap;
        cand.decoder.postprocess = pp;
        std::string key = dedup_ga_key(cand);
        TransformFitnessFull fit;
        auto it = cache.find(key);
        if (it != cache.end()) {
          fit = it->second;
        } else {
          fit = evaluate_genome(cand, train_pairs, cfg.distance_mode, collect_cases,
                                cfg.lexicase_case_mode, cfg.scene_cache);
          cache.emplace(key, fit);
        }
        if (is_better_fit(fit, best.fit, cfg.rank_specificity)) {
          best.genome = cand;
          best.fit = fit;
        }
      }
    }
  }
  return best;
}

// The scored population's case vectors as an error matrix, for epsilon-lexicase
// selection (select.hpp).
std::vector<std::vector<double>> collect_case_matrix(const std::vector<ScoredGenome>& scored) {
  std::vector<std::vector<double>> m;
  m.reserve(scored.size());
  for (const auto& sg : scored) m.push_back(sg.fit.case_errors);
  return m;
}

// ---------------------------------------------------------------------------
// Local refinement passes after the generation loop. Each pass scores through
// the SAME cached evaluate_genome the loop uses (dedup_ga_key cache), and
// consumes the SAME rng the whole attempt has been drawing from. Without a
// deadline every pass runs its gate and goes to its fixed step/stall cap, not
// a wall-clock fraction.
// ---------------------------------------------------------------------------

// score_fn helper: cached single-genome fitness for the refinement passes
// (dedup_ga_key cache).
TransformFitnessFull refine_score(const TransformGenome& g,
                                  const std::vector<TrainPair>& train_pairs,
                                  const SearchConfig& cfg,
                                  std::unordered_map<std::string, TransformFitnessFull>& cache) {
  bool collect_cases =
      (cfg.selection_mode == "lexicase" || cfg.selection_mode == "lexicase_inject");
  std::string key = dedup_ga_key(g);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  TransformFitnessFull fit = evaluate_genome(g, train_pairs, cfg.distance_mode,
                                             collect_cases, cfg.lexicase_case_mode,
                                             cfg.scene_cache);
  cache.emplace(key, fit);
  return fit;
}

// hill_climb (single-parent greedy climb). mutate_fn is either codec.mutate
// (structural) or the delta_guided_mutate wrapper (delta). Caps: structural
// (48/12), delta (24/8). With no deadline, no deadline guard fires. Returns
// the best genome/fit AFTER the climb (tries discarded).
//
// `commit_epoch` (optional): incremented every time a strictly-better child is
// committed as the new best. The delta wrapper uses it to detect a commit and
// refresh its hint (the best is a fixed reference, so a commit is signalled
// through the epoch).
template <typename MutateFn>
void hill_climb(TransformGenome& best_genome, TransformFitnessFull& best_fit,
                MutateFn&& mutate_fn, const std::vector<TrainPair>& train_pairs,
                const SearchConfig& cfg,
                std::unordered_map<std::string, TransformFitnessFull>& cache, ArcRandom& rng,
                int max_steps, int stall_limit, int* commit_epoch = nullptr) {
  int stall = 0;
  for (int step = 0; step < max_steps; ++step) {
    // deadline_exceeded() dormant; ARC_POST_GRACE bounds the climb.
    if (best_fit.solved || past_post_deadline(cfg)) break;
    TransformGenome cand = mutate_fn(best_genome, rng);
    TransformFitnessFull cand_fit = refine_score(cand, train_pairs, cfg, cache);
    if (is_better_fit(cand_fit, best_fit, cfg.rank_specificity)) {
      best_genome = std::move(cand);
      best_fit = cand_fit;
      stall = 0;
      if (commit_epoch) ++(*commit_epoch);
    } else {
      ++stall;
      if (stall >= stall_limit) break;
    }
  }
}

// arg_candidates: the values the numeric-arg sweep tries for `key`. Returns an
// empty list for any other key (symbolic args left to mutate).
std::vector<int> arg_candidates(const std::string& key) {
  if (key == "color" || key == "target_color" || key == "new_color") {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};  // colours
  }
  if (key == "mass" || key == "threshold" || key == "min_mass" || key == "max_mass" ||
      key == "count") {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // small ints
  }
  if (key == "dy" || key == "dx" || key == "offset_y" || key == "offset_x") {
    return {-3, -2, -1, 0, 1, 2, 3};  // translation offsets
  }
  return {};
}

// True iff DslValue `v` equals the int candidate `val`: a missing key
// (nullptr) or a string never matches; an int matches exactly; a float matches
// iff it is integral and equal (3 == 3.0); a bool counts as 0 / 1.
bool arg_equals_int(const DslValue* v, int val) {
  if (v == nullptr) return false;
  switch (v->tag) {
    case DslValue::Tag::Int:
      return v->i == static_cast<long long>(val);
    case DslValue::Tag::Float:
      return v->f == static_cast<double>(val);
    case DslValue::Tag::Bool:
      // true == 1, false == 0.
      return (v->b ? 1 : 0) == val;
    default:
      return false;
  }
}

// Set (replace-or-append preserving insertion order, ArgMap semantics) a named
// int arg on a DslArgs copy.
DslArgs set_int_arg(const DslArgs& args, const std::string& key, int val) {
  DslArgs out = args;
  DslValue dv;
  dv.tag = DslValue::Tag::Int;
  dv.i = val;
  for (auto& kv : out) {
    if (kv.first == key) {
      kv.second = dv;
      return out;
    }
  }
  out.emplace_back(key, dv);
  return out;
}

// refine_numeric_args. Sweep every numeric arg of every rule (selector args
// then action args, rule order), keep the rank-best variant. Deterministic (no
// rng). The clone rebuilds only the swept rule (Program + TransformGenome).
void refine_numeric_args(TransformGenome& best_genome, TransformFitnessFull& best_fit,
                         const std::vector<TrainPair>& train_pairs, const SearchConfig& cfg,
                         std::unordered_map<std::string, TransformFitnessFull>& cache) {
  for (size_t rule_idx = 0; rule_idx < best_genome.program.rules.size(); ++rule_idx) {
    // Selector args (iterate the current best_genome rule's keys in order).
    {
      std::vector<std::string> keys;
      for (const auto& kv : best_genome.program.rules[rule_idx].selector.args)
        keys.push_back(kv.first);
      for (const std::string& key : keys) {
        std::vector<int> cands = arg_candidates(key);
        // Capture the current value BY VALUE: an improving candidate below
        // move-assigns best_genome (freeing this args buffer), so a pointer into
        // it would dangle across the next iteration (heap-use-after-free).
        const DslValue* cur_src =
            args_get(best_genome.program.rules[rule_idx].selector.args, key);
        DslValue cur_copy;
        if (cur_src) cur_copy = *cur_src;
        const DslValue* cur = cur_src ? &cur_copy : nullptr;
        for (int val : cands) {
          if (past_post_deadline(cfg)) return;
          if (arg_equals_int(cur, val)) continue;
          TransformGenome cand = best_genome;
          cand.program.rules[rule_idx].selector.args =
              set_int_arg(best_genome.program.rules[rule_idx].selector.args, key, val);
          TransformFitnessFull cand_fit = refine_score(cand, train_pairs, cfg, cache);
          if (is_better_fit(cand_fit, best_fit, cfg.rank_specificity)) {
            best_genome = std::move(cand);
            best_fit = cand_fit;
          }
        }
      }
    }
    // Action args.
    {
      std::vector<std::string> keys;
      for (const auto& kv : best_genome.program.rules[rule_idx].action.args)
        keys.push_back(kv.first);
      for (const std::string& key : keys) {
        std::vector<int> cands = arg_candidates(key);
        // Capture BY VALUE (see the selector loop above): the improving-candidate
        // move-assign below frees this args buffer, so a pointer would dangle.
        const DslValue* cur_src =
            args_get(best_genome.program.rules[rule_idx].action.args, key);
        DslValue cur_copy;
        if (cur_src) cur_copy = *cur_src;
        const DslValue* cur = cur_src ? &cur_copy : nullptr;
        for (int val : cands) {
          if (past_post_deadline(cfg)) return;
          if (arg_equals_int(cur, val)) continue;
          TransformGenome cand = best_genome;
          cand.program.rules[rule_idx].action.args =
              set_int_arg(best_genome.program.rules[rule_idx].action.args, key, val);
          TransformFitnessFull cand_fit = refine_score(cand, train_pairs, cfg, cache);
          if (is_better_fit(cand_fit, best_fit, cfg.rank_specificity)) {
            best_genome = std::move(cand);
            best_fit = cand_fit;
          }
        }
      }
    }
  }
}

// compress_program. Greedy rule-drop: try dropping each
// rule; keep the drop when the shorter program is NOT worse (is_better(best,
// cand) is false), then restart the sweep. Deterministic (no rng).
void compress_program(TransformGenome& best_genome, TransformFitnessFull& best_fit,
                      const std::vector<TrainPair>& train_pairs, const SearchConfig& cfg,
                      std::unordered_map<std::string, TransformFitnessFull>& cache) {
  bool changed = true;
  while (changed) {
    changed = false;
    if (best_genome.program.rules.empty()) break;
    for (size_t rule_idx = 0; rule_idx < best_genome.program.rules.size(); ++rule_idx) {
      if (past_post_deadline(cfg)) return;
      TransformGenome cand = best_genome;
      cand.program.rules.erase(cand.program.rules.begin() +
                               static_cast<std::ptrdiff_t>(rule_idx));
      TransformFitnessFull cand_fit = refine_score(cand, train_pairs, cfg, cache);
      // !is_better(best, cand) -> commit (cand is at least as good).
      if (!is_better_fit(best_fit, cand_fit, cfg.rank_specificity)) {
        best_genome = std::move(cand);
        best_fit = cand_fit;
        changed = true;
        break;
      }
    }
  }
}

// ARC_GENERALISE=1: Occam pass on the evolved program. Try the identity global
// transform, and the plain all_subobjects selector (no guards) on every rule whose
// action is a concept op (those re-segment the grid themselves, so a narrower
// selector is usually an accident of the train pairs). Keep each change that is
// not worse on the train pairs. Deterministic (no rng).
void generalise_program(TransformGenome& best_genome, TransformFitnessFull& best_fit,
                        const std::vector<TrainPair>& train_pairs, const SearchConfig& cfg,
                        std::unordered_map<std::string, TransformFitnessFull>& cache) {
  static const bool on = [] {
    const char* e = std::getenv("ARC_GENERALISE");
    return e && std::string(e) == "1";
  }();
  if (!on) return;
  auto try_commit = [&](TransformGenome cand) {
    if (past_post_deadline(cfg)) return;
    TransformFitnessFull cand_fit = refine_score(cand, train_pairs, cfg, cache);
    if (is_better_fit(best_fit, cand_fit, cfg.rank_specificity)) return;
    best_genome = std::move(cand);
    best_fit = cand_fit;
  };
  if (best_genome.program.global_transform != GlobalTransform::IDENTITY) {
    TransformGenome cand = best_genome;
    cand.program.global_transform = GlobalTransform::IDENTITY;
    try_commit(std::move(cand));
  }
  for (size_t i = 0; i < best_genome.program.rules.size(); ++i) {
    const DslRule& r = best_genome.program.rules[i];
    if (!find_concept_op(r.action.kind)) continue;
    if (r.selector.kind == SelectorKind::ALL_SUBOBJECTS && r.guards.empty()) continue;
    TransformGenome cand = best_genome;
    cand.program.rules[i].selector = DslSelector{};
    cand.program.rules[i].guards.clear();
    try_commit(std::move(cand));
  }
}

int immigrants_per_gen() {
  static const int n = [] {
    const char* e = std::getenv("ARC_IMMIGRANTS");
    return e ? std::max(0, std::atoi(e)) : 0;
  }();
  return n;
}

// ARC_ALL_RESTARTS=1: keep running the restarts after the first one that fits
// the train pairs (each later restart gets half a restart's time budget), so
// the finalists hold programs from several independent searches.
bool all_restarts() {
  static const bool on = [] {
    const char* e = std::getenv("ARC_ALL_RESTARTS");
    return e && std::string(e) == "1";
  }();
  return on;
}

// ARC_HILLCLIMB=1: a (1 + lambda) hill climber instead of the GA: the best genome
// is the only elite and every child is a mutation of it (no crossover, no
// immigrants, no stagnation injections).
bool hillclimb() {
  static const bool on = [] {
    const char* e = std::getenv("ARC_HILLCLIMB");
    return e && std::string(e) == "1";
  }();
  return on;
}

// ARC_LAMBDA=k: each generation breeds k times the free slots and keeps the best
// children ((mu + lambda) style truncation; default 1 = no extra children).
int lambda_factor() {
  static const int n = [] {
    const char* e = std::getenv("ARC_LAMBDA");
    return e ? std::max(1, std::atoi(e)) : 1;
  }();
  return n;
}

// ARC_FITLOG=1: per generation, its index within the attempt and the best and
// mean fitness score of the population are kept in fitness_curve() (per thread,
// for the result file).
bool fitlog_on() {
  static const bool on = [] {
    const char* e = std::getenv("ARC_FITLOG");
    return e && std::string(e) == "1";
  }();
  return on;
}

// ARC_BROOD=k: each crossover child is the best of k siblings bred from the same
// two parents (default 1 = no brood selection).
int brood_size() {
  static const int n = [] {
    const char* e = std::getenv("ARC_BROOD");
    return e ? std::max(1, std::atoi(e)) : 1;
  }();
  return n;
}

// TrainPair -> SeedTrainPair (seed_enum's own pair struct; same layout).
// ARC_INFORMED_ARGS=1: per concept op, the argument sets it derives from the
// train pairs, offered to mutation (codec.set_concept_arg_pool). Empty otherwise.
std::vector<std::pair<int, std::vector<DslArgs>>> concept_arg_pool(const std::vector<TrainPair>& tp) {
  std::vector<std::pair<int, std::vector<DslArgs>>> pool;
  const char* e = std::getenv("ARC_INFORMED_ARGS");
  if (!e || std::string(e) != "1") return pool;
  std::vector<Grid> ins, outs;
  for (const auto& p : tp) {
    ins.push_back(p.input);
    outs.push_back(p.target);
  }
  for (const ConceptOp& op : concept_ops()) {
    std::vector<DslArgs> a = op.seed_args(ins, outs);
    if (!a.empty()) pool.emplace_back(static_cast<int>(op.kind), std::move(a));
  }
  return pool;
}

// ARC_INFORMED_ATTR=1: the by_attr argument sets (attr, op, value) that, on every
// same-size train pair, select exactly the objects whose cells change, or
// exactly the ones that stay. Offered to mutation (codec.set_by_attr_pool).
std::vector<DslArgs> by_attr_pool(const std::vector<TrainPair>& tp) {
  std::vector<DslArgs> pool;
  const char* e = std::getenv("ARC_INFORMED_ATTR");
  if (!e || std::string(e) != "1" || tp.empty()) return pool;
  struct PairSets {
    Scene scene{0, 0};
    std::set<std::string> changed, kept;
  };
  std::vector<PairSets> ps;
  for (const auto& p : tp) {
    if (p.input.height() != p.target.height() || p.input.width() != p.target.width()) return pool;
    PairSets s;
    s.scene = encode_minimal(p.input, GraphConfig{});
    for (const std::string& id : select_node_ids(s.scene, DslSelector{})) {
      const Node& n = s.scene.nodes()[s.scene.index_of(id)];
      bool diff = false;
      for (const Coord& c : n.pixel_list)
        if (p.input.at(c.first, c.second) != p.target.at(c.first, c.second)) diff = true;
      (diff ? s.changed : s.kept).insert(id);
    }
    if (s.changed.empty() || s.kept.empty()) return pool;
    ps.push_back(std::move(s));
  }
  auto matches = [&](const DslArgs& a) {
    DslSelector sel{SelectorKind::BY_ATTR, a};
    bool all_changed = true, all_kept = true;
    for (const auto& s : ps) {
      std::vector<std::string> ids = select_node_ids(s.scene, sel);
      std::set<std::string> got(ids.begin(), ids.end());
      all_changed = all_changed && got == s.changed;
      all_kept = all_kept && got == s.kept;
      if (!all_changed && !all_kept) return false;
    }
    return true;
  };
  for (const std::string& attr : object_attr_names()) {
    std::set<int> values;
    for (const auto& s : ps)
      for (const Node& n : s.scene.nodes()) {
        int v;
        if (node_attr(n, attr, &v)) values.insert(v);
      }
    std::vector<DslArgs> cands;
    for (const char* op : {"max", "min", "unique"})
      cands.push_back(DslArgs{{"attr", cdv_str(attr)}, {"op", cdv_str(op)}, {"value", cdv_int(0)}});
    for (int v : values)
      for (const char* op : {"eq", "ne", "ge", "le"})
        cands.push_back(DslArgs{{"attr", cdv_str(attr)}, {"op", cdv_str(op)}, {"value", cdv_int(v)}});
    for (auto& a : cands)
      if (matches(a)) pool.push_back(std::move(a));
  }
  return pool;
}

// ARC_INFORMED_COLOR=1: input colours of the changed cells (from) and their
// output colours (to) over the same-size train pairs, sorted (codec.set_change_colors).
void set_change_colors(TransformCodec& codec, const std::vector<TrainPair>& tp) {
  static const bool on = [] {
    const char* e = std::getenv("ARC_INFORMED_COLOR");
    return e && std::string(e) == "1";
  }();
  if (!on) return;
  std::set<int> from, to;
  for (const auto& p : tp) {
    if (p.input.height() != p.target.height() || p.input.width() != p.target.width()) continue;
    const auto& a = p.input.cells();
    const auto& b = p.target.cells();
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i]) {
        from.insert(a[i]);
        to.insert(b[i]);
      }
  }
  codec.set_change_colors(std::vector<int>(from.begin(), from.end()), std::vector<int>(to.begin(), to.end()));
}

// Phase-A cache (ARC_PHASEA_CACHE, on unless =0): the scores of Phase A (quick
// rank on pair 0, the seed population, the elites' decoder sweep) are the same in
// every restart and ensemble member of a task: the seeds do not depend on the
// rng, and a genome's fitness depends only on the genome and the train pairs. The
// first attempt keeps them per thread, keyed by the train pairs and the scoring
// mode; later attempts start from them instead of scoring the seeds again (on a
// 30x30 task that was 4-5 s of every restart's time budget). Results are those
// of a fresh score; only the time left for the GA changes.
struct PhaseACache {
  std::string key;
  std::unordered_map<std::string, TransformFitnessFull> fits;
  std::unordered_map<std::string, double> quick;
  bool have_seeds = false;
  std::vector<TransformGenome> seeds;  // Phase A's enumerate_seeds list
  bool have_init = false;
  std::vector<TransformGenome> init;   // Phase B's (with delta seeds)
};
bool phasea_cache_on() {
  static const bool on = [] {
    const char* e = std::getenv("ARC_PHASEA_CACHE");
    return !(e && std::string(e) == "0");
  }();
  return on;
}
PhaseACache* phasea_cache(const std::vector<TrainPair>& tp, const SearchConfig& cfg) {
  thread_local std::list<PhaseACache> caches;  // most recent first, at most 4
  if (!phasea_cache_on()) return nullptr;
  std::string key = cfg.distance_mode + "|" + cfg.lexicase_case_mode + "|" + cfg.selection_mode;
  for (const auto& p : tp)
    for (const Grid* g : {&p.input, &p.target}) {
      key += "|" + std::to_string(g->height()) + "x" + std::to_string(g->width()) + ":";
      for (int v : g->cells()) key += static_cast<char>('0' + v);
    }
  for (auto it = caches.begin(); it != caches.end(); ++it)
    if (it->key == key) {
      caches.splice(caches.begin(), caches, it);
      return &caches.front();
    }
  caches.push_front(PhaseACache{key, {}, {}});
  if (caches.size() > 4) caches.pop_back();
  return &caches.front();
}

std::vector<SeedTrainPair> to_seed_pairs(const std::vector<TrainPair>& tp) {
  std::vector<SeedTrainPair> out;
  out.reserve(tp.size());
  for (const auto& p : tp) out.push_back(SeedTrainPair{p.input, p.target});
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// run_single_attempt.
// ---------------------------------------------------------------------------

SearchTrace run_single_attempt(const std::vector<TrainPair>& train_pairs,
                               const std::string& seed_decimal, int seed_is_u64,
                               unsigned long long seed_u64, const SearchConfig& cfg) {
  ArcRandom rng;
  if (seed_is_u64) {
    rng.seed(seed_u64);
  } else {
    rng.seed_decimal(seed_decimal.c_str());
  }

  TransformCodec codec(cfg.max_rules);
  codec.set_color_prior(cfg.color_prior);
  codec.set_macro_library(cfg.macro_lib);  // nullptr (default) == OFF
  codec.set_concept_arg_pool(concept_arg_pool(train_pairs));
  codec.set_by_attr_pool(by_attr_pool(train_pairs));
  set_change_colors(codec, train_pairs);

  PhaseACache* pac = phasea_cache(train_pairs, cfg);
  std::unordered_map<std::string, TransformFitnessFull> cache;
  if (pac) cache = pac->fits;

  SearchTrace trace;

  // ---- Phase A -------------------------------------------------------------
  // heuristic_pop = enumerate_seeds(fit_pairs, max_seconds=2.0,
  //                                 include_delta_seeds=false).
  //   * enumerator OFF -> no seeds; heuristic_pop is bootstrapped with ONE
  //     codec.random_genome(rng) (a single genome -> quick-rank skipped).
  //   * enumerator ON  -> the ordered non-delta enumerate_seeds list (delta
  //     seeds have their own lane below). With more than one seed, the
  //     quick-rank re-orders heuristic_pop by the score on pair 0 (a stable
  //     sort; NO RNG). random_genome is the fallback ONLY when the enumerator
  //     emits nothing.
  std::vector<GraphConfig> encoder_seed_bank;
  DecoderConfig default_decoder;  // default decoder
  std::vector<SeedTrainPair> seed_pairs;
  if (cfg.enable_seed_enumerator) {
    encoder_seed_bank = build_encoder_seed_bank();
    seed_pairs = to_seed_pairs(train_pairs);
  }

  std::vector<TransformGenome> heuristic_pop;
  if (cfg.enable_seed_enumerator) {
    // Phase A: include_delta_seeds=False (delta scored in a separate lane).
    if (pac && pac->have_seeds) {
      heuristic_pop = pac->seeds;
    } else {
      heuristic_pop = enumerate_seeds(seed_pairs, codec, encoder_seed_bank, default_decoder,
                                      /*include_delta_seeds=*/false, cfg.macro_seed_lib,
                                      /*max_seconds=*/cfg.deadline.has_value() ? 2.0 : 0.0);
      if (pac) {
        pac->seeds = heuristic_pop;
        pac->have_seeds = true;
      }
    }
    // Record the ordered enumerate_seeds list (pre-quick-rank).
    trace.phasea_seed_hashes.reserve(heuristic_pop.size());
    for (const auto& g : heuristic_pop)
      trace.phasea_seed_hashes.push_back(hash_program(g.program));
  }
  if (heuristic_pop.empty()) {
    // No seeds -> one random_genome (covers enumerator-off AND the (rare)
    // enumerator-on-but-empty case).
    heuristic_pop.push_back(codec.random_genome(rng));
  } else if (!train_pairs.empty() && heuristic_pop.size() > 1) {
    // Quick-rank by pair-0 score (stable sort). NO RNG.
    // DEPLOY: on large grids scoring every seed can take minutes per attempt, so
    // with a deadline the ranking stops after min(max(1, remaining*0.15), 3)s and
    // the unranked seeds follow in enumeration order. No deadline -> all ranked.
    const Grid& inp0 = train_pairs[0].input;
    const Grid& out0 = train_pairs[0].target;
    std::optional<std::chrono::steady_clock::time_point> rank_deadline;
    if (cfg.deadline.has_value()) {
      auto now = std::chrono::steady_clock::now();
      double remaining = std::max(0.0, std::chrono::duration<double>(cfg.deadline.value() - now).count());
      rank_deadline = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(std::min(std::max(1.0, remaining * 0.15), 3.0)));
    }
    std::vector<std::pair<double, TransformGenome>> ranked;
    std::vector<TransformGenome> unranked;
    ranked.reserve(heuristic_pop.size());
    for (const auto& g : heuristic_pop) {
      if (rank_deadline.has_value() && std::chrono::steady_clock::now() >= rank_deadline.value()) {
        unranked.push_back(g);
        continue;
      }
      double s;
      std::string qk = pac ? dedup_ga_key(g) : std::string();
      auto qit = pac ? pac->quick.find(qk) : std::unordered_map<std::string, double>::iterator();
      if (pac && qit != pac->quick.end()) {
        s = qit->second;
      } else {
        s = quick_seed_score(inp0, out0, g.encoder_config, g.decoder, g.program, cfg.scene_cache);
        if (pac) pac->quick.emplace(qk, s);
      }
      ranked.emplace_back(s, g);
    }
    // Stable sort by the pair-0 score.
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const std::pair<double, TransformGenome>& a,
                        const std::pair<double, TransformGenome>& b) {
                       return a.first < b.first;
                     });
    heuristic_pop.clear();
    for (auto& kv : ranked) heuristic_pop.push_back(std::move(kv.second));
    for (auto& g : unranked) heuristic_pop.push_back(std::move(g));
  }
  // score_population(heuristic_pop), then the Phase-A decoder sweep (top_k=3).
  // The decoder sweep is deterministic (no RNG); it runs BEFORE the loop so the
  // swept elite seeds the elite carry (load-bearing for the loop's rank order).
  // DEPLOY Phase-A scoring budget: when a per-task deadline is set, score only
  // the prefix reachable within min(max(1.0, remaining*0.30), 6.0)s; with no
  // deadline, score all.
  std::optional<std::chrono::steady_clock::time_point> heuristic_deadline;
  if (cfg.deadline.has_value()) {
    auto now = std::chrono::steady_clock::now();
    double remaining = std::chrono::duration<double>(cfg.deadline.value() - now).count();
    if (remaining < 0.0) remaining = 0.0;
    double budget = std::min(std::max(1.0, remaining * 0.30), 6.0);  // FRACTION=0.30, MAX=6.0
    heuristic_deadline = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                   std::chrono::duration<double>(budget));
  }
  std::vector<ScoredGenome> scored =
      score_population(heuristic_pop, train_pairs, cfg, cache, heuristic_deadline);

  // Phase-A delta lane: delta_pop = emit_delta_seeds deduped by dedup_ga_key,
  // scored in its own lane, then merged into `scored` by (encoder_signature,
  // program_hash, decoder-fields) so a delta seed already in the general pool
  // isn't double-counted, then re-sorted by fitness rank. Gated by
  // cfg.include_delta_seeds.
  if (cfg.enable_seed_enumerator && cfg.include_delta_seeds) {
    // emit_delta_seeds, deduped by dedup_ga_key.
    std::vector<TransformGenome> raw = emit_delta_seeds(seed_pairs, GraphConfig(), default_decoder);
    std::vector<TransformGenome> delta_pop;
    std::set<std::string> dseen;
    for (const auto& g : raw) {
      std::string k = dedup_ga_key(g);
      if (dseen.count(k)) continue;
      dseen.insert(k);
      delta_pop.push_back(g);
    }
    if (!delta_pop.empty()) {
      // DEPLOY: its own Phase-A-sized budget.
      std::optional<std::chrono::steady_clock::time_point> delta_deadline;
      if (cfg.deadline.has_value()) {
        auto now = std::chrono::steady_clock::now();
        double remaining = std::max(0.0, std::chrono::duration<double>(cfg.deadline.value() - now).count());
        delta_deadline = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                   std::chrono::duration<double>(std::min(std::max(1.0, remaining * 0.30), 6.0)));
      }
      std::vector<ScoredGenome> delta_scored =
          score_population(delta_pop, train_pairs, cfg, cache, delta_deadline);
      std::set<std::string> seen_keys;
      for (const auto& sg : scored) seen_keys.insert(dedup_ga_key(sg.genome));
      for (auto& sg : delta_scored) {
        std::string k = dedup_ga_key(sg.genome);
        if (seen_keys.count(k)) continue;
        seen_keys.insert(k);
        scored.push_back(sg);
      }
      // Stable sort by fitness rank.
      std::stable_sort(scored.begin(), scored.end(),
                       [&](const ScoredGenome& a, const ScoredGenome& b) {
                         return rank_less(fitness_rank(a.fit.scalars(), cfg.rank_specificity),
                                          fitness_rank(b.fit.scalars(), cfg.rank_specificity));
                       });
    }
  }

  // decoder_sweep_done: keyed by (encoder_signature, program_hash). Every swept
  // genome is recorded here so a re-surfaced best (e.g. the loop-exit best == a
  // Phase-A elite) is not re-swept by the post-loop passes. The Phase-A sweep
  // below and the final decoder sweep share this set.
  std::set<std::string> decoder_sweep_done;
  auto decoder_key = [](const TransformGenome& g) {
    return encoder_signature_repr(g.encoder_config) + std::string(1, '\x1f') +
           hash_program(g.program);
  };
  // Phase-A decoder sweep (top_k=3): sweep the top-3 elites, keep the better of
  // (swept, original) per elite, then re-rank the whole list.
  {
    int top_k = 3;
    for (int i = 0; i < top_k && i < static_cast<int>(scored.size()); ++i) {
      // Decoder sweep: skip (and DO NOT re-add) if already swept.
      std::string k = decoder_key(scored[static_cast<size_t>(i)].genome);
      if (decoder_sweep_done.count(k)) continue;
      ScoredGenome swept = decoder_sweep(scored[static_cast<size_t>(i)], train_pairs, cfg, cache);
      decoder_sweep_done.insert(k);
      if (is_better_fit(swept.fit, scored[static_cast<size_t>(i)].fit, cfg.rank_specificity))
        scored[static_cast<size_t>(i)] = swept;
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [&](const ScoredGenome& a, const ScoredGenome& b) {
                       return rank_less(fitness_rank(a.fit.scalars(), cfg.rank_specificity),
                                        fitness_rank(b.fit.scalars(), cfg.rank_specificity));
                     });
  }
  if (pac)
    for (const auto& kv : cache) pac->fits.emplace(kv.first, kv.second);
  TransformGenome best_genome = scored[0].genome;
  TransformFitnessFull best_fit = scored[0].fit;

  // generations <= 0 or phase_a_shortcut-on-solve early return.
  if (cfg.generations <= 0 || (cfg.phase_a_shortcut && best_fit.solved && mdl_prior_cost(best_genome) == 0.0)) {
    trace.best_genome = best_genome;
    trace.best_fit = best_fit;
    trace.generations_used = 0;
    // finalists = rank_dedupe_finalists([(best,best)] + scored). The
    // early-return `scored` is the Phase-A swept, rank-sorted list.
    trace.final_scored = scored;
    return trace;
  }

  // ---- Phase B: initial population -------------------------------------------
  // enumerator OFF: no seeds; fill entirely with random_genome.
  // enumerator ON : seeds = enumerate_seeds(train_pairs) (default max_seconds=5,
  //   include_delta_seeds=true -> the non-delta blocks 1-4 PLUS the delta-guided
  //   block 5, tail-appended in emission order). population = the seeds; with
  //   more than pop_size: keep an elite prefix (max(2, pop_size / 4)), shuffle
  //   the tail with the rng, recombine; then fill to pop_size with
  //   random_genome; keep the first pop_size.
  std::vector<TransformGenome> population;
  if (cfg.enable_seed_enumerator) {
    // cfg.include_delta_seeds (on by default) switches the delta seeds on or
    // off.
    if (pac && pac->have_init) {
      population = pac->init;
    } else {
      population = enumerate_seeds(seed_pairs, codec, encoder_seed_bank, default_decoder,
                                   /*include_delta_seeds=*/cfg.include_delta_seeds,
                                   cfg.macro_seed_lib,
                                   /*max_seconds=*/cfg.deadline.has_value() ? 5.0 : 0.0);
      if (pac) {
        pac->init = population;
        pac->have_init = true;
      }
    }
    if (static_cast<int>(population.size()) > cfg.pop_size) {
      int elite_size = std::max(2, cfg.pop_size / 4);
      elite_size = std::min(elite_size, cfg.pop_size);
      std::vector<TransformGenome> elite(population.begin(),
                                         population.begin() + elite_size);
      std::vector<TransformGenome> tail(population.begin() + elite_size, population.end());
      // rng.shuffle(tail): Fisher-Yates over indices, then reorder tail.
      std::vector<size_t> perm(tail.size());
      for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
      rng.shuffle_indices(perm);
      std::vector<TransformGenome> shuffled;
      shuffled.reserve(tail.size());
      for (size_t idx : perm) shuffled.push_back(tail[idx]);
      population = std::move(elite);
      for (auto& g : shuffled) population.push_back(std::move(g));
    }
    while (static_cast<int>(population.size()) < cfg.pop_size)
      population.push_back(codec.random_genome(rng));
    if (static_cast<int>(population.size()) > cfg.pop_size)
      population.resize(static_cast<size_t>(cfg.pop_size));
  } else {
    population.reserve(static_cast<size_t>(cfg.pop_size));
    while (static_cast<int>(population.size()) < cfg.pop_size)
      population.push_back(codec.random_genome(rng));
  }
  // Record the initial population (pre-scoring order) in the trace.
  if (cfg.enable_seed_enumerator) {
    trace.phasea_initial_pop_hashes.reserve(population.size());
    for (const auto& g : population)
      trace.phasea_initial_pop_hashes.push_back(hash_program(g.program));
  }

  int elite_k = hillclimb() ? 1 : std::max(2, cfg.pop_size / 10);
  // population = the top elite_k scored genomes, then the first
  // pop_size - elite_k genomes of the population.
  {
    std::vector<TransformGenome> elites;
    for (int i = 0; i < elite_k && i < static_cast<int>(scored.size()); ++i)
      elites.push_back(scored[static_cast<size_t>(i)].genome);
    int keep = std::max(0, cfg.pop_size - static_cast<int>(elites.size()));
    std::vector<TransformGenome> merged = elites;
    for (int i = 0; i < keep && i < static_cast<int>(population.size()); ++i)
      merged.push_back(population[static_cast<size_t>(i)]);
    population = std::move(merged);
  }
  // DEPLOY: bounded by the attempt deadline (the loop below stops there anyway).
  scored = score_population(population, train_pairs, cfg, cache, cfg.deadline);
  if (!scored.empty() && is_better_fit(scored[0].fit, best_fit, cfg.rank_specificity)) {
    best_genome = scored[0].genome;
    best_fit = scored[0].fit;
  }

  // Record gens[0] = the initial (pre-loop) scored population.
  auto push_gen = [&](const std::vector<ScoredGenome>& sc, int stagnation) {
    GenerationTrace gt;
    gt.pop_hashes.reserve(sc.size());
    for (const auto& sg : sc) gt.pop_hashes.push_back(hash_program(sg.genome.program));
    gt.best_hash = hash_program(best_genome.program);
    gt.best_solved = best_fit.solved;
    gt.best_exact_ratio = best_fit.exact_ratio;
    gt.best_max_pair_mismatch = best_fit.max_pair_mismatch;
    gt.best_pixel_mismatch_mean = best_fit.pixel_mismatch_mean;
    gt.best_complexity = best_fit.complexity;
    gt.stagnation_count = stagnation;
    trace.gens.push_back(std::move(gt));
    if (fitlog_on() && !sc.empty()) {
      double mean = 0.0;
      for (const auto& sg : sc) mean += sg.fit.score;
      fitness_curve().push_back({static_cast<double>(trace.gens.size() - 1), best_fit.score,
                                 mean / static_cast<double>(sc.size())});
    }
  };

  int generations_used = 0;
  int stagnation_count = 0;
  int solved_flat = 0;
  bool have_prev_best_rank = !scored.empty();
  FitnessRank prev_best_rank =
      have_prev_best_rank ? fitness_rank(best_fit.scalars(), cfg.rank_specificity) : FitnessRank{};

  push_gen(scored, stagnation_count);

  // gen_cap = cfg.generations (a deadline, when set, breaks out earlier).
  int gen_cap = cfg.generations;
  for (int gen_idx = 0; gen_idx < gen_cap; ++gen_idx) {
    // DEPLOY per-task wall-clock deadline (dormant unless cfg.deadline is set).
    // Break to the best-so-far genome captured at loop entry / the prior gen.
    if (past_deadline(cfg)) {
      generations_used = gen_idx;
      break;
    }
    // solved-refine early exit.
    int refine = cfg.solved_refine_stagnation * (mdl_prior_cost(best_genome) > 0.0 ? 4 : 1);
    if (best_fit.solved && (cfg.solved_refine_stagnation <= 0 || solved_flat >= refine)) {
      generations_used = gen_idx;
      break;
    }

    // Elite carry: next_pop = the top elite_k scored; seen_hashes = their program
    // hashes.
    std::vector<TransformGenome> next_pop;
    std::set<std::string> seen_hashes;
    for (int i = 0; i < elite_k && i < static_cast<int>(scored.size()); ++i) {
      next_pop.push_back(scored[static_cast<size_t>(i)].genome);
      seen_hashes.insert(hash_program(scored[static_cast<size_t>(i)].genome.program));
    }

    // Diversity gate (default off -> stuck = True with no rng draw / structural
    // measurement).
    bool stuck = true;
    if (cfg.diversity_gate > 0.0 && !scored.empty()) {
      std::set<std::string> skels;
      for (const auto& sg : scored) {
        std::string sig;
        for (const auto& r : sg.genome.program.rules) {
          sig += to_string(r.selector.kind);
          sig.push_back('|');
          sig += to_string(r.action.kind);
          sig.push_back(';');
        }
        sig.push_back('#');
        sig += to_string(sg.genome.program.global_transform);
        skels.insert(sig);
      }
      stuck = (static_cast<double>(skels.size()) / static_cast<double>(scored.size()))
              <= cfg.diversity_gate;
    }

    // Stagnation injection / restart.
    int inject_count =
        (stagnation_count >= cfg.stagnation_inject_threshold && !scored.empty() && stuck)
            ? std::max(5, cfg.pop_size / 4)
            : 0;
    int random_inject_count = 0;
    if (stagnation_count >= cfg.stagnation_restart_threshold && !scored.empty() && stuck) {
      random_inject_count = cfg.pop_size / 2;
      stagnation_count = 0;
    }

    // ---- fill the generation ------------------------------------------------
    int attempts = 0;
    int max_attempts = cfg.pop_size * 6 * lambda_factor();
    // ARC_LAMBDA: breed lambda times the free slots, keep the best below.
    const int n_elite_carried = static_cast<int>(next_pop.size());
    const int fill_target = n_elite_carried + lambda_factor() * std::max(0, cfg.pop_size - n_elite_carried);
    if (hillclimb()) inject_count = random_inject_count = 0;

    // Epsilon-lexicase support: build the error matrix + eps ONCE per gen, gated
    // on every genome carrying a same-length non-empty case vector.
    bool lex_ready = false;
    std::vector<std::vector<double>> lex_errors;
    std::vector<double> lex_eps;
    if ((cfg.selection_mode == "lexicase" || cfg.selection_mode == "lexicase_inject") &&
        !scored.empty()) {
      bool all_nonempty = true;
      size_t len0 = scored[0].fit.case_errors.size();
      bool same_len = (len0 > 0);
      for (const auto& sg : scored) {
        if (sg.fit.case_errors.empty()) all_nonempty = false;
        if (sg.fit.case_errors.size() != len0) same_len = false;
      }
      if (all_nonempty && same_len) {
        lex_errors = collect_case_matrix(scored);
        lex_eps = lexicase_epsilon(lex_errors);
        lex_ready = true;
      }
    }
    bool lex_breed = lex_ready && cfg.selection_mode == "lexicase";
    bool lex_inject = lex_ready && cfg.selection_mode == "lexicase_inject";

    // Rank scalars for the tournament path (rebuilt once per generation).
    std::vector<FitnessScalars> tourn_fits;
    if (!lex_breed) {
      tourn_fits.reserve(scored.size());
      for (const auto& sg : scored) tourn_fits.push_back(sg.fit.scalars());
    }

    // LEX_CHAMPIONS carry (default 0 -> no draw).
    if (lex_ready && cfg.lex_champions > 0) {
      int n_cases = static_cast<int>(lex_errors[0].size());
      int k = std::min(cfg.lex_champions, n_cases);
      std::vector<size_t> sample = rng.sample_indices(static_cast<size_t>(n_cases),
                                                      static_cast<size_t>(k));
      for (size_t c : sample) {
        if (static_cast<int>(next_pop.size()) >= cfg.pop_size) break;
        // champ_i = the genome with the lowest error on case c; ties -> lowest
        // index.
        int champ_i = 0;
        double best_err = lex_errors[0][c];
        for (int i = 1; i < static_cast<int>(scored.size()); ++i) {
          if (lex_errors[static_cast<size_t>(i)][c] < best_err) {
            best_err = lex_errors[static_cast<size_t>(i)][c];
            champ_i = i;
          }
        }
        const TransformGenome& champ = scored[static_cast<size_t>(champ_i)].genome;
        std::string h = hash_program(champ.program);
        if (seen_hashes.count(h)) continue;
        seen_hashes.insert(h);
        next_pop.push_back(champ);
      }
    }

    // ARC_IMMIGRANTS=k: k single-rule immigrants per generation, each a fresh
    // random rule on an elite's encoder and decoder, with the identity transform
    // half of the time. Random genomes rarely yield a clean one-rule program
    // (0-5 rules, one transform in 26), and mutation reaches one only by
    // deleting every other rule of a parent.
    for (int k = 0; k < immigrants_per_gen() && !hillclimb() && !scored.empty() &&
                    static_cast<int>(next_pop.size()) < cfg.pop_size;
         ++k) {
      int bound = std::min(static_cast<int>(scored.size()), elite_k);
      const TransformGenome& host = scored[static_cast<size_t>(rng.randrange(bound))].genome;
      TransformGenome child;
      child.encoder_config = host.encoder_config;
      child.decoder = host.decoder;
      child.program.global_transform =
          rng.random() < 0.5 ? GlobalTransform::IDENTITY : host.program.global_transform;
      child.program.rules.push_back(codec.random_rule(rng));
      std::string h = hash_program(child.program);
      if (seen_hashes.count(h)) continue;
      seen_hashes.insert(h);
      next_pop.push_back(std::move(child));
    }

    while (static_cast<int>(next_pop.size()) < fill_target && attempts < max_attempts) {
      ++attempts;
      TransformGenome child;
      if (hillclimb()) {
        child = codec.mutate(scored[0].genome, rng);  // (1 + lambda): mutants of the best
      } else if (random_inject_count > 0) {
        child = codec.random_genome(rng);
        --random_inject_count;
      } else if (inject_count > 0) {
        TransformGenome parent;
        if (lex_inject) {
          int idx = lexicase_select_idx(lex_errors, lex_eps, rng, nullptr);
          parent = scored[static_cast<size_t>(idx)].genome;
        } else {
          // a uniform pick among the top min(scored size, elite_k)
          int bound = std::min(static_cast<int>(scored.size()), elite_k);
          long long ri = rng.randrange(bound);
          parent = scored[static_cast<size_t>(ri)].genome;
        }
        child = codec.mutate(parent, rng);
        child = codec.mutate(child, rng);  // double mutation
        --inject_count;
      } else {
        TransformGenome mom;
        TransformGenome dad;
        if (lex_breed) {
          int mi = lexicase_select_idx(lex_errors, lex_eps, rng, nullptr);
          mom = scored[static_cast<size_t>(mi)].genome;
          const std::vector<double>* bias = nullptr;
          std::vector<double> bias_vec;
          if (cfg.lex_complement) {
            bias_vec = lex_errors[static_cast<size_t>(mi)];
            bias = &bias_vec;
          }
          int di = lexicase_select_idx(lex_errors, lex_eps, rng, bias);
          dad = scored[static_cast<size_t>(di)].genome;
        } else {
          int mi = select_parent_tournament(tourn_fits, rng, 4, cfg.rank_specificity);
          mom = scored[static_cast<size_t>(mi)].genome;
          int di = select_parent_tournament(tourn_fits, rng, 4, cfg.rank_specificity);
          dad = scored[static_cast<size_t>(di)].genome;
        }
        child = codec.crossover(mom, dad, rng);
        if (rng.random() < 0.9) child = codec.mutate(child, rng);
        if (brood_size() > 1) {
          // Brood selection: the siblings not seen yet are scored (into the
          // fitness cache, so the generation's scoring reuses them) and the
          // best one is kept.
          std::vector<TransformGenome> brood;
          for (int b = 0; b < brood_size(); ++b) {
            TransformGenome sib = b == 0 ? child : codec.crossover(mom, dad, rng);
            if (b > 0 && rng.random() < 0.9) sib = codec.mutate(sib, rng);
            if (!seen_hashes.count(hash_program(sib.program))) brood.push_back(std::move(sib));
          }
          if (!brood.empty()) child = score_population(brood, train_pairs, cfg, cache)[0].genome;
        }
      }
      std::string h = hash_program(child.program);
      if (seen_hashes.count(h)) continue;
      seen_hashes.insert(h);
      next_pop.push_back(child);
    }
    // ARC_LAMBDA: keep the elites and the best children for the free slots.
    if (static_cast<int>(next_pop.size()) > cfg.pop_size) {
      std::vector<TransformGenome> kids(next_pop.begin() + n_elite_carried, next_pop.end());
      std::vector<ScoredGenome> ranked = score_population(kids, train_pairs, cfg, cache);
      next_pop.resize(static_cast<size_t>(n_elite_carried));
      for (const auto& sg : ranked) {
        if (static_cast<int>(next_pop.size()) >= cfg.pop_size) break;
        next_pop.push_back(sg.genome);
      }
    }
    // Fill any remainder with fresh random genomes.
    while (static_cast<int>(next_pop.size()) < cfg.pop_size)
      next_pop.push_back(codec.random_genome(rng));

    // ---- score + track best ------------------------------------------------
    scored = score_population(next_pop, train_pairs, cfg, cache);
    if (scored.empty()) break;

    const TransformFitnessFull& cand_fit = scored[0].fit;
    FitnessRank cur_rank = fitness_rank(cand_fit.scalars(), cfg.rank_specificity);
    bool improved = is_better_fit(cand_fit, best_fit, cfg.rank_specificity);
    if (improved) {
      best_genome = scored[0].genome;
      best_fit = cand_fit;
      stagnation_count = 0;
    } else if (have_prev_best_rank && !rank_less(cur_rank, prev_best_rank)) {
      // cur_rank >= prev_best_rank  <=>  NOT (cur_rank < prev_best_rank).
      ++stagnation_count;
    }
    prev_best_rank = cur_rank;
    have_prev_best_rank = true;
    if (best_fit.solved) solved_flat = improved ? 0 : solved_flat + 1;
    generations_used = gen_idx + 1;

    push_gen(scored, stagnation_count);
  }

  // ---- post-loop local refinement passes -----------------------------------
  // With no deadline every gate guard is true and each budget-fraction
  // sub-deadline is unset, so the passes run to their fixed step/stall caps,
  // not a wall fraction. They consume the SAME rng the whole attempt has drawn
  // from (structural + delta draw; arg / compress / decoder-sweep are
  // deterministic). Order and gating are fixed.
  if (cfg.run_refinement) {
    trace.refined = true;
    auto snapshot = [&](const char* name) {
      SearchTrace::RefinePass rp;
      rp.name = name;
      rp.genome = best_genome;
      rp.fit = best_fit;
      trace.refine_passes.push_back(std::move(rp));
    };

    // 1. Structural hill-climb (gate: not solved).
    if (!best_fit.solved) {
      hill_climb(
          best_genome, best_fit,
          [&](const TransformGenome& parent, ArcRandom& r) { return codec.mutate(parent, r); },
          train_pairs, cfg, cache, rng, /*max_steps=*/48, /*stall_limit=*/12);
    }
    snapshot("structural");

    // 2. Arg-jitter over the top-K=5 loop-exit scored candidates (gate: not
    //    solved). Each candidate is refined from ITS OWN fitness (not the
    //    running best); dedup by program hash; keep the refined result if it
    //    out-ranks best; break on a solved refinement. Deterministic.
    if (!best_fit.solved) {
      std::set<std::string> seen_hashes;
      int top_k = 5;
      for (int i = 0; i < top_k && i < static_cast<int>(scored.size()); ++i) {
        std::string h = hash_program(scored[static_cast<size_t>(i)].genome.program);
        if (seen_hashes.count(h)) continue;
        seen_hashes.insert(h);
        TransformGenome cand_g = scored[static_cast<size_t>(i)].genome;
        TransformFitnessFull cand_f = scored[static_cast<size_t>(i)].fit;
        refine_numeric_args(cand_g, cand_f, train_pairs, cfg, cache);
        if (is_better_fit(cand_f, best_fit, cfg.rank_specificity)) {
          best_genome = std::move(cand_g);
          best_fit = cand_f;
          if (best_fit.solved) break;
        }
      }
    }
    snapshot("arg");

    // 3. Delta-guided hill-climb (gate: not solved AND exact_ratio < 1.0 AND a
    //    hint exists). compute_delta_hint internally gates on
    //    max_pair_mismatch <= 0.20 and >= 2 wrong cells. The hint is recomputed
    //    after each accepted child (the mutate wrapper detects a parent change).
    if (!best_fit.solved && best_fit.exact_ratio < 1.0) {
      bool has_hint = false;
      DeltaHint hint = compute_delta_hint(best_genome, train_pairs, &has_hint);
      if (has_hint) {
        // Closure state: the hint is bound to the current tracked genome; on a
        // commit (hill_climb takes a NEW best) the wrapper refreshes it. The
        // best is a fixed reference, so hill_climb signals commits via
        // commit_epoch and the closure refreshes when the epoch it last saw
        // differs.
        DeltaHint cur_hint = hint;
        bool cur_has = true;
        int commit_epoch = 0;
        int last_epoch = 0;
        auto delta_mutate = [&](const TransformGenome& parent, ArcRandom& r) -> TransformGenome {
          if (commit_epoch != last_epoch) {
            last_epoch = commit_epoch;
            cur_hint = compute_delta_hint(parent, train_pairs, &cur_has);
          }
          if (!cur_has) return parent;  // no-op; hill_climb stalls out
          return delta_guided_mutate(codec, parent, cur_hint, r);
        };
        hill_climb(best_genome, best_fit, delta_mutate, train_pairs, cfg, cache, rng,
                   /*max_steps=*/24, /*stall_limit=*/8, &commit_epoch);
      }
      // A missing hint is still a pass that "ran" (the climb is skipped);
      // record the (unchanged) best.
      snapshot("delta");
    } else {
      snapshot("delta");
    }

    // 4. Program compression (always; deterministic).
    compress_program(best_genome, best_fit, train_pairs, cfg, cache);
    generalise_program(best_genome, best_fit, train_pairs, cfg, cache);
    snapshot("compress");

    // 5. Exhaustive decoder sweep (gate: not solved), keyed dedup against the
    //    Phase-A sweep + any prior sweep of the same (encoder, program).
    if (!best_fit.solved) {
      std::string k = decoder_key(best_genome);
      if (!decoder_sweep_done.count(k)) {
        ScoredGenome start{best_genome, best_fit};
        ScoredGenome swept = decoder_sweep(start, train_pairs, cfg, cache);
        decoder_sweep_done.insert(k);
        if (is_better_fit(swept.fit, best_fit, cfg.rank_specificity)) {
          best_genome = swept.genome;
          best_fit = swept.fit;
        }
      }
    }
    snapshot("decoder_sweep");
  }

  trace.best_genome = best_genome;
  trace.best_fit = best_fit;
  trace.generations_used = generations_used;
  // The loop-exit `scored` population. The post-loop refinement above does
  // NOT re-score into `scored`, so this is the pre-refinement final population
  // -- exactly what rank_dedupe_finalists([(best, best_fit)] + scored) reads.
  trace.final_scored = scored;
  return trace;
}

// ---------------------------------------------------------------------------
// run_task: the multi-restart loop.
// ---------------------------------------------------------------------------

namespace {

// Add a small non-negative increment to a non-negative decimal string
// (base_seed + restart_idx * RESTART_SEED_STEP in arbitrary precision).
// Handles seeds wider than 64 bits without overflow.
std::string decimal_add(const std::string& base, unsigned long long inc) {
  // Strip a leading sign / whitespace defensively (seeds are non-negative
  // here; seed_decimal strips the sign too).
  size_t start = 0;
  bool negative = false;
  while (start < base.size() && (base[start] == ' ' || base[start] == '\t')) ++start;
  if (start < base.size() && (base[start] == '+' || base[start] == '-')) {
    negative = (base[start] == '-');
    ++start;
  }
  std::string digits = base.substr(start);
  if (digits.empty()) digits = "0";
  // A negative base seed never occurs for a task seed (& 0x7fffffff), so we do
  // not implement signed magnitude arithmetic; assert the contract instead.
  (void)negative;

  // Add from least-significant digit up.
  std::string out;
  out.reserve(digits.size() + 20);
  int i = static_cast<int>(digits.size()) - 1;
  unsigned long long carry = inc;
  while (i >= 0 || carry != 0) {
    unsigned long long d = 0;
    if (i >= 0) {
      char c = digits[static_cast<size_t>(i)];
      d = static_cast<unsigned long long>(c - '0');
      --i;
    }
    unsigned long long s = d + (carry % 10ULL);
    carry /= 10ULL;
    if (s >= 10ULL) {
      s -= 10ULL;
      carry += 1ULL;
    }
    out.push_back(static_cast<char>('0' + static_cast<int>(s)));
  }
  std::reverse(out.begin(), out.end());
  // Trim any leading zeros (keep a single 0 for a zero result).
  size_t nz = 0;
  while (nz + 1 < out.size() && out[nz] == '0') ++nz;
  return out.substr(nz);
}

// Multiply a non-negative decimal string by a small unsigned factor in
// arbitrary precision (the cross-restart refine rng seed is seed * 7919).
// Handles seeds wider than 64 bits without overflow.
std::string decimal_mul(const std::string& base, unsigned long long factor) {
  size_t start = 0;
  while (start < base.size() && (base[start] == ' ' || base[start] == '\t')) ++start;
  if (start < base.size() && (base[start] == '+' || base[start] == '-')) ++start;
  std::string digits = base.substr(start);
  if (digits.empty()) digits = "0";
  std::string out;
  out.reserve(digits.size() + 24);
  unsigned long long carry = 0;
  for (int i = static_cast<int>(digits.size()) - 1; i >= 0; --i) {
    unsigned long long d = static_cast<unsigned long long>(digits[static_cast<size_t>(i)] - '0');
    unsigned long long prod = d * factor + carry;
    out.push_back(static_cast<char>('0' + static_cast<int>(prod % 10ULL)));
    carry = prod / 10ULL;
  }
  while (carry != 0) {
    out.push_back(static_cast<char>('0' + static_cast<int>(carry % 10ULL)));
    carry /= 10ULL;
  }
  std::reverse(out.begin(), out.end());
  size_t nz = 0;
  while (nz + 1 < out.size() && out[nz] == '0') ++nz;
  return out.substr(nz);
}

}  // namespace

TaskResult run_task(const std::vector<TrainPair>& train_pairs,
                    const std::string& base_seed_decimal, int seed_is_u64,
                    unsigned long long base_seed_u64, int n_restarts,
                    const SearchConfig& cfg) {
  TaskResult result;
  bool have_best = false;
  int solved_at_idx = -1;

  for (int restart_idx = 0; restart_idx < n_restarts; ++restart_idx) {
    // run_task has no deadline, so there is NO deadline break here.
    // Per-restart seed = base_seed + restart_idx * RESTART_SEED_STEP.
    unsigned long long inc =
        static_cast<unsigned long long>(restart_idx) * RESTART_SEED_STEP;
    SearchTrace st;
    unsigned long long restart_seed_u64 = 0;
    std::string restart_seed_dec;
    if (seed_is_u64) {
      restart_seed_u64 = base_seed_u64 + inc;
      st = run_single_attempt(train_pairs, "", /*seed_is_u64=*/1,
                              restart_seed_u64, cfg);
    } else {
      restart_seed_dec = decimal_add(base_seed_decimal, inc);
      st = run_single_attempt(train_pairs, restart_seed_dec, /*seed_is_u64=*/0,
                              0, cfg);
    }

    RestartResult rr;
    rr.seed = seed_is_u64 ? restart_seed_u64 : inc;  // u64 field informational
    rr.best_genome = st.best_genome;
    rr.best_fit = st.best_fit;
    rr.generations_used = st.generations_used;
    result.restarts.push_back(rr);

    // Cross-restart best (8-key is_better_fit). First restart seeds it.
    if (!have_best || is_better_fit(st.best_fit, result.best_fit, cfg.rank_specificity)) {
      result.best_genome = st.best_genome;
      result.best_fit = st.best_fit;
      result.best_restart_idx = restart_idx;
      have_best = true;
    }

    // Lock the FIRST train-solver, then (SOLVED_DIVERSITY_RESTARTS default 0)
    // break the moment a restart solves: restart_idx - solved_at_idx >= 0.
    if (st.best_fit.solved && solved_at_idx < 0) {
      solved_at_idx = restart_idx;
    }
    if (solved_at_idx >= 0 && (restart_idx - solved_at_idx) >= SOLVED_DIVERSITY_RESTARTS) {
      break;
    }
  }

  result.solved_at_idx = solved_at_idx;
  return result;
}

// ---------------------------------------------------------------------------
// run_task_full: restarts, then the solved / unsolved tail.
// ---------------------------------------------------------------------------

namespace {

// apply_view: map a grid into `view` space. The spatial views keep colours
// intact so the run_task color prior stays valid. The view continuation skips
// names that is_known_view rejects, so an unknown view never kills the task.
Grid apply_view_impl(const Grid& g, const std::string& view) {
  int h = g.height(), w = g.width();
  if (view == "transpose") {
    Grid out(w, h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) out.set(x, y, g.at(y, x));
    return out;
  }
  if (view == "fliplr") {
    Grid out(h, w);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) out.set(y, x, g.at(y, w - 1 - x));
    return out;
  }
  if (view == "flipud") {
    Grid out(h, w);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) out.set(y, x, g.at(h - 1 - y, x));
    return out;
  }
  if (view == "rot90") {
    // counter-clockwise turn: out[i,j] = g[j, w-1-i]; shape (w, h).
    Grid out(w, h);
    for (int i = 0; i < w; ++i)
      for (int j = 0; j < h; ++j) out.set(i, j, g.at(j, w - 1 - i));
    return out;
  }
  if (view == "rot270") {
    // clockwise turn: out[i,j] = g[h-1-j, i]; shape (w, h).
    Grid out(w, h);
    for (int i = 0; i < w; ++i)
      for (int j = 0; j < h; ++j) out.set(i, j, g.at(h - 1 - j, i));
    return out;
  }
  // "identity" and anything else -> return unchanged (identity is a no-op; the
  // caller filters unknown names out before reaching here).
  return g;
}

bool is_known_view(const std::string& view) {
  return view == "transpose" || view == "fliplr" || view == "flipud" ||
         view == "rot90" || view == "rot270" || view == "identity";
}

// rule_pins_color: True iff the rule's WHICH-OBJECTS part
// (selector args or guard args) pins one of `colors`. Action args are EXCLUDED.
// Booleans are skipped so a True flag never collides with colour 1.
bool args_pin_color(const DslArgs& args, const std::set<int>& colors) {
  for (const auto& kv : args) {
    const DslValue& v = kv.second;
    if (v.tag == DslValue::Tag::Bool) continue;  // bools are skipped
    if (v.tag == DslValue::Tag::Int &&
        colors.count(static_cast<int>(v.i))) {
      return true;
    }
  }
  return false;
}

bool rule_pins_color(const DslRule& rule, const std::set<int>& colors) {
  if (args_pin_color(rule.selector.args, colors)) return true;
  for (const auto& g : rule.guards) {
    if (args_pin_color(g.args, colors)) return true;
  }
  return false;
}

// Sorted unique colours in a grid (ascending). Reused for LOO.
std::set<int> grid_unique_colors(const Grid& g) {
  std::set<int> out;
  for (int c : g.cells()) out.insert(c);
  return out;
}

}  // namespace

Grid apply_view_grid(const Grid& g, const std::string& view) {
  return apply_view_impl(g, view);
}

int loo_generalization(const TransformGenome& genome,
                       const std::vector<TrainPair>& train_pairs,
                       const std::string& distance_mode,
                       SceneCache* scene_cache) {
  const auto& rules = genome.program.rules;
  if (rules.empty()) return 0;
  std::vector<std::set<int>> input_colors;
  input_colors.reserve(train_pairs.size());
  for (const auto& p : train_pairs) input_colors.push_back(grid_unique_colors(p.input));
  int n = static_cast<int>(input_colors.size());
  if (n <= 1) return 0;
  int loo = 0;
  for (int i = 0; i < n; ++i) {
    // only_i = colours present ONLY in pair i's input among all N inputs.
    std::set<int> only_i;
    for (int c : input_colors[static_cast<size_t>(i)]) {
      bool unique = true;
      for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        if (input_colors[static_cast<size_t>(j)].count(c)) {
          unique = false;
          break;
        }
      }
      if (unique) only_i.insert(c);
    }
    if (only_i.empty()) continue;
    // Strip every rule that pins one of only_i.
    std::vector<DslRule> kept;
    for (const auto& r : rules) {
      if (!rule_pins_color(r, only_i)) kept.push_back(r);
    }
    if (kept.size() == rules.size()) continue;  // nothing pair-i-specific pinned
    TransformGenome stripped;
    stripped.encoder_config = genome.encoder_config;
    stripped.program.rules = std::move(kept);
    stripped.program.global_transform = genome.program.global_transform;
    stripped.decoder = genome.decoder;
    // evaluate_genome never throws, so re-score and gate on solved directly.
    TransformFitnessFull f =
        evaluate_genome(stripped, train_pairs, distance_mode, false, "coarse", scene_cache);
    if (!f.solved) ++loo;
  }
  return loo;
}

namespace {

// One finalist candidate: genome + its full fitness (loo_fail folded in later).
using FinalistCand = std::pair<TransformGenome, TransformFitnessFull>;

// rank_dedupe_finalists. Rank best-first (8-key fitness rank via
// is_better_fit), dedupe by (hash_program, decoder repr) --
// keeping the rank-best per signature -- optionally fold in the LOO penalty for
// each distinct SOLVED candidate (compute_loo, run_task-level gather ONLY), then
// re-sort and keep the top N_FINALISTS.
std::vector<FinalistCand> rank_dedupe_finalists(
    std::vector<FinalistCand> candidates, const std::vector<TrainPair>& train_pairs,
    bool compute_loo, const SearchConfig& cfg) {
  // Stable sort by fitness rank.
  std::stable_sort(candidates.begin(), candidates.end(),
                   [&](const FinalistCand& a, const FinalistCand& b) {
                     return rank_less(fitness_rank(a.second.scalars(), cfg.rank_specificity),
                                      fitness_rank(b.second.scalars(), cfg.rank_specificity));
                   });
  std::set<std::string> seen;
  std::vector<FinalistCand> distinct;
  for (auto& gf : candidates) {
    std::string key = dedup_finalist_key(gf.first);
    if (seen.count(key)) continue;
    seen.insert(key);
    distinct.push_back(std::move(gf));
  }
  if (compute_loo) {
    for (auto& gf : distinct) {
      if (gf.second.solved) {
        gf.second.loo_fail = loo_generalization(gf.first, train_pairs, cfg.distance_mode, cfg.scene_cache);
      }
    }
    std::stable_sort(distinct.begin(), distinct.end(),
                     [&](const FinalistCand& a, const FinalistCand& b) {
                       return rank_less(fitness_rank(a.second.scalars(), cfg.rank_specificity),
                                        fitness_rank(b.second.scalars(), cfg.rank_specificity));
                     });
  }
  // ARC_ALL_RESTARTS keeps more finalists, for the train-fit vote.
  const int keep = all_restarts() ? 4 * N_FINALISTS : N_FINALISTS;
  if (static_cast<int>(distinct.size()) > keep) distinct.resize(static_cast<size_t>(keep));
  return distinct;
}

// Per-restart finalists = rank_dedupe_finalists([(best, best_fit)] + scored)
// WITHOUT LOO (the per-attempt call), built from the attempt's best and its
// final scored population.
std::vector<FinalistCand> per_attempt_finalists(const SearchTrace& st, const SearchConfig& cfg) {
  std::vector<FinalistCand> cands;
  cands.reserve(1 + st.final_scored.size());
  cands.emplace_back(st.best_genome, st.best_fit);
  for (const auto& sg : st.final_scored) cands.emplace_back(sg.genome, sg.fit);
  return rank_dedupe_finalists(std::move(cands), {}, /*compute_loo=*/false, cfg);
}

// Run one restart's full run_single_attempt (with the post-loop refinement) and
// package its loop-exit best + per-attempt finalists.
struct AttemptResult {
  TransformGenome best_genome;
  TransformFitnessFull best_fit;
  std::vector<FinalistCand> finalists;  // per-attempt rank_dedupe_finalists (no LOO)
  int generations_used = 0;
};

AttemptResult run_attempt_with_finalists(const std::vector<TrainPair>& train_pairs,
                                         const std::string& seed_dec, int seed_is_u64,
                                         unsigned long long seed_u64, const SearchConfig& cfg) {
  SearchConfig c = cfg;
  c.run_refinement = true;  // always the full single attempt, refinement included
  SearchTrace st = run_single_attempt(train_pairs, seed_dec, seed_is_u64, seed_u64, c);
  AttemptResult ar;
  ar.best_genome = st.best_genome;
  ar.best_fit = st.best_fit;
  ar.finalists = per_attempt_finalists(st, c);
  ar.generations_used = st.generations_used;
  return ar;
}

}  // namespace

TaskResultFull run_task_full(const std::vector<TrainPair>& train_pairs,
                             const std::string& base_seed_decimal, int seed_is_u64,
                             unsigned long long base_seed_u64, int n_restarts,
                             const SearchConfig& cfg, const TaskTailConfig& tail) {
  TaskResultFull result;

  // DEPLOY per-task wall-clock budget: stamp the absolute task deadline ONCE
  // here. The budget is then split per attempt: each restart k is capped at a
  // CUMULATIVE min(global, task_start + budget*(k+1)); each occam / view
  // continuation attempt at min(global, now + budget). This keeps a slow
  // restart-0 from starving the later differently-seeded restarts. When
  // max_task_seconds <= 0 the deadline stays nullopt and cfg_task == cfg
  // exactly, so the deadline-free path is unchanged.
  SearchConfig cfg_task = cfg;
  std::optional<std::chrono::steady_clock::time_point> task_start;
  std::chrono::steady_clock::duration per_attempt_budget{};
  bool budget_sliced = false;
  if (cfg_task.max_task_seconds > 0.0 && !cfg_task.deadline) {
    auto t0 = std::chrono::steady_clock::now();
    cfg_task.deadline =
        t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                 std::chrono::duration<double>(cfg_task.max_task_seconds));
    task_start = t0;
    // per_attempt_budget = max(max_task_seconds / n_restarts, 1.0).
    double b = cfg_task.max_task_seconds / static_cast<double>(std::max(1, n_restarts));
    if (b < 1.0) b = 1.0;
    per_attempt_budget = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(b));
    budget_sliced = true;
  }
  // Restart k local deadline = min(global, task_start + budget*(k+1)); with no
  // budget -> the task deadline (possibly nullopt).
  auto restart_deadline =
      [&](int k) -> std::optional<std::chrono::steady_clock::time_point> {
    if (!budget_sliced) return cfg_task.deadline;
    return std::min(*cfg_task.deadline, *task_start + per_attempt_budget * (k + 1));
  };
  // Continuation (occam/view) attempt deadline = min(global, now + budget).
  auto continuation_deadline =
      [&]() -> std::optional<std::chrono::steady_clock::time_point> {
    if (!budget_sliced) return cfg_task.deadline;
    return std::min(*cfg_task.deadline,
                    std::chrono::steady_clock::now() + per_attempt_budget);
  };
  // ARC_POST_GRACE=S: post-loop passes stop S s after the deadline they follow.
  static const double post_grace = [] {
    const char* e = std::getenv("ARC_POST_GRACE");
    return e ? std::max(0.0, std::atof(e)) : -1.0;
  }();
  auto post_stop = [&](const std::optional<std::chrono::steady_clock::time_point>& dl)
      -> std::optional<std::chrono::steady_clock::time_point> {
    if (post_grace < 0.0 || !budget_sliced || !dl) return std::nullopt;
    return *dl + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                     std::chrono::duration<double>(post_grace));
  };
  // A continuation stops when < 5s remain to the task deadline; with no budget
  // this guard is dormant (like past_deadline=false).
  auto continuation_should_break = [&]() -> bool {
    if (!budget_sliced) return past_deadline(cfg_task);
    return (*cfg_task.deadline - std::chrono::steady_clock::now()) < std::chrono::seconds(5);
  };

  // DEPLOY encoder scene cache: one per-task cache owned
  // here for the whole task. The input train grids recur across every restart /
  // occam / view continuation, so a single per-task cache maximizes reuse; it is
  // threaded down the fitness leaf chain via cfg_task (every sub-path copies
  // cfg_task, so the non-owning pointer propagates). OFF (nullptr) unless
  // cfg.encoder_cache, keeping the no-cache path unchanged. Each task runs
  // on one worker thread with a single-threaded GA, so no mutex is needed.
  std::optional<SceneCache> task_scene_cache;
  if (cfg_task.encoder_cache && cfg_task.scene_cache == nullptr) {
    std::size_t cap = cfg_task.encoder_cache_mb > 0
                          ? static_cast<std::size_t>(cfg_task.encoder_cache_mb) * 1024ULL * 1024ULL
                          : 1500ULL * 1024ULL * 1024ULL;
    task_scene_cache.emplace(cap);
    cfg_task.scene_cache = &(*task_scene_cache);
  }
  // Copy final scene-cache telemetry into the result before returning (called at
  // every run_task_full exit; no-op when the cache is off).
  // The post-restart passes (finalist LOO, cross-restart refine) score without a
  // deadline, but through the task's scene cache.
  SearchConfig cfg_post = cfg;
  cfg_post.scene_cache = cfg_task.scene_cache;
  cfg_post.post_deadline = post_stop(cfg_task.deadline);
  auto stamp_cache_stats = [&]() {
    if (task_scene_cache) {
      result.scene_cache_hits = static_cast<long long>(task_scene_cache->hits());
      result.scene_cache_misses = static_cast<long long>(task_scene_cache->misses());
      result.scene_cache_evictions = static_cast<long long>(task_scene_cache->evictions());
      result.scene_cache_peak_bytes = static_cast<long long>(task_scene_cache->peak_bytes());
    }
  };

  // ---- multi-restart loop (identical to run_task, but keeping per-restart
  //      finalists this time) -----------------------------------------------
  std::vector<AttemptResult> restart_results;
  bool have_best = false;
  TransformGenome best_genome;
  TransformFitnessFull best_fit;
  int solved_at_idx = -1;
  int best_restart_idx = 0;

  auto seed_for = [&](int idx, std::string* dec, unsigned long long* u64) {
    unsigned long long inc = static_cast<unsigned long long>(idx) * RESTART_SEED_STEP;
    if (seed_is_u64) {
      *u64 = base_seed_u64 + inc;
      dec->clear();
    } else {
      *dec = decimal_add(base_seed_decimal, inc);
      *u64 = 0;
    }
  };

  for (int restart_idx = 0; restart_idx < n_restarts; ++restart_idx) {
    if (past_deadline(cfg_task)) break;  // DEPLOY: stop launching restarts past budget
    std::string sdec;
    unsigned long long su64 = 0;
    seed_for(restart_idx, &sdec, &su64);
    SearchConfig cfg_restart = cfg_task;
    cfg_restart.deadline = restart_deadline(restart_idx);  // per-attempt budget
    if (all_restarts() && solved_at_idx >= 0 && budget_sliced)
      cfg_restart.deadline = std::min(*cfg_task.deadline, std::chrono::steady_clock::now() + per_attempt_budget / 2);
    cfg_restart.post_deadline = post_stop(cfg_restart.deadline);
    AttemptResult ar = run_attempt_with_finalists(train_pairs, sdec, seed_is_u64, su64, cfg_restart);
    result.generations_used += ar.generations_used;  // total GA generations this task

    if (!have_best || is_better_fit(ar.best_fit, best_fit, cfg.rank_specificity)) {
      best_genome = ar.best_genome;
      best_fit = ar.best_fit;
      best_restart_idx = restart_idx;
      have_best = true;
    }
    bool solved = ar.best_fit.solved;
    restart_results.push_back(std::move(ar));
    if (solved && solved_at_idx < 0) solved_at_idx = restart_idx;
    if (solved_at_idx >= 0 && (restart_idx - solved_at_idx) >= SOLVED_DIVERSITY_RESTARTS && !all_restarts()) break;
  }
  result.solved_at_idx = solved_at_idx;

  // ---- SOLVED path ---------------------------------------------------------
  if (solved_at_idx >= 0) {
    // solved_primary = the FIRST train-solver (the restart at solved_at_idx). The
    // break-on-solve above makes the last restart that solver (with
    // ARC_ALL_RESTARTS later restarts follow; the index still names it).
    AttemptResult& solved_primary = restart_results[static_cast<size_t>(solved_at_idx)];
    TransformGenome primary_genome = solved_primary.best_genome;
    TransformFitnessFull primary_fit = solved_primary.best_fit;
    std::vector<FinalistCand> primary_finalists = solved_primary.finalists;

    // Occam continuation (ON by default): depth-capped shallower-solver search,
    // walking the cap down after each success, stop on the first capped miss.
    // Every attempt is APPENDED to restart_results so the finalist gather sees it.
    if (tail.occam_continue) {
      int depth = static_cast<int>(primary_genome.program.rules.size());
      for (const auto& r : restart_results) {
        if (r.best_fit.solved)
          depth = std::min(depth, static_cast<int>(r.best_genome.program.rules.size()));
        for (const auto& fg : r.finalists) {
          if (fg.second.solved)
            depth = std::min(depth, static_cast<int>(fg.first.program.rules.size()));
        }
      }
      int cap = depth - 1;
      int attempt = 0;
      while (cap >= 1) {
        if (continuation_should_break()) break;  // <5s to the global deadline
        SearchConfig occam_cfg = cfg_task;
        occam_cfg.max_rules = cap;  // TransformCodec(max_rules=cap)
        occam_cfg.deadline = continuation_deadline();  // min(global, now + budget)
        occam_cfg.post_deadline = post_stop(occam_cfg.deadline);
        std::string sdec;
        unsigned long long su64 = 0;
        seed_for(OCCAM_SEED_OFFSET + attempt, &sdec, &su64);
        AttemptResult cand =
            run_attempt_with_finalists(train_pairs, sdec, seed_is_u64, su64, occam_cfg);
        ++attempt;
        bool solved = cand.best_fit.solved;
        int rules = static_cast<int>(cand.best_genome.program.rules.size());
        restart_results.push_back(std::move(cand));
        if (!solved) break;  // stop on first capped failure
        cap = rules - 1;
      }
    }

    // Tabu restarts (tail.tabu_restart, off by default) are not implemented:
    // the flag is accepted and ignored.

    // first_solver = (solved_primary.genome, solved_primary.fitness) BEFORE the
    // generality promotion, reported next to the promoted primary.
    result.has_first_solver = true;
    result.first_solver_genome = primary_genome;
    result.first_solver_fit = primary_fit;

    // Finalist gather + LOO/MDL promotion: merge the primary + its finalists +
    // every restart's (genome, finalists), then
    // rank_dedupe_finalists(compute_loo=true) folds in LOO and re-ranks.
    std::vector<FinalistCand> gathered;
    gathered.emplace_back(primary_genome, primary_fit);
    for (const auto& fg : primary_finalists) gathered.push_back(fg);
    for (const auto& r : restart_results) {
      gathered.emplace_back(r.best_genome, r.best_fit);
      for (const auto& fg : r.finalists) gathered.push_back(fg);
    }
    std::vector<FinalistCand> ranked =
        rank_dedupe_finalists(std::move(gathered), train_pairs, /*compute_loo=*/true, cfg_post);
    result.finalists = ranked;

    // Promote the most-general solving program (finalists[0]) to primary.
    if (!ranked.empty() && ranked[0].second.solved) {
      primary_genome = ranked[0].first;
      primary_fit = ranked[0].second;
    }
    result.primary_genome = primary_genome;
    result.primary_fit = primary_fit;
    result.solved = primary_fit.solved;

    // View continuation (ON by default): one attempt per view on the transformed
    // train pairs; each solving view is a view-space solver (LOO on view pairs).
    if (!tail.views.empty()) {
      for (size_t i = 0; i < tail.views.size(); ++i) {
        if (continuation_should_break()) break;  // <5s to the global deadline
        const std::string& view = tail.views[i];
        if (!is_known_view(view)) continue;  // unknown view -> skip it
        std::vector<TrainPair> v_pairs;
        v_pairs.reserve(train_pairs.size());
        for (const auto& p : train_pairs)
          v_pairs.push_back(TrainPair{apply_view_impl(p.input, view),
                                      apply_view_impl(p.target, view)});
        SearchConfig view_cfg = cfg_task;
        view_cfg.deadline = continuation_deadline();  // min(global, now + budget)
        view_cfg.post_deadline = post_stop(view_cfg.deadline);
        // The color prior stays valid (views keep colours intact); run_task set it
        // from the identity-space outputs, which equal the view-space outputs as a
        // multiset, so the prior is unchanged.
        std::string sdec;
        unsigned long long su64 = 0;
        seed_for(VIEW_SEED_OFFSET + static_cast<int>(i), &sdec, &su64);
        AttemptResult cand =
            run_attempt_with_finalists(v_pairs, sdec, seed_is_u64, su64, view_cfg);
        if (cand.best_fit.solved) {
          ViewSolver vs;
          vs.view = view;
          vs.genome = cand.best_genome;
          vs.fit = cand.best_fit;
          // LOO on the view-space pairs (downstream tie-break for the view theory).
          vs.fit.loo_fail = loo_generalization(vs.genome, v_pairs, view_cfg.distance_mode, view_cfg.scene_cache);
          result.view_solvers.push_back(std::move(vs));
        }
      }
    }
    stamp_cache_stats();
    return result;
  }

  // ---- UNSOLVED path -------------------------------------------------------
  // Cross-restart refine (best-effort): dedup restart genomes by program hash,
  // rank, structural-refine the pool, then compress + decoder-sweep the best.
  // Can surface / overwrite the best. Its rng is seeded with seed * 7919.
  if (have_best && !best_fit.solved && restart_results.size() > 1) {
    std::unordered_map<std::string, TransformFitnessFull> cache;
    std::set<std::string> seen;
    std::vector<FinalistCand> pool;
    for (const auto& r : restart_results) {
      std::string h = hash_program(r.best_genome.program);
      if (seen.count(h)) continue;
      seen.insert(h);
      pool.emplace_back(r.best_genome, r.best_fit);
    }
    // Stable sort by fitness rank.
    std::stable_sort(pool.begin(), pool.end(),
                     [&](const FinalistCand& a, const FinalistCand& b) {
                       return rank_less(fitness_rank(a.second.scalars(), cfg.rank_specificity),
                                        fitness_rank(b.second.scalars(), cfg.rank_specificity));
                     });
    // rng seeded with seed * 7919, where seed is the BASE task seed.
    ArcRandom cr_rng;
    if (seed_is_u64) {
      cr_rng.seed(base_seed_u64 * 7919ULL);
    } else {
      cr_rng.seed_decimal(decimal_mul(base_seed_decimal, 7919ULL).c_str());
    }
    SearchConfig c = cfg_post;
    for (auto& gf : pool) {
      // No deadline -> no break (ARC_POST_GRACE: the task's stop time).
      if (past_post_deadline(c)) break;
      TransformGenome refined = gf.first;
      TransformFitnessFull refined_fit = gf.second;
      // Structural refine = hill_climb (48/12) via codec.mutate on this pool's
      // own rng.
      TransformCodec codec(c.max_rules);
      codec.set_color_prior(c.color_prior);
      codec.set_macro_library(c.macro_lib);  // nullptr == OFF
      codec.set_concept_arg_pool(concept_arg_pool(train_pairs));
      codec.set_by_attr_pool(by_attr_pool(train_pairs));
  set_change_colors(codec, train_pairs);
      hill_climb(
          refined, refined_fit,
          [&](const TransformGenome& parent, ArcRandom& r) { return codec.mutate(parent, r); },
          train_pairs, c, cache, cr_rng, /*max_steps=*/48, /*stall_limit=*/12);
      if (is_better_fit(refined_fit, best_fit, c.rank_specificity)) {
        best_genome = refined;
        best_fit = refined_fit;
        if (best_fit.solved) break;
      }
    }
    // compress (no deadline -> always).
    compress_program(best_genome, best_fit, train_pairs, c, cache);
    generalise_program(best_genome, best_fit, train_pairs, c, cache);
    // decoder sweep (gate: not solved).
    if (!best_fit.solved) {
      ScoredGenome start{best_genome, best_fit};
      ScoredGenome swept = decoder_sweep(start, train_pairs, c, cache);
      if (is_better_fit(swept.fit, best_fit, c.rank_specificity)) {
        best_genome = swept.genome;
        best_fit = swept.fit;
      }
    }
    best_restart_idx = -1;  // refinement produced a synthetic best (idx not meaningful)
  }

  // finalists = rank_dedupe_finalists(gathered finalists of best + restarts, LOO).
  std::vector<FinalistCand> gathered;
  gathered.emplace_back(best_genome, best_fit);
  for (const auto& r : restart_results) {
    gathered.emplace_back(r.best_genome, r.best_fit);
    for (const auto& fg : r.finalists) gathered.push_back(fg);
  }
  std::vector<FinalistCand> ranked =
      rank_dedupe_finalists(std::move(gathered), train_pairs, /*compute_loo=*/true, cfg_post);
  result.finalists = ranked;

  TransformGenome primary_genome = best_genome;
  TransformFitnessFull primary_fit = best_fit;
  // If a solved program surfaced (e.g. via cross-restart refine), promote the
  // most-general one; otherwise finalists[0] is not solved -> no-op.
  if (!ranked.empty() && ranked[0].second.solved) {
    result.has_first_solver = true;
    result.first_solver_genome = best_genome;
    result.first_solver_fit = best_fit;
    primary_genome = ranked[0].first;
    primary_fit = ranked[0].second;
  }
  result.primary_genome = primary_genome;
  result.primary_fit = primary_fit;
  result.solved = primary_fit.solved;
  (void)best_restart_idx;
  stamp_cache_stats();
  return result;
}

}  // namespace arc
