// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_SEARCH_HPP
#define ARC_SEARCH_HPP

// The GA search: one attempt (run_single_attempt), the multi-restart task
// loops (run_task, run_task_full) and their helpers.
//
// run_single_attempt runs Phase A (the seed enumerator when
// enable_seed_enumerator is set, else a single codec.random_genome), builds the
// initial population, runs the generation loop and, when run_refinement is
// set, the post-loop refinement passes.
//
// Without a deadline (max_task_seconds <= 0) every loop runs to its fixed cap
// and the trajectory depends ONLY on (seed, generations, config). With a
// deadline the loops stop at the best-so-far once it passes, so the result
// also depends on timing.
//
// The RNG draw order across the whole loop is part of the behaviour. One
// extra/missing/reordered draw changes every later generation; the
// per-generation population program-hash trace (GenerationTrace) shows the
// first generation where two runs differ.

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "arc/codec.hpp"
#include "arc/config.hpp"
#include "arc/dsl.hpp"
#include "arc/grid.hpp"
#include "arc/rng.hpp"
#include "arc/select.hpp"

namespace arc {

// ARC_FITLOG=1: the generation index within its attempt and the best and mean
// fitness score of the population, one entry per generation of the task the calling thread is solving (cleared by
// the caller before each task).
std::vector<std::array<double, 3>>& fitness_curve();


// Per-task encoder scene cache (scene_cache.hpp). Forward-declared: SearchConfig
// only holds a non-owning pointer, so it needs no full definition here.
class SceneCache;

// One (input, target) training pair (raw grids; the fitness stack encodes /
// executes / decodes internally).
struct TrainPair {
  Grid input;
  Grid target;
};

// The full fitness the evolver ranks on (evaluate_genome's result). The
// scalars feed select.hpp's fitness_rank; case_errors is the lexicase vector
// (empty in tournament mode).
struct TransformFitnessFull {
  double pixel_mismatch_mean = 1.0;
  double max_pair_mismatch = 1.0;
  double complexity = 0.0;
  double score = 1.0;
  double exact_ratio = 0.0;
  bool solved = false;
  int loo_fail = 0;
  int spec = 0;
  std::vector<double> case_errors;

  // Project onto the select.hpp rank scalars (the GA hot loop never sets
  // loo_fail, so it stays 0 there).
  FitnessScalars scalars() const {
    FitnessScalars f;
    f.solved = solved;
    f.loo_fail = loo_fail;
    f.spec = spec;
    f.complexity = complexity;
    f.exact_ratio = exact_ratio;
    f.max_pair_mismatch = max_pair_mismatch;
    f.pixel_mismatch_mean = pixel_mismatch_mean;
    return f;
  }
};

// evaluate_genome: score a genome on ALL train pairs with a pixel-gated
// objective: the mean/max aggregation (the mean is a left-to-right sum / n),
// the pixel-exact solved gate, complexity (MDL complexity when solved, the rule
// count when not), specificity, and the lexicase case vector.
TransformFitnessFull evaluate_genome(const TransformGenome& genome,
                                     const std::vector<TrainPair>& train_pairs,
                                     const std::string& distance_mode,
                                     bool collect_cases,
                                     const std::string& case_mode,
                                     SceneCache* scene_cache = nullptr);

// One scored population member (index-aligned genome + its full fitness).
struct ScoredGenome {
  TransformGenome genome;
  TransformFitnessFull fit;
};

// Config knobs the generation loop reads. Defaults: stagnation_inject=2,
// stagnation_restart=4, diversity_gate=0.0 (off), selection_mode="tournament",
// lexicase_cases="coarse", lex_champions=0, lex_complement=0,
// solved_refine_stagnation=8, rank_specificity off, phase_a_shortcut off.
struct SearchConfig {
  int pop_size = 40;
  int generations = 20;
  int max_rules = 8;
  std::string selection_mode = "tournament";  // tournament|lexicase|lexicase_inject
  std::string lexicase_case_mode = "coarse";   // coarse|fine|fine2
  std::string distance_mode = "graph";         // graph|pixel
  int stagnation_inject_threshold = 2;
  int stagnation_restart_threshold = 4;
  double diversity_gate = 0.0;
  int lex_champions = 0;
  int lex_complement = 0;
  int solved_refine_stagnation = 8;
  bool rank_specificity = false;
  bool phase_a_shortcut = false;
  // Phase-A seed enumerator. When true, run_single_attempt runs the SEEDED
  // Phase A: enumerate_seeds (non-delta) -> quick-rank -> score -> decoder
  // sweep, and the seeded branch of the initial population. When false (the
  // struct default; the binary turns it on), Phase A bootstraps with one
  // random_genome.
  bool enable_seed_enumerator = false;
  // Include the delta-guided seeds (block 5) in the initial population's
  // enumerate_seeds call (Phase B). Phase A always excludes them (delta seeds
  // are scored in a separate lane). Default true. Only consulted when
  // enable_seed_enumerator is true.
  bool include_delta_seeds = true;
  // color prior applied before search (run_task sets it from observed output
  // colours). Empty -> no prior.
  std::vector<int> color_prior;
  // Post-loop local refinement passes. When true, run_single_attempt runs, in
  // this order and each behind its own gate: structural hill-climb, arg-jitter
  // top-K, delta-guided hill-climb, program compression, decoder sweep -- all on
  // the loop-exit best genome, consuming the SAME rng. Default false: the
  // attempt then returns its loop-exit best (run_task_full always sets it).
  bool run_refinement = false;
  // Macro library (OFF by default). When macro_lib is non-null with rate>0, the
  // search codec's random_genome / add-rule mutation realize mined macros. When
  // macro_seed_lib is non-null with seed_topn>0, the Phase-A enumerate_seeds
  // block 6 emits the mined macro seeds. Both NON-OWNING; nullptr (default)
  // leaves every sampling path as with macros OFF.
  const MacroLibrary* macro_lib = nullptr;
  const MacroLibrary* macro_seed_lib = nullptr;
  // ---- per-task wall-clock budget ------------------------------------------
  // max_task_seconds: the per-task wall budget in seconds. <= 0 -> OFF (the
  // default: the generation + restart loops run to their fixed caps, so the
  // result is deterministic). When > 0, run_task_full (the per-task entry)
  // stamps `deadline` below at task start, and the generation loop +
  // restart/occam/view loops break to best-so-far once it passes, so the result
  // then depends on timing. main.cpp wires it from --max-task-seconds.
  double max_task_seconds = 0.0;
  // Absolute per-task deadline (steady_clock). nullopt -> no deadline (OFF).
  // Set INTERNALLY (run_task_full) from max_task_seconds; never
  // a user-supplied field. past_deadline(cfg) gates every loop break.
  std::optional<std::chrono::steady_clock::time_point> deadline;
  // ARC_POST_GRACE=S: when the refinement passes after a GA loop must stop
  // (deadline + S, set by run_task_full). nullopt: they run to their caps.
  std::optional<std::chrono::steady_clock::time_point> post_deadline;
  // ---- encoder scene cache -------------------------------------------------
  // scene_cache: non-owning ptr to a per-task SceneCache (scene_cache.hpp),
  // built + owned by run_task_full when encoder_cache is true and threaded down
  // the fitness leaf chain to the initial encode. nullptr (the default) =>
  // uncached; results are byte-identical with or without the cache.
  // encoder_cache / encoder_cache_mb mirror the config knobs; main.cpp's
  // make_search_config sets them from the loaded ArcConfig (default on,
  // 1500 MiB).
  SceneCache* scene_cache = nullptr;
  bool encoder_cache = false;
  int encoder_cache_mb = 1500;
};

// True iff a per-task deadline is set AND has passed. When no deadline is set
// (the default) this is always false, so every guarded loop runs to its fixed
// cap.
inline bool past_deadline(const SearchConfig& cfg) {
  return cfg.deadline.has_value() &&
         std::chrono::steady_clock::now() >= cfg.deadline.value();
}

// True iff a post-loop stop time is set (ARC_POST_GRACE) and has passed.
inline bool past_post_deadline(const SearchConfig& cfg) {
  return cfg.post_deadline.has_value() &&
         std::chrono::steady_clock::now() >= cfg.post_deadline.value();
}

// One generation's recorded state for the search trace: the ORDERED population
// program hashes (next_pop AFTER scoring, i.e. rank-sorted `scored`), the best
// genome's program hash, and the best fitness scalars going forward.
struct GenerationTrace {
  std::vector<std::string> pop_hashes;  // rank-sorted population program hashes
  std::string best_hash;
  bool best_solved = false;
  double best_exact_ratio = 0.0;
  double best_max_pair_mismatch = 0.0;
  double best_pixel_mismatch_mean = 0.0;
  double best_complexity = 0.0;
  int stagnation_count = 0;
};

// The result of one run_single_attempt: the best genome (at LOOP EXIT, or after
// the refinement passes when they ran) and the full per-generation trace
// (index 0 = the pre-loop scored population).
struct SearchTrace {
  TransformGenome best_genome;
  TransformFitnessFull best_fit;
  int generations_used = 0;
  std::vector<GenerationTrace> gens;  // gens[0] = initial (pre-loop) population
  // Phase-A enumeration record (populated only when the enumerator is ON): the
  // ordered, deduplicated enumerate_seeds list (program hashes, in emission
  // order) and the initial population program hashes after the quick-rank /
  // initial-population construction (pre-scoring order). Empty when off.
  std::vector<std::string> phasea_seed_hashes;      // enumerate_seeds order
  std::vector<std::string> phasea_initial_pop_hashes;  // initial population order
  // Refinement pass snapshots (populated only when cfg.run_refinement).
  // One entry per pass that RAN (a gated-out pass is skipped); each carries the
  // pass name + the best genome/fit AFTER it.
  // The final entry is the fully refined genome (== best_genome when refinement
  // ran; the pass tail also updates best_genome/best_fit in place).
  struct RefinePass {
    std::string name;  // structural | arg | delta | compress | decoder_sweep
    TransformGenome genome;
    TransformFitnessFull fit;
  };
  std::vector<RefinePass> refine_passes;
  bool refined = false;  // true iff cfg.run_refinement ran the passes
  // Finalist gather input: the loop-exit `scored` population (the last
  // generation's rank-sorted population, or the Phase-A/B pre-loop scored on the
  // early-return paths). An attempt's finalists are the rank-deduplicated
  // [(best, best_fit)] + scored, so run_task_full's finalist gather needs every
  // restart's loop-exit `scored`. NOTE: the post-loop refinement does NOT
  // re-score into `scored`, so this snapshot is the pre-refinement final
  // population.
  std::vector<ScoredGenome> final_scored;
};

// run_single_attempt: Phase A (see enable_seed_enumerator) + the initial
// population + the GA generation loop, then the post-loop refinements
// (structural/arg/delta/compress/decoder sweep) when cfg.run_refinement is set.
// The seed is seed_u64 when seed_is_u64 != 0, else the decimal seed_decimal.
SearchTrace run_single_attempt(const std::vector<TrainPair>& train_pairs,
                               const std::string& seed_decimal, int seed_is_u64,
                               unsigned long long seed_u64,
                               const SearchConfig& cfg);

// ---------------------------------------------------------------------------
// Multi-restart run_task + seed streams.
//
// Per-restart seed stream: restart_idx in [0, n_restarts) draws
//   seed = base_seed + restart_idx * RESTART_SEED_STEP   (RESTART_SEED_STEP=1009)
// Each restart is an INDEPENDENT ArcRandom(seed); the base seed is an input
// (the binary derives it per task with task_seed).
//
// Cross-restart selection: best_result tracks the rank-best (8-key
// is_better_fit) across restarts. The first train-solver is locked; with
// SOLVED_DIVERSITY_RESTARTS=0 (default) the loop BREAKS the moment a restart
// solves train (restart_idx - solved_at_idx >= 0), so extra restarts only run
// when nothing has solved yet.
//
// run_task is the plain restart loop: no deadline, no finalist gather or
// promotion, no cross-restart refine, no post-solve continuations. It returns
// the cross-restart best of the restarts' run_single_attempt results.
// run_task_full (below) adds all of these.

// RESTART_SEED_STEP: per-restart seed increment.
constexpr unsigned long long RESTART_SEED_STEP = 1009ULL;
// SOLVED_DIVERSITY_RESTARTS: extra restarts after first solve (default 0 ->
// stop-on-solve).
constexpr int SOLVED_DIVERSITY_RESTARTS = 0;

// One restart's contribution to the task result: its loop-exit best genome +
// fitness (the run_single_attempt output) plus the seed it ran on.
struct RestartResult {
  unsigned long long seed = 0;
  TransformGenome best_genome;
  TransformFitnessFull best_fit;
  int generations_used = 0;
};

// run_task result: the cross-restart best (returned genome) + every restart's
// per-restart best.
struct TaskResult {
  TransformGenome best_genome;
  TransformFitnessFull best_fit;
  int best_restart_idx = 0;      // which restart produced the returned best
  int solved_at_idx = -1;        // restart_idx of first train-solver, -1 if none
  std::vector<RestartResult> restarts;
};

// run_task: run up to n_restarts run_single_attempt calls
// on the +RESTART_SEED_STEP seed stream and return the cross-restart best.
// `base_seed` is the per-task seed (u64 or decimal). The SearchConfig knobs are
// shared across every restart (the codec is rebuilt per restart inside
// run_single_attempt from cfg.max_rules + cfg.color_prior).
TaskResult run_task(const std::vector<TrainPair>& train_pairs,
                    const std::string& base_seed_decimal, int seed_is_u64,
                    unsigned long long base_seed_u64, int n_restarts,
                    const SearchConfig& cfg);

// ---------------------------------------------------------------------------
// compute_delta_hint. Runs the genome on every train pair (predict_grid: the
// program executed with the genome's decoder), collects the wrong pixels, and
// aggregates the DeltaHint the delta-guided hill-climb consumes. `has_hint` is
// false when: any pair's prediction shape != target, the best-pair mismatch is
// 0 (solved) or > 0.20, or the total wrong-cell count is < 2. Ties for the
// most common color break to the FIRST-SEEN one, i.e. row-major scan order
// across the concatenated pairs.
DeltaHint compute_delta_hint(const TransformGenome& genome,
                             const std::vector<TrainPair>& train_pairs,
                             bool* has_hint);

// ---------------------------------------------------------------------------
// run_task tail (run_task_full).
//
// After the restarts: the post-solve continuations (occam ON, view ON, tabu
// OFF -- the arc_config.conf DEFAULT), the cross-restart refine (unsolved path
// only), and the finalist gather + LOO/spec/MDL promotion (rank-deduplicated
// finalists, LOO computed). Returns the PROMOTED primary genome + the full
// ordered finalist list + the view_solvers (feed attempt_2, never merged into
// finalists).
//
// Continuation seed offsets: each continuation's attempts
// seed at base_seed + OFFSET*RESTART_SEED_STEP, disjoint from the restart block.
constexpr int OCCAM_SEED_OFFSET = 101;   // depth-capped shallower-solver search
constexpr int TABU_SEED_OFFSET = 211;    // banned-selector restart (OFF by default)
constexpr int VIEW_SEED_OFFSET = 307;    // per-view continuation block
// N_FINALISTS: distinct (program, decoder) finalists retained.
constexpr int N_FINALISTS = 5;

// The DEFAULT continuation config (occam+view ON, tabu OFF). Each field can be
// toggled on its own.
struct TaskTailConfig {
  bool occam_continue = true;                // occam_continue (default 1)
  bool tabu_restart = false;                 // tabu_restart (default 0)
  // The view names (view_continue); "1" -> {"transpose","fliplr","flipud"}. Empty
  // -> off. apply_view supports transpose/fliplr/flipud/rot90/rot270/identity.
  std::vector<std::string> views = {"transpose", "fliplr", "flipud"};
};

// One augmentation-view solver: the view name + the view-space genome/fitness
// (with loo_fail computed on the view-space pairs). Never merged into finalists.
struct ViewSolver {
  std::string view;
  TransformGenome genome;
  TransformFitnessFull fit;
};

// The full run_task result: the promoted primary + the ordered finalist list +
// the view solvers + the first train-solver (kept for comparison).
struct TaskResultFull {
  TransformGenome primary_genome;
  TransformFitnessFull primary_fit;
  bool solved = false;                       // primary_fit.solved (task solved on train)
  std::vector<std::pair<TransformGenome, TransformFitnessFull>> finalists;
  std::vector<ViewSolver> view_solvers;
  bool has_first_solver = false;             // set on the solved path
  TransformGenome first_solver_genome;
  TransformFitnessFull first_solver_fit;
  int solved_at_idx = -1;
  int generations_used = 0;                  // total GA generations across restarts (search effort)
  // Per-task encoder scene cache telemetry (0 when the cache is off). Aggregated
  // into the end-of-run SUMMARY by main.cpp.
  long long scene_cache_hits = 0;
  long long scene_cache_misses = 0;
  long long scene_cache_evictions = 0;
  long long scene_cache_peak_bytes = 0;
};

// apply_view: map a grid into `view` space, with the executor's whole-grid
// ops. transpose/fliplr/flipud are self-inverse; rot90/rot270 pair up.
Grid apply_view_grid(const Grid& g, const std::string& view);

// loo_generalization: leave-one-pair-out generalization
// penalty for a TRAIN-SOLVED genome (lower == more general). Strips rules that pin
// a pair-unique input color and re-scores on ALL pairs; +1 per pair whose strip
// breaks the all-pairs solve. distance_mode gates the soft objective but `solved`
// is a pixel-exact gate, so distance_mode is irrelevant to the returned count.
int loo_generalization(const TransformGenome& genome,
                       const std::vector<TrainPair>& train_pairs,
                       const std::string& distance_mode,
                       SceneCache* scene_cache = nullptr);

// run_task_full: the per-task entry. Runs the multi-restart loop (with the
// deadline and scene cache cfg asks for) then the tail `tail` configures:
// occam continuation, tabu, the cross-restart refine (unsolved path only),
// finalist gather + LOO + MDL promotion, view continuation. The seed
// enumerator, macros and delta seeds follow cfg.
TaskResultFull run_task_full(const std::vector<TrainPair>& train_pairs,
                             const std::string& base_seed_decimal, int seed_is_u64,
                             unsigned long long base_seed_u64, int n_restarts,
                             const SearchConfig& cfg, const TaskTailConfig& tail);

}  // namespace arc

#endif  // ARC_SEARCH_HPP
