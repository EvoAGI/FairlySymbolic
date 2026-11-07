// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_FITNESS_HPP
#define ARC_FITNESS_HPP

// Fitness:
//   * object_distance (sorted greedy IoU match + 5 weighted terms; square roots
//     are an UNFOLDED pow(x, 0.5), never sqrt).
//   * target_bounded_mismatch.
//   * the combined soft score (graph = obj_d + 0.3*pix_d, pixel = pix_d), the
//     MAX_RENDER_CELLS guard, and the coarse / fine / fine2 lexicase case
//     vectors.
//
// FLOAT ORDER. The distance terms accumulate float sub-values in a FIXED order
// (the greedy-match order over the candidates sorted in descending order, then
// the per-matched-pair scan order). Every square root is pow_half() -- a
// genuine pow(x, 0.5) call kept off the sqrt fast path via an fp barrier and
// the build's -fno-builtin-pow flag -- so its result, which can differ from
// sqrt(x) in the last bit, does not depend on compiler optimizations. Squares
// are x*x. The build has -ffp-contract=off so no term fuses into an FMA.

#include <string>
#include <vector>

#include "arc/config.hpp"
#include "arc/dsl.hpp"
#include "arc/grid.hpp"

namespace arc {

// Per-task encoder scene cache (scene_cache.hpp). Non-owning pointer only.
class SceneCache;

// ---------------------------------------------------------------------------
// Object extraction. A 4-connected, single-color region of non-background
// cells. Determinism: row-major scan + stack flood-fill -> stable object order.
// ---------------------------------------------------------------------------

struct FitObject {
  int color = 0;
  // Absolute pixel set, insertion order irrelevant (membership only). Stored as
  // a sorted vector of encoded (y*W + x) for O(log n) intersection.
  std::vector<long long> pixels;   // sorted, absolute y*Wgrid + x keys
  int ymin = 0, xmin = 0, ymax = 0, xmax = 0;
  int size = 0;
  // Normalized shape: sorted vector of encoded (dy * STRIDE + dx), dy=y-ymin.
  std::vector<long long> norm;     // sorted, translation-normalized shape keys
};

// Most-frequent color (ties -> smallest color): the encoder-free background
// proxy.
int fit_background_color(const Grid& grid);

// 4-connected single-color components of non-background cells.
std::vector<FitObject> extract_objects(const Grid& grid);
std::vector<FitObject> extract_objects(const Grid& grid, int background);

// ---------------------------------------------------------------------------
// object_distance. The 5 raw components in [0,1]:
// (count_d, match_d, color_d, shape_d, bbox_d); combine = weighted sum / W_SUM
// clamped to 1.0. object_distance == combine(components).
// ---------------------------------------------------------------------------

struct ObjDistComponents {
  double count_d = 0.0;
  double match_d = 0.0;
  double color_d = 0.0;
  double shape_d = 0.0;
  double bbox_d = 0.0;
};

ObjDistComponents object_distance_components(const std::vector<FitObject>& pred_objs,
                                             const std::vector<FitObject>& gt_objs,
                                             int grid_h, int grid_w);

double combine_components(const ObjDistComponents& c);

double object_distance(const std::vector<FitObject>& pred_objs,
                       const std::vector<FitObject>& gt_objs,
                       int grid_h, int grid_w);

// ---------------------------------------------------------------------------
// Pixel-bounded mismatch: pixel
// disagreement normalized by the target's non-background pixel count (envelope
// fallback when the target is all-background). Returns [0,1].
// ---------------------------------------------------------------------------

double target_bounded_mismatch(const Grid& pred, const Grid& target);

// Per-color pixel disagreement: one
// value in [0,1] per color in `target_colors` (sorted distinct target colors).
std::vector<double> per_color_pixel_errors(const Grid& pred, const Grid& target,
                                           const std::vector<int>& target_colors);

// Pixel mismatch per target quadrant (TL,TR,BL,BR), each [0,1].
std::vector<double> quadrant_pixel_errors(const Grid& pred, const Grid& target);

// ---------------------------------------------------------------------------
// The square-root primitive object_distance uses: a genuine pow(x, 0.5) that
// must NOT fold to sqrt (the two can differ in the last bit).
// ---------------------------------------------------------------------------

double pow_half(double x);

// ---------------------------------------------------------------------------
// The MAX_RENDER_CELLS guard. Projected output cells for
// a whole-grid transform on an (h x w) input.
// ---------------------------------------------------------------------------

constexpr int MAX_RENDER_CELLS = 4096;
long long projected_output_cells(int h, int w, GlobalTransform t);

// ---------------------------------------------------------------------------
// Full per-pair fitness. Renders the prediction (encode, execute, decode) for
// the given (grid, program, encoder_config, decoder_config), then scores it
// against the target. On a single (grid,target) pair this equals the genome
// evaluation on a one-pair train set (aggregate score IS the pair soft score).
// ---------------------------------------------------------------------------

// distance_mode: "graph" (obj_d + 0.3*pix_d) or "pixel" (pix_d).
double fitness_score(const Grid& grid, const Grid& target,
                     const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                     const DslProgram& program, const std::string& distance_mode,
                     SceneCache* scene_cache = nullptr);

// case_mode: "coarse" | "fine" | "fine2". distance_mode: "graph" | "pixel".
// Returns the lexicase case vector for the single pair.
std::vector<double> fitness_cases(const Grid& grid, const Grid& target,
                                  const GraphConfig& encoder_cfg,
                                  const DecoderConfig& decoder_cfg,
                                  const DslProgram& program,
                                  const std::string& distance_mode,
                                  const std::string& case_mode,
                                  SceneCache* scene_cache = nullptr);

// Combined single-pair evaluation for evaluate_genome: renders the prediction
// ONCE and returns whether the pair is pixel-exact, the pair soft score, and
// (when collect_cases) its lexicase case vector. `exact` is the true
// pixel-equality gate; when exact the soft score is 0.0 and the cases are the
// const-0 vector.
// On the MAX_RENDER_CELLS guard the soft score is 1.0 + PIXEL_SMOOTH_WEIGHT and
// the cases are the const-1 vector. Reuses fitness_score / fitness_cases logic
// but avoids a double render.
struct PairEval {
  bool exact = false;
  double soft = 1.0;
  std::vector<double> cases;
};

PairEval fitness_pair_eval(const Grid& grid, const Grid& target,
                           const GraphConfig& encoder_cfg,
                           const DecoderConfig& decoder_cfg,
                           const DslProgram& program,
                           const std::string& distance_mode,
                           bool collect_cases, const std::string& case_mode,
                           SceneCache* scene_cache = nullptr);

// quick_seed_score: fast single-pair mismatch estimate
// used to RANK Phase-A seeds before the full multi-pair evaluate. Renders the
// prediction with the genome's encoder + decoder (predict_grid), then applies
// the exact float formula:
//   MAX_RENDER_CELLS guard  -> 1e9
//   mismatch = (diff_common + extra) / denom
//   + 0.1 * size_gap + 0.01 * n_rules
// (the executor never throws, so only the render-cells guard produces the 1e9
// sentinel.)
double quick_seed_score(const Grid& inp, const Grid& target,
                        const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                        const DslProgram& program,
                        SceneCache* scene_cache = nullptr);

// predict_grid: the decoded prediction with the genome's OWN DecoderConfig +
// input grid (all three executor branches). This is exactly the `pred`
// compute_delta_hint diffs against the target (fitness_pair_eval uses this
// internally too).
Grid predict_grid(const Grid& grid, const DslProgram& program,
                  const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                  SceneCache* scene_cache = nullptr);

}  // namespace arc

#endif  // ARC_FITNESS_HPP
