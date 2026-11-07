// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Fitness: object distance, target-bounded pixel mismatch, and the per-pair
// scores and case vectors of a transform genome.
//
// See fitness.hpp for the floating-point contract. Every square root goes
// through pow_half() (an unfolded pow(x, 0.5), NOT sqrt); float terms
// accumulate in a fixed order; the build supplies -ffp-contract=off +
// -fno-builtin-pow.

#include "arc/fitness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_map>

#include "arc/encode.hpp"

namespace arc {

// ---------------------------------------------------------------------------
// Float barrier + pow_half.
//
// Forces the operand through volatile memory so the optimizer cannot fold
// pow(x, 0.5) into a sqrt (or otherwise reassociate). Same fp_barrier pattern
// as rng.cpp (which keeps cos()/sin() as separate libm calls). Combined with
// the build's -fno-builtin-pow this always calls the C library pow(x, 0.5),
// which differs from sqrt(x) on ~0.1% of arguments.
// ---------------------------------------------------------------------------
namespace {
inline double fp_barrier(double x) {
  volatile double t = x;
  return t;
}
}  // namespace

double pow_half(double x) {
  // std::pow with the exponent routed through the barrier; the barrier + the
  // -fno-builtin-pow build flag keep this a genuine libm pow call.
  return std::pow(x, fp_barrier(0.5));
}

// ---------------------------------------------------------------------------
// Component weights of the object distance.
// ---------------------------------------------------------------------------
namespace {
constexpr double W_COUNT = 1.0;
constexpr double W_MATCH = 2.0;
constexpr double W_COLOR = 1.0;
constexpr double W_SHAPE = 1.0;
constexpr double W_BBOX = 0.5;
constexpr double W_SUM = W_COUNT + W_MATCH + W_COLOR + W_SHAPE + W_BBOX;  // 5.5
constexpr int MAX_MATCH_OBJECTS = 200;
constexpr double PIXEL_SMOOTH_WEIGHT = 0.3;
}  // namespace

// ---------------------------------------------------------------------------
// Object extraction (fit_background_color / extract_objects).
// ---------------------------------------------------------------------------

int fit_background_color(const Grid& grid) {
  // The most frequent color; on a tie, the smallest color (ascending scan over
  // the count table with strict '>').
  int counts[256];
  std::memset(counts, 0, sizeof(counts));
  int h = grid.height(), w = grid.width();
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int c = grid.at(y, x);
      if (c >= 0 && c < 256) ++counts[c];
    }
  int best = 0;
  long long best_count = -1;
  for (int c = 0; c < 256; ++c) {
    if (counts[c] > 0 && counts[c] > best_count) {
      best_count = counts[c];
      best = c;
    }
  }
  return best;
}

std::vector<FitObject> extract_objects(const Grid& grid) {
  return extract_objects(grid, fit_background_color(grid));
}

std::vector<FitObject> extract_objects(const Grid& grid, int background) {
  int h = grid.height(), w = grid.width();
  std::vector<char> visited(static_cast<size_t>(h) * w, 0);
  std::vector<FitObject> objects;
  // 4-neighbour order: (1,0),(-1,0),(0,1),(0,-1). The order does not change
  // the final object (a pixel set).
  static const int NY[4] = {1, -1, 0, 0};
  static const int NX[4] = {0, 0, 1, -1};
  std::vector<std::pair<int, int>> stack, pixels;  // reused per object
  for (int sy = 0; sy < h; ++sy) {
    for (int sx = 0; sx < w; ++sx) {
      size_t sidx = static_cast<size_t>(sy) * w + sx;
      if (visited[sidx]) continue;
      int color = grid.at(sy, sx);
      if (color == background) {
        visited[sidx] = 1;
        continue;
      }
      // Stack flood-fill (row-major scan + LIFO stack).
      stack.clear();
      stack.emplace_back(sy, sx);
      visited[sidx] = 1;
      pixels.clear();
      int ymin = sy, ymax = sy, xmin = sx, xmax = sx;
      while (!stack.empty()) {
        auto [cy, cx] = stack.back();
        stack.pop_back();
        pixels.emplace_back(cy, cx);
        if (cy < ymin) ymin = cy;
        else if (cy > ymax) ymax = cy;
        if (cx < xmin) xmin = cx;
        else if (cx > xmax) xmax = cx;
        for (int d = 0; d < 4; ++d) {
          int ny = cy + NY[d], nx = cx + NX[d];
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          size_t nidx = static_cast<size_t>(ny) * w + nx;
          if (visited[nidx]) continue;
          if (grid.at(ny, nx) == color) {
            visited[nidx] = 1;
            stack.emplace_back(ny, nx);
          }
        }
      }
      FitObject obj;
      obj.color = color;
      obj.ymin = ymin;
      obj.xmin = xmin;
      obj.ymax = ymax;
      obj.xmax = xmax;
      obj.size = static_cast<int>(pixels.size());
      obj.pixels.reserve(pixels.size());
      obj.norm.reserve(pixels.size());
      // Encode absolute pixels as y*STRIDE + x and the translation-normalized
      // shape as (y-ymin)*STRIDE + (x-xmin). A FIXED STRIDE (independent of the
      // grid width) is required so absolute keys are comparable across grids of
      // DIFFERENT widths: IoU is computed between pred and target objects whose
      // grids may differ in shape. STRIDE > any grid coordinate so (y,x) packs
      // uniquely.
      const long long STRIDE = 4096;  // > max grid dim.
      for (auto& [py, px] : pixels) {
        obj.pixels.push_back(static_cast<long long>(py) * STRIDE + px);
        long long dy = py - ymin, dx = px - xmin;
        obj.norm.push_back(dy * STRIDE + dx);
      }
      std::sort(obj.pixels.begin(), obj.pixels.end());
      std::sort(obj.norm.begin(), obj.norm.end());
      objects.push_back(std::move(obj));
    }
  }
  return objects;
}

// ---------------------------------------------------------------------------
// object_distance internals.
// ---------------------------------------------------------------------------
namespace {

bool bbox_overlap(const FitObject& a, const FitObject& b) {
  return !(a.ymax < b.ymin || b.ymax < a.ymin || a.xmax < b.xmin || b.xmax < a.xmin);
}

// |a & b| over two sorted key vectors.
int set_intersection_size(const std::vector<long long>& a,
                          const std::vector<long long>& b) {
  int inter = 0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) ++i;
    else if (b[j] < a[i]) ++j;
    else { ++inter; ++i; ++j; }
  }
  return inter;
}

double iou(const FitObject& a, const FitObject& b) {
  int inter = set_intersection_size(a.pixels, b.pixels);
  if (inter == 0) return 0.0;
  // inter / (|a| + |b| - inter); the integer denominator is exact.
  return static_cast<double>(inter) /
         static_cast<double>(static_cast<int>(a.pixels.size()) +
                             static_cast<int>(b.pixels.size()) - inter);
}

struct Candidate {
  double iou;
  int i;
  int j;
};

}  // namespace

ObjDistComponents object_distance_components(const std::vector<FitObject>& pred_objs,
                                             const std::vector<FitObject>& gt_objs,
                                             int grid_h, int grid_w) {
  int n_pred = static_cast<int>(pred_objs.size());
  int n_gt = static_cast<int>(gt_objs.size());
  if (n_pred == 0 && n_gt == 0) {
    return {0.0, 0.0, 0.0, 0.0, 0.0};
  }
  int denom_i = n_pred > n_gt ? n_pred : n_gt;
  // count_d = abs(n_pred - n_gt) / denom  (int diff, int denom -> exact double).
  double count_d = static_cast<double>(std::abs(n_pred - n_gt)) /
                   static_cast<double>(denom_i);

  if (n_pred > MAX_MATCH_OBJECTS || n_gt > MAX_MATCH_OBJECTS) {
    return {count_d, count_d, count_d, count_d, count_d};
  }

  // Candidate list: pred-major, gt-minor (i outer, j inner).
  std::vector<Candidate> candidates;
  for (int i = 0; i < n_pred; ++i) {
    for (int j = 0; j < n_gt; ++j) {
      if (!bbox_overlap(pred_objs[i], gt_objs[j])) continue;
      double v = iou(pred_objs[i], gt_objs[j]);
      if (v > 0.0) candidates.push_back({v, i, j});
    }
  }
  // Sort descending on the tuple (iou, i, j): stable_sort with a strict
  // descending tuple compare.
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& a, const Candidate& b) {
                     if (a.iou != b.iou) return a.iou > b.iou;
                     if (a.i != b.i) return a.i > b.i;
                     return a.j > b.j;
                   });

  std::vector<char> used_p(n_pred, 0);
  std::vector<char> used_g(n_gt, 0);
  // matches in greedy-accept order (this is the accumulation order for the
  // matched_cost sum and the per-pair attribute scan).
  std::vector<Candidate> matches;
  for (const Candidate& c : candidates) {
    if (!used_p[c.i] && !used_g[c.j]) {
      used_p[c.i] = 1;
      used_g[c.j] = 1;
      matches.push_back(c);
    }
  }

  int n_matched = static_cast<int>(matches.size());
  // matched_cost = sum(1.0 - iou) in match order.
  double matched_cost = 0.0;
  for (const Candidate& m : matches) matched_cost += 1.0 - m.iou;
  int unmatched = denom_i - n_matched;
  // match_d = (matched_cost + unmatched) / denom.
  double match_d = (matched_cost + static_cast<double>(unmatched)) /
                   static_cast<double>(denom_i);

  if (n_matched == 0) {
    return {count_d, match_d, 1.0, 1.0, 1.0};
  }

  int color_mis = 0;
  int shape_mis = 0;
  double bbox_acc = 0.0;
  // diag = sqrt(grid_h^2 + grid_w^2), or 1.0 when that is 0.
  double diag = pow_half(static_cast<double>(grid_h) * grid_h +
                         static_cast<double>(grid_w) * grid_w);
  if (diag == 0.0) diag = 1.0;
  for (const Candidate& m : matches) {
    const FitObject& po = pred_objs[m.i];
    const FitObject& go = gt_objs[m.j];
    if (po.color != go.color) ++color_mis;
    if (po.norm != go.norm) ++shape_mis;  // normalized pixel sets, as sorted vectors
    // centers: (ymin+ymax)/2.0, (xmin+xmax)/2.0.
    double pcy = (static_cast<double>(po.ymin) + po.ymax) / 2.0;
    double pcx = (static_cast<double>(po.xmin) + po.xmax) / 2.0;
    double gcy = (static_cast<double>(go.ymin) + go.ymax) / 2.0;
    double gcx = (static_cast<double>(go.xmin) + go.xmax) / 2.0;
    double ddy = pcy - gcy;
    double ddx = pcx - gcx;
    // center_d = sqrt(ddy^2 + ddx^2) / diag.
    double center_d = pow_half(ddy * ddy + ddx * ddx) / diag;
    // size_d = (|pred h - gt h| + |pred w - gt w|) / (grid_h+grid_w), with
    // h = ymax - ymin and w = xmax - xmin.
    int psh = po.ymax - po.ymin;
    int gsh = go.ymax - go.ymin;
    int psw = po.xmax - po.xmin;
    int gsw = go.xmax - go.xmin;
    double size_d = static_cast<double>(std::abs(psh - gsh) + std::abs(psw - gsw)) /
                    static_cast<double>(grid_h + grid_w);
    // bbox_acc += min((center_d + size_d) / 2.0, 1.0).
    double term = (center_d + size_d) / 2.0;
    if (term > 1.0) term = 1.0;
    bbox_acc += term;
  }

  double color_d = static_cast<double>(color_mis) / static_cast<double>(n_matched);
  double shape_d = static_cast<double>(shape_mis) / static_cast<double>(n_matched);
  double bbox_d = bbox_acc / static_cast<double>(n_matched);
  return {count_d, match_d, color_d, shape_d, bbox_d};
}

double combine_components(const ObjDistComponents& c) {
  // total = W_COUNT*count + W_MATCH*match + W_COLOR*color + W_SHAPE*shape + W_BBOX*bbox
  double total = W_COUNT * c.count_d + W_MATCH * c.match_d + W_COLOR * c.color_d +
                 W_SHAPE * c.shape_d + W_BBOX * c.bbox_d;
  double v = total / W_SUM;
  return v < 1.0 ? v : 1.0;
}

double object_distance(const std::vector<FitObject>& pred_objs,
                       const std::vector<FitObject>& gt_objs,
                       int grid_h, int grid_w) {
  return combine_components(
      object_distance_components(pred_objs, gt_objs, grid_h, grid_w));
}

// ---------------------------------------------------------------------------
// Pixel mismatch (target_bounded_mismatch).
// ---------------------------------------------------------------------------

double target_bounded_mismatch(const Grid& pred, const Grid& target) {
  int ph = pred.height(), pw = pred.width();
  int th = target.height(), tw = target.width();
  int common_h = ph < th ? ph : th;
  int common_w = pw < tw ? pw : tw;
  long long diff_common = 0;
  for (int y = 0; y < common_h; ++y)
    for (int x = 0; x < common_w; ++x)
      if (pred.at(y, x) != target.at(y, x)) ++diff_common;
  long long extra = (static_cast<long long>(ph) * pw -
                     static_cast<long long>(common_h) * common_w) +
                    (static_cast<long long>(th) * tw -
                     static_cast<long long>(common_h) * common_w);
  long long wrong = diff_common + extra;

  long long target_content = 0;
  for (int y = 0; y < th; ++y)
    for (int x = 0; x < tw; ++x)
      if (target.at(y, x) != 0) ++target_content;

  if (target_content == 0) {
    long long mh = ph > th ? ph : th;
    long long mw = pw > tw ? pw : tw;
    long long denom = mh * mw;
    if (denom < 1) denom = 1;
    // wrong / max(denom, 1), as a double.
    return static_cast<double>(wrong) / static_cast<double>(denom);
  }
  double v = static_cast<double>(wrong) / static_cast<double>(target_content);
  return v < 1.0 ? v : 1.0;
}

std::vector<double> per_color_pixel_errors(const Grid& pred, const Grid& target,
                                           const std::vector<int>& target_colors) {
  int ph = pred.height(), pw = pred.width();
  int th = target.height(), tw = target.width();
  if (ph != th || pw != tw) {
    return std::vector<double>(target_colors.size(), 1.0);
  }
  long long n = static_cast<long long>(th) * tw;
  std::vector<double> out;
  out.reserve(target_colors.size());
  for (int c : target_colors) {
    // mean( (target==c) != (pred==c) ) over all cells.
    long long diff = 0;
    for (int y = 0; y < th; ++y)
      for (int x = 0; x < tw; ++x) {
        bool tc = target.at(y, x) == c;
        bool pc = pred.at(y, x) == c;
        if (tc != pc) ++diff;
      }
    // The fraction of cells that differ.
    out.push_back(static_cast<double>(diff) / static_cast<double>(n));
  }
  return out;
}

std::vector<double> quadrant_pixel_errors(const Grid& pred, const Grid& target) {
  int ph = pred.height(), pw = pred.width();
  int th = target.height(), tw = target.width();
  if (ph != th || pw != tw) {
    return {1.0, 1.0, 1.0, 1.0};
  }
  int h = th, w = tw;
  int hy = (h + 1) / 2;
  int hx = (w + 1) / 2;
  // Quadrants as half-open [y0:y1, x0:x1) ranges:
  // (0:hy,0:hx),(0:hy,hx:w),(hy:h,0:hx),(hy:h,hx:w).
  const int ys0[4] = {0, 0, hy, hy};
  const int ys1[4] = {hy, hy, h, h};
  const int xs0[4] = {0, hx, 0, hx};
  const int xs1[4] = {hx, w, hx, w};
  std::vector<double> out;
  out.reserve(4);
  for (int q = 0; q < 4; ++q) {
    long long size = static_cast<long long>(ys1[q] - ys0[q]) * (xs1[q] - xs0[q]);
    if (size == 0) {
      out.push_back(0.0);  // empty quadrant -> 0.0
      continue;
    }
    long long diff = 0;
    for (int y = ys0[q]; y < ys1[q]; ++y)
      for (int x = xs0[q]; x < xs1[q]; ++x)
        if (pred.at(y, x) != target.at(y, x)) ++diff;
    out.push_back(static_cast<double>(diff) / static_cast<double>(size));
  }
  return out;
}

// ---------------------------------------------------------------------------
// MAX_RENDER_CELLS guard (projected_output_cells).
// ---------------------------------------------------------------------------

long long projected_output_cells(int h, int w, GlobalTransform t) {
  long long hh = h, ww = w;
  switch (t) {
    case GlobalTransform::IDENTITY:
    case GlobalTransform::ROT90:
    case GlobalTransform::ROT180:
    case GlobalTransform::ROT270:
    case GlobalTransform::FLIP_H:
    case GlobalTransform::FLIP_V:
    case GlobalTransform::TRANSPOSE:
    case GlobalTransform::ANTI_TRANSPOSE:
    case GlobalTransform::CROP_FOREGROUND:
    case GlobalTransform::FILL_PERIODIC_BACKGROUND:
      return hh * ww;
    case GlobalTransform::SCALE2_NEAREST:
      return (hh * 2) * (ww * 2);
    case GlobalTransform::SCALE3_NEAREST:
      return (hh * 3) * (ww * 3);
    case GlobalTransform::KRON_SELF_MASK:
    case GlobalTransform::KRON_MASK_ANCHOR_COLOR:
      return (hh * hh) * (ww * ww);
    case GlobalTransform::MIRROR_CONCAT_H:
    case GlobalTransform::MIRROR_CONCAT_H_REV:
    case GlobalTransform::TILE_H:
    case GlobalTransform::MIRROR_CONCAT_V:
    case GlobalTransform::MIRROR_CONCAT_V_REV:
    case GlobalTransform::TILE_V:
      return (hh * ww) * 2;
    case GlobalTransform::MIRROR_QUAD_TL:
    case GlobalTransform::MIRROR_QUAD_TR:
    case GlobalTransform::MIRROR_QUAD_BL:
    case GlobalTransform::MIRROR_QUAD_BR:
      return (hh * ww) * 4;
    default:
      // DEDUP_RUNS / BLOCK_REDUCE2 only shrink (<= h*w).
      return hh * ww;
  }
}

// ---------------------------------------------------------------------------
// Prediction: run the program and render with the genome's NON-DEFAULT
// DecoderConfig + the input grid.
//
// execute_program_final_grid renders with the DEFAULT DecoderConfig via
// render_scene_default. Fitness needs the genome's decoder, so the three
// branches are repeated here with render_scene_to_grid(scene, cfg, &grid).
// ---------------------------------------------------------------------------

Grid predict_grid(const Grid& grid, const DslProgram& program,
                  const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                  SceneCache* scene_cache) {
  // Branch 1: empty-rule + fast-path transform -> whole-grid function on RAW input.
  if (program.rules.empty() && is_fast_path_transform(program.global_transform)) {
    return apply_full_grid_global(grid, program.global_transform);
  }
  // Run the rules to get the mutated G_out scene.
  Scene scene = execute_program_gout(grid, program, encoder_cfg, scene_cache);
  // Branch 2: with-rules full-grid transform -> render (decoder+input), then op.
  if (is_full_grid_transform(program.global_transform)) {
    Grid rendered = render_scene_to_grid(scene, decoder_cfg, &grid);
    return apply_full_grid_global(rendered, program.global_transform);
  }
  // Branch 3: foreground-only scene transform, then decode with decoder+input.
  apply_global_transform_to_scene(scene, program.global_transform);
  return render_scene_to_grid(scene, decoder_cfg, &grid);
}

namespace {

bool grids_equal(const Grid& a, const Grid& b) {
  if (a.height() != b.height() || a.width() != b.width()) return false;
  return a.cells() == b.cells();
}

std::vector<int> sorted_distinct_colors(const Grid& g) {
  bool seen[256];
  std::memset(seen, 0, sizeof(seen));
  int h = g.height(), w = g.width();
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int c = g.at(y, x);
      if (c >= 0 && c < 256) seen[c] = true;
    }
  std::vector<int> out;
  for (int c = 0; c < 256; ++c)
    if (seen[c]) out.push_back(c);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// fitness_score / fitness_cases: single-pair evaluate_genome.
// ---------------------------------------------------------------------------

double fitness_score(const Grid& grid, const Grid& target,
                     const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                     const DslProgram& program, const std::string& distance_mode,
                     SceneCache* scene_cache) {
  bool pixel_mode = (distance_mode == "pixel");
  // MAX_RENDER_CELLS guard: fail the pair BEFORE rendering.
  if (projected_output_cells(grid.height(), grid.width(),
                             program.global_transform) > MAX_RENDER_CELLS) {
    return 1.0 + PIXEL_SMOOTH_WEIGHT;
  }
  Grid pred = predict_grid(grid, program, encoder_cfg, decoder_cfg, scene_cache);
  // Pixel-exact gate -> score 0.0.
  if (grids_equal(pred, target)) return 0.0;

  double pix_d = target_bounded_mismatch(pred, target);
  if (pixel_mode) return pix_d;
  std::vector<FitObject> pred_objs = extract_objects(pred);
  std::vector<FitObject> tgt_objs = extract_objects(target);
  double obj_d = object_distance(pred_objs, tgt_objs, pred.height(), pred.width());
  return obj_d + PIXEL_SMOOTH_WEIGHT * pix_d;
}

std::vector<double> fitness_cases(const Grid& grid, const Grid& target,
                                  const GraphConfig& encoder_cfg,
                                  const DecoderConfig& decoder_cfg,
                                  const DslProgram& program,
                                  const std::string& distance_mode,
                                  const std::string& case_mode,
                                  SceneCache* scene_cache) {
  bool pixel_mode = (distance_mode == "pixel");
  bool fine = (case_mode == "fine" || case_mode == "fine2");
  bool quads = (case_mode == "fine2");
  // Target colors: the FIXED key set for the fine per-color cases.
  std::vector<int> target_colors = sorted_distinct_colors(target);
  int n_color = static_cast<int>(target_colors.size());

  auto fine_len = [&]() -> int {
    int base = pixel_mode ? (1 + n_color) : (5 + 1 + n_color);
    return base + (quads ? 4 : 0);
  };
  auto const_cases = [&](double val) -> std::vector<double> {
    if (fine) return std::vector<double>(fine_len(), val);
    return pixel_mode ? std::vector<double>{val}
                      : std::vector<double>{val, val};
  };

  // MAX_RENDER_CELLS guard -> constant 1.0 case vector.
  if (projected_output_cells(grid.height(), grid.width(),
                             program.global_transform) > MAX_RENDER_CELLS) {
    return const_cases(1.0);
  }
  Grid pred = predict_grid(grid, program, encoder_cfg, decoder_cfg, scene_cache);
  // Pixel-exact gate -> constant 0.0 case vector.
  if (grids_equal(pred, target)) return const_cases(0.0);

  double pix_d = target_bounded_mismatch(pred, target);
  if (!fine) {
    // Coarse: pixel -> (pix_d); graph -> (graph_d, pix_d).
    if (pixel_mode) return {pix_d};
    std::vector<FitObject> pred_objs = extract_objects(pred);
    std::vector<FitObject> tgt_objs = extract_objects(target);
    double graph_d = object_distance(pred_objs, tgt_objs, pred.height(), pred.width());
    return {graph_d, pix_d};
  }

  // Fine / fine2.
  std::vector<double> per_color = per_color_pixel_errors(pred, target, target_colors);
  std::vector<double> cases;
  if (pixel_mode) {
    cases.push_back(pix_d);
    for (double v : per_color) cases.push_back(v);
  } else {
    std::vector<FitObject> pred_objs = extract_objects(pred);
    std::vector<FitObject> tgt_objs = extract_objects(target);
    ObjDistComponents comps =
        object_distance_components(pred_objs, tgt_objs, pred.height(), pred.width());
    cases.push_back(comps.count_d);
    cases.push_back(comps.match_d);
    cases.push_back(comps.color_d);
    cases.push_back(comps.shape_d);
    cases.push_back(comps.bbox_d);
    cases.push_back(pix_d);
    for (double v : per_color) cases.push_back(v);
  }
  if (quads) {
    std::vector<double> qs = quadrant_pixel_errors(pred, target);
    for (double v : qs) cases.push_back(v);
  }
  return cases;
}

namespace {

// extract_objects of a train target, kept per thread for the last few targets
// (a search scores every genome against the same few grids).
const std::vector<FitObject>& target_objects(const Grid& target) {
  struct Entry {
    Grid grid;
    std::vector<FitObject> objs;
  };
  thread_local std::vector<Entry> recent;  // most recent last, at most 8
  for (size_t i = recent.size(); i-- > 0;)
    if (recent[i].grid.height() == target.height() && recent[i].grid.width() == target.width() &&
        recent[i].grid.cells() == target.cells()) {
      if (i + 1 != recent.size()) std::swap(recent[i], recent.back());
      return recent.back().objs;
    }
  if (recent.size() == 8) recent.erase(recent.begin());
  recent.push_back(Entry{target, extract_objects(target)});
  return recent.back().objs;
}

// The scores of a prediction that misses its target are a function of the
// predicted grid, the target and the scoring mode alone, and a population
// repeats predictions (neutral mutations, shared programs). Kept per thread
// for up to 2048 (prediction, target, mode) triples, the whole table dropped
// when full; a hit is checked against the stored grids and mode, so it gives
// exactly what scoring again would.
struct PairMemo {
  struct Entry {
    Grid pred;
    std::shared_ptr<const Grid> target;
    std::string mode;
    double soft;
    std::vector<double> cases;
  };
  std::unordered_multimap<uint64_t, Entry> table;
  std::vector<std::shared_ptr<const Grid>> targets;  // recent targets, shared by the entries

  static uint64_t hash_grid(const Grid& g, uint64_t h) {
    h = (h ^ static_cast<uint64_t>(g.height())) * 1099511628211ULL;
    h = (h ^ static_cast<uint64_t>(g.width())) * 1099511628211ULL;
    for (int v : g.cells()) h = (h ^ static_cast<uint64_t>(static_cast<uint32_t>(v))) * 1099511628211ULL;
    return h;
  }
  static bool same(const Grid& a, const Grid& b) {
    return a.height() == b.height() && a.width() == b.width() && a.cells() == b.cells();
  }
  std::shared_ptr<const Grid> target_copy(const Grid& t) {
    for (const auto& g : targets)
      if (same(*g, t)) return g;
    if (targets.size() == 16) targets.erase(targets.begin());
    targets.push_back(std::make_shared<const Grid>(t));
    return targets.back();
  }
};

PairMemo& pair_memo() {
  thread_local PairMemo m;
  return m;
}

void score_miss(PairEval& r, const Grid& pred, const Grid& target, bool pixel_mode, bool collect_cases,
                bool fine, bool quads, const std::vector<int>& target_colors);

}  // namespace

PairEval fitness_pair_eval(const Grid& grid, const Grid& target,
                           const GraphConfig& encoder_cfg,
                           const DecoderConfig& decoder_cfg,
                           const DslProgram& program,
                           const std::string& distance_mode,
                           bool collect_cases, const std::string& case_mode,
                           SceneCache* scene_cache) {
  bool pixel_mode = (distance_mode == "pixel");
  bool fine = collect_cases && (case_mode == "fine" || case_mode == "fine2");
  bool quads = (case_mode == "fine2");
  std::vector<int> target_colors;
  if (fine) target_colors = sorted_distinct_colors(target);
  int n_color = static_cast<int>(target_colors.size());
  auto fine_len = [&]() -> int {
    int base = pixel_mode ? (1 + n_color) : (5 + 1 + n_color);
    return base + (quads ? 4 : 0);
  };
  auto const_cases = [&](double val) -> std::vector<double> {
    if (!collect_cases) return {};
    if (fine) return std::vector<double>(fine_len(), val);
    return pixel_mode ? std::vector<double>{val}
                      : std::vector<double>{val, val};
  };

  PairEval r;
  // MAX_RENDER_CELLS guard: fail the pair BEFORE rendering.
  if (projected_output_cells(grid.height(), grid.width(),
                             program.global_transform) > MAX_RENDER_CELLS) {
    r.exact = false;
    r.soft = 1.0 + PIXEL_SMOOTH_WEIGHT;
    r.cases = const_cases(1.0);
    return r;
  }
  Grid pred = predict_grid(grid, program, encoder_cfg, decoder_cfg, scene_cache);
  // Pixel-exact gate.
  if (grids_equal(pred, target)) {
    r.exact = true;
    r.soft = 0.0;
    r.cases = const_cases(0.0);
    return r;
  }
  r.exact = false;
  PairMemo& memo = pair_memo();
  std::string mode = distance_mode + '\x1f' + (collect_cases ? '1' : '0') + case_mode;
  uint64_t key = PairMemo::hash_grid(target, PairMemo::hash_grid(pred, 1469598103934665603ULL));
  for (char ch : mode) key = (key ^ static_cast<unsigned char>(ch)) * 1099511628211ULL;
  auto range = memo.table.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    const PairMemo::Entry& e = it->second;
    if (e.mode == mode && PairMemo::same(e.pred, pred) && PairMemo::same(*e.target, target)) {
      r.soft = e.soft;
      r.cases = e.cases;
      return r;
    }
  }
  score_miss(r, pred, target, pixel_mode, collect_cases, fine, quads, target_colors);
  if (memo.table.size() >= 2048) memo.table.clear();
  memo.table.emplace(key, PairMemo::Entry{std::move(pred), memo.target_copy(target), std::move(mode), r.soft, r.cases});
  return r;
}

namespace {

// fitness_pair_eval's scores of a prediction that is not its target.
void score_miss(PairEval& r, const Grid& pred, const Grid& target, bool pixel_mode, bool collect_cases,
                bool fine, bool quads, const std::vector<int>& target_colors) {
  double pix_d = target_bounded_mismatch(pred, target);
  // Graph distance: the objects of pred and target and their matching once,
  // shared by the soft score and the case vector.
  std::vector<FitObject> pred_objs;
  ObjDistComponents comps{};
  if (!pixel_mode) {
    pred_objs = extract_objects(pred);
    comps = object_distance_components(pred_objs, target_objects(target), pred.height(), pred.width());
  }
  if (pixel_mode) {
    r.soft = pix_d;
  } else {
    r.soft = combine_components(comps) + PIXEL_SMOOTH_WEIGHT * pix_d;
  }
  if (!collect_cases) return;
  if (!fine) {
    if (pixel_mode) {
      r.cases = {pix_d};
    } else {
      r.cases = {combine_components(comps), pix_d};
    }
    return;
  }
  // Fine / fine2.
  std::vector<double> per_color = per_color_pixel_errors(pred, target, target_colors);
  std::vector<double> cases;
  if (pixel_mode) {
    cases.push_back(pix_d);
    for (double v : per_color) cases.push_back(v);
  } else {
    cases.push_back(comps.count_d);
    cases.push_back(comps.match_d);
    cases.push_back(comps.color_d);
    cases.push_back(comps.shape_d);
    cases.push_back(comps.bbox_d);
    cases.push_back(pix_d);
    for (double v : per_color) cases.push_back(v);
  }
  if (quads) {
    std::vector<double> qs = quadrant_pixel_errors(pred, target);
    for (double v : qs) cases.push_back(v);
  }
  r.cases = std::move(cases);
}

}  // namespace

// ---------------------------------------------------------------------------
// quick_seed_score: a cheap score for ranking seed genomes. Renders the
// prediction with the genome's encoder + decoder through predict_grid (which
// honours the genome's DecoderConfig + input grid), then the exact float formula.
// ---------------------------------------------------------------------------
double quick_seed_score(const Grid& inp, const Grid& target,
                        const GraphConfig& encoder_cfg, const DecoderConfig& decoder_cfg,
                        const DslProgram& program,
                        SceneCache* scene_cache) {
  // Fast reject: same MAX_RENDER_CELLS bound the full fitness uses.
  if (projected_output_cells(inp.height(), inp.width(), program.global_transform) >
      MAX_RENDER_CELLS) {
    return 1e9;
  }
  Grid pred = predict_grid(inp, program, encoder_cfg, decoder_cfg, scene_cache);
  long long ph = pred.height(), pw = pred.width();
  long long th = target.height(), tw = target.width();
  long long common_h = std::min(ph, th);
  long long common_w = std::min(pw, tw);
  long long diff_common = 0;
  for (long long y = 0; y < common_h; ++y)
    for (long long x = 0; x < common_w; ++x)
      if (pred.at(static_cast<int>(y), static_cast<int>(x)) !=
          target.at(static_cast<int>(y), static_cast<int>(x)))
        ++diff_common;
  long long extra = (ph * pw - common_h * common_w) + (th * tw - common_h * common_w);
  long long denom_ll = std::max<long long>(std::max(ph, th) * std::max(pw, tw), 1);
  double mismatch = static_cast<double>(diff_common + extra) / static_cast<double>(denom_ll);
  long long ppw = ph * pw, ttw = th * tw;
  long long size_denom = std::max<long long>(std::max(ppw, ttw), 1);
  long long diff_abs = ppw >= ttw ? (ppw - ttw) : (ttw - ppw);
  double size_gap = static_cast<double>(diff_abs) / static_cast<double>(size_denom);
  return mismatch + 0.1 * size_gap +
         0.01 * static_cast<double>(program.rules.size());
}

}  // namespace arc
