// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL: the 26 GlobalTransforms. Shared helpers live in arg_coerce.hpp /
// scene_ops.hpp, which also state the ordering rules.

#include "arc/dsl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arc/concepts.hpp"
#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

#include "scene_ops.hpp"

namespace arc {

// ===========================================================================
// The 26 GlobalTransforms in the executor path.
//
// Two families:
//   * permutation | full-grid transforms (the fast-path set): whole-grid
//     functions (apply_full_grid_global). The empty-rule fast path applies
//     these to the RAW input grid; the with-rules path renders the mutated
//     scene to a grid, then applies them to the whole grid.
//   * everything else (IDENTITY, ROT*, FLIP*, TRANSPOSE, ANTI_TRANSPOSE, KRON_*,
//     CROP_FOREGROUND, SCALE*_NEAREST, FILL_PERIODIC_BACKGROUND) goes through the
//     foreground-only apply_global_transform_to_scene, then render_scene_default.
//
// NOTE (permutation transforms appear in BOTH sets): with no rules they use the
// FULL-grid function (fast path); with rules they are not full-grid
// transforms, so they take the scene path. See execute_program_final_grid.
// ===========================================================================

bool is_permutation_transform(GlobalTransform t) {
  switch (t) {
    case GlobalTransform::ROT90:
    case GlobalTransform::ROT180:
    case GlobalTransform::ROT270:
    case GlobalTransform::FLIP_H:
    case GlobalTransform::FLIP_V:
    case GlobalTransform::TRANSPOSE:
    case GlobalTransform::ANTI_TRANSPOSE:
      return true;
    default:
      return false;
  }
}

bool is_full_grid_transform(GlobalTransform t) {
  switch (t) {
    case GlobalTransform::MIRROR_CONCAT_H:
    case GlobalTransform::MIRROR_CONCAT_H_REV:
    case GlobalTransform::MIRROR_CONCAT_V:
    case GlobalTransform::MIRROR_CONCAT_V_REV:
    case GlobalTransform::MIRROR_QUAD_TL:
    case GlobalTransform::MIRROR_QUAD_TR:
    case GlobalTransform::MIRROR_QUAD_BL:
    case GlobalTransform::MIRROR_QUAD_BR:
    case GlobalTransform::TILE_H:
    case GlobalTransform::TILE_V:
    case GlobalTransform::DEDUP_RUNS:
    case GlobalTransform::BLOCK_REDUCE2:
      return true;
    default:
      return false;
  }
}

// Fast-path transforms = permutation | full-grid.
bool is_fast_path_transform(GlobalTransform t) {
  return is_permutation_transform(t) || is_full_grid_transform(t);
}

// permute_full_grid(grid, transform): apply a permutation transform to EVERY
// cell (background included), using the same coordinate maps as
// apply_global_transform_to_scene.
Grid permute_full_grid(const Grid& g, GlobalTransform t) {
  int h = g.height(), w = g.width();
  bool swaps = (t == GlobalTransform::ROT90 || t == GlobalTransform::ROT270 ||
                t == GlobalTransform::TRANSPOSE || t == GlobalTransform::ANTI_TRANSPOSE);
  Grid out(swaps ? w : h, swaps ? h : w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int c = g.at(y, x);
      switch (t) {
        case GlobalTransform::ROT90:          out.set(w - 1 - x, y, c); break;
        case GlobalTransform::ROT180:         out.set(h - 1 - y, w - 1 - x, c); break;
        case GlobalTransform::ROT270:         out.set(x, h - 1 - y, c); break;
        case GlobalTransform::FLIP_H:         out.set(y, w - 1 - x, c); break;
        case GlobalTransform::FLIP_V:         out.set(h - 1 - y, x, c); break;
        case GlobalTransform::TRANSPOSE:      out.set(x, y, c); break;
        case GlobalTransform::ANTI_TRANSPOSE: out.set(w - 1 - x, h - 1 - y, c); break;
        default: break;
      }
    }
  }
  return out;
}

// dedup_runs(grid): drop consecutive duplicate rows, then consecutive
// duplicate columns.
Grid dedup_runs(const Grid& g) {
  int h = g.height(), w = g.width();
  // keep_rows = [0] + rows i where row i != row i-1.
  std::vector<int> keep_rows;
  if (h > 0) keep_rows.push_back(0);
  for (int i = 1; i < h; ++i) {
    bool same = true;
    for (int x = 0; x < w; ++x) if (g.at(i, x) != g.at(i - 1, x)) { same = false; break; }
    if (!same) keep_rows.push_back(i);
  }
  int h2 = static_cast<int>(keep_rows.size());
  // Now on the row-reduced grid, keep columns j where col j != col j-1.
  std::vector<int> keep_cols;
  if (w > 0) keep_cols.push_back(0);
  for (int j = 1; j < w; ++j) {
    bool same = true;
    for (int r = 0; r < h2; ++r) {
      if (g.at(keep_rows[r], j) != g.at(keep_rows[r], j - 1)) { same = false; break; }
    }
    if (!same) keep_cols.push_back(j);
  }
  int w2 = static_cast<int>(keep_cols.size());
  Grid out(h2, w2);
  for (int r = 0; r < h2; ++r)
    for (int c = 0; c < w2; ++c) out.set(r, c, g.at(keep_rows[r], keep_cols[c]));
  return out;
}

// apply_full_grid_global(grid, transform): whole-grid op (every cell, bg
// included). Handles the permutation set + mirror-tiling / tile / dedup /
// block-reduce family. Returns the grid unchanged for out-of-family transforms
// (defensive).
Grid apply_full_grid_global(const Grid& g, GlobalTransform t) {
  if (is_permutation_transform(t)) return permute_full_grid(g, t);
  int h = g.height(), w = g.width();
  // fh = g with its columns flipped, fv = g with its rows flipped, fhv = both.
  auto fh_at = [&](int y, int x) { return g.at(y, w - 1 - x); };
  auto fv_at = [&](int y, int x) { return g.at(h - 1 - y, x); };
  auto fhv_at = [&](int y, int x) { return g.at(h - 1 - y, w - 1 - x); };
  switch (t) {
    case GlobalTransform::MIRROR_CONCAT_H: {
      Grid out(h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, g.at(y, x)); out.set(y, w + x, fh_at(y, x)); }
      return out;
    }
    case GlobalTransform::MIRROR_CONCAT_H_REV: {
      Grid out(h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, fh_at(y, x)); out.set(y, w + x, g.at(y, x)); }
      return out;
    }
    case GlobalTransform::MIRROR_CONCAT_V: {
      Grid out(2 * h, w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, g.at(y, x)); out.set(h + y, x, fv_at(y, x)); }
      return out;
    }
    case GlobalTransform::MIRROR_CONCAT_V_REV: {
      Grid out(2 * h, w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, fv_at(y, x)); out.set(h + y, x, g.at(y, x)); }
      return out;
    }
    // Quad layout [[A, B], [C, D]]: top-left A, top-right B, bottom-left C, bottom-right D.
    case GlobalTransform::MIRROR_QUAD_TL: {  // [[g, fh], [fv, fhv]]
      Grid out(2 * h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          out.set(y, x, g.at(y, x)); out.set(y, w + x, fh_at(y, x));
          out.set(h + y, x, fv_at(y, x)); out.set(h + y, w + x, fhv_at(y, x));
        }
      return out;
    }
    case GlobalTransform::MIRROR_QUAD_TR: {  // [[fh, g], [fhv, fv]]
      Grid out(2 * h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          out.set(y, x, fh_at(y, x)); out.set(y, w + x, g.at(y, x));
          out.set(h + y, x, fhv_at(y, x)); out.set(h + y, w + x, fv_at(y, x));
        }
      return out;
    }
    case GlobalTransform::MIRROR_QUAD_BL: {  // [[fv, fhv], [g, fh]]
      Grid out(2 * h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          out.set(y, x, fv_at(y, x)); out.set(y, w + x, fhv_at(y, x));
          out.set(h + y, x, g.at(y, x)); out.set(h + y, w + x, fh_at(y, x));
        }
      return out;
    }
    case GlobalTransform::MIRROR_QUAD_BR: {  // [[fhv, fv], [fh, g]]
      Grid out(2 * h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          out.set(y, x, fhv_at(y, x)); out.set(y, w + x, fv_at(y, x));
          out.set(h + y, x, fh_at(y, x)); out.set(h + y, w + x, g.at(y, x));
        }
      return out;
    }
    case GlobalTransform::TILE_H: {  // [g, g] side by side
      Grid out(h, 2 * w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, g.at(y, x)); out.set(y, w + x, g.at(y, x)); }
      return out;
    }
    case GlobalTransform::TILE_V: {  // g stacked on g
      Grid out(2 * h, w);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) { out.set(y, x, g.at(y, x)); out.set(h + y, x, g.at(y, x)); }
      return out;
    }
    case GlobalTransform::DEDUP_RUNS:
      return dedup_runs(g);
    case GlobalTransform::BLOCK_REDUCE2: {  // every second row and column
      int h2 = (h + 1) / 2, w2 = (w + 1) / 2;
      Grid out(h2, w2);
      for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) out.set(y, x, g.at(2 * y, 2 * x));
      return out;
    }
    default:
      return g;  // out-of-family (defensive)
  }
}

// infer_periodic_tile(grid, bg): smallest (py, px) with 1<=py<=h, 1<=px<=w,
// (py,px) != (h,w), such that every non-bg cell agrees with a single color per
// (y%py, x%px) orbit. Returns {false,...} if none. Iteration order: py outer,
// px inner.
struct PeriodResult { bool ok; int py; int px; };
PeriodResult infer_periodic_tile(const Grid& g, int bg) {
  int h = g.height(), w = g.width();
  for (int py = 1; py <= h; ++py) {
    for (int px = 1; px <= w; ++px) {
      if (py == h && px == w) continue;
      bool ok = true;
      for (int y0 = 0; y0 < py && ok; ++y0) {
        for (int x0 = 0; x0 < px && ok; ++x0) {
          int color = bg;
          for (int yy = y0; yy < h && ok; yy += py) {
            for (int xx = x0; xx < w; xx += px) {
              int c = g.at(yy, xx);
              if (c == bg) continue;
              if (color == bg) color = c;
              else if (color != c) { ok = false; break; }
            }
          }
        }
      }
      if (ok) return {true, py, px};
    }
  }
  return {false, 0, 0};
}

// apply_global_transform_to_scene(scene, transform): the FOREGROUND-only path.
// Remaps SubObject pixels (bg excluded), rebuilds via reset_scene_subobjects.
// Full-grid transforms are guarded to a no-op (they never reach this path).
// Returns via scene mutation.
void apply_global_transform_to_scene(Scene& scene, GlobalTransform t) {
  if (t == GlobalTransform::IDENTITY) return;
  if (is_full_grid_transform(t)) return;  // never reaches here (fast/full-grid path)

  int h = scene.height(), w = scene.width();
  std::vector<std::tuple<int, int, int>> source = collect_colored_pixels(scene);
  std::vector<std::tuple<int, int, int>> mapped;
  int new_h = h, new_w = w;

  auto emit = [&](int y, int x, int c) { mapped.emplace_back(y, x, c); };

  if (t == GlobalTransform::ROT90) {
    new_h = w; new_w = h;
    for (auto& s : source) emit(w - 1 - std::get<1>(s), std::get<0>(s), std::get<2>(s));
  } else if (t == GlobalTransform::ROT180) {
    for (auto& s : source) emit(h - 1 - std::get<0>(s), w - 1 - std::get<1>(s), std::get<2>(s));
  } else if (t == GlobalTransform::ROT270) {
    new_h = w; new_w = h;
    for (auto& s : source) emit(std::get<1>(s), h - 1 - std::get<0>(s), std::get<2>(s));
  } else if (t == GlobalTransform::FLIP_H) {
    for (auto& s : source) emit(std::get<0>(s), w - 1 - std::get<1>(s), std::get<2>(s));
  } else if (t == GlobalTransform::FLIP_V) {
    for (auto& s : source) emit(h - 1 - std::get<0>(s), std::get<1>(s), std::get<2>(s));
  } else if (t == GlobalTransform::TRANSPOSE) {
    new_h = w; new_w = h;
    for (auto& s : source) emit(std::get<1>(s), std::get<0>(s), std::get<2>(s));
  } else if (t == GlobalTransform::ANTI_TRANSPOSE) {
    new_h = w; new_w = h;
    for (auto& s : source) emit(w - 1 - std::get<1>(s), h - 1 - std::get<0>(s), std::get<2>(s));
  } else if (t == GlobalTransform::KRON_SELF_MASK) {
    // anchors = sorted set of (y,x); new dims h*h x w*w.
    std::set<Coord> aset;
    for (auto& s : source) aset.insert({std::get<0>(s), std::get<1>(s)});
    new_h = h * h; new_w = w * w;
    for (const auto& a : aset)  // std::set iterates (y,x) ascending
      for (auto& s : source)
        emit(a.first * h + std::get<0>(s), a.second * w + std::get<1>(s), std::get<2>(s));
  } else if (t == GlobalTransform::KRON_MASK_ANCHOR_COLOR) {
    // anchors = sorted set of (y,x,c); source_mask = sorted set of (y,x).
    std::set<std::tuple<int, int, int>> aset(source.begin(), source.end());
    std::set<Coord> mask;
    for (auto& s : source) mask.insert({std::get<0>(s), std::get<1>(s)});
    new_h = h * h; new_w = w * w;
    for (const auto& a : aset)  // ascending (y,x,c)
      for (const auto& m : mask)
        emit(std::get<0>(a) * h + m.first, std::get<1>(a) * w + m.second, std::get<2>(a));
  } else if (t == GlobalTransform::CROP_FOREGROUND) {
    if (source.empty()) return;  // scene left as is (no reset)
    int ymin = std::get<0>(source[0]), xmin = std::get<1>(source[0]);
    int ymax = ymin, xmax = xmin;
    for (auto& s : source) {
      ymin = std::min(ymin, std::get<0>(s)); xmin = std::min(xmin, std::get<1>(s));
      ymax = std::max(ymax, std::get<0>(s)); xmax = std::max(xmax, std::get<1>(s));
    }
    new_h = ymax - ymin + 1; new_w = xmax - xmin + 1;
    for (auto& s : source) emit(std::get<0>(s) - ymin, std::get<1>(s) - xmin, std::get<2>(s));
  } else if (t == GlobalTransform::SCALE2_NEAREST) {
    new_h = h * 2; new_w = w * 2;
    for (auto& s : source) {
      int by = std::get<0>(s) * 2, bx = std::get<1>(s) * 2, c = std::get<2>(s);
      emit(by, bx, c); emit(by + 1, bx, c); emit(by, bx + 1, c); emit(by + 1, bx + 1, c);
    }
  } else if (t == GlobalTransform::SCALE3_NEAREST) {
    new_h = h * 3; new_w = w * 3;
    for (auto& s : source) {
      int by = std::get<0>(s) * 3, bx = std::get<1>(s) * 3, c = std::get<2>(s);
      for (int dy = 0; dy < 3; ++dy)
        for (int dx = 0; dx < 3; ++dx) emit(by + dy, bx + dx, c);
    }
  } else if (t == GlobalTransform::FILL_PERIODIC_BACKGROUND) {
    // Holes are interpreted as color 0 (NOT scene bg).
    int scene_bg = scene.has_background() ? scene.background() : 0;
    Grid grid(h, w);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) grid.set(y, x, scene_bg);
    for (auto& s : source) {
      int y = std::get<0>(s), x = std::get<1>(s);
      if (y >= 0 && y < h && x >= 0 && x < w) grid.set(y, x, std::get<2>(s));
    }
    PeriodResult pr = infer_periodic_tile(grid, 0);
    if (!pr.ok) return;  // scene left as is (no reset)
    int py = pr.py, px = pr.px;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (grid.at(y, x) != 0) { emit(y, x, grid.at(y, x)); continue; }
        int y0 = y % py, x0 = x % px, color = 0;
        for (int yy = y0; yy < h && color == 0; yy += py)
          for (int xx = x0; xx < w; xx += px) {
            int c = grid.at(yy, xx);
            if (c != 0) { color = c; break; }
          }
        if (color != 0) emit(y, x, color);
      }
    }
    new_h = h; new_w = w;
  }

  reset_scene_subobjects(scene, mapped, new_h, new_w);
}

// execute_program_final_grid: the FULL executor path incl. the global transform,
// returning the final DECODED grid:
//   1. empty-rule + FAST_PATH transform -> apply_full_grid_global(RAW input).
//   2. rule loop (select/guard/apply/re-encode) as in execute_program_gout.
//   3. with-rules + FULL_GRID transform -> render scene, apply_full_grid_global.
//   4. otherwise -> apply_global_transform_to_scene + render_scene_default.
Grid execute_program_final_grid(const Grid& grid, const DslProgram& program,
                                const GraphConfig& encoder_cfg) {
  // Empty-rule full-grid fast path (before the rule loop).
  if (program.rules.empty() && is_fast_path_transform(program.global_transform)) {
    return apply_full_grid_global(grid, program.global_transform);
  }
  // Run the rule loop to the mutated G_out scene.
  Scene scene = execute_program_gout(grid, program, encoder_cfg);
  // With-rules full-grid transform: render the mutated scene, apply to whole grid.
  if (is_full_grid_transform(program.global_transform)) {
    Grid rendered = render_scene_default(scene);
    return apply_full_grid_global(rendered, program.global_transform);
  }
  // Otherwise: foreground-only scene transform, then default-decode render.
  // Under the default DecoderConfig (background_policy=
  // "scene_background_or_zero") the input grid is not used for bg, so
  // render_scene_default gives the same grid as render_scene_to_grid(input=grid).
  apply_global_transform_to_scene(scene, program.global_transform);
  return render_scene_default(scene);
}

Scene execute_program_gout(const Grid& grid, const DslProgram& program,
                           const GraphConfig& encoder_cfg,
                           SceneCache* scene_cache) {
  // Initial encode. The decode / global transform are NOT applied here: G_out
  // is the rule-mutated scene (the caller decodes it and applies the global
  // transform). When a per-task scene cache is supplied (deploy only), the
  // initial encode of the raw input grid is served from it (a hit returns a
  // private copy, identical to a fresh encode); the mid-loop re-encode below
  // stays uncached.
  Scene scene = scene_cache ? scene_cache->get(grid, encoder_cfg)
                            : encode_minimal(grid, encoder_cfg);

  // The safe selectors: selectors that read only attributes the
  // actions keep in sync (color/mass/bbox/centroid), so no re-encode is needed
  // before them.
  auto is_safe = [](SelectorKind k) {
    switch (k) {
      case SelectorKind::ALL_SUBOBJECTS:
      case SelectorKind::BY_COLOR:
      case SelectorKind::NON_BACKGROUND:
      case SelectorKind::BY_MASS:
      case SelectorKind::BY_BOUNDARY_TOUCH:
      case SelectorKind::EXTREME:
      case SelectorKind::BY_COLOR_FREQUENCY:
        return true;
      default:
        return false;
    }
  };

  const size_t n_rules = program.rules.size();
  for (size_t idx = 0; idx < n_rules; ++idx) {
    const DslRule& rule = program.rules[idx];
    std::vector<std::string> selected_ids = select_node_ids(scene, rule.selector);
    bool passed = guards_pass(selected_ids, rule.guards);
    // Concept ops read the whole grid, so they also run on a scene with no objects.
    bool whole_grid = rule.selector.kind == SelectorKind::ALL_SUBOBJECTS && find_concept_op(rule.action.kind);
    if (passed && (!selected_ids.empty() || whole_grid)) {
      apply_action_to_scene(scene, selected_ids, rule.action);
      if (idx + 1 < n_rules) {
        SelectorKind next_kind = program.rules[idx + 1].selector.kind;
        if (!is_safe(next_kind)) {
          Grid rerender = render_scene_default(scene);
          // ARC_REENC_CACHE=1: the re-encode goes through the scene cache too
          // (same scene as a fresh encode; children of one parent often share
          // their first rules and so their intermediate grids).
          static const bool reenc_cache = [] {
            const char* e = std::getenv("ARC_REENC_CACHE");
            return e && std::string(e) == "1";
          }();
          scene = (scene_cache && reenc_cache) ? scene_cache->get(rerender, encoder_cfg)
                                               : encode_minimal(rerender, encoder_cfg);
        }
      }
    }
  }
  return scene;
}


}  // namespace arc
