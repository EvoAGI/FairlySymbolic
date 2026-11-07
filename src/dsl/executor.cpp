// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL: default renderer and program executor. Shared helpers live in
// arg_coerce.hpp / scene_ops.hpp, which also state the ordering rules.

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

#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

#include "scene_ops.hpp"

namespace arc {

// ===========================================================================
// Default renderer + executor.
// ===========================================================================

Grid render_scene_default(const Scene& scene) {
  // DecoderConfig defaults: background_policy="scene_background_or_zero",
  // paint_order="as_is", overlap_policy="overwrite", postprocess="none", and NO
  // input grid. So bg = scene bg or 0, paint SubObjects in iter_subobjects
  // (lexicographic id) order, overwrite. Same result as render_scene_to_grid's
  // else branch.
  int bg = scene.has_background() ? scene.background() : 0;
  int h = scene.height(), w = scene.width();
  Grid grid(h, w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) grid.set(y, x, bg);
  for (const Node* n : iter_subobjects(scene)) {
    int color = node_color(*n, bg);
    for (const auto& px : n->pixel_list) {
      int y = px.first, x = px.second;
      if (y >= 0 && y < h && x >= 0 && x < w) grid.set(y, x, color);
    }
  }
  return grid;
}

namespace {

// Most common value over a (value, count) list given in first-seen order:
// return the value with the highest count; ties go to the FIRST-SEEN value.
// The caller passes values in first-seen order (row-major grid scan for
// most_frequent_input; iter_subobjects color stream for
// most_frequent_predicted). Returns 0 when the stream is empty.
int counter_most_common_first(const std::vector<int>& insertion_order,
                              const std::map<int, long long>& counts) {
  int best = 0;
  long long best_count = -1;
  for (int v : insertion_order) {
    auto it = counts.find(v);
    long long c = (it == counts.end()) ? 0 : it->second;
    if (c > best_count) {  // strict > keeps the first-inserted maximum
      best_count = c;
      best = v;
    }
  }
  return best;
}

}  // namespace

// render_scene_to_grid(scene, cfg, input_grid): full decoder honoring all four
// policy axes. See the header for the contract.
Grid render_scene_to_grid(const Scene& scene, const DecoderConfig& cfg,
                          const Grid* input_grid) {
  int h = scene.height(), w = scene.width();
  int scene_bg = scene.has_background() ? scene.background() : 0;

  // --- background fill --------------------------------------------------
  // The input grid is used only when its shape is (scene.height, scene.width);
  // a null / mismatched-shape input falls through to the else branch.
  bool have_input = input_grid != nullptr &&
                    input_grid->height() == h && input_grid->width() == w;

  bool copy_input = (cfg.background_policy == "input_copy") && have_input;
  int bg;
  if (copy_input) {
    // grid = copy of input; bg = scene bg or 0 (used for keep_first / crop).
    bg = scene_bg;
  } else if (cfg.background_policy == "zero") {
    bg = 0;
  } else if (cfg.background_policy == "most_frequent_input" && have_input) {
    // Colour counts over the input grid, in row-major scan order.
    std::vector<int> order;
    std::map<int, long long> counts;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int v = input_grid->at(y, x);
        auto ins = counts.emplace(v, 0);
        if (ins.second) order.push_back(v);
        ins.first->second += 1;
      }
    }
    bg = order.empty() ? 0 : counter_most_common_first(order, counts);
  } else if (cfg.background_policy == "most_frequent_predicted") {
    // pred_vals: for each iter_subobjects node with a color, append color mass
    // times; first-seen order = first appearance in that stream. Nodes with no
    // color (nullopt) are skipped.
    std::vector<int> order;
    std::map<int, long long> counts;
    bool any = false;
    for (const Node* n : iter_subobjects(scene)) {
      if (!n->color.has_value()) continue;
      int c = *n->color;
      long long m = n->mass;
      if (m <= 0) continue;  // mass 0 adds nothing
      any = true;
      auto ins = counts.emplace(c, 0);
      if (ins.second) order.push_back(c);
      ins.first->second += m;
    }
    bg = any ? counter_most_common_first(order, counts) : 0;
  } else {
    // scene_background_or_zero (default), OR any policy whose precondition
    // failed (most_frequent_input with no input, input_copy with no/mismatched
    // input): scene bg or 0.
    bg = scene_bg;
  }

  // --- paint order ------------------------------------------------------
  // Base order = iter_subobjects (lexicographic node-id). Every sort key ends
  // with node_id as its tie-break; since the base order is already
  // lexicographic and std::stable_sort preserves it, sorting by the primary
  // key alone gives the full (primary, node_id) order.
  std::vector<const Node*> rows = iter_subobjects(scene);
  if (cfg.paint_order == "mass_desc") {
    std::stable_sort(rows.begin(), rows.end(),
                     [](const Node* a, const Node* b) { return a->mass > b->mass; });
  } else if (cfg.paint_order == "mass_asc") {
    std::stable_sort(rows.begin(), rows.end(),
                     [](const Node* a, const Node* b) { return a->mass < b->mass; });
  } else if (cfg.paint_order == "y_then_x") {
    // key = (bbox_ymin, bbox_xmin); absent bbox -> (1e9, 1e9). node-id
    // secondary handled by stable_sort over the lex base order.
    std::stable_sort(rows.begin(), rows.end(), [](const Node* a, const Node* b) {
      int ay = a->has_bbox ? a->bbox_ymin : 1000000000;
      int ax = a->has_bbox ? a->bbox_xmin : 1000000000;
      int by = b->has_bbox ? b->bbox_ymin : 1000000000;
      int bx = b->has_bbox ? b->bbox_xmin : 1000000000;
      if (ay != by) return ay < by;
      return ax < bx;
    });
  } else if (cfg.paint_order == "z_index_asc") {
    std::stable_sort(rows.begin(), rows.end(), [](const Node* a, const Node* b) {
      return a->z_index < b->z_index;
    });
  } else if (cfg.paint_order == "z_index_desc") {
    std::stable_sort(rows.begin(), rows.end(), [](const Node* a, const Node* b) {
      return a->z_index > b->z_index;
    });
  }
  // "as_is" (and any unknown) -> no sort (base iter_subobjects order).

  // --- paint ------------------------------------------------------------
  Grid grid(h, w);
  if (copy_input) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) grid.set(y, x, input_grid->at(y, x));
  } else {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) grid.set(y, x, bg);
  }

  bool can_keep_first =
      (cfg.overlap_policy == "keep_first") && (cfg.background_policy != "input_copy");
  for (const Node* n : rows) {
    int color = node_color(*n, bg);
    for (const auto& px : n->pixel_list) {
      int y = px.first, x = px.second;
      if (y >= 0 && y < h && x >= 0 && x < w) {
        if (can_keep_first && grid.at(y, x) != bg) continue;
        grid.set(y, x, color);
      }
    }
  }

  // --- postprocess: keep_lines -------------------------------------------
  // The separator lines the encoder set apart (structural SubObjects) are not
  // painted above; keep_lines paints them back on top.
  if (cfg.postprocess == "keep_lines") {
    for (size_t i : scene.get_nodes_by_type("SubObject", true)) {
      const Node& n = scene.nodes()[i];
      if (!n.is_structural) continue;
      int color = node_color(n, bg);
      if (color == bg) continue;
      for (const auto& px : n.pixel_list)
        if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w) grid.set(px.first, px.second, color);
    }
  }

  // --- postprocess: crop_foreground -------------------------------------
  if (cfg.postprocess == "crop_foreground") {
    int ymin = 1000000000, ymax = -1, xmin = 1000000000, xmax = -1;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (grid.at(y, x) != bg) {
          if (y < ymin) ymin = y;
          if (y > ymax) ymax = y;
          if (x < xmin) xmin = x;
          if (x > xmax) xmax = x;
        }
      }
    }
    if (ymax >= 0) {  // at least one foreground cell
      int ch = ymax - ymin + 1, cw = xmax - xmin + 1;
      Grid cropped(ch, cw);
      for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) cropped.set(y, x, grid.at(ymin + y, xmin + x));
      return cropped;
    }
  }
  return grid;
}


}  // namespace arc
