// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 2: zone detection (detect_zones: find_grid_lines +
// build_gridline_subobjects + carve_zones).
//
// Determinism notes:
//   * Grid line = a full row/col of a single non-background color when
//     grid_line_threshold >= 1.0 (the default). The partial-threshold branch
//     takes each line's most common color (count desc, first-seen tie-break)
//     and, when several grid colors survive, keeps the best one, scanning the
//     candidate colors in ascending order (see the threshold<1.0 comments).
//   * Gridline SubObjects are is_structural=True -> excluded from the canonical
//     serializer, but they DO advance the shared node counter, so their id
//     assignment order matters for the zone_{n} ids that follow.
//   * ZoneNode zone_color: the most frequent non-background color; on a tie,
//     the first seen in pixel scan order (y outer).
//   * The grid-line pixel set is returned to feed step 3's exclude_pixels.

#include "arc/encode.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace arc {

namespace {

struct Line {
  int pos;
  int color;
  double frac;
};

// Most common non-background value: highest count, ties -> first-seen.
// vals is the raw value stream; the first-seen order defines the tie-break.
bool most_common_nonbg(const std::vector<int>& vals, bool has_bg, int bg,
                       int& out_color, int& out_count) {
  std::vector<int> keys;
  std::map<int, int> idx;
  std::vector<int> counts;
  for (int v : vals) {
    if (has_bg && v == bg) continue;
    auto it = idx.find(v);
    if (it == idx.end()) {
      idx[v] = static_cast<int>(keys.size());
      keys.push_back(v);
      counts.push_back(1);
    } else {
      counts[it->second] += 1;
    }
  }
  if (keys.empty()) return false;
  int best = 0;
  int best_c = counts[0];
  for (size_t i = 1; i < keys.size(); ++i) {
    if (counts[i] > best_c) {
      best_c = counts[i];
      best = static_cast<int>(i);
    }
  }
  out_color = keys[best];
  out_count = counts[best];
  return true;
}

// Phase 1: find_grid_lines.
void find_grid_lines(const Grid& grid, const GraphConfig& cfg, std::optional<int> bg,
                     std::vector<Line>& h_lines, std::vector<Line>& v_lines) {
  int h = grid.height(), w = grid.width();
  double threshold = cfg.grid_line_threshold;
  bool has_bg = bg.has_value();
  int bgv = has_bg ? *bg : 0;

  for (int y = 0; y < h; ++y) {
    if (threshold >= 1.0) {
      // The row holds a single color.
      int c0 = grid.at(y, 0);
      bool uniform = true;
      for (int x = 1; x < w; ++x)
        if (grid.at(y, x) != c0) {
          uniform = false;
          break;
        }
      if (uniform) {
        if (!has_bg || c0 != bgv) h_lines.push_back({y, c0, 1.0});
      }
    } else {
      std::vector<int> vals(w);
      for (int x = 0; x < w; ++x) vals[x] = grid.at(y, x);
      int bc, bcnt;
      if (most_common_nonbg(vals, has_bg, bgv, bc, bcnt)) {
        double frac = static_cast<double>(bcnt) / w;
        if (frac >= threshold) h_lines.push_back({y, bc, frac});
      }
    }
  }

  for (int x = 0; x < w; ++x) {
    if (threshold >= 1.0) {
      int c0 = grid.at(0, x);
      bool uniform = true;
      for (int y = 1; y < h; ++y)
        if (grid.at(y, x) != c0) {
          uniform = false;
          break;
        }
      if (uniform) {
        if (!has_bg || c0 != bgv) v_lines.push_back({x, c0, 1.0});
      }
    } else {
      std::vector<int> vals(h);
      for (int y = 0; y < h; ++y) vals[y] = grid.at(y, x);
      int bc, bcnt;
      if (most_common_nonbg(vals, has_bg, bgv, bc, bcnt)) {
        double frac = static_cast<double>(bcnt) / h;
        if (frac >= threshold) v_lines.push_back({x, bc, frac});
      }
    }
  }

  if (threshold < 1.0 && (!h_lines.empty() || !v_lines.empty())) {
    // h_colors / v_colors: line count per color, in first-seen order.
    // full_span: count of full-span lines per color.
    std::vector<int> hc_keys;
    std::map<int, int> hc_idx;
    std::vector<int> hc_val;
    std::vector<int> vc_keys;
    std::map<int, int> vc_idx;
    std::vector<int> vc_val;
    std::map<int, int> full_span;  // color -> count
    auto bump = [](std::vector<int>& keys, std::map<int, int>& idx, std::vector<int>& val, int k) {
      auto it = idx.find(k);
      if (it == idx.end()) {
        idx[k] = static_cast<int>(keys.size());
        keys.push_back(k);
        val.push_back(1);
      } else {
        val[it->second] += 1;
      }
    };
    for (const auto& L : h_lines) {
      bump(hc_keys, hc_idx, hc_val, L.color);
      if (L.frac >= 1.0) full_span[L.color] += 1;
    }
    for (const auto& L : v_lines) {
      bump(vc_keys, vc_idx, vc_val, L.color);
      if (L.frac >= 1.0) full_span[L.color] += 1;
    }
    // both = intersection of h_colors and v_colors keys.
    std::set<int> hset(hc_keys.begin(), hc_keys.end());
    std::set<int> vset(vc_keys.begin(), vc_keys.end());
    std::vector<int> candidates;
    bool both_nonempty = false;
    for (int c : hset)
      if (vset.count(c)) {
        candidates.push_back(c);
        both_nonempty = true;
      }
    if (!both_nonempty) {
      // Otherwise the union of h_colors and v_colors, in ascending color order.
      std::set<int> uni;
      for (int c : hc_keys) uni.insert(c);
      for (int c : vc_keys) uni.insert(c);
      candidates.assign(uni.begin(), uni.end());
    }
    if (!candidates.empty()) {
      auto hget = [&](int c) {
        auto it = hc_idx.find(c);
        return it == hc_idx.end() ? 0 : hc_val[it->second];
      };
      auto vget = [&](int c) {
        auto it = vc_idx.find(c);
        return it == vc_idx.end() ? 0 : vc_val[it->second];
      };
      auto fsget = [&](int c) {
        auto it = full_span.find(c);
        return it == full_span.end() ? 0 : it->second;
      };
      // Pick the candidate with the most full-span lines, then the most lines;
      // on a full tie the first candidate (lowest color) wins.
      int grid_color = candidates[0];
      int best_fs = fsget(grid_color);
      int best_hv = hget(grid_color) + vget(grid_color);
      for (size_t i = 1; i < candidates.size(); ++i) {
        int c = candidates[i];
        int fs = fsget(c);
        int hv = hget(c) + vget(c);
        if (fs > best_fs || (fs == best_fs && hv > best_hv)) {
          best_fs = fs;
          best_hv = hv;
          grid_color = c;
        }
      }
      std::vector<Line> hf, vf;
      for (const auto& L : h_lines)
        if (L.color == grid_color) hf.push_back(L);
      for (const auto& L : v_lines)
        if (L.color == grid_color) vf.push_back(L);
      h_lines.swap(hf);
      v_lines.swap(vf);
    }
  }
}

// Phase 2: build_gridline_subobjects. Registers grid-line pixels and emits
// structural SubObject(s). Returns the grid-line pixel set.
std::set<Coord> build_gridline_subobjects(Scene& scene, const Grid& grid, const GraphConfig& cfg,
                                          const std::vector<Line>& h_lines,
                                          const std::vector<Line>& v_lines) {
  int h = grid.height(), w = grid.width();
  double threshold = cfg.grid_line_threshold;

  std::vector<Coord> line_pixels;
  int line_color = 0;
  bool have_color = false;
  for (const auto& L : h_lines) {
    line_color = L.color;
    have_color = true;
    for (int x = 0; x < w; ++x)
      if (threshold >= 1.0 || grid.at(L.pos, x) == L.color) line_pixels.emplace_back(L.pos, x);
  }
  std::set<Coord> h_pixel_set(line_pixels.begin(), line_pixels.end());
  for (const auto& L : v_lines) {
    line_color = L.color;
    have_color = true;
    for (int y = 0; y < h; ++y) {
      Coord p(y, L.pos);
      if (h_pixel_set.find(p) == h_pixel_set.end()) {
        if (threshold >= 1.0 || grid.at(y, L.pos) == L.color) {
          line_pixels.push_back(p);
          h_pixel_set.insert(p);
        }
      }
    }
  }

  if (line_pixels.empty()) return {};

  std::set<Coord> grid_line_pixels(line_pixels.begin(), line_pixels.end());

  if (cfg.decompose_grid_lines) {
    for (const auto& L : h_lines) {
      scene.node_counter += 1;
      std::vector<Coord> px;
      for (int x = 0; x < w; ++x)
        if (threshold >= 1.0 || grid.at(L.pos, x) == L.color) px.emplace_back(L.pos, x);
      Node sub;
      sub.node_id = "sub_" + std::to_string(scene.node_counter);
      sub.node_type = "SubObject";
      sub.color = L.color;
      sub.is_structural = true;
      sub.pixel_list = px;
      sub.post_init();
      sub.check_boundary_touches(w, h);
      sub.compute_shape_hash();
      sub.classify_shape();
      scene.add_node(sub);
    }
    for (const auto& L : v_lines) {
      scene.node_counter += 1;
      std::vector<Coord> px;
      for (int y = 0; y < h; ++y)
        if (threshold >= 1.0 || grid.at(y, L.pos) == L.color) px.emplace_back(y, L.pos);
      Node sub;
      sub.node_id = "sub_" + std::to_string(scene.node_counter);
      sub.node_type = "SubObject";
      sub.color = L.color;
      sub.is_structural = true;
      sub.pixel_list = px;
      sub.post_init();
      sub.check_boundary_touches(w, h);
      sub.compute_shape_hash();
      sub.classify_shape();
      scene.add_node(sub);
    }
  } else {
    scene.node_counter += 1;
    Node gl;
    gl.node_id = "sub_" + std::to_string(scene.node_counter);
    gl.node_type = "SubObject";
    if (have_color) gl.color = line_color;
    gl.shape_mask = std::string("GridLine");
    gl.is_structural = true;
    gl.is_gridline_meta = true;
    for (const auto& L : h_lines) gl.meta_h_lines.push_back(L.pos);
    for (const auto& L : v_lines) gl.meta_v_lines.push_back(L.pos);
    gl.pixel_list = line_pixels;
    gl.post_init();
    gl.check_boundary_touches(w, h);
    // Note: shape_mask preset to "GridLine"; NO compute_shape_hash/classify.
    scene.add_node(gl);
  }

  return grid_line_pixels;
}

// Phase 3: carve_zones. Emits one ZoneNode per non-empty cell.
void carve_zones(Scene& scene, const Grid& grid, std::optional<int> bg,
                 const std::vector<Line>& h_lines, const std::vector<Line>& v_lines) {
  int h = grid.height(), w = grid.width();
  bool has_bg = bg.has_value();
  int bgv = has_bg ? *bg : 0;

  std::vector<int> h_bounds;
  h_bounds.push_back(0);
  for (const auto& L : h_lines) h_bounds.push_back(L.pos);
  h_bounds.push_back(h);
  std::vector<int> v_bounds;
  v_bounds.push_back(0);
  for (const auto& L : v_lines) v_bounds.push_back(L.pos);
  v_bounds.push_back(w);

  std::set<int> line_rows, line_cols;
  for (const auto& L : h_lines) line_rows.insert(L.pos);
  for (const auto& L : v_lines) line_cols.insert(L.pos);

  for (size_t i = 0; i + 1 < h_bounds.size(); ++i) {
    for (size_t j = 0; j + 1 < v_bounds.size(); ++j) {
      int y_start = h_bounds[i];
      int y_end = h_bounds[i + 1];
      int x_start = v_bounds[j];
      int x_end = v_bounds[j + 1];
      if (line_rows.count(y_start)) y_start += 1;
      if (line_cols.count(x_start)) x_start += 1;

      std::vector<Coord> zone_pixels;
      // zone_colors: insertion-ordered non-bg color histogram (scan order).
      std::vector<int> zc_keys;
      std::map<int, int> zc_idx;
      std::vector<int> zc_val;
      for (int y = y_start; y < y_end; ++y) {
        if (line_rows.count(y)) continue;
        for (int x = x_start; x < x_end; ++x) {
          if (line_cols.count(x)) continue;
          int c = grid.at(y, x);
          zone_pixels.emplace_back(y, x);
          if (!has_bg || c != bgv) {
            auto it = zc_idx.find(c);
            if (it == zc_idx.end()) {
              zc_idx[c] = static_cast<int>(zc_keys.size());
              zc_keys.push_back(c);
              zc_val.push_back(1);
            } else {
              zc_val[it->second] += 1;
            }
          }
        }
      }
      if (zone_pixels.empty()) continue;

      // zone_color: the most frequent non-bg color (first seen on a tie), else bg.
      std::optional<int> zone_color;
      if (!zc_keys.empty()) {
        int best = zc_keys[0];
        int best_v = zc_val[0];
        for (size_t k = 1; k < zc_keys.size(); ++k) {
          if (zc_val[k] > best_v) {
            best_v = zc_val[k];
            best = zc_keys[k];
          }
        }
        zone_color = best;
      } else {
        zone_color = bg;  // nullopt when there is no background
      }

      scene.node_counter += 1;
      Node zone;
      zone.node_id = "zone_" + std::to_string(scene.node_counter);
      zone.node_type = "ZoneNode";
      zone.zone_color = zone_color;
      zone.pixel_list = zone_pixels;
      zone.post_init();
      zone.check_boundary_touches(w, h);
      scene.add_node(zone);
    }
  }
}

}  // namespace

std::set<Coord> detect_zones(Scene& scene, const Grid& grid, const GraphConfig& cfg) {
  std::optional<int> bg = scene.has_background() ? std::optional<int>(scene.background())
                                                 : std::nullopt;
  std::vector<Line> h_lines, v_lines;
  find_grid_lines(grid, cfg, bg, h_lines, v_lines);
  if (h_lines.empty() && v_lines.empty()) return {};
  // A line on the grid edge bounds the grid, it separates nothing (a border frame
  // is an object).
  {
    auto edge_h = [&](const Line& L) { return L.pos == 0 || L.pos == grid.height() - 1; };
    auto edge_v = [&](const Line& L) { return L.pos == 0 || L.pos == grid.width() - 1; };
    h_lines.erase(std::remove_if(h_lines.begin(), h_lines.end(), edge_h), h_lines.end());
    v_lines.erase(std::remove_if(v_lines.begin(), v_lines.end(), edge_v), v_lines.end());
    if (h_lines.empty() && v_lines.empty()) return {};
  }
  // Separators share one colour: keep the lines of the most common line colour
  // (full rows / columns of other colours are parts of objects).
  {
    std::map<int, int> n;
    for (const auto& L : h_lines) n[L.color]++;
    for (const auto& L : v_lines) n[L.color]++;
    if (n.size() > 1) {
      int c = n.begin()->first;
      for (const auto& kv : n)
        if (kv.second > n[c]) c = kv.first;
      auto other = [c](const Line& L) { return L.color != c; };
      h_lines.erase(std::remove_if(h_lines.begin(), h_lines.end(), other), h_lines.end());
      v_lines.erase(std::remove_if(v_lines.begin(), v_lines.end(), other), v_lines.end());
    }
  }
  // Lines with nothing else on the grid separate nothing: they are the objects.
  {
    std::set<int> rows, cols;
    for (const auto& L : h_lines) rows.insert(L.pos);
    for (const auto& L : v_lines) cols.insert(L.pos);
    bool other = false;
    for (int y = 0; y < grid.height() && !other; ++y)
      for (int x = 0; x < grid.width() && !other; ++x)
        other = !rows.count(y) && !cols.count(x) && (!bg.has_value() || grid.at(y, x) != *bg);
    if (!other) return {};
  }
  std::set<Coord> gl = build_gridline_subobjects(scene, grid, cfg, h_lines, v_lines);
  carve_zones(scene, grid, bg, h_lines, v_lines);
  return gl;
}

}  // namespace arc
