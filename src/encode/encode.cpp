// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Scene encoder: step 1 (background), step 3 (connected-component SubObject
// extraction), the driver that runs every encoder step in order, and the
// canonical scene serializers.
//
// Determinism notes:
//   * color_counts iterates colors in ascending order.
//   * corner_counts / border_counts iterate colors in first-seen order.
//   * BFS scans y-outer/x-inner, is FIFO, and visits neighbours in the order
//     of get_orthogonal_neighbors / get_8_neighbors.
//   * multi-color dominant pick: the most frequent color; on a tie, the one
//     first reached in BFS pixel-discovery order.
//   * node ids are "sub_{n}" with a single global counter starting at 0.
//   * serialized nodes are byte-wise lexicographically sorted by id.

#include "arc/encode.hpp"
#include "arc/object_attrs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "arc/cpyset.hpp"
#include "arc/md5.hpp"

namespace arc {

// ==========================================================================
// Step 1: background heuristic (calculate_background_heuristic).
// ==========================================================================

namespace {

// color_counts: pixel count per color, iterated in ascending color order.
std::map<int, int> color_counts_of(const Grid& grid) {
  std::map<int, int> counts;
  for (int v : grid.cells()) counts[v] += 1;
  return counts;
}

// Insertion-ordered small map: iterates keys in first-seen order, which
// several background strategies use to break ties.
struct OrderedCounts {
  std::vector<int> keys;
  std::map<int, int> idx;   // key -> position in keys
  std::vector<int> values;  // parallel to keys

  void add(int k, int by = 1) {
    auto it = idx.find(k);
    if (it == idx.end()) {
      idx[k] = static_cast<int>(keys.size());
      keys.push_back(k);
      values.push_back(by);
    } else {
      values[it->second] += by;
    }
  }
  int get(int k) const {
    auto it = idx.find(k);
    return it == idx.end() ? 0 : values[it->second];
  }
  int max_value() const {
    int m = 0;
    bool first = true;
    for (int v : values) {
      if (first || v > m) {
        m = v;
        first = false;
      }
    }
    return m;
  }
  size_t size() const { return keys.size(); }
};

int grid_total_pixels(const Grid& g) { return g.height() * g.width(); }

// Strategy 4b gate.
bool bg_rules_out_any(const Grid& grid, const std::map<int, int>& color_counts) {
  int total = grid_total_pixels(grid);
  bool has_zero = color_counts.find(0) != color_counts.end();
  int num_colors = static_cast<int>(color_counts.size());
  int maxc = 0;
  for (const auto& kv : color_counts) maxc = std::max(maxc, kv.second);
  double max_freq = total ? static_cast<double>(maxc) / total : 0.0;
  return !has_zero && num_colors >= 5 && max_freq < 0.40;
}

// corner_counts: iterate corners [TL, TR, BL, BR] with dim guards.
OrderedCounts bg_corner_counts(const Grid& grid) {
  int h = grid.height(), w = grid.width();
  std::vector<int> corners;
  corners.push_back(grid.at(0, 0));
  if (w > 1) corners.push_back(grid.at(0, w - 1));
  if (h > 1) {
    corners.push_back(grid.at(h - 1, 0));
    if (w > 1) corners.push_back(grid.at(h - 1, w - 1));
  }
  OrderedCounts out;
  for (int c : corners) out.add(c);
  return out;
}

// Strategy 1: all corners agree.
bool bg_strategy_corners_all_same(const OrderedCounts& cc, int& out) {
  if (cc.size() == 1) {
    out = cc.keys[0];
    return true;
  }
  return false;
}

// Strategy 2: single color touching all 4 edges, coverage >= 0.2.
bool bg_strategy_frame(const Grid& grid, const std::map<int, int>& color_counts, int& out) {
  int h = grid.height(), w = grid.width();
  int total = h * w;
  // Build each edge set in first-seen order of the row / column scan, without
  // duplicates; the hash-set iteration order computed by
  // cpython_set_intersection_int depends on this insertion order.
  auto edge_insertion_order = [](std::vector<int> vals) {
    std::vector<int> uniq;
    for (int v : vals) {
      bool seen = false;
      for (int u : uniq)
        if (u == v) {
          seen = true;
          break;
        }
      if (!seen) uniq.push_back(v);
    }
    return uniq;
  };
  std::vector<int> top_vals, bottom_vals, left_vals, right_vals;
  for (int x = 0; x < w; ++x) top_vals.push_back(grid.at(0, x));
  std::vector<int> top_row = edge_insertion_order(top_vals);
  std::vector<int> bottom_row;
  if (h > 1) {
    for (int x = 0; x < w; ++x) bottom_vals.push_back(grid.at(h - 1, x));
    bottom_row = edge_insertion_order(bottom_vals);
  } else {
    bottom_row = top_row;
  }
  for (int y = 0; y < h; ++y) left_vals.push_back(grid.at(y, 0));
  std::vector<int> left_col = edge_insertion_order(left_vals);
  std::vector<int> right_col;
  if (w > 1) {
    for (int y = 0; y < h; ++y) right_vals.push_back(grid.at(y, w - 1));
    right_col = edge_insertion_order(right_vals);
  } else {
    right_col = left_col;
  }

  // candidates = top_row & bottom_row & left_col & right_col (left-associative),
  // iterated in hash-set order (see cpyset.hpp), not ascending color order. The
  // strict '>' below keeps the FIRST-iterated color on a coverage tie, so this
  // order decides the winner when two colors frame the grid at equal coverage.
  std::vector<int> candidates = cpython_set_intersection_int(top_row, bottom_row);
  candidates = cpython_set_intersection_int(candidates, left_col);
  candidates = cpython_set_intersection_int(candidates, right_col);
  if (candidates.empty()) return false;

  bool have_best = false;
  int best = 0;
  double best_cov = 0.0;
  for (int c : candidates) {
    auto it = color_counts.find(c);
    int cnt = it == color_counts.end() ? 0 : it->second;
    double cov = total ? static_cast<double>(cnt) / total : 0.0;
    if (cov > best_cov) {
      best_cov = cov;
      best = c;
      have_best = true;
    }
  }
  if (have_best && best_cov >= 0.2) {
    out = best;
    return true;
  }
  return false;
}

// Strategy 3: 3+ corners agree AND that color covers >= 25%.
bool bg_strategy_corners_majority(const OrderedCounts& cc, const std::map<int, int>& color_counts,
                                  int& out) {
  if (cc.size() == 0) return false;
  int max_corner = cc.max_value();
  if (max_corner < 3) return false;
  int total = 0;
  for (const auto& kv : color_counts) total += kv.second;
  for (size_t i = 0; i < cc.keys.size(); ++i) {
    int c = cc.keys[i];
    int cnt = cc.values[i];
    auto it = color_counts.find(c);
    int ccount = it == color_counts.end() ? 0 : it->second;
    // The corner color must also cover at least 25% of all pixels.
    if (cnt == max_corner && static_cast<double>(ccount) >= total * 0.25) {
      out = c;
      return true;
    }
  }
  return false;
}

// border_counts: top row, bottom row, then interior side columns.
OrderedCounts bg_border_counts(const Grid& grid) {
  int h = grid.height(), w = grid.width();
  OrderedCounts counts;
  for (int x = 0; x < w; ++x) counts.add(grid.at(0, x));
  if (h > 1)
    for (int x = 0; x < w; ++x) counts.add(grid.at(h - 1, x));
  if (h > 2) {
    for (int y = 1; y < h - 1; ++y) counts.add(grid.at(y, 0));
    if (w > 1)
      for (int y = 1; y < h - 1; ++y) counts.add(grid.at(y, w - 1));
  }
  return counts;
}

// Strategies 4/5/6: always returns.
int bg_strategy_border(const OrderedCounts& border, const std::map<int, int>& color_counts) {
  if (border.size() == 0) return 0;
  int total_border = 0;
  for (int v : border.values) total_border += v;
  int total_pixels = 0;
  for (const auto& kv : color_counts) total_pixels += kv.second;
  int max_border = border.max_value();

  // border_candidates preserve border insertion order.
  std::vector<int> candidates;
  for (size_t i = 0; i < border.keys.size(); ++i)
    if (border.values[i] == max_border) candidates.push_back(border.keys[i]);

  // Strategy 4.
  for (int cand : candidates) {
    double border_ratio = static_cast<double>(border.get(cand)) / total_border;
    auto it = color_counts.find(cand);
    int ccount = it == color_counts.end() ? 0 : it->second;
    double overall_ratio = total_pixels ? static_cast<double>(ccount) / total_pixels : 0.0;
    if (border_ratio >= 0.5 && overall_ratio >= 0.3) return cand;
  }

  if (candidates.size() == 1) return candidates[0];

  // Strategy 5: prefer 0.
  for (int c : candidates)
    if (c == 0) return 0;

  // Strategy 6: overall frequency tie-break.
  int best = candidates[0];
  auto itb = color_counts.find(best);
  int best_count = itb == color_counts.end() ? 0 : itb->second;
  for (size_t i = 1; i < candidates.size(); ++i) {
    int c = candidates[i];
    auto it = color_counts.find(c);
    int cnt = it == color_counts.end() ? 0 : it->second;
    if (cnt > best_count) {
      best = c;
      best_count = cnt;
    }
  }
  return best;
}

}  // namespace

std::optional<int> calculate_background_heuristic(const Grid& grid) {
  int h = grid.height(), w = grid.width();
  if (h == 0 || w == 0) return 0;

  std::map<int, int> color_counts = color_counts_of(grid);

  if (bg_rules_out_any(grid, color_counts)) return std::nullopt;

  OrderedCounts corner_counts = bg_corner_counts(grid);

  int result = 0;
  if (bg_strategy_corners_all_same(corner_counts, result)) return result;
  if (bg_strategy_frame(grid, color_counts, result)) return result;
  if (bg_strategy_corners_majority(corner_counts, color_counts, result)) return result;

  OrderedCounts border_counts = bg_border_counts(grid);
  return bg_strategy_border(border_counts, color_counts);
}

std::optional<int> resolve_background(const Grid& grid, const GraphConfig& cfg) {
  // An explicit override wins; mode "none" means no background; otherwise the
  // heuristic decides.
  if (cfg.background_color_override >= 0) return cfg.background_color_override;
  if (cfg.background_mode == "none") return std::nullopt;
  return calculate_background_heuristic(grid);
}

// ==========================================================================
// Step 3: connected-component extraction (extract_step3).
// ==========================================================================

namespace {

// compute_spatial_ordering: stable-sort by bbox x / bbox y / mass(desc).
// Operates on a flat vector of freshly-created SubObjects (order == creation).
void compute_spatial_ordering(std::vector<Node>& objs) {
  if (objs.empty()) return;
  size_t n = objs.size();
  std::vector<size_t> idxs(n);
  for (size_t i = 0; i < n; ++i) idxs[i] = i;

  // from_left / from_right: sorted by bbox xmin.
  std::vector<size_t> by_x(idxs);
  std::stable_sort(by_x.begin(), by_x.end(), [&](size_t a, size_t b) {
    int ka = objs[a].has_bbox ? objs[a].bbox_xmin : 0;
    int kb = objs[b].has_bbox ? objs[b].bbox_xmin : 0;
    return ka < kb;
  });
  for (size_t i = 0; i < n; ++i) {
    Node& o = objs[by_x[i]];
    o.spatial_order.from_left = static_cast<int>(i) + 1;
    o.spatial_order.from_right = static_cast<int>(n) - static_cast<int>(i);
  }

  // from_top / from_bottom: sorted by bbox ymin.
  std::vector<size_t> by_y(idxs);
  std::stable_sort(by_y.begin(), by_y.end(), [&](size_t a, size_t b) {
    int ka = objs[a].has_bbox ? objs[a].bbox_ymin : 0;
    int kb = objs[b].has_bbox ? objs[b].bbox_ymin : 0;
    return ka < kb;
  });
  for (size_t i = 0; i < n; ++i) {
    Node& o = objs[by_y[i]];
    o.spatial_order.from_top = static_cast<int>(i) + 1;
    o.spatial_order.from_bottom = static_cast<int>(n) - static_cast<int>(i);
  }

  // by_size: sorted by mass, descending (stable).
  std::vector<size_t> by_mass(idxs);
  std::stable_sort(by_mass.begin(), by_mass.end(),
                   [&](size_t a, size_t b) { return objs[a].mass > objs[b].mass; });
  for (size_t i = 0; i < n; ++i) {
    Node& o = objs[by_mass[i]];
    o.spatial_order.by_size = static_cast<int>(i) + 1;
  }
}

}  // namespace

std::vector<Node> extract_step3(Scene& scene, const Grid& grid, const GraphConfig& cfg,
                                const std::set<Coord>& exclude_pixels) {
  int h = grid.height(), w = grid.width();
  const bool bg_as_object = (cfg.background_mode == "bg_as_object");
  const bool has_bg = scene.has_background();
  const int bg = has_bg ? scene.background() : -1;
  const bool multi = (cfg.color_mode == "multi");

  std::vector<uint8_t> visited(static_cast<size_t>(h) * w, 0);
  auto vis = [&](int y, int x) -> uint8_t& { return visited[static_cast<size_t>(y) * w + x]; };

  // exclude_pixels are pre-marked visited; out-of-grid entries are ignored.
  for (const auto& p : exclude_pixels) {
    if (0 <= p.first && p.first < h && 0 <= p.second && p.second < w) vis(p.first, p.second) = 1;
  }

  // SubObjects created this pass, in creation order -- feeds spatial ordering.
  std::vector<Node> created;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (vis(y, x)) continue;

      int color = grid.at(y, x);
      bool skip_bg = has_bg && color == bg && !bg_as_object;
      if (skip_bg) {
        vis(y, x) = 1;
        continue;
      }

      // BFS (FIFO): `pixels` is the queue, in discovery order. Neighbours in
      // the order of get_orthogonal_neighbors / get_8_neighbors.
      std::vector<Coord> pixels;
      pixels.emplace_back(y, x);
      vis(y, x) = 1;
      auto visit = [&](int ny, int nx) {
        if (ny < 0 || ny >= h || nx < 0 || nx >= w || vis(ny, nx)) return;
        if (multi) {
          int nc = grid.at(ny, nx);
          bool is_bg = has_bg && nc == bg && !bg_as_object;
          if (!is_bg) {
            vis(ny, nx) = 1;
            pixels.emplace_back(ny, nx);
          }
        } else if (grid.at(ny, nx) == color) {
          vis(ny, nx) = 1;
          pixels.emplace_back(ny, nx);
        }
      };
      for (size_t head = 0; head < pixels.size(); ++head) {
        const int cy = pixels[head].first, cx = pixels[head].second;
        if (cfg.connectivity == 8) {
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              if (dy != 0 || dx != 0) visit(cy + dy, cx + dx);
        } else {
          visit(cy - 1, cx);
          visit(cy + 1, cx);
          visit(cy, cx - 1);
          visit(cy, cx + 1);
        }
      }

      scene.node_counter += 1;

      // Multi-color dominant pick: the most frequent color; on a tie, the one
      // first reached in BFS pixel-discovery order.
      if (multi) {
        std::vector<int> keys;
        std::map<int, int> idx;
        std::vector<int> vals;
        for (const auto& p : pixels) {
          int c = grid.at(p.first, p.second);
          auto it = idx.find(c);
          if (it == idx.end()) {
            idx[c] = static_cast<int>(keys.size());
            keys.push_back(c);
            vals.push_back(1);
          } else {
            vals[it->second] += 1;
          }
        }
        int best = keys.empty() ? color : keys[0];
        int best_v = keys.empty() ? 0 : vals[0];
        for (size_t i = 1; i < keys.size(); ++i) {
          if (vals[i] > best_v) {
            best_v = vals[i];
            best = keys[i];
          }
        }
        color = best;
      }

      Node subobj;
      subobj.node_id = "sub_" + std::to_string(scene.node_counter);
      subobj.node_type = "SubObject";
      subobj.color = color;
      subobj.pixel_list = std::move(pixels);
      subobj.post_init();  // bbox, centroid, dims, mass, density
      subobj.check_boundary_touches(w, h);
      subobj.compute_perimeter_pixels();
      subobj.compute_interior_pixels();
      if (cfg.compute_shape_hash) subobj.compute_shape_hash();
      if (cfg.compute_shape_classification) subobj.classify_shape();

      created.push_back(std::move(subobj));
    }
  }

  compute_spatial_ordering(created);
  return created;
}

// Step 3c inline logic (multi_color_as_cluster): pull multi-color SubObjects out
// into ClusterObjects. cluster ids are assigned interleaved with the subobject
// scan, so the shared counter advances in that same order.
namespace {

void step3c_multi_color_as_cluster(Scene& scene, const Grid& grid, std::vector<Node>& subobjects,
                                   std::vector<Node>& clusters_out) {
  std::vector<Node> new_subs;
  for (auto& s : subobjects) {
    // Only objects with more than one color become clusters.
    std::set<int> colors_in_obj;
    for (const auto& p : s.pixel_list) colors_in_obj.insert(grid.at(p.first, p.second));
    if (colors_in_obj.size() > 1) {
      // color_hist: colors in first-seen order of the pixel_list scan.
      std::vector<std::pair<int, int>> hist;
      std::map<int, int> hidx;
      for (const auto& p : s.pixel_list) {
        int c = grid.at(p.first, p.second);
        auto it = hidx.find(c);
        if (it == hidx.end()) {
          hidx[c] = static_cast<int>(hist.size());
          hist.emplace_back(c, 1);
        } else {
          hist[it->second].second += 1;
        }
      }
      scene.node_counter += 1;
      Node cluster;
      cluster.node_id = "cluster_" + std::to_string(scene.node_counter);
      cluster.node_type = "ClusterObject";
      cluster.pixel_list = s.pixel_list;  // same pixels
      cluster.color_histogram = hist;
      cluster.post_init();
      cluster.check_boundary_touches(scene.width(), scene.height());
      cluster.compute_shape_hash();
      cluster.classify_shape_cluster();
      // As in step 5c, colors are the color_histogram keys in ascending order.
      std::vector<int> keys;
      for (const auto& kv : hist) keys.push_back(kv.first);
      std::sort(keys.begin(), keys.end());
      cluster.colors = keys;
      cluster.has_colors = true;
      clusters_out.push_back(std::move(cluster));
    } else {
      new_subs.push_back(std::move(s));
    }
  }
  subobjects.swap(new_subs);
}

}  // namespace

Scene encode_minimal(const Grid& grid, const GraphConfig& cfg) {
  Scene scene(grid.width(), grid.height());

  // Keep a copy of the input grid on the scene: a few DSL actions read it
  // directly (PROJECT_SINGLETONS_TO_FULL_LINES).
  scene.set_grid(grid);

  // Degenerate (zero-width/height) grid: a deep-search program can decode to an
  // empty grid, and re-encoding it (the fitness cycle) would read out of bounds --
  // e.g. find_grid_lines does grid.at(y,0) with width 0 -> null deref (SEGV
  // grid.hpp:29). Nothing to encode; return the minimal scene so the genome is
  // scored as a non-match instead of crashing the whole run.
  if (grid.width() <= 0 || grid.height() <= 0) return scene;

  // Step 1: background.
  scene.set_background(resolve_background(grid, cfg));

  // Step 2: zones (structural gridline SubObjects + ZoneNodes). Returns the
  // grid-line pixel set fed to step 3's exclude_pixels. Grid lines are scanned
  // only when detect_zones is on; otherwise no pixels are excluded.
  std::set<Coord> exclude_pixels;
  if (cfg.detect_zones) {
    exclude_pixels = detect_zones(scene, grid, cfg);
  }
  // Step 9b (compute_zone_features) consumes the grid-line pixel set; store it
  // on the scene so it survives past this function's local scope.
  scene.set_grid_line_pixels(exclude_pixels);

  // Step 3: connected-component extraction (not yet added to the scene).
  std::vector<Node> subobjects = extract_step3(scene, grid, cfg, exclude_pixels);

  // Step 3b: decompose irregular objects.
  if (cfg.decompose_objects) {
    subobjects = decompose_objects(scene, std::move(subobjects), grid, cfg);
  }

  // Step 3c: multi-color -> cluster.
  std::vector<Node> clusters_from_multi;
  if (cfg.color_mode == "multi" && cfg.multi_color_as_cluster) {
    step3c_multi_color_as_cluster(scene, grid, subobjects, clusters_from_multi);
  }

  // Step 3d: merge same-color objects by containment.
  if (cfg.merge_by_containment) {
    subobjects = merge_objects_by_containment(scene, std::move(subobjects), grid);
  }

  // min_object_size filter (size_first): drop small objects before cavities and
  // clusters are built. Merged objects are exempt. The default 0 disables it.
  if (cfg.noise_filter_order == "size_first" && cfg.min_object_size > 0) {
    std::vector<Node> filtered;
    for (auto& s : subobjects) {
      if (s.mass >= cfg.min_object_size || s.merged) filtered.push_back(std::move(s));
    }
    subobjects.swap(filtered);
  }

  // merge_distance has no effect: nearby objects are not merged.

  // Add the surviving SubObjects, then the ClusterObjects.
  scene.reserve_nodes(subobjects.size() + clusters_from_multi.size());
  for (const auto& s : subobjects) scene.add_node(s);
  for (const auto& c : clusters_from_multi) scene.add_node(c);

  // Step 4: detect cavities (foreground = post-3d subobjects only).
  if (cfg.detect_cavities) {
    std::vector<Node> cavities = detect_cavities(scene, grid, subobjects);
    scene.reserve_nodes(cavities.size());
    for (auto& cav : cavities) scene.add_node(std::move(cav));
  }

  // Step 5: group into clusters. Step 5c (colors) is applied inline per cluster.
  if (cfg.detect_clusters) {
    std::vector<Node> clusters = detect_clusters(scene, subobjects, cfg);
    scene.reserve_nodes(clusters.size());
    for (auto& c : clusters) scene.add_node(std::move(c));
  }

  // Step 5d: split by containment (mutates the scene). Default-off.
  if (cfg.split_by_containment) {
    split_by_containment(scene);
  }

  // Step 6: spatial edges (TOUCHES / OVERLAPS / directional).
  if (cfg.compute_spatial_edges) {
    compute_spatial_edges(scene, cfg);
  }

  // Step 7: noise scoring (requires spatial edges for the isolation signal).
  if (cfg.compute_noise_scoring) {
    score_noise_candidates(scene, cfg);
    // Auto-reclassify if threshold < 1.0 (default 1.0 -> off). Iterating
    // get_nodes_by_type("SubObject") in lexicographic order; reclassify is a
    // node-type + layer-map mutation, so the visitation order is irrelevant to
    // the resulting set.
    if (cfg.noise_threshold < 1.0) {
      std::vector<size_t> subs = scene.get_nodes_by_type("SubObject");
      std::vector<std::string> to_reclass;
      for (size_t idx : subs)
        if (scene.nodes()[idx].noise_score > cfg.noise_threshold)
          to_reclass.push_back(scene.nodes()[idx].node_id);
      for (const auto& nid : to_reclass) scene.reclassify_as_noise(nid);
    }
  }

  // noise_first: apply min_object_size AFTER noise reclassification.
  if (cfg.noise_filter_order == "noise_first" && cfg.min_object_size > 0) {
    std::vector<size_t> subs = scene.get_nodes_by_type("SubObject");
    std::vector<std::string> to_remove;
    for (size_t idx : subs)
      if (scene.nodes()[idx].mass < cfg.min_object_size)
        to_remove.push_back(scene.nodes()[idx].node_id);
    scene.remove_nodes(to_remove);
  }

  // Step 8: containment edges (CONTAINS / IS_PART_OF + cluster/cavity edges).
  if (cfg.compute_containment_edges) {
    compute_containment_edges(scene);
  }

  // Step 9: local coordinates + IS_INSIDE_ZONE (requires zones).
  if (cfg.compute_local_coordinates) {
    assign_local_coordinates(scene);
  }

  // Step 9b: zone features. Runs only when detect_zones is also on; without
  // zones no grid lines are scanned, so there are no zone features to compute.
  if (cfg.compute_zone_features && cfg.detect_zones) {
    compute_zone_features(scene, grid);
  }

  // Step 10: z_index over the CONTAINS containment DAG.
  if (cfg.compute_z_index) {
    compute_z_indices(scene);
  }

  // Step 11: alignment edges (SHARES_ROW/COLUMN, COLLINEAR_TO, DIAGONAL_TO,
  // SAME_CENTROID_ROW/COLUMN).
  if (cfg.compute_alignment_edges) {
    compute_alignment_edges(scene, cfg);
  }

  // Step 12: ray edges (PROJECTS_RAY_TO).
  if (cfg.compute_ray_edges) {
    compute_ray_edges(scene);
  }

  // Step 13: gravity edges (FALLS_TOWARDS).
  if (cfg.compute_gravity_edges) {
    compute_gravity_edges(scene);
  }

  // Step 14: bridge edges (BRIDGED_TO). Requires spatial (TOUCHES) edges.
  if (cfg.compute_bridge_edges) {
    compute_bridge_edges(scene);
  }

  // Step 15: logic edges (SAME_SHAPE, SAME_COLOR, IS_SYMMETRIC_TO).
  if (cfg.compute_logic_edges) {
    compute_logic_edges(scene);
  }

  // Step 16: patterns (PatternObject nodes + FORMS_REPEATING_PATTERN / IS_PART_OF).
  if (cfg.detect_patterns) {
    detect_patterns(scene, cfg);
  }

  // Step 17: symmetries (ImplicitObject axis nodes + IS_SYMMETRIC_TO edges).
  if (cfg.detect_symmetries) {
    detect_symmetries(scene, grid, cfg);
  }

  // Step 18: object attributes (counts, shape predicates, ranks) for BY_ATTR.
  compute_object_attributes(scene);

  scene.drop_node_handles();
  return scene;
}

namespace {

void json_escape(std::ostringstream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << c;
    }
  }
  os << '"';
}

// A double as the lowercase hex of its 8 little-endian IEEE-754 bytes, with no
// separators.
std::string fbits(double x) {
  uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  static const char* hexd = "0123456789abcdef";
  std::string out;
  out.reserve(16);
  for (int i = 0; i < 8; ++i) {
    uint8_t b = static_cast<uint8_t>((bits >> (8 * i)) & 0xff);
    out.push_back(hexd[b >> 4]);
    out.push_back(hexd[b & 0xf]);
  }
  return out;
}

// Sorted (y,x) coordinate list as a JSON array of [y,x] pairs.
void emit_coords(std::ostringstream& os, std::vector<Coord> coords) {
  std::stable_sort(coords.begin(), coords.end());
  os << '[';
  for (size_t i = 0; i < coords.size(); ++i) {
    if (i) os << ',';
    os << '[' << coords[i].first << ',' << coords[i].second << ']';
  }
  os << ']';
}

// Emits everything from the opening '{' up to (but NOT including) the closing
// '}', so callers can append extra fields before closing. emit_node (step1..8)
// and emit_node_step10 both wrap this.
void emit_node_common(std::ostringstream& os, const Node& n) {
  os << '{';
  os << "\"node_id\":"; json_escape(os, n.node_id);
  os << ",\"node_type\":"; json_escape(os, n.node_type);
  os << ",\"color\":";
  if (n.color.has_value()) os << *n.color; else os << "null";
  os << ",\"bbox\":";
  if (n.has_bbox)
    os << '[' << n.bbox_ymin << ',' << n.bbox_xmin << ',' << n.bbox_ymax << ',' << n.bbox_xmax << ']';
  else
    os << "null";
  os << ",\"width\":" << n.width;
  os << ",\"height\":" << n.height;
  os << ",\"centroid\":";
  if (n.has_centroid) {
    os << '[';
    json_escape(os, fbits(n.centroid_y));
    os << ',';
    json_escape(os, fbits(n.centroid_x));
    os << ']';
  } else {
    os << "null";
  }
  os << ",\"mass\":" << n.mass;
  os << ",\"density\":"; json_escape(os, fbits(n.density));
  os << ",\"z_index\":" << n.z_index;
  os << ",\"spatial_order\":{";
  os << "\"from_left\":" << n.spatial_order.from_left;
  os << ",\"from_right\":" << n.spatial_order.from_right;
  os << ",\"from_top\":" << n.spatial_order.from_top;
  os << ",\"from_bottom\":" << n.spatial_order.from_bottom;
  os << ",\"by_size\":" << n.spatial_order.by_size;
  os << '}';
  os << ",\"shape_hash\":";
  if (n.shape_hash.has_value()) json_escape(os, *n.shape_hash); else os << "null";
  os << ",\"shape_mask\":";
  if (n.shape_mask.has_value()) json_escape(os, *n.shape_mask); else os << "null";
  os << ",\"is_structural\":" << (n.is_structural ? "true" : "false");
  os << ",\"touches_top\":" << (n.touches_top ? "true" : "false");
  os << ",\"touches_bottom\":" << (n.touches_bottom ? "true" : "false");
  os << ",\"touches_left\":" << (n.touches_left ? "true" : "false");
  os << ",\"touches_right\":" << (n.touches_right ? "true" : "false");
  os << ",\"noise_score\":"; json_escape(os, fbits(n.noise_score));
  os << ",\"pixel_list\":"; emit_coords(os, n.pixel_list);
  os << ",\"perimeter_pixels\":"; emit_coords(os, n.perimeter_pixels.get());
  os << ",\"interior_pixels\":"; emit_coords(os, n.interior_pixels.get());
  // Type-specific trailing fields, after the common ones.
  if (n.node_type == "ZoneNode") {
    os << ",\"zone_color\":";
    if (n.zone_color.has_value()) os << *n.zone_color; else os << "null";
  } else if (n.node_type == "ClusterObject") {
    os << ",\"color_histogram\":[";
    for (size_t i = 0; i < n.color_histogram.size(); ++i) {
      if (i) os << ',';
      os << '[' << n.color_histogram[i].first << ',' << n.color_histogram[i].second << ']';
    }
    os << ']';
    os << ",\"colors\":[";
    for (size_t i = 0; i < n.colors.size(); ++i) {
      if (i) os << ',';
      os << n.colors[i];
    }
    os << ']';
  } else if (n.node_type == "CavityObject") {
    os << ",\"enclosing_object_id\":";
    if (n.enclosing_object_id.has_value()) json_escape(os, *n.enclosing_object_id);
    else os << "null";
    os << ",\"cavity_color\":";
    if (n.cavity_color.has_value()) os << *n.cavity_color; else os << "null";
  } else if (n.node_type == "PatternObject") {
    os << ",\"pattern_type\":";
    if (n.pattern_type.has_value()) json_escape(os, *n.pattern_type); else os << "null";
    os << ",\"constituent_ids\":[";
    for (size_t i = 0; i < n.constituent_ids.size(); ++i) {
      if (i) os << ',';
      json_escape(os, n.constituent_ids[i]);
    }
    os << ']';
  } else if (n.node_type == "ImplicitObject") {
    os << ",\"implicit_type\":";
    if (n.implicit_type.has_value()) json_escape(os, *n.implicit_type); else os << "null";
    // parameters: exactly one key -- "y"/"x" (double as hex bits) or "type"
    // (string).
    os << ",\"parameters\":{";
    if (n.has_param_y) {
      os << "\"y\":";
      json_escape(os, fbits(n.param_y));
    } else if (n.has_param_x) {
      os << "\"x\":";
      json_escape(os, fbits(n.param_x));
    } else if (n.param_type.has_value()) {
      os << "\"type\":";
      json_escape(os, *n.param_type);
    }
    os << '}';
  }
}

// Step 1..8 node object: common fields, closed.
void emit_node(std::ostringstream& os, const Node& n) {
  emit_node_common(os, n);
  os << '}';
}

// Step 1..10 node object: common fields + step 9/9b/10 fields, in a fixed key
// order (local_bounding_box, local_centroid, zone_row, zone_col, then the
// ZoneNode-only zone-feature fields).
void emit_node_step10(std::ostringstream& os, const Node& n) {
  emit_node_common(os, n);

  os << ",\"local_bounding_box\":";
  if (n.has_local_bbox)
    os << '[' << n.local_bbox_ymin << ',' << n.local_bbox_xmin << ','
       << n.local_bbox_ymax << ',' << n.local_bbox_xmax << ']';
  else
    os << "null";

  os << ",\"local_centroid\":";
  if (n.has_local_centroid) {
    os << '[';
    json_escape(os, fbits(n.local_centroid_y));
    os << ',';
    json_escape(os, fbits(n.local_centroid_x));
    os << ']';
  } else {
    os << "null";
  }

  os << ",\"zone_row\":";
  if (n.zone_row.has_value()) os << *n.zone_row; else os << "null";
  os << ",\"zone_col\":";
  if (n.zone_col.has_value()) os << *n.zone_col; else os << "null";

  if (n.node_type == "ZoneNode") {
    // Zone-feature fields: null when never computed (has_zone_features false).
    if (n.has_zone_features) {
      os << ",\"n_colors\":" << n.zf_n_colors;
      os << ",\"dominant_color\":";
      if (n.zf_dominant_color.has_value()) os << *n.zf_dominant_color; else os << "null";
      os << ",\"n_objects\":" << n.zf_n_objects;
      os << ",\"fill_ratio\":"; json_escape(os, fbits(n.zf_fill_ratio));
      os << ",\"n_inside_objects\":" << n.zf_n_inside_objects;
      os << ",\"n_outside_objects\":" << n.zf_n_outside_objects;
      os << ",\"inside_colors\":[";
      for (size_t i = 0; i < n.zf_inside_colors.size(); ++i) {
        if (i) os << ',';
        os << n.zf_inside_colors[i];
      }
      os << ']';
      os << ",\"outside_colors\":[";
      for (size_t i = 0; i < n.zf_outside_colors.size(); ++i) {
        if (i) os << ',';
        os << n.zf_outside_colors[i];
      }
      os << ']';
      for (int cv = 0; cv < 10; ++cv) {
        os << ",\"has_color_" << cv << "\":" << (n.zf_has_color[cv] ? "true" : "false");
      }
    } else {
      os << ",\"n_colors\":null";
      os << ",\"dominant_color\":null";
      os << ",\"n_objects\":null";
      os << ",\"fill_ratio\":null";
      os << ",\"n_inside_objects\":null";
      os << ",\"n_outside_objects\":null";
      os << ",\"inside_colors\":null";
      os << ",\"outside_colors\":null";
      for (int cv = 0; cv < 10; ++cv) os << ",\"has_color_" << cv << "\":null";
    }
  }

  os << '}';
}

// All non-structural node ids across every type, byte-wise lexicographically
// sorted.
std::vector<size_t> all_serialized_nodes(const Scene& scene) {
  std::vector<std::pair<std::string, size_t>> entries;
  for (size_t i = 0; i < scene.nodes().size(); ++i) {
    const Node& n = scene.nodes()[i];
    if (n.is_structural) continue;
    entries.emplace_back(n.node_id, i);
  }
  std::stable_sort(entries.begin(), entries.end(),
                   [](const std::pair<std::string, size_t>& a,
                      const std::pair<std::string, size_t>& b) { return a.first < b.first; });
  std::vector<size_t> out;
  out.reserve(entries.size());
  for (const auto& e : entries) out.push_back(e.second);
  return out;
}

}  // namespace

std::string scene_to_canonical_json(const Scene& scene) {
  std::vector<size_t> ids = all_serialized_nodes(scene);

  std::ostringstream os;
  os << '{';
  os << "\"background\":";
  if (scene.has_background()) os << scene.background(); else os << "null";
  os << ",\"width\":" << scene.width();
  os << ",\"height\":" << scene.height();
  os << ",\"nodes\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) os << ',';
    emit_node(os, scene.nodes()[ids[i]]);
  }
  os << "]}";
  return os.str();
}

std::string scene_to_canonical_json_with_edges(const Scene& scene) {
  std::vector<size_t> ids = all_serialized_nodes(scene);

  std::ostringstream os;
  os << '{';
  os << "\"background\":";
  if (scene.has_background()) os << scene.background(); else os << "null";
  os << ",\"width\":" << scene.width();
  os << ",\"height\":" << scene.height();
  os << ",\"nodes\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) os << ',';
    emit_node(os, scene.nodes()[ids[i]]);
  }
  os << ']';

  // Edges sorted by (source_id, target_id, edge_type). overlap_size is the only
  // step1..8 attribute; emit it as a number for OVERLAPS edges, null otherwise.
  std::vector<size_t> eidx(scene.edges().size());
  for (size_t i = 0; i < eidx.size(); ++i) eidx[i] = i;
  const std::vector<Edge>& edges = scene.edges();
  std::stable_sort(eidx.begin(), eidx.end(), [&](size_t a, size_t b) {
    const Edge& ea = edges[a];
    const Edge& eb = edges[b];
    if (ea.source_id != eb.source_id) return ea.source_id < eb.source_id;
    if (ea.target_id != eb.target_id) return ea.target_id < eb.target_id;
    return ea.edge_type < eb.edge_type;
  });
  os << ",\"edges\":[";
  for (size_t i = 0; i < eidx.size(); ++i) {
    if (i) os << ',';
    const Edge& e = edges[eidx[i]];
    os << '{';
    os << "\"src\":"; json_escape(os, e.source_id);
    os << ",\"dst\":"; json_escape(os, e.target_id);
    os << ",\"type\":"; json_escape(os, e.edge_type);
    os << ",\"overlap_size\":";
    if (e.has_int_attr("overlap_size")) os << e.get_int_attr("overlap_size"); else os << "null";
    os << '}';
  }
  os << ']';
  os << '}';
  return os.str();
}

std::string scene_to_canonical_json_step10(const Scene& scene) {
  std::vector<size_t> ids = all_serialized_nodes(scene);

  std::ostringstream os;
  os << '{';
  os << "\"background\":";
  if (scene.has_background()) os << scene.background(); else os << "null";
  os << ",\"width\":" << scene.width();
  os << ",\"height\":" << scene.height();
  os << ",\"nodes\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) os << ',';
    emit_node_step10(os, scene.nodes()[ids[i]]);
  }
  os << ']';

  // Edges sorted by (source_id, target_id, edge_type). The step1..10 edge set
  // also includes IS_INSIDE_ZONE edges (step 9); overlap_size is the only
  // integer attribute.
  std::vector<size_t> eidx(scene.edges().size());
  for (size_t i = 0; i < eidx.size(); ++i) eidx[i] = i;
  const std::vector<Edge>& edges = scene.edges();
  std::stable_sort(eidx.begin(), eidx.end(), [&](size_t a, size_t b) {
    const Edge& ea = edges[a];
    const Edge& eb = edges[b];
    if (ea.source_id != eb.source_id) return ea.source_id < eb.source_id;
    if (ea.target_id != eb.target_id) return ea.target_id < eb.target_id;
    return ea.edge_type < eb.edge_type;
  });
  os << ",\"edges\":[";
  for (size_t i = 0; i < eidx.size(); ++i) {
    if (i) os << ',';
    const Edge& e = edges[eidx[i]];
    os << '{';
    os << "\"src\":"; json_escape(os, e.source_id);
    os << ",\"dst\":"; json_escape(os, e.target_id);
    os << ",\"type\":"; json_escape(os, e.edge_type);
    os << ",\"overlap_size\":";
    if (e.has_int_attr("overlap_size")) os << e.get_int_attr("overlap_size"); else os << "null";
    os << '}';
  }
  os << ']';
  os << '}';
  return os.str();
}

std::string scene_to_canonical_json_step15(const Scene& scene) {
  std::vector<size_t> ids = all_serialized_nodes(scene);

  std::ostringstream os;
  os << '{';
  os << "\"background\":";
  if (scene.has_background()) os << scene.background(); else os << "null";
  os << ",\"width\":" << scene.width();
  os << ",\"height\":" << scene.height();
  os << ",\"nodes\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) os << ',';
    emit_node_step10(os, scene.nodes()[ids[i]]);  // steps 11..15 add no node fields
  }
  os << ']';

  // Edges sorted by (source_id, target_id, edge_type). The step1..15 edge set
  // adds alignment / ray / gravity / bridge / logic edges. Each edge carries a
  // FIXED attribute-slot block: overlap_size (int|null), axis (str|null),
  // direction (str|null), distance (double hex bits|null), fall_distance
  // (int|null), bridge_node (str|null). Absent slots serialize as null.
  std::vector<size_t> eidx(scene.edges().size());
  for (size_t i = 0; i < eidx.size(); ++i) eidx[i] = i;
  const std::vector<Edge>& edges = scene.edges();
  std::stable_sort(eidx.begin(), eidx.end(), [&](size_t a, size_t b) {
    const Edge& ea = edges[a];
    const Edge& eb = edges[b];
    if (ea.source_id != eb.source_id) return ea.source_id < eb.source_id;
    if (ea.target_id != eb.target_id) return ea.target_id < eb.target_id;
    return ea.edge_type < eb.edge_type;
  });
  os << ",\"edges\":[";
  for (size_t i = 0; i < eidx.size(); ++i) {
    if (i) os << ',';
    const Edge& e = edges[eidx[i]];
    os << '{';
    os << "\"src\":"; json_escape(os, e.source_id);
    os << ",\"dst\":"; json_escape(os, e.target_id);
    os << ",\"type\":"; json_escape(os, e.edge_type);
    os << ",\"overlap_size\":";
    if (e.has_int_attr("overlap_size")) os << e.get_int_attr("overlap_size"); else os << "null";
    os << ",\"axis\":";
    if (e.has_str_attr("axis")) json_escape(os, e.get_str_attr("axis")); else os << "null";
    os << ",\"direction\":";
    if (e.has_str_attr("direction")) json_escape(os, e.get_str_attr("direction")); else os << "null";
    os << ",\"distance\":";
    if (e.has_dbl_attr("distance")) json_escape(os, fbits(e.get_dbl_attr("distance")));
    else os << "null";
    os << ",\"fall_distance\":";
    if (e.has_int_attr("fall_distance")) os << e.get_int_attr("fall_distance"); else os << "null";
    os << ",\"bridge_node\":";
    if (e.has_str_attr("bridge_node")) json_escape(os, e.get_str_attr("bridge_node"));
    else os << "null";
    os << '}';
  }
  os << ']';
  os << '}';
  return os.str();
}

std::string scene_to_canonical_json_full(const Scene& scene) {
  // The full step1..17 encoding. Nodes use the step10 emitter (emit_node_common
  // writes the PatternObject / ImplicitObject trailing fields; step9/9b/10
  // fields serialize as null for these node types). Edges use the step15 block
  // (FORMS_REPEATING_PATTERN / IS_PART_OF / IS_SYMMETRIC_TO edges carry no
  // extra attributes).
  std::vector<size_t> ids = all_serialized_nodes(scene);

  std::ostringstream os;
  os << '{';
  os << "\"background\":";
  if (scene.has_background()) os << scene.background(); else os << "null";
  os << ",\"width\":" << scene.width();
  os << ",\"height\":" << scene.height();
  os << ",\"nodes\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) os << ',';
    emit_node_step10(os, scene.nodes()[ids[i]]);
  }
  os << ']';

  std::vector<size_t> eidx(scene.edges().size());
  for (size_t i = 0; i < eidx.size(); ++i) eidx[i] = i;
  const std::vector<Edge>& edges = scene.edges();
  std::stable_sort(eidx.begin(), eidx.end(), [&](size_t a, size_t b) {
    const Edge& ea = edges[a];
    const Edge& eb = edges[b];
    if (ea.source_id != eb.source_id) return ea.source_id < eb.source_id;
    if (ea.target_id != eb.target_id) return ea.target_id < eb.target_id;
    return ea.edge_type < eb.edge_type;
  });
  os << ",\"edges\":[";
  for (size_t i = 0; i < eidx.size(); ++i) {
    if (i) os << ',';
    const Edge& e = edges[eidx[i]];
    os << '{';
    os << "\"src\":"; json_escape(os, e.source_id);
    os << ",\"dst\":"; json_escape(os, e.target_id);
    os << ",\"type\":"; json_escape(os, e.edge_type);
    os << ",\"overlap_size\":";
    if (e.has_int_attr("overlap_size")) os << e.get_int_attr("overlap_size"); else os << "null";
    os << ",\"axis\":";
    if (e.has_str_attr("axis")) json_escape(os, e.get_str_attr("axis")); else os << "null";
    os << ",\"direction\":";
    if (e.has_str_attr("direction")) json_escape(os, e.get_str_attr("direction")); else os << "null";
    os << ",\"distance\":";
    if (e.has_dbl_attr("distance")) json_escape(os, fbits(e.get_dbl_attr("distance")));
    else os << "null";
    os << ",\"fall_distance\":";
    if (e.has_int_attr("fall_distance")) os << e.get_int_attr("fall_distance"); else os << "null";
    os << ",\"bridge_node\":";
    if (e.has_str_attr("bridge_node")) json_escape(os, e.get_str_attr("bridge_node"));
    else os << "null";
    os << '}';
  }
  os << ']';
  os << '}';
  return os.str();
}
}  // namespace arc
