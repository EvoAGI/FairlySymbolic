// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 9b: zone features (compute_zone_features).
//
// Determinism notes:
//   * Each ZoneNode's features are written independently, so the zone iteration
//     order does not affect the output; zones are visited in lexicographic
//     node-id order anyway.
//   * physical_objects is collected from the node vector in insertion order,
//     filtered to the physical node types and non-structural, needing a bbox.
//     Its order only feeds order-independent counts here.
//   * pixel_colors is an insertion-ordered (scan-order) histogram; dominant_color
//     is the most frequent color, the first inserted on a tie.
//   * fill_ratio = round(total_fg / zone_area, 4): correctly rounded to 4
//     decimals with ties to even, via snprintf("%.4f")+strtod.
//   * zone_row/col: unique sorted ymins/xmins -> index maps; each zone's row/col
//     is then propagated to its children over CONTAINS out-edges then
//     IS_INSIDE_ZONE in-edges (edge vector insertion order), with a
//     `propagated` dedup set.

#include "arc/encode.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace arc {

namespace {

// round(x, 4): x correctly rounded to 4 decimal places, ties to even.
// snprintf("%.4f") rounds the exact binary value and strtod reads the text back
// as the nearest double.
double py_round4(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", x);
  return std::strtod(buf, nullptr);
}

}  // namespace

void compute_zone_features(Scene& scene, const Grid& grid) {
  std::optional<int> bg = scene.has_background() ? std::optional<int>(scene.background())
                                                 : std::nullopt;
  bool has_bg = bg.has_value();
  int bgv = has_bg ? *bg : 0;

  // Cache physical objects up front (insertion order, non-structural, bbox).
  struct Phys {
    int oy0, ox0, oy1, ox1;
    std::optional<int> color;
  };
  std::vector<Phys> physical_objects;
  const auto& nodes = scene.nodes();
  for (const Node& nd : nodes) {
    const std::string& t = nd.node_type;
    bool physical = (t == "SubObject" || t == "ClusterObject" || t == "CavityObject" ||
                     t == "NoiseObject" || t == "PatternObject");
    if (!physical) continue;
    if (nd.is_structural) continue;
    if (!nd.has_bbox) continue;
    physical_objects.push_back(
        {nd.bbox_ymin, nd.bbox_xmin, nd.bbox_ymax, nd.bbox_xmax, nd.color});
  }

  const std::set<Coord>& gl = scene.grid_line_pixels();

  // get_nodes_by_type("ZoneNode") -> lexicographic, non-structural.
  std::vector<size_t> zone_idxs = scene.get_nodes_by_type("ZoneNode");

  for (size_t zi : zone_idxs) {
    Node& zone = scene.node_ref(zi);
    if (!zone.has_bbox) continue;

    int ymin = zone.bbox_ymin, xmin = zone.bbox_xmin, ymax = zone.bbox_ymax, xmax = zone.bbox_xmax;
    long long zone_area =
        static_cast<long long>(ymax - ymin + 1) * static_cast<long long>(xmax - xmin + 1);

    // n_objects = count of child_nodes still present in the graph.
    int n_objects = 0;
    for (const std::string& c : zone.child_nodes)
      if (scene.has_node(c)) ++n_objects;

    // pixel_colors: scan-order insertion histogram over the zone bbox, skipping
    // background and grid-line pixels.
    std::vector<int> pc_keys;
    std::map<int, int> pc_idx;
    std::vector<int> pc_val;
    int total_fg_pixels = 0;
    for (int y = ymin; y <= ymax; ++y) {
      for (int x = xmin; x <= xmax; ++x) {
        int c = grid.at(y, x);
        if (has_bg && c == bgv) continue;
        if (gl.find(Coord(y, x)) != gl.end()) continue;
        auto it = pc_idx.find(c);
        if (it == pc_idx.end()) {
          pc_idx[c] = static_cast<int>(pc_keys.size());
          pc_keys.push_back(c);
          pc_val.push_back(1);
        } else {
          pc_val[it->second] += 1;
        }
        total_fg_pixels += 1;
      }
    }

    int n_colors = static_cast<int>(pc_keys.size());
    // dominant_color: the most frequent color, the first inserted on a tie;
    // else bg (which may be unset).
    std::optional<int> dominant_color;
    if (!pc_keys.empty()) {
      int best = pc_keys[0];
      int best_v = pc_val[0];
      for (size_t k = 1; k < pc_keys.size(); ++k) {
        if (pc_val[k] > best_v) {
          best_v = pc_val[k];
          best = pc_keys[k];
        }
      }
      dominant_color = best;
    } else {
      dominant_color = bg;
    }
    double fill_ratio = zone_area > 0
                            ? static_cast<double>(total_fg_pixels) / static_cast<double>(zone_area)
                            : 0.0;

    // inside/outside object buckets.
    int n_inside = 0, n_outside = 0;
    std::set<int> inside_colors, outside_colors;
    for (const Phys& p : physical_objects) {
      bool fully_inside = (p.oy0 >= ymin && p.ox0 >= xmin && p.oy1 <= ymax && p.ox1 <= xmax);
      bool fully_outside = (p.oy1 < ymin || p.oy0 > ymax || p.ox1 < xmin || p.ox0 > xmax);
      if (fully_inside) {
        ++n_inside;
        if (p.color.has_value()) inside_colors.insert(*p.color);
      } else if (fully_outside) {
        ++n_outside;
        if (p.color.has_value()) outside_colors.insert(*p.color);
      }
    }

    zone.has_zone_features = true;
    zone.zf_n_colors = n_colors;
    zone.zf_dominant_color = dominant_color;
    zone.zf_n_objects = n_objects;
    zone.zf_fill_ratio = py_round4(fill_ratio);
    zone.zf_n_inside_objects = n_inside;
    zone.zf_n_outside_objects = n_outside;
    zone.zf_inside_colors.assign(inside_colors.begin(), inside_colors.end());
    zone.zf_outside_colors.assign(outside_colors.begin(), outside_colors.end());
    for (int cv = 0; cv < 10; ++cv)
      zone.zf_has_color[cv] = (pc_idx.find(cv) != pc_idx.end());
  }

  // --- Zone grid position (zone_row, zone_col) ---
  struct ZB {
    size_t idx;
    int ymin, xmin;
  };
  std::vector<ZB> zone_bboxes;
  for (size_t zi : zone_idxs) {
    const Node& z = scene.nodes()[zi];
    if (!z.has_bbox) continue;
    zone_bboxes.push_back({zi, z.bbox_ymin, z.bbox_xmin});
  }
  if (zone_bboxes.empty()) return;

  std::set<int> uniq_ymins, uniq_xmins;
  for (const ZB& z : zone_bboxes) {
    uniq_ymins.insert(z.ymin);
    uniq_xmins.insert(z.xmin);
  }
  std::map<int, int> ymin_to_row, xmin_to_col;
  {
    int i = 0;
    for (int y : uniq_ymins) ymin_to_row[y] = i++;
    i = 0;
    for (int x : uniq_xmins) xmin_to_col[x] = i++;
  }

  const std::vector<EdgeRec>& edges = scene.edge_recs();
  for (const ZB& zb : zone_bboxes) {
    int row_idx = ymin_to_row[zb.ymin];
    int col_idx = xmin_to_col[zb.xmin];
    const uint32_t zone_h = scene.find_edge_node(scene.nodes()[zb.idx].node_id);
    scene.node_ref(zb.idx).zone_row = row_idx;
    scene.node_ref(zb.idx).zone_col = col_idx;
    if (zone_h == Scene::kNoEdgeNode) continue;  // no edge names the zone

    // Propagate to children: CONTAINS out-edges then IS_INSIDE_ZONE in-edges,
    // with a dedup set (the double-propagation guard).
    std::set<uint32_t> propagated;
    auto propagate = [&](uint32_t child_h) {
      if (!propagated.insert(child_h).second) return;
      const std::string& child_id = scene.edge_node_id(child_h);
      if (scene.has_node(child_id)) {
        Node& child = scene.node_ref(scene.index_of(child_id));
        child.zone_row = row_idx;
        child.zone_col = col_idx;
      }
    };
    for (const EdgeRec& e : edges)
      if (e.src == zone_h && e.type == EdgeType::CONTAINS) propagate(e.dst);
    for (const EdgeRec& e : edges)
      if (e.dst == zone_h && e.type == EdgeType::IS_INSIDE_ZONE) propagate(e.src);
  }
}

}  // namespace arc
