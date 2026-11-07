// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 12: ray edges (compute_ray_edges).
//
// Determinism notes:
//   * physical_nodes = SubObject/ClusterObject, not structural, in nodes_
//     vector insertion order. Early-out when fewer than 2.
//   * For each source (with a centroid) and each of the 4 cardinal directions
//     (up, down, left, right -- in that order), scan ALL physical_nodes (same
//     order) for the nearest bbox-aligned target. best_dist starts at +inf;
//     the '<' comparison keeps the FIRST-seen minimum on ties (insertion order).
//     The four directions share one scan; the edges go out in direction order.
//   * The distance is a float (cy_a - ymax_b etc., cy/cx are floats). The
//     PROJECTS_RAY_TO edge carries direction (str) then distance (float).
//   * A target with no bbox is skipped. Self is skipped by id.

#include "arc/encode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc {

namespace {

bool is_ray_physical(const std::string& t) {
  return t == "SubObject" || t == "ClusterObject";
}

}  // namespace

void compute_ray_edges(Scene& scene) {
  const std::vector<Node>& nodes = scene.nodes();

  std::vector<const Node*> phys;
  for (const Node& n : nodes) {
    if (!is_ray_physical(n.node_type)) continue;
    if (n.is_structural) continue;
    phys.push_back(&n);
  }

  if (phys.size() < 2) return;

  // directions in fixed order: up, down, left, right.
  static const char* const dir_names[4] = {"up", "down", "left", "right"};

  // The targets' boxes in one compact array; `same` is the first index with
  // the node's id (self is skipped by id).
  struct Box {
    int ymin, xmin, ymax, xmax;
    bool has_bbox;
    int same;
  };
  std::vector<Box> box(phys.size());
  std::unordered_map<std::string, int> first;
  for (size_t i = 0; i < phys.size(); ++i) {
    const Node& b = *phys[i];
    box[i] = Box{b.bbox_ymin, b.bbox_xmin, b.bbox_ymax, b.bbox_xmax, b.has_bbox,
                 first.emplace(b.node_id, static_cast<int>(i)).first->second};
  }

  // A target passes the up / down test only if it covers column
  // floor(cx_a) (its bounds are integers), and the left / right test only if
  // it covers row floor(cy_a): scan just the targets covering that column /
  // row, in phys order, with the same tests. Each direction keeps its
  // first-seen minimum, as over the whole list.
  int ylo = 0, yhi = -1, xlo = 0, xhi = -1;
  bool any = false;
  for (const Box& b : box) {
    if (!b.has_bbox) continue;
    if (!any) { ylo = b.ymin; yhi = b.ymax; xlo = b.xmin; xhi = b.xmax; any = true; }
    ylo = std::min(ylo, b.ymin); yhi = std::max(yhi, b.ymax);
    xlo = std::min(xlo, b.xmin); xhi = std::max(xhi, b.xmax);
  }
  if (!any) return;
  std::vector<std::vector<int>> by_col(xhi - xlo + 1), by_row(yhi - ylo + 1);
  for (size_t j = 0; j < box.size(); ++j) {
    const Box& b = box[j];
    if (!b.has_bbox) continue;
    for (int x = b.xmin; x <= b.xmax; ++x) by_col[x - xlo].push_back(static_cast<int>(j));
    for (int y = b.ymin; y <= b.ymax; ++y) by_row[y - ylo].push_back(static_cast<int>(j));
  }
  static const std::vector<int> none;
  for (size_t i = 0; i < phys.size(); ++i) {
    const Node& a = *phys[i];
    if (!a.has_centroid) continue;
    const double cy_a = a.centroid_y, cx_a = a.centroid_x;
    const int self = box[i].same;
    int best[4] = {-1, -1, -1, -1};
    double best_dist[4];
    for (double& d : best_dist) d = std::numeric_limits<double>::infinity();
    const double fx = std::floor(cx_a), fy = std::floor(cy_a);
    const std::vector<int>& col =
        fx >= xlo && fx <= xhi ? by_col[static_cast<int>(fx) - xlo] : none;
    const std::vector<int>& row =
        fy >= ylo && fy <= yhi ? by_row[static_cast<int>(fy) - ylo] : none;
    for (int j : col) {
      const Box& b = box[j];
      if (b.same == self) continue;
      if (b.xmin <= cx_a && cx_a <= b.xmax) {
        if (b.ymax < cy_a) {  // up
          double dist = cy_a - b.ymax;
          if (dist < best_dist[0]) { best_dist[0] = dist; best[0] = j; }
        }
        if (b.ymin > cy_a) {  // down
          double dist = b.ymin - cy_a;
          if (dist < best_dist[1]) { best_dist[1] = dist; best[1] = j; }
        }
      }
    }
    for (int j : row) {
      const Box& b = box[j];
      if (b.same == self) continue;
      if (b.ymin <= cy_a && cy_a <= b.ymax) {
        if (b.xmax < cx_a) {  // left
          double dist = cx_a - b.xmax;
          if (dist < best_dist[2]) { best_dist[2] = dist; best[2] = j; }
        }
        if (b.xmin > cx_a) {  // right
          double dist = b.xmin - cx_a;
          if (dist < best_dist[3]) { best_dist[3] = dist; best[3] = j; }
        }
      }
    }
    // A target passes one direction's test only (the four exclude one
    // another), so a source's edges have distinct keys, and no other step
    // adds PROJECTS_RAY_TO: append them.
    for (int d = 0; d < 4; ++d)
      if (best[d] >= 0) {
        const uint32_t ha = scene.edge_node_at(static_cast<size_t>(phys[i] - scene.nodes().data()));
        const uint32_t hb = scene.edge_node_at(static_cast<size_t>(phys[best[d]] - scene.nodes().data()));
        EdgeAttrs at;
        at.str_attrs.emplace_back("direction", dir_names[d]);
        at.dbl_attrs.emplace_back("distance", best_dist[d]);
        scene.add_new_edge(ha, hb, EdgeType::PROJECTS_RAY_TO, scene.edge_attr_record(std::move(at)));
      }
  }
}

}  // namespace arc
