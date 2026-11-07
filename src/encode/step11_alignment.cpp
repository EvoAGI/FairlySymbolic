// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 11: alignment edges (compute_alignment_edges).
//
// Determinism notes:
//   * nodes_with_bbox: the physical-typed, non-structural nodes with a bbox, in
//     nodes_ vector insertion order.
//   * O(n^2) i<j pairs over nodes_with_bbox. Each predicate emits BOTH
//     directions (a->b then b->a); the edge key is (src,dst,type), so a
//     re-added pair updates in place.
//   * Row/Column overlap use bbox integer comparisons (no float).
//   * COLLINEAR_TO carries an "axis" string attr ("horizontal" / "vertical").
//     A pair can be BOTH horizontal- and vertical-collinear; both edges share
//     the (a,b,COLLINEAR_TO) key, so the vertical one OVERWRITES the horizontal
//     one. Such a pair is emitted once, axis vertical, in the place the
//     horizontal edge would take.
//   * Centroid tests use alignment_tolerance (a float) with strict '<'.
//   * round(cy) rounds half to even (banker's rounding): std::nearbyint under
//     the default FE_TONEAREST rounding mode.

#include "arc/encode.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace arc {

namespace {

bool is_alignment_physical(const std::string& t) {
  return t == "SubObject" || t == "ClusterObject" || t == "CavityObject" ||
         t == "NoiseObject" || t == "PatternObject";
}

}  // namespace

void compute_alignment_edges(Scene& scene, const GraphConfig& cfg) {
  const std::vector<Node>& nodes = scene.nodes();

  // nodes_with_bbox: physical-typed, has bbox, not structural -- insertion order.
  std::vector<const Node*> nb;
  for (const Node& n : nodes) {
    if (!is_alignment_physical(n.node_type)) continue;
    if (!n.has_bbox) continue;
    if (n.is_structural) continue;
    nb.push_back(&n);
  }

  const double tol = cfg.alignment_tolerance;
  const size_t n = nb.size();
  // Edge-node handles, interned in the order the node ids first reach an edge
  // (the handles are internal: any order gives the same edges). Every pair is
  // visited once and no earlier step adds these types: all keys are new.
  std::vector<uint32_t> hs(n, Scene::kNoEdgeNode);
  auto handle = [&](size_t i) {
    if (hs[i] == Scene::kNoEdgeNode) hs[i] = scene.edge_node_at(static_cast<size_t>(nb[i] - nodes.data()));
    return hs[i];
  };
  // The COLLINEAR_TO attribute records, one per axis, made when first used.
  uint32_t rec_h = 0, rec_v = 0;
  auto axis_rec = [&scene](const char* axis, uint32_t& rec) {
    if (!rec) {
      EdgeAttrs a;
      a.str_attrs.emplace_back("axis", axis);
      rec = scene.edge_attr_record(std::move(a));
    }
    return rec;
  };
  // The pair loop reads compact per-node values; the rounded centroids are
  // computed once per node (round(c) is the same number for every pair).
  struct Box {
    int y0, x0, y1, x1;
    bool has_c;
    double cy, cx, ry, rx;
  };
  std::vector<Box> bx(n);
  for (size_t i = 0; i < n; ++i) {
    const Node& a = *nb[i];
    bx[i] = Box{a.bbox_ymin, a.bbox_xmin, a.bbox_ymax, a.bbox_xmax, a.has_centroid, a.centroid_y,
                a.centroid_x, a.has_centroid ? std::nearbyint(a.centroid_y) : 0.0,
                a.has_centroid ? std::nearbyint(a.centroid_x) : 0.0};
  }
  for (size_t i = 0; i < n; ++i) {
    const Box& a = bx[i];
    for (size_t j = i + 1; j < n; ++j) {
      const Box& b = bx[j];

      // Row overlap: not (ymax_a < ymin_b or ymin_a > ymax_b).
      if (!(a.y1 < b.y0 || a.y0 > b.y1)) {
        scene.add_new_edge(handle(i), handle(j), EdgeType::SHARES_ROW);
        scene.add_new_edge(handle(j), handle(i), EdgeType::SHARES_ROW);
      }

      // Column overlap: not (xmax_a < xmin_b or xmin_a > xmax_b).
      if (!(a.x1 < b.x0 || a.x0 > b.x1)) {
        scene.add_new_edge(handle(i), handle(j), EdgeType::SHARES_COLUMN);
        scene.add_new_edge(handle(j), handle(i), EdgeType::SHARES_COLUMN);
      }

      // Centroid alignment: both nodes must have a centroid.
      if (a.has_c && b.has_c) {
        // Horizontal then vertical: a pair that is both keeps the place of the
        // horizontal edges and the vertical axis, so emit once, axis vertical.
        const bool col_h = std::fabs(a.cy - b.cy) < tol, col_v = std::fabs(a.cx - b.cx) < tol;
        if (col_h || col_v) {
          const uint32_t rec = col_v ? axis_rec("vertical", rec_v) : axis_rec("horizontal", rec_h);
          scene.add_new_edge(handle(i), handle(j), EdgeType::COLLINEAR_TO, rec);
          scene.add_new_edge(handle(j), handle(i), EdgeType::COLLINEAR_TO, rec);
        }

        double dy = std::fabs(a.cy - b.cy);
        double dx = std::fabs(a.cx - b.cx);
        if (dy > tol && dx > tol && std::fabs(dy - dx) < tol) {
          scene.add_new_edge(handle(i), handle(j), EdgeType::DIAGONAL_TO);
          scene.add_new_edge(handle(j), handle(i), EdgeType::DIAGONAL_TO);
        }

        // round(cy_a) == round(cy_b), rounding half to even.
        if (a.ry == b.ry) {
          scene.add_new_edge(handle(i), handle(j), EdgeType::SAME_CENTROID_ROW);
          scene.add_new_edge(handle(j), handle(i), EdgeType::SAME_CENTROID_ROW);
        }
        if (a.rx == b.rx) {
          scene.add_new_edge(handle(i), handle(j), EdgeType::SAME_CENTROID_COLUMN);
          scene.add_new_edge(handle(j), handle(i), EdgeType::SAME_CENTROID_COLUMN);
        }
      }
    }
  }
}

}  // namespace arc
