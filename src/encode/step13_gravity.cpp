// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 13: gravity edges (compute_gravity_edges).
//
// Determinism notes:
//   * physical_nodes = SubObject/ClusterObject, not structural, in nodes_
//     vector insertion order.
//   * occupied: (y,x) -> node_id, built by scanning physical_nodes in order and
//     each node's pixel_list; a later write wins for a shared cell (physical
//     objects are disjoint components, so no overlap in practice).
//   * Per source, for every column x in [xmin..xmax], scan rows below the
//     bottom (bottom_row+1 .. height-1). On the first occupied cell owned by
//     another node, take dist=y-bottom_row-1 and stop scanning the column; a
//     column with no hit uses the grid-floor distance (height-1-bottom_row)
//     with fall_target = GRID_BOTTOM sentinel.
//   * min_fall tracks the smallest dist (first-seen wins on '<' ties). An edge
//     is emitted ONLY when fall_target is a real node (not unset, not the floor
//     sentinel). fall_distance is always an integer and serializes as an
//     integer attribute.

#include "arc/encode.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc {

namespace {

bool is_gravity_physical(const std::string& t) {
  return t == "SubObject" || t == "ClusterObject";
}

}  // namespace

void compute_gravity_edges(Scene& scene) {
  const std::vector<Node>& nodes = scene.nodes();
  const int height = scene.height();

  std::vector<const Node*> phys;
  for (const Node& n : nodes) {
    if (!is_gravity_physical(n.node_type)) continue;
    if (n.is_structural) continue;
    phys.push_back(&n);
  }

  if (phys.empty()) return;

  // occupied: (y,x) -> owner, physical_nodes-order writes (last wins), as a
  // dense array of phys indices over the pixels' extent (-1: free). Owners
  // are compared by id through `same`, the first phys index with the id.
  std::unordered_map<std::string, int> first;
  std::vector<int> same(phys.size());
  int py0 = 0, py1 = -1, px0 = 0, px1 = -1;
  bool any = false;
  for (size_t i = 0; i < phys.size(); ++i) {
    same[i] = first.emplace(phys[i]->node_id, static_cast<int>(i)).first->second;
    for (const auto& p : phys[i]->pixel_list) {
      if (!any) { py0 = py1 = p.first; px0 = px1 = p.second; any = true; }
      py0 = std::min(py0, p.first); py1 = std::max(py1, p.first);
      px0 = std::min(px0, p.second); px1 = std::max(px1, p.second);
    }
  }
  const int OW = px1 - px0 + 1;
  std::vector<int> occupied(any ? static_cast<size_t>(py1 - py0 + 1) * OW : 0, -1);
  for (size_t i = 0; i < phys.size(); ++i)
    for (const auto& p : phys[i]->pixel_list)
      occupied[static_cast<size_t>(p.first - py0) * OW + (p.second - px0)] = static_cast<int>(i);
  auto owner = [&](int y, int x) {
    if (!any || y < py0 || y > py1 || x < px0 || x > px1) return -1;
    return occupied[static_cast<size_t>(y - py0) * OW + (x - px0)];
  };

  const int GRID_BOTTOM = -2;  // "fell to grid floor" (no target node)

  for (size_t ia = 0; ia < phys.size(); ++ia) {
    const Node& a = *phys[ia];
    if (!a.has_bbox) continue;
    int xmin_a = a.bbox_xmin, xmax_a = a.bbox_xmax, ymax_a = a.bbox_ymax;
    int bottom_row = ymax_a;

    double min_fall = std::numeric_limits<double>::infinity();
    bool have_target = false;
    int fall_target = -1;  // phys index or GRID_BOTTOM; valid only when have_target

    for (int x = xmin_a; x <= xmax_a; ++x) {
      bool broke = false;
      for (int y = bottom_row + 1; y < height; ++y) {
        const int o = owner(y, x);
        if (o >= 0 && same[o] != same[ia]) {
          int dist = y - bottom_row - 1;
          if (dist < min_fall) {
            min_fall = dist;
            have_target = true;
            fall_target = o;
          }
          broke = true;
          break;
        }
      }
      if (!broke) {  // no hit in this column: fell to the grid floor
        int dist = height - 1 - bottom_row;
        if (dist < min_fall) {
          min_fall = dist;
          have_target = true;
          fall_target = GRID_BOTTOM;
        }
      }
    }

    if (have_target && fall_target != GRID_BOTTOM) {
      // fall_distance is an integer: emit it as an int attr.
      // One edge per source and no other step adds FALLS_TOWARDS: append.
      const uint32_t ha = scene.edge_node_at(static_cast<size_t>(phys[ia] - scene.nodes().data()));
      const uint32_t hb = scene.edge_node_at(static_cast<size_t>(phys[fall_target] - scene.nodes().data()));
      EdgeAttrs at;
      at.int_attrs.emplace_back("fall_distance", static_cast<int>(min_fall));
      scene.add_new_edge(ha, hb, EdgeType::FALLS_TOWARDS, scene.edge_attr_record(std::move(at)));
    }
  }
}

}  // namespace arc
