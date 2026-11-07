// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 17: symmetry detection (detect_symmetries and
// connect_symmetric_objects).
//
// Determinism notes:
//   * Symmetry predicates compare the raw grid: horizontal = every row y equals
//     row h-1-y for y in [0, h/2); vertical = every col x equals col w-1-x for
//     x in [0, w/2); diagonal main = the grid equals its transpose (square
//     only); diagonal anti = the column-reversed grid equals its own transpose.
//     Exact integer equality.
//   * add_implicit_node fires H, then V, then diag-main, then diag-anti (in that
//     order), each advancing the shared node_counter -> implicit_{n} ids.
//     HorizontalAxis {"y": (h-1)/2.0}, VerticalAxis {"x": (w-1)/2.0},
//     DiagonalAxis {"type": "main"|"anti"}. Diagonal axes connect no objects.
//   * connect_symmetric_objects: subobjects = get_nodes_by_type("SubObject") --
//     lexicographic node-id order. Items without a centroid are skipped. For
//     each i<j pair: expected_mirror = 2*axis - mc_a; connect BOTH a and b (once
//     each, tracked in a `connected` set) iff |mc_b-expected|<tol AND
//     |oc_a-oc_b|<tol AND sh_a==sh_b AND sh_a is set. The IS_SYMMETRIC_TO edge
//     set is order-independent (the serializer sorts edges), so lex order is
//     safe.

#include "arc/encode.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace arc {

namespace {

// connect_symmetric_objects: axis_id already added. axis_is_horizontal selects
// (mirror=cy, other=cx) vs (mirror=cx, other=cy).
void connect_symmetric_objects(Scene& scene, const std::string& axis_id,
                               const std::vector<size_t>& sub_idx, bool axis_is_horizontal,
                               double axis_value, double tolerance) {
  struct Item {
    std::string nid;
    double mc;  // mirror coordinate
    double oc;  // other coordinate
    bool has_hash;
    std::string sh;
  };
  std::vector<Item> items;
  for (size_t idx : sub_idx) {
    const Node& n = scene.nodes()[idx];
    if (!n.has_centroid) continue;  // no centroid: skip
    double cy = n.centroid_y, cx = n.centroid_x;
    double mc = axis_is_horizontal ? cy : cx;
    double oc = axis_is_horizontal ? cx : cy;
    items.push_back({n.node_id, mc, oc, n.shape_hash.has_value(),
                     n.shape_hash.has_value() ? *n.shape_hash : std::string()});
  }

  std::map<std::string, bool> connected;
  auto is_connected = [&](const std::string& id) {
    auto it = connected.find(id);
    return it != connected.end() && it->second;
  };

  for (size_t i = 0; i < items.size(); ++i) {
    const Item& a = items[i];
    double expected_mirror = 2.0 * axis_value - a.mc;
    for (size_t j = i + 1; j < items.size(); ++j) {
      const Item& b = items[j];
      if (std::fabs(b.mc - expected_mirror) < tolerance && std::fabs(a.oc - b.oc) < tolerance &&
          a.has_hash && b.has_hash && a.sh == b.sh) {
        // axis_id is a new node and `connected` emits each object once.
        if (!is_connected(a.nid)) {
          scene.add_new_edge(scene.edge_node(axis_id), scene.edge_node(a.nid), EdgeType::IS_SYMMETRIC_TO);
          connected[a.nid] = true;
        }
        if (!is_connected(b.nid)) {
          scene.add_new_edge(scene.edge_node(axis_id), scene.edge_node(b.nid), EdgeType::IS_SYMMETRIC_TO);
          connected[b.nid] = true;
        }
      }
    }
  }
}

}  // namespace

void detect_symmetries(Scene& scene, const Grid& grid, const GraphConfig& cfg) {
  int h = grid.height(), w = grid.width();
  double tolerance = cfg.symmetry_tolerance;

  std::vector<size_t> sub_idx = scene.get_nodes_by_type("SubObject");

  // Horizontal symmetry: row y == row h-1-y for y in [0, h/2).
  bool is_h = true;
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w; ++x)
      if (grid.at(y, x) != grid.at(h - 1 - y, x)) {
        is_h = false;
        break;
      }
    if (!is_h) break;
  }
  if (is_h) {
    double axis_y = static_cast<double>(h - 1) / 2.0;
    std::string axis_id = scene.add_implicit_node_coord("HorizontalAxis", "y", axis_y);
    connect_symmetric_objects(scene, axis_id, sub_idx, /*horizontal=*/true, axis_y, tolerance);
  }

  // Vertical symmetry: col x == col w-1-x for x in [0, w/2).
  bool is_v = true;
  for (int x = 0; x < w / 2; ++x) {
    for (int y = 0; y < h; ++y)
      if (grid.at(y, x) != grid.at(y, w - 1 - x)) {
        is_v = false;
        break;
      }
    if (!is_v) break;
  }
  if (is_v) {
    double axis_x = static_cast<double>(w - 1) / 2.0;
    std::string axis_id = scene.add_implicit_node_coord("VerticalAxis", "x", axis_x);
    connect_symmetric_objects(scene, axis_id, sub_idx, /*horizontal=*/false, axis_x, tolerance);
  }

  // Diagonal symmetry (square grids only).
  if (h == w) {
    // main: the grid equals its transpose, grid[y][x] == grid[x][y] for all y,x.
    bool is_main = true;
    for (int y = 0; y < h && is_main; ++y)
      for (int x = 0; x < w; ++x)
        if (grid.at(y, x) != grid.at(x, y)) {
          is_main = false;
          break;
        }
    if (is_main) scene.add_implicit_node_type("DiagonalAxis", "main");

    // anti: the column-reversed grid f (f[a][b] = grid[a][w-1-b]) equals its
    // transpose, f[y][x] == f[x][y], i.e. grid[y][w-1-x] == grid[x][w-1-y] for
    // all y,x.
    bool is_anti = true;
    for (int y = 0; y < h && is_anti; ++y)
      for (int x = 0; x < w; ++x)
        if (grid.at(y, w - 1 - x) != grid.at(x, w - 1 - y)) {
          is_anti = false;
          break;
        }
    if (is_anti) scene.add_implicit_node_type("DiagonalAxis", "anti");
  }
}

}  // namespace arc
