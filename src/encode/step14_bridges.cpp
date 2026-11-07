// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 14: bridge edges (compute_bridge_edges).
//
// Determinism notes:
//   * subobjects = get_nodes_by_type("SubObject") -- lexicographic node-id
//     order. `bridges` are the Line/Rectangle, density==1.0, min(w,h)==1
//     SubObjects, in that order.
//   * For each bridge, `touching` collects neighbours reached by a TOUCHES edge:
//     out-neighbours first, then in-neighbours. Each side lists neighbours in
//     FIRST-APPEARANCE-as-adjacent order (grouped by neighbour, not global edge
//     creation order): one scan of the global edge vector builds the
//     out-neighbour and in-neighbour first-appearance orders, then they are
//     filtered to TOUCHES.
//       - out loop: append target when a TOUCHES edge exists, target != bridge,
//         target not structural.
//       - in loop: append source when a TOUCHES edge exists, source != bridge,
//         source NOT already in touching, source not structural.
//   * Then all i<j pairs of `touching` emit BRIDGED_TO (both directions) with a
//     "bridge_node" string attr = the bridge id.
//   * density is a float, compared with `density == 1.0` exactly.

#include "arc/encode.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace arc {

void compute_bridge_edges(Scene& scene) {
  // subobjects in lexicographic id order (canonical get_nodes_by_type).
  std::vector<size_t> sub_idx = scene.get_nodes_by_type("SubObject");

  std::vector<std::string> bridges;
  for (size_t idx : sub_idx) {
    const Node& s = scene.nodes()[idx];
    // Line or Rectangle shape, density == 1.0 and min(w,h) == 1.
    const std::string shape = s.shape_mask.has_value() ? *s.shape_mask : "";
    if ((shape == "Line" || shape == "Rectangle") && s.density == 1.0 &&
        std::min(s.width, s.height) == 1) {
      bridges.push_back(s.node_id);
    }
  }

  if (bridges.empty()) return;

  // One pass over the edges collects every bridge's out-targets and
  // in-sources (any type, creation order) and its TOUCHES neighbours, by
  // edge-node handle. The BRIDGED_TO edges added below never change a later
  // bridge's list: a neighbour reached only through them has no TOUCHES edge,
  // and an already-seen one keeps its place.
  const size_t nh = scene.edge_node_count();
  std::vector<int> bridge_of(nh, -1);
  std::vector<uint32_t> bridge_h(bridges.size(), Scene::kNoEdgeNode);
  for (size_t b = 0; b < bridges.size(); ++b) {
    bridge_h[b] = scene.find_edge_node(bridges[b]);
    if (bridge_h[b] != Scene::kNoEdgeNode) bridge_of[bridge_h[b]] = static_cast<int>(b);
  }
  struct Adj {
    std::vector<uint32_t> out, in, touch_out, touch_in;
  };
  std::vector<Adj> adj(bridges.size());
  for (const EdgeRec& e : scene.edge_recs()) {
    const int bo = bridge_of[e.src], bi = bridge_of[e.dst];
    if (bo >= 0) {
      adj[bo].out.push_back(e.dst);
      if (e.type == EdgeType::TOUCHES) adj[bo].touch_out.push_back(e.dst);
    }
    if (bi >= 0) {
      adj[bi].in.push_back(e.src);
      if (e.type == EdgeType::TOUCHES) adj[bi].touch_in.push_back(e.src);
    }
  }

  // Stamps per handle, one fresh value per list, instead of per-bridge sets.
  std::vector<int> seen(nh, 0), touch(nh, 0), taken(nh, 0);
  int stamp = 0;
  auto structural = [&scene](uint32_t h) {
    const std::string& id = scene.edge_node_id(h);
    return scene.has_node(id) && scene.nodes()[scene.index_of(id)].is_structural;
  };
  for (size_t b = 0; b < bridges.size(); ++b) {
    if (bridge_h[b] == Scene::kNoEdgeNode) continue;  // no edges: touches nothing
    const uint32_t self = bridge_h[b];
    const Adj& ad = adj[b];
    std::vector<uint32_t> touching;
    const int take = ++stamp;
    // out-targets in first-appearance order, kept when a TOUCHES edge exists.
    const int so = ++stamp;
    for (uint32_t t : ad.touch_out) touch[t] = so;
    for (uint32_t t : ad.out) {
      if (seen[t] == so) continue;
      seen[t] = so;
      if (touch[t] != so || t == self || structural(t)) continue;
      touching.push_back(t);
      taken[t] = take;
    }
    // then the in-sources not already taken.
    const int si = ++stamp;
    for (uint32_t t : ad.touch_in) touch[t] = si;
    for (uint32_t t : ad.in) {
      if (seen[t] == si) continue;
      seen[t] = si;
      if (touch[t] != si || t == self || taken[t] == take || structural(t)) continue;
      touching.push_back(t);
      taken[t] = take;
    }
    if (touching.size() < 2) continue;

    EdgeAttrs at;
    at.str_attrs.emplace_back("bridge_node", bridges[b]);
    const uint32_t rec = scene.edge_attr_record(std::move(at));
    for (size_t i = 0; i < touching.size(); ++i) {
      for (size_t j = i + 1; j < touching.size(); ++j) {
        scene.add_edge(touching[i], touching[j], EdgeType::BRIDGED_TO, rec);
        scene.add_edge(touching[j], touching[i], EdgeType::BRIDGED_TO, rec);
      }
    }
  }
}

}  // namespace arc
