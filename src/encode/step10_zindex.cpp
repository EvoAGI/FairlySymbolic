// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 10: z_index from containment topology (compute_z_indices).
//
// Determinism notes:
//   * children is built by scanning the edge vector in insertion order and
//     keeping CONTAINS edges; has_parent tracks every CONTAINS target. Edge
//     order only affects the per-parent child order, which the BFS visits FIFO.
//   * roots = nodes with no CONTAINS parent, in node-vector insertion order.
//     Each root's z_index is set to 0 (this OVERWRITES any prior z_index).
//   * BFS with a FIFO queue. On revisit, z_index is raised only when child_z
//     (= parent_z + 1) exceeds the current value; the child is re-enqueued
//     every time.
//   * Every node is either a root or a CONTAINS target, so all are covered: a
//     node with no CONTAINS parent is a root (z=0); a node with a parent is
//     reached via BFS. The default Node.z_index is already 0.

#include "arc/encode.hpp"

#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace arc {

void compute_z_indices(Scene& scene) {
  const std::vector<Node>& nodes = scene.nodes();

  // children: parent -> child handles (CONTAINS), insertion order.
  const size_t nh = scene.edge_node_count();
  std::vector<std::vector<uint32_t>> children(nh);
  std::vector<char> has_parent(nh, 0);
  for (const EdgeRec& e : scene.edge_recs()) {
    if (e.type == EdgeType::CONTAINS) {
      children[e.src].push_back(e.dst);
      has_parent[e.dst] = 1;
    }
  }
  // Node index of a handle, looked up once.
  std::vector<size_t> node_of(nh, static_cast<size_t>(-1));
  auto node_index = [&](uint32_t h) {
    if (node_of[h] == static_cast<size_t>(-1)) node_of[h] = scene.index_of(scene.edge_node_id(h));
    return node_of[h];
  };

  // roots = nodes with no CONTAINS parent, in node insertion order. z_index=0
  // (overwrites default / any earlier value). Queue entries: (node index,
  // handle or kNoEdgeNode when no edge names the node).
  std::deque<std::pair<size_t, uint32_t>> queue;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const uint32_t h = scene.find_edge_node(nodes[i].node_id);
    if (h != Scene::kNoEdgeNode && has_parent[h]) continue;
    scene.node_ref(i).z_index = 0;
    queue.emplace_back(i, h);
  }

  while (!queue.empty()) {
    const auto [idx, h] = queue.front();
    queue.pop_front();
    if (h == Scene::kNoEdgeNode) continue;
    int parent_z = scene.nodes()[idx].z_index;
    for (uint32_t c : children[h]) {
      int child_z = parent_z + 1;
      const size_t ci = node_index(c);
      Node& cnode = scene.node_ref(ci);
      if (child_z > cnode.z_index) cnode.z_index = child_z;
      queue.emplace_back(ci, c);
    }
  }
}

}  // namespace arc
