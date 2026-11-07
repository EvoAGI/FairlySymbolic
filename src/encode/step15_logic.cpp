// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 15: logic edges (compute_logic_edges).
//
// Determinism notes:
//   * subobjects = get_nodes_by_type("SubObject") -- lexicographic node-id
//     order. Group member order == that lexicographic iteration order.
//   * color_groups keyed by color (skip missing), shape_groups by shape_hash
//     (skip missing), matrix_shape_groups by the normalized matrix's (h,w).
//     A SubObject always has a normalized matrix here, so it lands in exactly
//     one matrix-shape group.
//   * SAME_COLOR then SAME_SHAPE: for each group, emit both directions of every
//     i<j pair (emit_undirected). Groups are visited in first-appearance order
//     of their key in the lexicographic subobject scan.
//   * IS_SYMMETRIC_TO: per matrix-shape group, all i<j pairs, gated by
//     are_symmetric: matrix_a equals b with its columns reversed, OR its rows
//     reversed, OR rotated by 180 degrees. Same-shape is guaranteed by the
//     group; the checks short circuit in that order.
//   * Group KEY order is insertion order (first-seen), reproduced with an
//     ordered-keys + index-map structure (not std::map, which would sort).

#include "arc/encode.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace arc {

namespace {

// Ordered group table: keys in first-appearance order, each mapping to a member
// index list. K is the key type (int color, string shape_hash, or (h,w) shape).
template <typename K>
struct OrderedGroups {
  std::vector<K> keys;
  std::map<K, size_t> idx;                 // key -> position in keys/values
  std::vector<std::vector<size_t>> values;  // parallel to keys; member node indices

  void add(const K& k, size_t member) {
    auto it = idx.find(k);
    if (it == idx.end()) {
      idx[k] = keys.size();
      keys.push_back(k);
      values.emplace_back();
      values.back().push_back(member);
    } else {
      values[it->second].push_back(member);
    }
  }
};

// are_symmetric: the matrices already share a shape (grouped). Returns true if
// a equals b with its columns reversed, OR its rows reversed, OR both (180
// rotation). Empty matrices of equal shape (h>=1, w==0 -> 0 cells) compare
// equal.
struct NormMatrix {
  int h = 0, w = 0;
  std::vector<uint8_t> m;
};

bool are_symmetric(const NormMatrix& na, const NormMatrix& nb) {
  const int ah = na.h, aw = na.w, bh = nb.h, bw = nb.w;
  const std::vector<uint8_t>& ma = na.m;
  const std::vector<uint8_t>& mb = nb.m;
  if (ah != bh || aw != bw) return false;  // grouped by shape, but be safe
  const int h = ah, w = aw;
  auto A = [&](int y, int x) -> uint8_t { return ma[static_cast<size_t>(y) * w + x]; };
  auto B = [&](int y, int x) -> uint8_t { return mb[static_cast<size_t>(y) * w + x]; };

  // Columns reversed: B(y, w-1-x).
  bool eq = true;
  for (int y = 0; y < h && eq; ++y)
    for (int x = 0; x < w; ++x)
      if (A(y, x) != B(y, w - 1 - x)) { eq = false; break; }
  if (eq) return true;

  // Rows reversed: B(h-1-y, x).
  eq = true;
  for (int y = 0; y < h && eq; ++y)
    for (int x = 0; x < w; ++x)
      if (A(y, x) != B(h - 1 - y, x)) { eq = false; break; }
  if (eq) return true;

  // 180 rotation: B(h-1-y, w-1-x).
  eq = true;
  for (int y = 0; y < h && eq; ++y)
    for (int x = 0; x < w; ++x)
      if (A(y, x) != B(h - 1 - y, w - 1 - x)) { eq = false; break; }
  return eq;
}

// Every i < j pair of the group, both directions: a clique. Each node is in
// one group of the type, so every key is new.
void emit_undirected(Scene& scene, const std::vector<size_t>& group, EdgeType edge_type) {
  if (group.size() < 2) return;
  std::vector<uint32_t> h(group.size());
  for (size_t i = 0; i < group.size(); ++i) h[i] = scene.edge_node_at(group[i]);
  scene.add_clique(edge_type, std::move(h));
}

}  // namespace

void compute_logic_edges(Scene& scene) {
  std::vector<size_t> sub_idx = scene.get_nodes_by_type("SubObject");

  OrderedGroups<int> color_groups;
  OrderedGroups<std::string> shape_groups;
  OrderedGroups<std::pair<int, int>> matrix_shape_groups;

  for (size_t idx : sub_idx) {
    const Node& n = scene.nodes()[idx];
    if (n.color.has_value()) color_groups.add(*n.color, idx);
    if (n.shape_hash.has_value()) shape_groups.add(*n.shape_hash, idx);
    // The normalized matrix shape: an empty pixel_list yields (1,0); a
    // non-empty one yields (h,w) from the tight bbox.
    int mh = 0, mw = 0;
    (void)n.normalized_matrix(mh, mw);
    matrix_shape_groups.add({mh, mw}, idx);
  }

  for (const auto& group : color_groups.values) emit_undirected(scene, group, EdgeType::SAME_COLOR);
  for (const auto& group : shape_groups.values) emit_undirected(scene, group, EdgeType::SAME_SHAPE);

  for (const auto& group : matrix_shape_groups.values) {
    // Each member's matrix, computed once.
    std::vector<NormMatrix> mats(group.size());
    std::vector<uint32_t> h(group.size(), Scene::kNoEdgeNode);  // interned when first used
    for (size_t i = 0; i < group.size(); ++i)
      mats[i].m = scene.nodes()[group[i]].normalized_matrix(mats[i].h, mats[i].w);
    auto handle = [&](size_t i) {
      if (h[i] == Scene::kNoEdgeNode) h[i] = scene.edge_node_at(group[i]);
      return h[i];
    };
    // All members alike: every pair is symmetric iff one is with itself, so
    // the group is a clique or has no edge.
    bool alike = true;
    for (size_t i = 1; i < group.size() && alike; ++i) alike = mats[i].m == mats[0].m;
    if (alike) {
      if (group.size() >= 2 && are_symmetric(mats[0], mats[0])) {
        for (size_t i = 0; i < group.size(); ++i) handle(i);
        scene.add_clique(EdgeType::IS_SYMMETRIC_TO, h);
      }
      continue;
    }
    for (size_t i = 0; i < group.size(); ++i) {
      for (size_t j = i + 1; j < group.size(); ++j) {
        if (are_symmetric(mats[i], mats[j])) {
          const uint32_t ha = handle(i), hb = handle(j);
          scene.add_new_edge(ha, hb, EdgeType::IS_SYMMETRIC_TO);  // one matrix-shape group per node
          scene.add_new_edge(hb, ha, EdgeType::IS_SYMMETRIC_TO);
        }
      }
    }
  }
}

}  // namespace arc
