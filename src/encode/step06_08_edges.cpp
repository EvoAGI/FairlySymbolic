// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder steps 6 (spatial edges), 7 (noise scoring), 8 (containment edges).
//
// Determinism notes:
//   * Steps 6 & 8 iterate the scene node list in INSERTION order (the nodes_
//     vector, after any step-5d removals), so edges are created in the order
//     of the nested pair loop over that list.
//   * Step 7 averages the SubObject centroids. A floating-point sum depends on
//     the order of its terms, so it is accumulated in lexicographic node-id
//     order; noise_score bits are therefore fully determined by the scene.
//   * Overlap set-intersection only reads |overlap| (order-independent).
//   * median_mass is element len/2 of the sorted masses; max is a first-max
//     scan; both are order-independent given the canonical inputs.

#include "arc/encode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace arc {

namespace {

// physical_types = {SubObject, ClusterObject, CavityObject, NoiseObject, PatternObject}.
bool is_physical_type(const std::string& t) {
  return t == "SubObject" || t == "ClusterObject" || t == "CavityObject" ||
         t == "NoiseObject" || t == "PatternObject";
}

}  // namespace

// ==========================================================================
// Step 6: spatial edges.
// ==========================================================================

void compute_spatial_edges(Scene& scene, const GraphConfig& cfg) {
  // physical_nodes: insertion order, physical type, non-empty pixel_list, not
  // structural. Store indices into scene.nodes().
  std::vector<size_t> physical;
  const std::vector<Node>& nodes = scene.nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Node& d = nodes[i];
    if (!is_physical_type(d.node_type)) continue;
    if (d.pixel_list.empty()) continue;
    if (d.is_structural) continue;
    physical.push_back(i);
  }

  const int n = static_cast<int>(physical.size());
  const int touch_conn = cfg.spatial_touch_connectivity;

  // Each node's pixel set, and the cells next to it (touch_conn neighbours), as
  // bitsets over the box spanned by every physical pixel: a pixel outside it is
  // in no set, so overlap = |A & B| and touching = (next(A) & B) != 0 are plain
  // bitset tests.
  int y0 = 0, x0 = 0, y1 = -1, x1 = -1;
  for (size_t k : physical)
    for (const auto& p : nodes[k].pixel_list) {
      if (y1 < y0) { y0 = y1 = p.first; x0 = x1 = p.second; }
      y0 = std::min(y0, p.first); y1 = std::max(y1, p.first);
      x0 = std::min(x0, p.second); x1 = std::max(x1, p.second);
    }
  const int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const size_t nw = n > 0 ? (static_cast<size_t>(bw) * bh + 63) / 64 : 0;
  std::vector<uint64_t> cells(static_cast<size_t>(n) * nw, 0), next(static_cast<size_t>(n) * nw, 0);
  auto set_bit = [&](std::vector<uint64_t>& v, int k, int y, int x) {
    if (y < y0 || y > y1 || x < x0 || x > x1) return;
    size_t b = static_cast<size_t>(y - y0) * bw + (x - x0);
    v[static_cast<size_t>(k) * nw + b / 64] |= uint64_t{1} << (b % 64);
  };
  for (int k = 0; k < n; ++k)
    for (const auto& p : nodes[physical[k]].pixel_list) {
      set_bit(cells, k, p.first, p.second);
      if (touch_conn == 4) {
        set_bit(next, k, p.first - 1, p.second);
        set_bit(next, k, p.first + 1, p.second);
        set_bit(next, k, p.first, p.second - 1);
        set_bit(next, k, p.first, p.second + 1);
      } else {
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            if (dy || dx) set_bit(next, k, p.first + dy, p.second + dx);
      }
    }

  // The pair loop reads compact boxes and edge-node handles. Each pair is
  // visited once and no earlier step adds these types, so every key is new.
  struct Box {
    bool has;
    int y0, x0, y1, x1;
  };
  std::vector<Box> bb(n);
  std::vector<uint32_t> hs(n);
  for (int k = 0; k < n; ++k) {
    const Node& d = nodes[physical[k]];
    bb[k] = Box{d.has_bbox, d.bbox_ymin, d.bbox_xmin, d.bbox_ymax, d.bbox_xmax};
  }
  for (int k = 0; k < n; ++k) hs[k] = scene.edge_node_at(physical[k]);
  for (int i = 0; i < n; ++i) {
    const Box& a = bb[i];
    const uint64_t* ca = &cells[static_cast<size_t>(i) * nw];
    const uint64_t* na = &next[static_cast<size_t>(i) * nw];

    for (int j = i + 1; j < n; ++j) {
      const Box& b = bb[j];

      // Skip the pair when both have a bbox and the boxes, grown by 1 cell, do
      // not touch.
      if (a.has && b.has &&
          (a.y1 + 1 < b.y0 || a.y0 - 1 > b.y1 || a.x1 + 1 < b.x0 || a.x0 - 1 > b.x1))
        continue;

      const uint64_t* cb = &cells[static_cast<size_t>(j) * nw];
      // overlap = pixels_a & pixels_b.
      int overlap = 0;
      for (size_t w = 0; w < nw; ++w) overlap += __builtin_popcountll(ca[w] & cb[w]);
      if (overlap > 0) {
        EdgeAttrs at;
        at.int_attrs.emplace_back("overlap_size", overlap);
        const uint32_t rec = scene.edge_attr_record(std::move(at));
        scene.add_new_edge(hs[i], hs[j], EdgeType::OVERLAPS, rec);
        scene.add_new_edge(hs[j], hs[i], EdgeType::OVERLAPS, rec);
        continue;
      }

      // touch test: some neighbour of a pixel of a is a pixel of b.
      bool is_touching = false;
      for (size_t w = 0; w < nw && !is_touching; ++w) is_touching = (na[w] & cb[w]) != 0;

      if (is_touching) {
        scene.add_new_edge(hs[i], hs[j], EdgeType::TOUCHES);
        scene.add_new_edge(hs[j], hs[i], EdgeType::TOUCHES);
        if (a.has && b.has) {  // directional edges from relative bbox positions
          if (a.y1 < b.y0) {
            scene.add_new_edge(hs[i], hs[j], EdgeType::IS_IMMEDIATELY_ABOVE);
            scene.add_new_edge(hs[j], hs[i], EdgeType::IS_IMMEDIATELY_BELOW);
          } else if (a.y0 > b.y1) {
            scene.add_new_edge(hs[i], hs[j], EdgeType::IS_IMMEDIATELY_BELOW);
            scene.add_new_edge(hs[j], hs[i], EdgeType::IS_IMMEDIATELY_ABOVE);
          }
          if (a.x1 < b.x0) {
            scene.add_new_edge(hs[i], hs[j], EdgeType::IS_IMMEDIATELY_LEFT_OF);
            scene.add_new_edge(hs[j], hs[i], EdgeType::IS_IMMEDIATELY_RIGHT_OF);
          } else if (a.x0 > b.x1) {
            scene.add_new_edge(hs[i], hs[j], EdgeType::IS_IMMEDIATELY_RIGHT_OF);
            scene.add_new_edge(hs[j], hs[i], EdgeType::IS_IMMEDIATELY_LEFT_OF);
          }
        }
      }
    }
  }
}

// ==========================================================================
// Step 7: noise scoring (score_noise_candidates).
// ==========================================================================

void score_noise_candidates(Scene& scene, const GraphConfig& cfg) {
  // get_nodes_by_type("SubObject") -> lexicographic node-id order (canonical).
  std::vector<size_t> subs = scene.get_nodes_by_type("SubObject");
  if (subs.empty()) return;
  const std::vector<Node>& nodes = scene.nodes();

  // color_counts, shape_counts, masses, centroids (built in lex order).
  std::vector<std::pair<int, int>> color_counts;  // in first-seen order
  std::vector<std::pair<std::string, int>> shape_counts;
  auto cc_add = [&](int c) {
    for (auto& kv : color_counts)
      if (kv.first == c) {
        kv.second += 1;
        return;
      }
    color_counts.emplace_back(c, 1);
  };
  auto cc_get = [&](int c) -> int {
    for (const auto& kv : color_counts)
      if (kv.first == c) return kv.second;
    return 0;
  };
  auto sh_add = [&](const std::string& s) {
    for (auto& kv : shape_counts)
      if (kv.first == s) {
        kv.second += 1;
        return;
      }
    shape_counts.emplace_back(s, 1);
  };
  auto sh_get = [&](const std::string& s) -> int {
    for (const auto& kv : shape_counts)
      if (kv.first == s) return kv.second;
    return 0;
  };

  std::vector<int> masses;
  std::vector<std::pair<double, double>> centroids;

  for (size_t idx : subs) {
    const Node& d = nodes[idx];
    if (d.color.has_value()) cc_add(*d.color);
    if (d.shape_hash.has_value()) sh_add(*d.shape_hash);
    masses.push_back(d.mass);  // every SubObject has a mass
    if (d.has_centroid) centroids.emplace_back(d.centroid_y, d.centroid_x);
  }

  // median_mass: element len/2 of the sorted masses, 0 when there are none.
  int median_mass = 0;
  if (!masses.empty()) {
    std::vector<int> sorted_masses = masses;
    std::stable_sort(sorted_masses.begin(), sorted_masses.end());
    median_mass = sorted_masses[sorted_masses.size() / 2];
  }
  (void)median_mass;  // computed but not used in scoring

  // max_mass: the largest mass, 0 when there are none.
  int max_mass = 0;
  if (!masses.empty()) {
    max_mass = masses[0];
    for (int m : masses) max_mass = std::max(max_mass, m);
  }

  double mean_cy = 0.0, mean_cx = 0.0, max_dist = 1.0;
  if (!centroids.empty()) {
    // Mean centroid, summed in lexicographic node-id order.
    double sy = 0.0, sx = 0.0;
    for (const auto& c : centroids) {
      sy += c.first;
      sx += c.second;
    }
    double len = static_cast<double>(centroids.size());
    mean_cy = sy / len;
    mean_cx = sx / len;
    if (centroids.size() > 1) {
      // max_dist = max(sqrt((cy-mean_cy)^2 + (cx-mean_cx)^2)).
      bool first = true;
      double best = 0.0;
      for (const auto& c : centroids) {
        double dy = c.first - mean_cy;
        double dx = c.second - mean_cx;
        double dist = std::sqrt(dy * dy + dx * dx);
        if (first || dist > best) {
          best = dist;
          first = false;
        }
      }
      max_dist = best;
    } else {
      max_dist = 1.0;
    }
    max_dist = std::max(max_dist, 1.0);
  } else {
    mean_cy = mean_cx = 0.0;
    max_dist = 1.0;
  }

  // Per-node scoring loop. Order of this loop is irrelevant (each score is
  // independent); iterate lex for tidiness.
  const std::unordered_map<std::string, int> touch_deg = scene.edge_degrees(EdgeType::TOUCHES);
  for (size_t idx : subs) {
    Node& d = scene.node_ref(idx);
    double score = 0.0;

    // Signal 1: isolation (no TOUCHES edges, either direction).
    if (!touch_deg.count(d.node_id)) score += cfg.noise_weight_isolation;

    // Signal 2: singleton color.
    if (d.color.has_value() && cc_get(*d.color) == 1) score += cfg.noise_weight_singleton_color;

    // Signal 3: size outlier.
    int mass = d.mass;
    if (mass <= 2 && max_mass >= 5) {
      score += cfg.noise_weight_size_outlier;
    } else if (max_mass <= 3 && static_cast<int>(masses.size()) >= 3) {
      // Uniform-mass fallback: mean_mass = sum(masses)/len(masses).
      double sum_mass = 0.0;
      for (int m : masses) sum_mass += static_cast<double>(m);
      double mean_mass = sum_mass / static_cast<double>(masses.size());
      if (mass > 0 && static_cast<double>(mass) < mean_mass) score += cfg.noise_weight_size_outlier;
    }

    // Signal 4: shape outlier.
    if (d.shape_hash.has_value() && sh_get(*d.shape_hash) == 1)
      score += cfg.noise_weight_shape_outlier;

    // Signal 5: position outlier.
    if (d.has_centroid && centroids.size() > 1) {
      double dy = d.centroid_y - mean_cy;
      double dx = d.centroid_x - mean_cx;
      double dist = std::sqrt(dy * dy + dx * dx);
      if (dist / max_dist > 0.8) score += cfg.noise_weight_position_outlier;
    }

    d.noise_score = std::min(score, 1.0);
  }
}

// ==========================================================================
// Step 8: containment edges (compute_containment_edges).
// ==========================================================================

void compute_containment_edges(Scene& scene) {
  // node_ids = insertion order, non-structural.
  const std::vector<Node>& nodes = scene.nodes();
  std::vector<size_t> node_ids;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (!nodes[i].is_structural) node_ids.push_back(i);

  // The ordered pair loop over the nodes with a bbox, read from a compact array.
  struct Box {
    size_t idx;
    int y0, x0, y1, x1;
  };
  std::vector<Box> bb;
  for (size_t i : node_ids)
    if (nodes[i].has_bbox)
      bb.push_back(Box{i, nodes[i].bbox_ymin, nodes[i].bbox_xmin, nodes[i].bbox_ymax, nodes[i].bbox_xmax});
  for (const Box& a : bb) {
    for (const Box& b : bb) {
      if (a.idx == b.idx) continue;
      if (b.y0 >= a.y0 && b.y1 <= a.y1 && b.x0 >= a.x0 && b.x1 <= a.x1 &&
          (b.y0 > a.y0 || b.y1 < a.y1 || b.x0 > a.x0 || b.x1 < a.x1)) {
        scene.add_edge(nodes[a.idx].node_id, nodes[b.idx].node_id, EdgeType::CONTAINS);
        scene.add_edge(nodes[b.idx].node_id, nodes[a.idx].node_id, EdgeType::IS_PART_OF);
      }
    }
  }

  // Cluster -> subobject relationships (get_nodes_by_type -> lex order).
  std::vector<size_t> clusters = scene.get_nodes_by_type("ClusterObject");
  for (size_t ci : clusters) {
    const Node& cluster = nodes[ci];
    std::string cluster_id = cluster.node_id;
    for (const std::string& subobj_id : cluster.subobject_ids) {
      if (scene.has_node(subobj_id)) {
        scene.add_edge(cluster_id, subobj_id, EdgeType::CONTAINS);
        scene.add_edge(subobj_id, cluster_id, EdgeType::IS_PART_OF);
      }
    }
  }

  // Cavity -> enclosing object (get_nodes_by_type -> lex order).
  std::vector<size_t> cavities = scene.get_nodes_by_type("CavityObject");
  for (size_t vi : cavities) {
    const Node& cavity = nodes[vi];
    if (cavity.enclosing_object_id.has_value()) {
      const std::string& enclosing_id = *cavity.enclosing_object_id;
      if (!enclosing_id.empty() && scene.has_node(enclosing_id)) {
        scene.add_edge(enclosing_id, cavity.node_id, EdgeType::CONTAINS);
      }
    }
  }
}

}  // namespace arc
