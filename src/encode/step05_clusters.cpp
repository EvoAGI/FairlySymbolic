// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 5: cluster detection (detect_clusters) and step 5c
// (ClusterObject.colors = sorted color-histogram keys).
//
// Groups pixel-adjacent SubObjects into ClusterObjects; only components with
// more than one distinct color become clusters.
//
// Determinism notes:
//   * adjacency is built over the post-3d subobject list, upper-triangular
//     (j > i). A pair is adjacent iff their bboxes are within 1 AND some pixel
//     of A has a spatial_touch_connectivity-neighbor in B.
//   * The component BFS scans each node's neighbours in hash-set iteration
//     order, NOT ascending index order. Each adjacency set is recorded in
//     insertion order (all lower adjacent indices ascending, then all higher
//     adjacent indices ascending -- the order the upper-triangular pair loop
//     adds them) and cpython_set_order_int turns that into the iteration order
//     (see cpyset.hpp). The order is visible in the output: e.g. {2,5,8}
//     inserted 2,5,8 iterates [8,2,5], which decides the color_histogram tie
//     order of two equal-mass colors discovered via that node.
//   * cluster_decompose_two_color has no effect: every multi-color component
//     becomes one cluster, never split into two-color parts.
//   * color_histogram sums mass per color in component (BFS-discovery) order.
//   * ClusterObject creation calls check_boundary_touches + compute_shape_hash +
//     classify_shape (the cluster variant), then step 5c sets colors=sorted(keys).

#include "arc/cpyset.hpp"
#include "arc/encode.hpp"

#include <algorithm>
#include <deque>
#include <set>
#include <string>
#include <vector>

namespace arc {

std::vector<Node> detect_clusters(Scene& scene, const std::vector<Node>& subobjects,
                                  const GraphConfig& cfg) {
  std::vector<Node> clusters;
  if (subobjects.empty()) return clusters;

  const int n = static_cast<int>(subobjects.size());

  // Pixel sets per subobject for the neighbor membership test: an occupancy
  // bitmap over the pixels' extent. And the bboxes as a compact array.
  struct PixelSet {
    int y0 = 0, x0 = 0, h = 0, w = 0;
    std::vector<uint8_t> on;
    bool count(const Coord& c) const {
      const int y = c.first - y0, x = c.second - x0;
      return y >= 0 && y < h && x >= 0 && x < w && on[static_cast<size_t>(y) * w + x];
    }
  };
  std::vector<PixelSet> pixel_sets(n);
  for (int i = 0; i < n; ++i) {
    const std::vector<Coord>& px = subobjects[i].pixel_list;
    if (px.empty()) continue;
    PixelSet& ps = pixel_sets[i];
    int y1 = px[0].first, x1 = px[0].second;
    ps.y0 = y1;
    ps.x0 = x1;
    for (const auto& p : px) {
      ps.y0 = std::min(ps.y0, p.first); y1 = std::max(y1, p.first);
      ps.x0 = std::min(ps.x0, p.second); x1 = std::max(x1, p.second);
    }
    ps.h = y1 - ps.y0 + 1;
    ps.w = x1 - ps.x0 + 1;
    ps.on.assign(static_cast<size_t>(ps.h) * ps.w, 0);
    for (const auto& p : px) ps.on[static_cast<size_t>(p.first - ps.y0) * ps.w + (p.second - ps.x0)] = 1;
  }
  struct BBox {
    bool has;
    int y0, x0, y1, x1;
  };
  std::vector<BBox> bb(n);
  for (int i = 0; i < n; ++i) {
    const Node& o = subobjects[i];
    bb[i] = BBox{o.has_bbox, o.bbox_ymin, o.bbox_xmin, o.bbox_ymax, o.bbox_xmax};
  }
  // Bounding boxes within 1 cell of each other, on the compact boxes.
  auto adjacent_boxes = [&bb](int i, int j) {
    const BBox& a = bb[i];
    const BBox& b = bb[j];
    if (!a.has || !b.has) return false;
    return !(a.y1 + 1 < b.y0 || a.y0 - 1 > b.y1 || a.x1 + 1 < b.x0 || a.x0 - 1 > b.x1);
  };

  // adjacency[i] records the adjacency set's INSERTION order. The
  // upper-triangular pair loop adds j to adjacency[i] and i to adjacency[j], so
  // for a fixed index k the adds arrive as: every adjacent lower index in
  // ascending order (pairs (i,k), i<k, during outer iterations i=0..k-1), THEN
  // every adjacent higher index in ascending order (pairs (k,j), j>k, during
  // outer iteration i=k). cpython_set_order_int then maps this insertion order
  // to the hash-set iteration order used by the BFS neighbour scan.
  std::vector<std::vector<int>> adjacency(n);
  auto adj_add = [&](int a, int b) {
    // Set semantics: the pair loop visits each pair once so a duplicate cannot
    // occur, but guard anyway.
    for (int v : adjacency[a])
      if (v == b) return;
    adjacency[a].push_back(b);
  };

  const int touch_conn = cfg.spatial_touch_connectivity;

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (!adjacent_boxes(i, j)) continue;
      const PixelSet& pixels_b = pixel_sets[j];

      bool is_adjacent = false;
      // Stop at the first pixel of A with a neighbour in B. The result
      // (adjacent or not) does not depend on the scan order of A's pixels.
      for (const auto& p : subobjects[i].pixel_list) {
        int py = p.first, px = p.second;
        if (touch_conn == 4) {
          static const int d4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
          for (const auto& d : d4) {
            if (pixels_b.count(Coord(py + d[0], px + d[1]))) {
              is_adjacent = true;
              break;
            }
          }
        } else {
          for (int dy = -1; dy <= 1 && !is_adjacent; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dy == 0 && dx == 0) continue;
              if (pixels_b.count(Coord(py + dy, px + dx))) {
                is_adjacent = true;
                break;
              }
            }
        }
        if (is_adjacent) break;
      }

      if (is_adjacent) {
        adj_add(i, j);
        adj_add(j, i);
      }
    }
  }

  // Precompute each adjacency set's hash-set iteration order once (the BFS
  // scans it in that order).
  std::vector<std::vector<int>> adjacency_iter(n);
  for (int i = 0; i < n; ++i)
    adjacency_iter[i] = cpython_set_order_int(adjacency[i]);

  std::vector<uint8_t> visited(n, 0);

  for (int start = 0; start < n; ++start) {
    if (visited[start]) continue;

    std::vector<int> component;
    std::deque<int> queue;
    queue.push_back(start);
    visited[start] = 1;

    while (!queue.empty()) {
      int idx = queue.front();
      queue.pop_front();
      component.push_back(idx);
      for (int neighbor : adjacency_iter[idx]) {  // hash-set iteration order
        if (!visited[neighbor]) {
          visited[neighbor] = 1;
          queue.push_back(neighbor);
        }
      }
    }

    // distinct colors in component.
    std::set<int> colors_in_cluster;
    for (int idx : component)
      if (subobjects[idx].color.has_value()) colors_in_cluster.insert(*subobjects[idx].color);

    if (colors_in_cluster.size() <= 1) continue;

    // cluster_decompose_two_color is ignored: the whole component is one cluster.
    std::vector<Coord> all_pixels;
    // color_histogram: color -> summed mass, insertion order = component scan.
    std::vector<std::pair<int, int>> hist;
    auto hist_add = [&](int c, int m) {
      for (auto& kv : hist)
        if (kv.first == c) {
          kv.second += m;
          return;
        }
      hist.emplace_back(c, m);
    };
    for (int idx : component) {
      const Node& obj = subobjects[idx];
      all_pixels.insert(all_pixels.end(), obj.pixel_list.begin(), obj.pixel_list.end());
      if (obj.color.has_value()) hist_add(*obj.color, obj.mass);
    }

    // subobject_ids: member node ids in component BFS order (step 8 reads them).
    std::vector<std::string> subobj_ids;
    for (int idx : component) subobj_ids.push_back(subobjects[idx].node_id);

    scene.node_counter += 1;
    Node cluster;
    cluster.node_id = "cluster_" + std::to_string(scene.node_counter);
    cluster.node_type = "ClusterObject";
    cluster.pixel_list = all_pixels;
    cluster.color_histogram = hist;
    cluster.subobject_ids = subobj_ids;
    cluster.post_init();
    cluster.check_boundary_touches(scene.width(), scene.height());
    cluster.compute_shape_hash();
    cluster.classify_shape_cluster();

    // Step 5c: colors = sorted(color_histogram keys).
    std::vector<int> keys;
    for (const auto& kv : hist) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    cluster.colors = keys;
    cluster.has_colors = true;

    clusters.push_back(std::move(cluster));
  }

  return clusters;
}

}  // namespace arc
