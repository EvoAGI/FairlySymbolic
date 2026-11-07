// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 3d: merge objects by containment (merge_objects_by_containment).
//
// Determinism notes:
//   * color_counts is an insertion-ordered map keyed in SubObject-list order.
//     majority_color: the color with the most mass; on a tie, the first
//     inserted.
//   * container = the FIRST object of non-majority color with strictly greater
//     mass than the running best (strict '>' -> first-max-wins).
//   * pixel_buckets is keyed by (color, "inside"/"outside") in first-seen order
//     while scanning subobjects (skipping the container) then each pixel_list.
//     No hash-set order drives the output.

#include "arc/encode.hpp"

#include <string>
#include <vector>

namespace arc {

std::vector<Node> merge_objects_by_containment(Scene& scene, std::vector<Node> subobjects,
                                               const Grid& grid) {
  (void)grid;  // unused; kept for symmetry with the other steps
  if (subobjects.size() < 3) return subobjects;

  // color_counts: color -> total mass, insertion-ordered by subobject order.
  std::vector<int> cc_keys;
  std::vector<int> cc_val;  // parallel
  auto cc_index = [&](int c) -> int {
    for (size_t i = 0; i < cc_keys.size(); ++i)
      if (cc_keys[i] == c) return static_cast<int>(i);
    return -1;
  };
  for (const auto& s : subobjects) {
    if (!s.color.has_value()) continue;
    int c = *s.color;
    int idx = cc_index(c);
    if (idx < 0) {
      cc_keys.push_back(c);
      cc_val.push_back(s.mass);
    } else {
      cc_val[idx] += s.mass;
    }
  }
  if (cc_keys.size() < 2) return subobjects;

  // majority_color: the color with the most mass, the first one on a tie.
  int majority_color = cc_keys[0];
  int best_mass = cc_val[0];
  for (size_t i = 1; i < cc_keys.size(); ++i) {
    if (cc_val[i] > best_mass) {
      best_mass = cc_val[i];
      majority_color = cc_keys[i];
    }
  }

  // container = largest-mass object whose color != majority; strict '>'.
  int container_idx = -1;
  for (size_t i = 0; i < subobjects.size(); ++i) {
    if (subobjects[i].color.has_value() && *subobjects[i].color != majority_color) {
      if (container_idx < 0 || subobjects[i].mass > subobjects[container_idx].mass)
        container_idx = static_cast<int>(i);
    }
  }
  if (container_idx < 0) return subobjects;
  if (!subobjects[container_idx].has_bbox) return subobjects;  // container has no bbox

  int cymin = subobjects[container_idx].bbox_ymin;
  int cxmin = subobjects[container_idx].bbox_xmin;
  int cymax = subobjects[container_idx].bbox_ymax;
  int cxmax = subobjects[container_idx].bbox_xmax;

  // pixel_buckets keyed by (color, inside?) in first-seen order.
  struct Bucket {
    int color;
    bool inside;
    std::vector<Coord> pixels;
  };
  std::vector<Bucket> buckets;
  auto bucket_for = [&](int color, bool inside) -> Bucket& {
    for (auto& b : buckets)
      if (b.color == color && b.inside == inside) return b;
    buckets.push_back({color, inside, {}});
    return buckets.back();
  };

  std::vector<Node> kept;
  kept.push_back(subobjects[container_idx]);

  for (size_t i = 0; i < subobjects.size(); ++i) {
    if (static_cast<int>(i) == container_idx) continue;
    Node& s = subobjects[i];
    if (!s.color.has_value()) {
      kept.push_back(s);
      continue;
    }
    int color = *s.color;
    for (const auto& p : s.pixel_list) {
      bool inside = (cymin <= p.first && p.first <= cymax && cxmin <= p.second && p.second <= cxmax);
      bucket_for(color, inside).pixels.push_back(p);
    }
  }

  std::vector<Node> result = kept;
  for (const auto& b : buckets) {
    if (b.pixels.empty()) continue;
    scene.node_counter += 1;
    Node merged;
    merged.node_id = "sub_" + std::to_string(scene.node_counter);
    merged.node_type = "SubObject";
    merged.color = b.color;
    merged.pixel_list = b.pixels;
    merged.post_init();
    merged.check_boundary_touches(scene.width(), scene.height());
    merged.compute_shape_hash();
    merged.classify_shape();
    merged.merged = true;
    result.push_back(std::move(merged));
  }

  return result;
}

}  // namespace arc
