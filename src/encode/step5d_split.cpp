// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 5d: split by containment (split_by_containment).
// DEFAULT OFF (config.split_by_containment=false).
//
// For each ordered pair (A, B) with A larger, if some of B's pixels fall inside
// A's bbox and some outside (both parts >= 2 pixels), B is removed and replaced
// by two new SubObjects (inside_A, outside_A).
//
// Determinism notes:
//   * The candidate list is a snapshot of get_nodes_by_type("SubObject") taken
//     at entry (byte-wise lexicographic id order). Nodes created by the split
//     are NOT candidates this pass.
//   * The list is sorted by mass DESCENDING, ties broken by node id
//     (lexicographic): stable_sort over the already-lexicographic order plus a
//     lexicographic tie comparator, so the result does not depend on the
//     snapshot order.
//   * mass_b >= mass_a skips (strict '<' required). Inner loop breaks after the
//     first successful split of the current A ("only split B once").
//   * new ids are "sub_{n}", "_in" first then "_out". Neither part is marked
//     merged (unlike step 3d): they are ordinary SubObjects with shape hash +
//     classification computed.

#include "arc/encode.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace arc {

void split_by_containment(Scene& scene) {
  // Snapshot SubObject (id, node-copy) in lexicographic id order.
  std::vector<size_t> idxs = scene.get_nodes_by_type("SubObject");
  if (idxs.size() < 2) return;

  struct Cand {
    std::string nid;
    int mass;
    bool has_bbox;
    int ymin, xmin, ymax, xmax;
    std::optional<int> color;
    std::vector<Coord> pixels;
  };
  std::vector<Cand> cands;
  cands.reserve(idxs.size());
  for (size_t idx : idxs) {
    const Node& nd = scene.nodes()[idx];
    cands.push_back({nd.node_id, nd.mass, nd.has_bbox, nd.bbox_ymin, nd.bbox_xmin,
                     nd.bbox_ymax, nd.bbox_xmax, nd.color, nd.pixel_list});
  }

  // Sort mass-desc, node-id lexicographic tie-break.
  std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    if (a.mass != b.mass) return a.mass > b.mass;
    return a.nid < b.nid;
  });

  std::vector<std::string> splits_done;
  auto is_done = [&](const std::string& id) {
    for (const auto& d : splits_done)
      if (d == id) return true;
    return false;
  };

  const int n = static_cast<int>(cands.size());
  for (int i = 0; i < n; ++i) {
    const Cand& a = cands[i];
    if (!a.has_bbox) continue;  // A has no bbox
    const int mass_a = a.mass;

    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const Cand& b = cands[j];
      if (is_done(b.nid)) continue;
      if (b.mass >= mass_a) continue;
      if (b.pixels.empty()) continue;

      std::vector<Coord> inside, outside;
      for (const auto& p : b.pixels) {
        if (a.ymin <= p.first && p.first <= a.ymax && a.xmin <= p.second && p.second <= a.xmax)
          inside.push_back(p);
        else
          outside.push_back(p);
      }

      if (!inside.empty() && !outside.empty() && inside.size() >= 2 && outside.size() >= 2) {
        std::optional<int> color_b = b.color;

        scene.remove_node(b.nid);

        // "_in" then "_out".
        const std::pair<const char*, const std::vector<Coord>*> parts[2] = {
            {"_in", &inside}, {"_out", &outside}};
        for (const auto& part : parts) {
          scene.node_counter += 1;
          Node new_sub;
          new_sub.node_id = "sub_" + std::to_string(scene.node_counter);
          new_sub.node_type = "SubObject";
          new_sub.color = color_b;
          new_sub.pixel_list = *part.second;
          new_sub.post_init();
          new_sub.check_boundary_touches(scene.width(), scene.height());
          new_sub.compute_shape_hash();
          new_sub.classify_shape();
          scene.add_node(new_sub);
          (void)part.first;  // suffix is documentation only; ids come from counter
        }

        splits_done.push_back(b.nid);
        break;  // only split B once (exit inner loop for this A)
      }
    }
  }
}

}  // namespace arc
