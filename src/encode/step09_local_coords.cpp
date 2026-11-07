// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 9: local coordinates + IS_INSIDE_ZONE (assign_local_coordinates).
//
// Determinism notes:
//   * The zone candidate list comes from get_nodes_by_type("ZoneNode"), in
//     lexicographic node-id order, so the smallest-AREA tie-break (strict '<',
//     first wins) is deterministic.
//   * The object loop iterates the node vector in insertion order and includes
//     structural nodes (they get
//     local coords / IS_INSIDE_ZONE / child_nodes just like non-structural ones,
//     but are excluded from serialization). ZoneNode / ImplicitObject are skipped
//     by node_type.
//   * local coord math is exact integer subtraction; local_centroid is the double
//     centroid minus the integer zone ymin/xmin (double - int -> double).

#include "arc/encode.hpp"

#include <limits>
#include <string>
#include <vector>

namespace arc {

void assign_local_coordinates(Scene& scene) {
  // zone_data: (zone node index, bbox) for every ZoneNode that has a bbox.
  // get_nodes_by_type returns lexicographic node-id order.
  std::vector<size_t> zone_idxs = scene.get_nodes_by_type("ZoneNode");
  if (zone_idxs.empty()) return;

  struct ZoneBox {
    size_t idx;
    int ymin, xmin, ymax, xmax;
  };
  std::vector<ZoneBox> zone_data;
  for (size_t zi : zone_idxs) {
    const Node& z = scene.nodes()[zi];
    if (!z.has_bbox) continue;
    zone_data.push_back({zi, z.bbox_ymin, z.bbox_xmin, z.bbox_ymax, z.bbox_xmax});
  }

  // Iterate every node in insertion order.
  const size_t n = scene.nodes().size();
  for (size_t i = 0; i < n; ++i) {
    Node& node = scene.node_ref(i);
    const std::string& ntype = node.node_type;
    if (ntype == "ZoneNode" || ntype == "ImplicitObject") continue;
    if (!node.has_centroid || !node.has_bbox) continue;

    double cy = node.centroid_y;
    double cx = node.centroid_x;

    bool have_best = false;
    double best_area = std::numeric_limits<double>::infinity();
    int best_zone = -1;
    for (size_t zk = 0; zk < zone_data.size(); ++zk) {
      const ZoneBox& zb = zone_data[zk];
      // zymin <= cy <= zymax and zxmin <= cx <= zxmax (int compared to double).
      if (zb.ymin <= cy && cy <= zb.ymax && zb.xmin <= cx && cx <= zb.xmax) {
        double area = static_cast<double>((zb.ymax - zb.ymin + 1)) *
                      static_cast<double>((zb.xmax - zb.xmin + 1));
        if (area < best_area) {  // strict '<' -> first-seen (lex) wins on tie
          best_area = area;
          best_zone = static_cast<int>(zk);
          have_best = true;
        }
      }
    }
    if (!have_best) continue;

    const ZoneBox& zb = zone_data[static_cast<size_t>(best_zone)];
    int zymin = zb.ymin, zxmin = zb.xmin;

    node.local_bbox_ymin = node.bbox_ymin - zymin;
    node.local_bbox_xmin = node.bbox_xmin - zxmin;
    node.local_bbox_ymax = node.bbox_ymax - zymin;
    node.local_bbox_xmax = node.bbox_xmax - zxmin;
    node.has_local_bbox = true;
    node.local_centroid_y = cy - static_cast<double>(zymin);
    node.local_centroid_x = cx - static_cast<double>(zxmin);
    node.has_local_centroid = true;

    const std::string node_id = node.node_id;      // copy before add_edge/refs
    const std::string zone_id = scene.nodes()[zb.idx].node_id;
    scene.add_edge(node_id, zone_id, EdgeType::IS_INSIDE_ZONE);

    // Append to the zone's child_nodes if not already present. node_ref(zb.idx)
    // is valid: add_node was done in steps 1..5, add_edge does not mutate the
    // node vector.
    Node& zone_node = scene.node_ref(zb.idx);
    bool present = false;
    for (const auto& c : zone_node.child_nodes)
      if (c == node_id) {
        present = true;
        break;
      }
    if (!present) zone_node.child_nodes.push_back(node_id);
  }
}

}  // namespace arc
