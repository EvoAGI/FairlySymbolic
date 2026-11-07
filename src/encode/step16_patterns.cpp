// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Encoder step 16: repeating-pattern detection (detect_patterns) and its
// helpers (find_axis_patterns, find_diagonal_patterns, emit_pattern).
//
// Determinism notes:
//   * subobjects = get_nodes_by_type("SubObject") -- lexicographic node-id
//     order. This order drives (a) the (color, shape_hash) group insertion
//     order, (b) each group's member order, (c) the axis-bucket member order
//     (hence the stable secondary-axis sort tie-break), and (d) the diagonal
//     greedy chain order.
//   * groups keyed (color, shape_hash); no shape_hash -> skip. Groups keep
//     first-appearance insertion order, via an ordered-keys table.
//   * Per group (>= min_sz members): HLine axis, then VLine axis, then diagonal
//     over the members NOT yet in used_nodes.
//   * find_axis_patterns: bucket by round(primary, 1) (1 decimal, half to even);
//     per bucket (>= min_sz) stable-sort by secondary, compute spacings, avg,
//     skip avg<1.0, tolerance = max(avg*0.2, 0.5), all |s-avg|<=tol -> regular;
//     skip if any member already used; else emit.
//   * find_diagonal_patterns: sort remaining by centroid y; for each start
//     index build a greedy chain where each next member has dy>0 and |dy-|dx||<1
//     from the LAST chained member; emit if chain >= min_sz and none used.
//   * emit_pattern: pixel_list = concat of constituents' pixel_lists (order is
//     irrelevant -- the serializer sorts coords); counter++ -> "pattern_{n}";
//     post_init (bbox/centroid/dims/mass/density); check_boundary_touches. No
//     shape_hash / shape_mask (never computed -> null).
//   * Edges: for each constituent that still exists, FORMS_REPEATING_PATTERN
//     (pattern -> constituent) then IS_PART_OF (constituent -> pattern).

#include "arc/encode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace arc {

namespace {

// round(x, 1): x correctly rounded to 1 decimal place, ties to even.
// snprintf("%.1f") rounds the exact binary value and strtod reads the text back
// as the nearest double (same technique as step9b's py_round4). The rounded
// value is only used as a bucket key, so an equal-key match is what matters.
double py_round1(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f", x);
  return std::strtod(buf, nullptr);
}

// Ordered (color, shape_hash) group table: keys in first-appearance order, each
// mapping to the member node indices in scan order.
struct GroupTable {
  std::vector<std::pair<int, std::string>> keys;  // (color, shape_hash)
  std::map<std::pair<int, std::string>, size_t> idx;
  std::vector<std::vector<size_t>> values;

  void add(const std::pair<int, std::string>& k, size_t member) {
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

// Ordered axis-bucket table keyed by round(primary,1): first-appearance order.
struct AxisBuckets {
  std::vector<double> keys;
  std::map<double, size_t> idx;  // rounded-primary -> position
  std::vector<std::vector<size_t>> values;

  void add(double k, size_t member) {
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

}  // namespace

void detect_patterns(Scene& scene, const GraphConfig& cfg) {
  const int min_sz = cfg.pattern_min_size;
  std::vector<size_t> sub_idx = scene.get_nodes_by_type("SubObject");
  if (static_cast<int>(sub_idx.size()) < min_sz) return;

  // Build (color, shape_hash) groups; skip members without a shape_hash.
  GroupTable groups;
  for (size_t idx : sub_idx) {
    const Node& n = scene.nodes()[idx];
    if (!n.shape_hash.has_value()) continue;
    int color = n.color.has_value() ? *n.color : 0;  // a SubObject always has a color
    groups.add({color, *n.shape_hash}, idx);
  }

  // used_nodes: node ids consumed by an emitted pattern.
  std::map<std::string, bool> used;
  auto is_used = [&](const std::string& id) {
    auto it = used.find(id);
    return it != used.end() && it->second;
  };

  // Emit a PatternObject over `members` (indices) with the given type, register
  // the pattern node + its edges, and mark constituents used.
  auto emit_pattern = [&](const std::string& ptype, const std::vector<size_t>& members) {
    std::vector<Coord> all_pixels;
    for (size_t m : members) {
      const Node& n = scene.nodes()[m];
      all_pixels.insert(all_pixels.end(), n.pixel_list.begin(), n.pixel_list.end());
    }
    std::vector<std::string> member_ids;
    member_ids.reserve(members.size());
    for (size_t m : members) member_ids.push_back(scene.nodes()[m].node_id);

    scene.node_counter += 1;
    Node p;
    p.node_id = "pattern_" + std::to_string(scene.node_counter);
    p.node_type = "PatternObject";
    p.pixel_list = all_pixels;
    p.pattern_type = ptype;
    p.constituent_ids = member_ids;
    p.post_init();
    p.check_boundary_touches(scene.width(), scene.height());
    scene.add_node(p);

    for (const auto& id : member_ids) used[id] = true;

    // Edges: for each constituent that exists,
    // FORMS_REPEATING_PATTERN then IS_PART_OF. All constituents exist here.
    for (const auto& cid : member_ids) {
      if (!scene.has_node(cid)) continue;
      scene.add_edge(p.node_id, cid, EdgeType::FORMS_REPEATING_PATTERN);
      scene.add_edge(cid, p.node_id, EdgeType::IS_PART_OF);
    }
  };

  // find_axis_patterns for HLine (primary=y, secondary=x) then VLine
  // (primary=x, secondary=y). primary_is_y selects which centroid coord is the
  // bucket axis vs. the sort axis.
  auto find_axis_patterns = [&](const std::vector<size_t>& members, const std::string& axis_name,
                                bool primary_is_y) {
    AxisBuckets buckets;
    for (size_t m : members) {
      const Node& n = scene.nodes()[m];
      double primary = primary_is_y ? n.centroid_y : n.centroid_x;
      buckets.add(py_round1(primary), m);
    }
    for (const auto& line_members_in : buckets.values) {
      if (static_cast<int>(line_members_in.size()) < min_sz) continue;
      // Stable sort by secondary axis: ties keep the bucket insertion order,
      // which is the lexicographic scan order.
      std::vector<size_t> line_members = line_members_in;
      auto secondary = [&](size_t m) {
        const Node& n = scene.nodes()[m];
        return primary_is_y ? n.centroid_x : n.centroid_y;
      };
      std::stable_sort(line_members.begin(), line_members.end(),
                       [&](size_t a, size_t b) { return secondary(a) < secondary(b); });
      std::vector<double> secondary_vals;
      secondary_vals.reserve(line_members.size());
      for (size_t m : line_members) secondary_vals.push_back(secondary(m));
      // spacings between consecutive secondary values.
      std::vector<double> spacings;
      for (size_t i = 0; i + 1 < secondary_vals.size(); ++i)
        spacings.push_back(secondary_vals[i + 1] - secondary_vals[i]);
      if (spacings.empty()) continue;
      double sum = 0.0;
      for (double s : spacings) sum += s;
      double avg = sum / static_cast<double>(spacings.size());
      if (avg < 1.0) continue;
      double tolerance = avg * 0.2;
      if (tolerance < 0.5) tolerance = 0.5;  // max(avg*0.2, 0.5)
      bool regular = true;
      for (double s : spacings)
        if (std::fabs(s - avg) > tolerance) {
          regular = false;
          break;
        }
      if (!regular) continue;
      bool any_used = false;
      for (size_t m : line_members)
        if (is_used(scene.nodes()[m].node_id)) {
          any_used = true;
          break;
        }
      if (any_used) continue;
      emit_pattern(axis_name, line_members);
    }
  };

  // find_diagonal_patterns over `remaining` (indices, not yet used).
  auto find_diagonal_patterns = [&](std::vector<size_t> remaining) {
    // Stable sort by centroid y; ties keep the lexicographic scan order.
    std::stable_sort(remaining.begin(), remaining.end(), [&](size_t a, size_t b) {
      return scene.nodes()[a].centroid_y < scene.nodes()[b].centroid_y;
    });
    // Start indices 0 .. len-3; at least 3 members are needed.
    if (remaining.size() < 3) return;
    for (size_t start_idx = 0; start_idx + 2 < remaining.size(); ++start_idx) {
      std::vector<size_t> diag_group{remaining[start_idx]};
      for (size_t k = start_idx + 1; k < remaining.size(); ++k) {
        const Node& last = scene.nodes()[diag_group.back()];
        const Node& curr = scene.nodes()[remaining[k]];
        double dy = curr.centroid_y - last.centroid_y;
        double dx = std::fabs(curr.centroid_x - last.centroid_x);
        if (dy > 0 && std::fabs(dy - dx) < 1.0) diag_group.push_back(remaining[k]);
      }
      if (static_cast<int>(diag_group.size()) < min_sz) continue;
      bool any_used = false;
      for (size_t m : diag_group)
        if (is_used(scene.nodes()[m].node_id)) {
          any_used = true;
          break;
        }
      if (any_used) continue;
      emit_pattern("DiagLine", diag_group);
    }
  };

  for (const auto& members : groups.values) {
    if (static_cast<int>(members.size()) < min_sz) continue;

    find_axis_patterns(members, "HLine", /*primary_is_y=*/true);
    find_axis_patterns(members, "VLine", /*primary_is_y=*/false);

    // remaining = members not yet used, preserving group member order.
    std::vector<size_t> remaining;
    for (size_t m : members)
      if (!is_used(scene.nodes()[m].node_id)) remaining.push_back(m);
    if (static_cast<int>(remaining.size()) >= min_sz) find_diagonal_patterns(remaining);
  }
}

}  // namespace arc
