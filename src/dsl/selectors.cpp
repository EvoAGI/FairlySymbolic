// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL: selection helpers, select_node_ids, guards_pass. Shared helpers live in
// arg_coerce.hpp / scene_ops.hpp, which also state the ordering rules.

#include "arc/dsl.hpp"
#include "arc/object_attrs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

#include "arg_coerce.hpp"
#include "scene_ops.hpp"

namespace arc {

// ===========================================================================
// Selection helpers: node-data accessors, the subobjects sort, hole_count and
// edge walking.
// ===========================================================================

namespace {

// A SubObject as (index into scene.nodes, node pointer); comparators read the
// node's fields through it.
struct SubRow {
  size_t idx;
  const Node* node;
};

// subobjects(scene): get_nodes_by_type("SubObject") (lexicographic id) then a
// stable sort by (mass, node_id). get_nodes_by_type already returns lexicographic
// id order, so a stable sort on mass alone gives the (mass, id) order.
std::vector<SubRow> subobjects(const Scene& scene) {
  std::vector<size_t> idxs = scene.get_nodes_by_type("SubObject", false);
  std::vector<SubRow> rows;
  rows.reserve(idxs.size());
  for (size_t i : idxs) rows.push_back({i, &scene.nodes()[i]});
  std::stable_sort(rows.begin(), rows.end(), [](const SubRow& a, const SubRow& b) {
    if (a.node->mass != b.node->mass) return a.node->mass < b.node->mass;
    return a.node->node_id < b.node->node_id;
  });
  return rows;
}

std::vector<std::string> ids_of(const std::vector<SubRow>& rows) {
  std::vector<std::string> out;
  out.reserve(rows.size());
  for (const auto& r : rows) out.push_back(r.node->node_id);
  return out;
}

// hole_count(data): 4-connected complement-of-bbox components not touching the
// bbox border.
int hole_count(const Node& n) {
  const std::vector<Coord>& pixels = n.pixel_list;
  if (pixels.empty()) return 0;
  int y0 = pixels[0].first, x0 = pixels[0].second;
  int y1 = y0, x1 = x0;
  for (const auto& p : pixels) {
    y0 = std::min(y0, p.first);
    x0 = std::min(x0, p.second);
    y1 = std::max(y1, p.first);
    x1 = std::max(x1, p.second);
  }
  int h = y1 - y0 + 1;
  int w = x1 - x0 + 1;
  if (static_cast<size_t>(h) * static_cast<size_t>(w) == pixels.size()) return 0;
  std::vector<std::vector<char>> filled(h, std::vector<char>(w, 0));
  for (const auto& p : pixels) filled[p.first - y0][p.second - x0] = 1;
  std::vector<std::vector<char>> seen(h, std::vector<char>(w, 0));
  int holes = 0;
  for (int sy = 0; sy < h; ++sy) {
    for (int sx = 0; sx < w; ++sx) {
      if (filled[sy][sx] || seen[sy][sx]) continue;
      std::vector<Coord> stack;
      stack.push_back({sy, sx});
      seen[sy][sx] = 1;
      bool touches_border = false;
      while (!stack.empty()) {
        Coord cur = stack.back();
        stack.pop_back();
        int y = cur.first, x = cur.second;
        if (y == 0 || y == h - 1 || x == 0 || x == w - 1) touches_border = true;
        static const int dyv[4] = {-1, 1, 0, 0};
        static const int dxv[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
          int ny = y + dyv[d], nx = x + dxv[d];
          if (ny >= 0 && ny < h && nx >= 0 && nx < w && !filled[ny][nx] && !seen[ny][nx]) {
            seen[ny][nx] = 1;
            stack.push_back({ny, nx});
          }
        }
      }
      if (!touches_border) ++holes;
    }
  }
  return holes;
}

// neighbor_offsets(connectivity): 4- or 8-connected (dy,dx). Default 8 for
// anything but 4.

// --- bbox tuple helpers (missing-bbox sentinels) ----------------------------
// A node without a bbox (has_bbox=false) reads the caller's sentinel value.
long long bbox_component(const Node& n, int idx, long long missing) {
  if (!n.has_bbox) return missing;
  switch (idx) {
    case 0: return n.bbox_ymin;
    case 1: return n.bbox_xmin;
    case 2: return n.bbox_ymax;
    default: return n.bbox_xmax;
  }
}

// extremal_by_bbox: id of the sub with min (or max) bbox[idx], tie-break node-id.
std::string extremal_by_bbox(const std::vector<SubRow>& rows, int bbox_idx, bool take_max) {
  // missing = (-1,-1,-1,-1) for max, (1e9,1e9,0,0) for min.
  const SubRow* best = nullptr;
  long long best_key = 0;
  for (const auto& r : rows) {
    long long missing;
    if (take_max) missing = -1;
    else missing = (bbox_idx <= 1) ? 1000000000LL : 0LL;
    long long key = bbox_component(*r.node, bbox_idx, missing);
    if (best == nullptr) {
      best = &r; best_key = key; continue;
    }
    bool better;
    if (take_max) {
      better = (key > best_key) || (key == best_key && r.node->node_id > best->node->node_id);
    } else {
      better = (key < best_key) || (key == best_key && r.node->node_id < best->node->node_id);
    }
    if (better) { best = &r; best_key = key; }
  }
  return best ? best->node->node_id : std::string();
}

// non_dominant_color_candidates: subs whose color != the most-massive color;
// falls back to the full list if every sub shares the dominant color.
std::vector<SubRow> non_dominant_color_candidates(const std::vector<SubRow>& rows) {
  // color_mass in first-seen order, pick max by (mass, -color).
  std::vector<std::pair<int, int>> color_mass;  // (color, total_mass) in first-seen order
  auto bump = [&](int color, int mass) {
    for (auto& kv : color_mass)
      if (kv.first == color) { kv.second += mass; return; }
    color_mass.emplace_back(color, mass);
  };
  for (const auto& r : rows) bump(node_color(*r.node, -1), r.node->mass);
  if (color_mass.empty()) return {};
  // max by (mass, -color): higher mass wins; on mass tie, LOWER color wins.
  int dominant = color_mass[0].first;
  int best_mass = color_mass[0].second;
  for (const auto& kv : color_mass) {
    if (kv.second > best_mass || (kv.second == best_mass && (-kv.first) > (-dominant))) {
      dominant = kv.first;
      best_mass = kv.second;
    }
  }
  std::vector<SubRow> filtered;
  for (const auto& r : rows)
    if (node_color(*r.node, -1) != dominant) filtered.push_back(r);
  if (filtered.empty()) return rows;
  return filtered;
}

int bbox_gap_distance(long long ay0, long long ax0, long long ay1, long long ax1,
                      long long by0, long long bx0, long long by1, long long bx1) {
  long long gap_y = std::max({0LL, by0 - ay1 - 1, ay0 - by1 - 1});
  long long gap_x = std::max({0LL, bx0 - ax1 - 1, ax0 - bx1 - 1});
  return static_cast<int>(gap_x + gap_y);
}

// pair_distance(candidate, reference, mode). Returns +inf when centroids are
// missing.
double pair_distance(const Node& cand, const Node& ref, const std::string& mode) {
  if (mode == "bbox_gap") {
    if (cand.has_bbox && ref.has_bbox) {
      return static_cast<double>(bbox_gap_distance(
          cand.bbox_ymin, cand.bbox_xmin, cand.bbox_ymax, cand.bbox_xmax,
          ref.bbox_ymin, ref.bbox_xmin, ref.bbox_ymax, ref.bbox_xmax));
    }
  }
  if (!cand.has_centroid || !ref.has_centroid) return std::numeric_limits<double>::infinity();
  double dy = cand.centroid_y - ref.centroid_y;
  double dx = cand.centroid_x - ref.centroid_x;
  // Euclidean distance, computed as pow(d2, 0.5).
  return std::pow(dy * dy + dx * dx, 0.5);
}

}  // namespace

// ---------------------------------------------------------------------------
// select_node_ids: the full 34-selector dispatch.
// ---------------------------------------------------------------------------


// Per-selector handlers. select_node_ids tries them in a fixed order: nine
// kinds appear in more than one condition because some handlers cover a group
// and re-branch internally, so the dispatch is deliberately NOT a switch.
// Every handler ends in an unconditional return, so none falls through.

namespace {

struct SelCtx {
  const Scene& scene;
  const std::vector<SubRow>& subs;
  const DslSelector& selector;
  const DslArgs& args;
  int depth;
};

std::vector<std::string> sel_all_subobjects(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    return ids_of(subs);
}

std::vector<std::string> sel_selector_set_op(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::string op = to_lower(strip(arg_str(args, "op", "intersection")));
    std::string left_kind_raw = strip(arg_str(args, "left_selector_kind",
                                                to_string(SelectorKind::ALL_SUBOBJECTS)));
    std::string right_kind_raw = strip(arg_str(args, "right_selector_kind",
                                                 to_string(SelectorKind::ALL_SUBOBJECTS)));
    const DslValue* la = args_get(args, "left_selector_args");
    const DslValue* ra = args_get(args, "right_selector_args");
    DslArgs left_args = (la && la->tag == DslValue::Tag::Dict && la->dict) ? *la->dict : DslArgs{};
    DslArgs right_args = (ra && ra->tag == DslValue::Tag::Dict && ra->dict) ? *ra->dict : DslArgs{};
    SelectorKind left_kind, right_kind;
    if (!selector_from_name(left_kind_raw, &left_kind, &left_args)) left_kind = SelectorKind::ALL_SUBOBJECTS;
    if (!selector_from_name(right_kind_raw, &right_kind, &right_args)) right_kind = SelectorKind::ALL_SUBOBJECTS;
    if (left_kind == SelectorKind::SELECTOR_SET_OP && depth >= 2) {
      left_kind = SelectorKind::ALL_SUBOBJECTS;
      left_args = DslArgs{};
    }
    if (right_kind == SelectorKind::SELECTOR_SET_OP && depth >= 2) {
      right_kind = SelectorKind::ALL_SUBOBJECTS;
      right_args = DslArgs{};
    }
    DslSelector left_sel; left_sel.kind = left_kind; left_sel.args = left_args;
    DslSelector right_sel; right_sel.kind = right_kind; right_sel.args = right_args;
    std::vector<std::string> left_ids = select_node_ids(scene, left_sel, depth + 1);
    std::vector<std::string> right_ids = select_node_ids(scene, right_sel, depth + 1);
    std::set<std::string> right_set(right_ids.begin(), right_ids.end());
    if (op == "difference") {
      std::vector<std::string> out;
      for (const auto& nid : left_ids)
        if (right_set.find(nid) == right_set.end()) out.push_back(nid);
      return out;
    }
    if (op == "union") {
      std::vector<std::string> out = left_ids;
      std::set<std::string> seen(out.begin(), out.end());
      for (const auto& nid : right_ids)
        if (seen.insert(nid).second) out.push_back(nid);
      return out;
    }
    // intersection (default)
    std::vector<std::string> out;
    for (const auto& nid : left_ids)
      if (right_set.find(nid) != right_set.end()) out.push_back(nid);
    return out;
}

std::vector<std::string> sel_by_color(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    int color = arg_int(args, "color", -1);
    std::vector<std::string> out;
    for (const auto& r : subs)
      if (node_color(*r.node, -99) == color) out.push_back(r.node->node_id);
    return out;
}

std::vector<std::string> sel_non_background(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    if (!scene.has_background()) return ids_of(subs);
    int bg = scene.background();
    std::vector<std::string> out;
    for (const auto& r : subs)
      if (node_color(*r.node, -99) != bg) out.push_back(r.node->node_id);
    return out;
}

std::vector<std::string> sel_by_shape(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::string shape = strip(arg_str(args, "shape", ""));
    if (shape.empty()) return {};
    std::vector<std::string> out;
    for (const auto& r : subs) {
      std::string sm = r.node->shape_mask.has_value() ? *r.node->shape_mask : "";
      if (sm == shape) out.push_back(r.node->node_id);
    }
    return out;
}

std::vector<std::string> sel_by_mass(const SelCtx& c) {
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslArgs& args = c.args;

    // op: ge (the old by_mass_at_least), le (at_most), eq (exact).
    std::string op = arg_str(args, "op", "ge");
    int th = arg_int(args, "mass", 1);
    std::vector<std::string> out;
    for (const auto& r : subs) {
      int m = r.node->mass;
      if (op == "ge" ? m >= th : op == "le" ? m <= th : op == "eq" ? m == th : false)
        out.push_back(r.node->node_id);
    }
    return out;
}

std::vector<std::string> sel_by_hole_count(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    int t = arg_int(args, "holes", 1);
    std::vector<std::string> out;
    for (const auto& r : subs) if (hole_count(*r.node) == t) out.push_back(r.node->node_id);
    return out;
}

// BY_ATTR: objects whose step-18 attribute `attr` satisfies `op` (eq, ne, ge,
// le against `value`; max, min over the scene; unique = the value no other
// object has).
std::vector<std::string> sel_by_attr(const SelCtx& c) {
  std::string attr = arg_str(c.args, "attr", "rank_size");
  std::string op = arg_str(c.args, "op", "eq");
  int value = arg_int(c.args, "value", 1);
  std::vector<std::pair<const Node*, int>> vals;
  for (const auto& r : c.subs) {
    int v;
    if (node_attr(*r.node, attr, &v)) vals.emplace_back(r.node, v);
  }
  if (vals.empty()) return {};
  int lo = vals[0].second, hi = vals[0].second;
  std::map<int, int> count;
  for (const auto& p : vals) {
    lo = std::min(lo, p.second);
    hi = std::max(hi, p.second);
    count[p.second]++;
  }
  std::vector<std::string> out;
  for (const auto& p : vals) {
    int v = p.second;
    bool keep = op == "eq" ? v == value : op == "ne" ? v != value : op == "ge" ? v >= value
              : op == "le" ? v <= value : op == "max" ? v == hi : op == "min" ? v == lo
              : op == "unique" ? count[v] == 1 : false;
    if (keep) out.push_back(p.first->node_id);
  }
  return out;
}

std::vector<std::string> sel_by_boundary_touch(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::string side = to_lower(strip(arg_str(args, "side", "any")));
    std::vector<std::string> out;
    if (side == "any") {
      for (const auto& r : subs) {
        const Node& n = *r.node;
        if (n.touches_top || n.touches_bottom || n.touches_left || n.touches_right)
          out.push_back(n.node_id);
      }
      return out;
    }
    for (const auto& r : subs) {
      const Node& n = *r.node;
      bool hit = false;
      if (side == "top") hit = n.touches_top;
      else if (side == "bottom") hit = n.touches_bottom;
      else if (side == "left") hit = n.touches_left;
      else if (side == "right") hit = n.touches_right;
      else return {};  // unknown side -> no key -> empty selection
      if (hit) out.push_back(n.node_id);
    }
    return out;
}

std::vector<std::string> sel_by_color_frequency(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::vector<std::pair<int, int>> counts;  // (color, mass-sum) insertion-ordered
    auto bump = [&](int c, int m) {
      for (auto& kv : counts) if (kv.first == c) { kv.second += m; return; }
      counts.emplace_back(c, m);
    };
    for (const auto& r : subs) bump(node_color(*r.node, -1), r.node->mass);
    if (counts.empty()) return {};
    int target;
    // rank: most (the old most_frequent_color_objects) or least.
    const std::string rank = arg_str(args, "rank", "most");
    if (rank != "most" && rank != "least") return {};
    if (rank == "most") {
      // max by (count, -color)
      target = counts[0].first; int bc = counts[0].second;
      for (const auto& kv : counts)
        if (kv.second > bc || (kv.second == bc && (-kv.first) > (-target))) {
          target = kv.first; bc = kv.second;
        }
    } else {
      // min by (count, color)
      target = counts[0].first; int bc = counts[0].second;
      for (const auto& kv : counts)
        if (kv.second < bc || (kv.second == bc && kv.first < target)) {
          target = kv.first; bc = kv.second;
        }
    }
    std::vector<std::string> out;
    for (const auto& r : subs)
      if (node_color(*r.node, -1) == target) out.push_back(r.node->node_id);
    return out;
}

std::vector<std::string> sel_touching_color(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    // edge_type: the old edge_to_color (TOUCHES when absent, as touching_color).
    int color = arg_int(args, "color", -999);
    std::string edge_type_name = to_upper(strip(arg_str(args, "edge_type", "TOUCHES")));
    std::string direction = to_lower(arg_str(args, "direction", "both"));
    std::vector<std::string> out;
    const EdgeNeighbors nbr(scene, edge_type_name, direction);
    for (const auto& r : subs) {
      bool found = false;
      nbr.for_each(r.node->node_id, [&](const std::string& nb) {
        const Node* nn = find_node(scene, nb);
        int nc = nn ? node_color(*nn, -999) : -999;
        found = nc == color;
        return found;
      });
      if (found) out.push_back(r.node->node_id);
    }
    return out;
}

std::vector<std::string> sel_adjacent_to_color(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    int color = arg_int(args, "color", -999);
    int connectivity = as_int(args_get(args, "connectivity"), 4);
    if (connectivity != 4 && connectivity != 8) connectivity = 4;
    std::vector<Coord> offs = neighbor_offsets(connectivity);
    std::set<Coord> target_pixels;
    for (const auto& r : subs) {
      if (node_color(*r.node, -999) != color) continue;
      for (const auto& px : r.node->pixel_list) target_pixels.insert(px);
    }
    if (target_pixels.empty()) return {};
    std::vector<std::string> out;
    for (const auto& r : subs) {
      const Node& n = *r.node;
      if (n.pixel_list.empty()) continue;
      std::set<Coord> own;
      if (node_color(n, -999) == color) own.insert(n.pixel_list.begin(), n.pixel_list.end());
      bool found = false;
      for (const auto& px : n.pixel_list) {
        for (const auto& off : offs) {
          Coord pos = {px.first + off.first, px.second + off.second};
          if (target_pixels.find(pos) != target_pixels.end() && own.find(pos) == own.end()) {
            found = true; break;
          }
        }
        if (found) break;
      }
      if (found) out.push_back(n.node_id);
    }
    return out;
}

std::vector<std::string> sel_edge_count(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    // op: any (the old has_edge_type), none (has_no_edge_type), ge with count
    // (edge_count_at_least).
    std::string op = arg_str(args, "op", "any");
    if (op != "any" && op != "none" && op != "ge") return {};
    std::string edge_type_name = to_upper(strip(arg_str(args, "edge_type", "TOUCHES")));
    std::string direction = to_lower(arg_str(args, "direction", "both"));
    int min_count = std::max(1, arg_int(args, "count", 1));
    std::vector<std::string> out;
    const EdgeNeighbors nbr(scene, edge_type_name, direction);
    for (const auto& r : subs) {
      int ec = nbr.count(r.node->node_id);
      if (op == "any" && ec > 0) out.push_back(r.node->node_id);
      else if (op == "none" && ec == 0) out.push_back(r.node->node_id);
      else if (op == "ge" && ec >= min_count)
        out.push_back(r.node->node_id);
    }
    return out;
}

std::vector<std::string> sel_largest_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    // max by (mass, node_id). subs already sorted ascending by (mass, id), so
    // the last element is the max under that exact key.
    return {subs.back().node->node_id};
}

std::vector<std::string> sel_smallest_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    // min by (mass, node_id) -> first element of the (mass, id) ascending sort.
    return {subs.front().node->node_id};
}

std::vector<std::string> sel_leftmost_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;
return {extremal_by_bbox(subs, 1, false)};
}

std::vector<std::string> sel_rightmost_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;
return {extremal_by_bbox(subs, 3, true)};
}

std::vector<std::string> sel_topmost_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;
return {extremal_by_bbox(subs, 0, false)};
}

std::vector<std::string> sel_bottommost_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;
return {extremal_by_bbox(subs, 2, true)};
}

std::vector<std::string> sel_top_leftmost_subobject(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    // min by (bbox[0], bbox[1], node_id) with missing sentinel (1e9,1e9,0,0).
    const SubRow* best = nullptr;
    for (const auto& r : subs) {
      if (best == nullptr) { best = &r; continue; }
      long long by = bbox_component(*r.node, 0, 1000000000LL);
      long long bx = bbox_component(*r.node, 1, 1000000000LL);
      long long cy = bbox_component(*best->node, 0, 1000000000LL);
      long long cx = bbox_component(*best->node, 1, 1000000000LL);
      if (by < cy || (by == cy && (bx < cx || (bx == cx && r.node->node_id < best->node->node_id))))
        best = &r;
    }
    return {best->node->node_id};
}

std::vector<std::string> sel_top_leftmost_non_most_frequent(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::vector<SubRow> cands = non_dominant_color_candidates(subs);
    if (cands.empty()) return {};
    const SubRow* best = nullptr;
    for (const auto& r : cands) {
      if (best == nullptr) { best = &r; continue; }
      long long by = bbox_component(*r.node, 0, 1000000000LL);
      long long bx = bbox_component(*r.node, 1, 1000000000LL);
      long long cy = bbox_component(*best->node, 0, 1000000000LL);
      long long cx = bbox_component(*best->node, 1, 1000000000LL);
      if (by < cy || (by == cy && (bx < cx || (bx == cx && r.node->node_id < best->node->node_id))))
        best = &r;
    }
    return {best->node->node_id};
}

std::vector<std::string> sel_leftmost_non_most_frequent(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::vector<SubRow> cands = non_dominant_color_candidates(subs);
    if (cands.empty()) return {};
    // min by (bbox[1], bbox[0], node_id)
    const SubRow* best = nullptr;
    for (const auto& r : cands) {
      if (best == nullptr) { best = &r; continue; }
      long long bx = bbox_component(*r.node, 1, 1000000000LL);
      long long by = bbox_component(*r.node, 0, 1000000000LL);
      long long cx = bbox_component(*best->node, 1, 1000000000LL);
      long long cy = bbox_component(*best->node, 0, 1000000000LL);
      if (bx < cx || (bx == cx && (by < cy || (by == cy && r.node->node_id < best->node->node_id))))
        best = &r;
    }
    return {best->node->node_id};
}

std::vector<std::string> sel_inside_largest_bbox(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    bool include_border = as_bool(args_get(args, "include_border"), false);
    // largest = max by (mass, node_id) = subs.back()
    const Node& largest = *subs.back().node;
    if (!largest.has_bbox) return {};
    long long ymin = largest.bbox_ymin, xmin = largest.bbox_xmin;
    long long ymax = largest.bbox_ymax, xmax = largest.bbox_xmax;
    std::vector<std::string> out;
    for (const auto& r : subs) {
      const Node& n = *r.node;
      if (n.node_id == largest.node_id) continue;
      if (n.pixel_list.empty()) continue;
      bool inside = true;
      for (const auto& px : n.pixel_list) {
        long long y = px.first, x = px.second;
        bool ok = include_border ? (ymin <= y && y <= ymax && xmin <= x && x <= xmax)
                                  : (ymin < y && y < ymax && xmin < x && x < xmax);
        if (!ok) { inside = false; break; }
      }
      if (inside) out.push_back(n.node_id);
    }
    return out;
}

std::vector<std::string> sel_same_shape_as_reference(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::string ref_kind_raw = strip(arg_str(args, "reference_selector_kind",
                                              "top_leftmost_subobject"));
    const DslValue* ra = args_get(args, "reference_selector_args");
    DslArgs ref_args = (ra && ra->tag == DslValue::Tag::Dict && ra->dict) ? *ra->dict : DslArgs{};
    bool exclude_reference = as_bool(args_get(args, "exclude_reference"), true);
    SelectorKind ref_kind;
    if (!selector_from_name(ref_kind_raw, &ref_kind, &ref_args))
      selector_from_name("top_leftmost_subobject", &ref_kind, &ref_args);
    if (ref_kind == SelectorKind::SAME_SHAPE_AS_REFERENCE) {
      ref_args = DslArgs{};
      selector_from_name("top_leftmost_subobject", &ref_kind, &ref_args);
    }
    DslSelector ref_sel; ref_sel.kind = ref_kind; ref_sel.args = ref_args;
    std::vector<std::string> references = select_node_ids(scene, ref_sel, depth + 1);
    if (references.empty()) return {};
    std::set<std::string> reference_set(references.begin(), references.end());
    std::set<std::string> target_shapes;
    for (const auto& rid : references) {
      const Node* rn = find_node(scene, rid);
      if (rn && rn->shape_hash.has_value() && !rn->shape_hash->empty())
        target_shapes.insert(*rn->shape_hash);
    }
    if (target_shapes.empty()) return {};
    std::vector<std::string> out;
    for (const auto& r : subs) {
      const Node& n = *r.node;
      bool has_shape = n.shape_hash.has_value() &&
                       target_shapes.find(*n.shape_hash) != target_shapes.end();
      if (has_shape && (!exclude_reference || reference_set.find(n.node_id) == reference_set.end()))
        out.push_back(n.node_id);
    }
    return out;
}

std::vector<std::string> sel_reference_singletons(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    int max_mass = std::max(1, arg_int(args, "max_mass", 1));
    const DslValue* ec_v = args_get(args, "exclude_color");
    bool has_exclude = false;
    int exclude_color = 0;
    if (ec_v != nullptr && ec_v->tag != DslValue::Tag::Null) {
      // int/float/bool convert; a string must be a whole decimal integer, else no exclusion.
      if (ec_v->tag == DslValue::Tag::Int) { exclude_color = static_cast<int>(ec_v->i); has_exclude = true; }
      else if (ec_v->tag == DslValue::Tag::Bool) { exclude_color = ec_v->b ? 1 : 0; has_exclude = true; }
      else if (ec_v->tag == DslValue::Tag::Float) { exclude_color = static_cast<int>(ec_v->f); has_exclude = true; }
      else if (ec_v->tag == DslValue::Tag::Str) {
        char* endp = nullptr;
        long r = std::strtol(ec_v->s.c_str(), &endp, 10);
        if (endp != ec_v->s.c_str() && *endp == '\0') { exclude_color = static_cast<int>(r); has_exclude = true; }
      }
    }
    // outside_bbox_of_largest: a bool as is; any other value is true when its
    // lower-cased, stripped text is 1, true, yes, y or on.
    const DslValue* ob_v = args_get(args, "outside_bbox_of_largest");
    bool outside_bbox = false;
    if (ob_v != nullptr) {
      if (ob_v->tag == DslValue::Tag::Bool) outside_bbox = ob_v->b;
      else {
        std::string t;
        if (ob_v->tag == DslValue::Tag::Str) t = ob_v->s;
        else if (ob_v->tag == DslValue::Tag::Int) t = std::to_string(ob_v->i);
        t = to_lower(strip(t));
        outside_bbox = (t == "1" || t == "true" || t == "yes" || t == "y" || t == "on");
      }
    }
    const Node* largest = outside_bbox ? subs.back().node : nullptr;
    bool largest_has_bbox = largest && largest->has_bbox;
    std::vector<std::string> out;
    for (const auto& r : subs) {
      const Node& n = *r.node;
      int mass = n.mass;
      if (mass < 1 || mass > max_mass) continue;
      int color = node_color(n, -999);
      if (has_exclude && color == exclude_color) continue;
      if (outside_bbox && largest_has_bbox) {
        long long ymin = largest->bbox_ymin, xmin = largest->bbox_xmin;
        long long ymax = largest->bbox_ymax, xmax = largest->bbox_xmax;
        bool any_inside = false;
        for (const auto& px : n.pixel_list) {
          long long y = px.first, x = px.second;
          if (ymin <= y && y <= ymax && xmin <= x && x <= xmax) { any_inside = true; break; }
        }
        if (any_inside) continue;
      }
      out.push_back(n.node_id);
    }
    return out;
}

std::vector<std::string> sel_by_noise_score(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    double min_score = arg_float(args, "min_score", 0.0);
    double max_score = arg_float(args, "max_score", 1.0);
    if (min_score > max_score) std::swap(min_score, max_score);
    std::string order = to_lower(strip(arg_str(args, "order", "none")));
    const DslValue* tk = args_get(args, "top_k");
    bool has_top_k = tk != nullptr && tk->tag != DslValue::Tag::Null;
    int top_k = has_top_k ? std::max(1, as_int(tk, 1)) : 0;
    // filtered: preserve subobjects order (mass,id).
    std::vector<std::pair<double, std::string>> filtered;  // (noise, id) in sub order
    for (const auto& r : subs) {
      double ns = r.node->noise_score;
      if (min_score <= ns && ns <= max_score) filtered.emplace_back(ns, r.node->node_id);
    }
    if (order == "highest") {
      std::stable_sort(filtered.begin(), filtered.end(),
                       [](const auto& a, const auto& b) {
                         if (a.first != b.first) return a.first > b.first;
                         return a.second < b.second;
                       });
    } else if (order == "lowest") {
      std::stable_sort(filtered.begin(), filtered.end(),
                       [](const auto& a, const auto& b) {
                         if (a.first != b.first) return a.first < b.first;
                         return a.second < b.second;
                       });
    }
    std::vector<std::string> out;
    for (const auto& kv : filtered) out.push_back(kv.second);
    if (has_top_k && static_cast<int>(out.size()) > top_k) out.resize(top_k);
    return out;
}

std::vector<std::string> sel_distance_to_selector(const SelCtx& c) {
  [[maybe_unused]] const Scene& scene = c.scene;
  [[maybe_unused]] const std::vector<SubRow>& subs = c.subs;
  [[maybe_unused]] const DslSelector& selector = c.selector;
  [[maybe_unused]] const DslArgs& args = c.args;
  [[maybe_unused]] const int depth = c.depth;
  [[maybe_unused]] const SelectorKind kind = c.selector.kind;

    std::string other_raw = strip(arg_str(args, "other_selector_kind", "largest_subobject"));
    const DslValue* oa = args_get(args, "other_selector_args");
    DslArgs other_args = (oa && oa->tag == DslValue::Tag::Dict && oa->dict) ? *oa->dict : DslArgs{};
    bool include_references = as_bool(args_get(args, "include_references"), false);
    const DslValue* tk = args_get(args, "top_k");
    bool has_top_k = tk != nullptr && tk->tag != DslValue::Tag::Null;
    int top_k = has_top_k ? std::max(1, as_int(tk, 1)) : 0;
    std::string distance_mode = to_lower(strip(arg_str(args, "distance", "centroid")));
    if (distance_mode != "centroid" && distance_mode != "bbox_gap") distance_mode = "centroid";
    SelectorKind other_kind;
    if (!selector_from_name(other_raw, &other_kind, &other_args))
      selector_from_name("largest_subobject", &other_kind, &other_args);
    if (other_kind == SelectorKind::DISTANCE_TO_SELECTOR) {
      other_args = DslArgs{};
      selector_from_name("largest_subobject", &other_kind, &other_args);
    }
    DslSelector other_sel; other_sel.kind = other_kind; other_sel.args = other_args;
    std::vector<std::string> reference_ids = select_node_ids(scene, other_sel, depth + 1);
    if (reference_ids.empty()) return {};
    std::set<std::string> reference_set(reference_ids.begin(), reference_ids.end());
    std::vector<const Node*> reference_data;
    for (const auto& rid : reference_ids) {
      const Node* rn = find_node(scene, rid);
      if (rn) reference_data.push_back(rn);
    }
    std::vector<std::pair<std::string, double>> ranked;  // in subobjects order
    const double INF = std::numeric_limits<double>::infinity();
    for (const auto& r : subs) {
      const Node& n = *r.node;
      if (!include_references && reference_set.find(n.node_id) != reference_set.end()) continue;
      double best = INF;
      for (const Node* ref : reference_data)
        best = std::min(best, pair_distance(n, *ref, distance_mode));
      if (best != INF) ranked.emplace_back(n.node_id, best);
    }
    // dir: nearest (the old nearest_to_selector) or farthest.
    const std::string dir = arg_str(args, "dir", "nearest");
    if (dir != "nearest" && dir != "farthest") return {};
    if (dir == "nearest") {
      std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
      });
    } else {
      std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return (-a.second) < (-b.second);
        return a.first < b.first;
      });
    }
    std::vector<std::string> out;
    for (const auto& kv : ranked) out.push_back(kv.first);
    if (has_top_k && static_cast<int>(out.size()) > top_k) out.resize(top_k);
    return out;
}

// EXTREME: exactly one object, the extreme one by `pick` (ties broken by node
// id), each pick the handler of the kind it replaces.
std::vector<std::string> sel_extreme(const SelCtx& c) {
  const std::string pick = arg_str(c.args, "pick", "largest");
  if (pick == "largest") return sel_largest_subobject(c);
  if (pick == "smallest") return sel_smallest_subobject(c);
  if (pick == "leftmost") return sel_leftmost_subobject(c);
  if (pick == "rightmost") return sel_rightmost_subobject(c);
  if (pick == "topmost") return sel_topmost_subobject(c);
  if (pick == "bottommost") return sel_bottommost_subobject(c);
  if (pick == "top_left") return sel_top_leftmost_subobject(c);
  if (pick == "top_left_nondominant") return sel_top_leftmost_non_most_frequent(c);
  if (pick == "left_nondominant") return sel_leftmost_non_most_frequent(c);
  return {};
}

}  // namespace

std::vector<std::string> select_node_ids(const Scene& scene, const DslSelector& selector,
                                         int depth) {
  if (depth >= 4) return {};
  std::vector<SubRow> subs = subobjects(scene);
  if (subs.empty()) return {};

  const SelectorKind kind = selector.kind;
  const SelCtx c{scene, subs, selector, selector.args, depth};

  if (kind == SelectorKind::ALL_SUBOBJECTS) return sel_all_subobjects(c);
  if (kind == SelectorKind::SELECTOR_SET_OP) return sel_selector_set_op(c);
  if (kind == SelectorKind::BY_COLOR) return sel_by_color(c);
  if (kind == SelectorKind::NON_BACKGROUND) return sel_non_background(c);
  if (kind == SelectorKind::BY_SHAPE) return sel_by_shape(c);
  if (kind == SelectorKind::BY_MASS) return sel_by_mass(c);
  if (kind == SelectorKind::BY_HOLE_COUNT) return sel_by_hole_count(c);
  if (kind == SelectorKind::BY_ATTR) return sel_by_attr(c);
  if (kind == SelectorKind::BY_BOUNDARY_TOUCH) return sel_by_boundary_touch(c);
  if (kind == SelectorKind::BY_COLOR_FREQUENCY) return sel_by_color_frequency(c);
  if (kind == SelectorKind::TOUCHING_COLOR) return sel_touching_color(c);
  if (kind == SelectorKind::ADJACENT_TO_COLOR) return sel_adjacent_to_color(c);
  if (kind == SelectorKind::EDGE_COUNT) return sel_edge_count(c);
  if (kind == SelectorKind::EXTREME) return sel_extreme(c);
  if (kind == SelectorKind::INSIDE_LARGEST_BBOX) return sel_inside_largest_bbox(c);
  if (kind == SelectorKind::SAME_SHAPE_AS_REFERENCE) return sel_same_shape_as_reference(c);
  if (kind == SelectorKind::REFERENCE_SINGLETONS) return sel_reference_singletons(c);
  if (kind == SelectorKind::BY_NOISE_SCORE) return sel_by_noise_score(c);
  if (kind == SelectorKind::DISTANCE_TO_SELECTOR) return sel_distance_to_selector(c);

  return {};
}

// ---------------------------------------------------------------------------
// guards_pass.
// ---------------------------------------------------------------------------

bool guards_pass(const std::vector<std::string>& selected_ids,
                 const std::vector<DslGuard>& guards) {
  int count = static_cast<int>(selected_ids.size());
  for (const auto& g : guards) {
    int value = arg_int(g.args, "value", 0);
    if (g.kind == GuardKind::MIN_COUNT) {
      if (count < value) return false;
    } else if (g.kind == GuardKind::MAX_COUNT) {
      if (count > value) return false;
    } else if (g.kind == GuardKind::EXACT_COUNT) {
      if (count != value) return false;
    }
  }
  return true;
}


}  // namespace arc
