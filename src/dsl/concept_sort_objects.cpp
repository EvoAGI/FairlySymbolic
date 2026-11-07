// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SORT_OBJECTS: the objects are sorted by a measure and packed along an axis,
// from the position of the first one, with a fixed gap; the other coordinate
// of each object is kept. ConceptARC "Order".
//
// args: key   mass | height | width | pos (their place along the axis; with desc,
//             the order is reversed) | ends (pos with the first and last swapped)
//       desc  descending order
//       axis  x | y | auto (the axis along which the objects do not overlap) |
//             bars (each object becomes a bar as long as its measure, standing
//             on one grid edge, packed side by side)
//       base  bottom | top | left | right (bars: the edge they stand on)
//       from  start | end (bars: packed from the left / top, or the right / bottom)
//       gap   cells between consecutive objects
//       diag  8-connected objects
//       dots  lone cells of a colour at most two apart are one object (a dotted
//             outline)
// key panel_minority | panel_nonbg | panel_minority_row (with desc, axis v | h,
// clean): the grid's panels are sorted instead (formerly the op sort_panels;
// concept_sort_panels.cpp).

#include <algorithm>
#include <tuple>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void sort_panels(Scene& scene, const DslAction& action);
DslArgs sort_panels_sample(ArcRandom& rng);
std::vector<DslArgs> sort_panels_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  if (arg_str(action.args, "key", "mass").rfind("panel_", 0) == 0) return sort_panels(scene, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  bool diag = as_bool(args_get(action.args, "diag"), false);
  auto comps = as_bool(args_get(action.args, "dots"), false) ? concept_dotted_groups(g, bg, diag)
                                                              : concept_components(g, bg, diag, true);
  if (comps.size() < 2) return;
  std::string key = arg_str(action.args, "key", "mass");
  auto measure = [&](const std::vector<Coord>& c) {
    Box b = bbox_of(c);
    if (key == "height") return b.y1 - b.y0 + 1;
    if (key == "width") return b.x1 - b.x0 + 1;
    return static_cast<int>(c.size());
  };
  std::vector<std::tuple<int, int, int, size_t>> order;  // (measure, y0, x0, index)
  for (size_t i = 0; i < comps.size(); ++i) {
    Box b = bbox_of(comps[i]);
    order.emplace_back(measure(comps[i]), b.y0, b.x0, i);
  }
  std::sort(order.begin(), order.end());
  if (as_bool(args_get(action.args, "desc"), false)) std::reverse(order.begin(), order.end());

  std::string axis = arg_str(action.args, "axis", "auto");
  if (axis == "bars") {
    std::string base = arg_str(action.args, "base", "bottom");
    bool vertical = base == "bottom" || base == "top", end = arg_str(action.args, "from", "start") == "end";
    int gap = std::max(0, arg_int(action.args, "gap", 1)), slot = 0;
    int along = vertical ? W : H, depth = vertical ? H : W;
    CGrid out(H, std::vector<int>(W, bg));
    for (const auto& t : order) {
      const auto& c = comps[std::get<3>(t)];
      int len = std::get<0>(t);
      if (slot >= along || len > depth) return;
      int pos = end ? along - 1 - slot : slot;
      for (int d = 0; d < len; ++d) {
        int dd = (base == "bottom" || base == "right") ? depth - 1 - d : d;
        int y = vertical ? dd : pos, x = vertical ? pos : dd;
        out[y][x] = g[c[0].first][c[0].second];
      }
      slot += 1 + gap;
    }
    concept_install(scene, out);
    return;
  }
  if (axis == "auto") {
    bool sep_x = true, sep_y = true;
    for (size_t i = 0; i < comps.size(); ++i)
      for (size_t j = i + 1; j < comps.size(); ++j) {
        Box a = bbox_of(comps[i]), b = bbox_of(comps[j]);
        sep_x = sep_x && (a.x1 < b.x0 || b.x1 < a.x0);
        sep_y = sep_y && (a.y1 < b.y0 || b.y1 < a.y0);
      }
    if (sep_x == sep_y) return;
    axis = sep_x ? "x" : "y";
  }
  bool ax = axis == "x";
  if (key == "pos" || key == "ends") {
    std::sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
      return ax ? std::get<2>(a) < std::get<2>(b) : std::get<1>(a) < std::get<1>(b);
    });
    if (key == "ends") std::swap(order.front(), order.back());
    else if (as_bool(args_get(action.args, "desc"), false)) std::reverse(order.begin(), order.end());
  }
  int gap = std::max(0, arg_int(action.args, "gap", 1));
  int pos = 1 << 30;
  for (const auto& c : comps) pos = std::min(pos, ax ? bbox_of(c).x0 : bbox_of(c).y0);
  CGrid out(H, std::vector<int>(W, bg));
  for (const auto& t : order) {
    const auto& c = comps[std::get<3>(t)];
    Box b = bbox_of(c);
    int dy = ax ? 0 : pos - b.y0, dx = ax ? pos - b.x0 : 0;
    for (const Coord& p : c) {
      int y = p.first + dy, x = p.second + dx;
      if (y < 0 || y >= H || x < 0 || x >= W) return;
      out[y][x] = g[p.first][p.second];
    }
    pos += (ax ? b.x1 - b.x0 : b.y1 - b.y0) + 1 + gap;
  }
  concept_install(scene, out);
}

DslArgs make(const std::string& key, bool desc, const std::string& axis, int gap, bool diag, bool dots = false) {
  return {{"key", cdv_str(key)}, {"desc", cdv_bool(desc)}, {"axis", cdv_str(axis)},
          {"gap", cdv_int(gap)}, {"diag", cdv_bool(diag)}, {"dots", cdv_bool(dots)}};
}

// Half the draws are panel_* (the weight of the former SORT_PANELS op).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 1})) return sort_panels_sample(rng);
  std::string key = pick_str(rng, {"mass", "height", "width", "pos", "ends"});
  bool desc = pick_int(rng, {0, 1}) == 1;
  std::string axis = pick_str(rng, {"auto", "auto", "x", "y", "bars"});
  int gap = pick_int(rng, {0, 1, 1, 2});
  bool diag = pick_int(rng, {0, 1}) == 1;
  bool dots = pick_int(rng, {0, 1}) == 1;
  DslArgs a = make(key, desc, axis, gap, diag, dots);
  if (axis == "bars") {
    a.emplace_back("base", cdv_str(pick_str(rng, {"bottom", "top", "left", "right"})));
    a.emplace_back("from", cdv_str(pick_str(rng, {"start", "end"})));
  }
  return a;
}

// Seeds when every pair keeps the grid size and the colour counts (objects move).
// sort_panels' seeds first, the order the registry gave them.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = sort_panels_seeds(ins, outs);
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    std::vector<int> a = ins[i].cells(), b = outs[i].cells();
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) return res;
  }
  for (const char* k : {"mass", "height", "width"})
    for (bool desc : {false, true})
      for (const char* axis : {"auto", "x", "y", "bars"})
        for (int gap : {0, 1})
          for (bool d : {false, true}) res.push_back(make(k, desc, axis, gap, d));
  return res;
}

}  // namespace

ConceptOp sort_objects_op() { return {ActionKind::SORT_OBJECTS, 4, apply, sample, seeds}; }

}  // namespace arc
