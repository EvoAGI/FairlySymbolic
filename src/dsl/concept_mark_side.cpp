// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// MARK_SIDE: the cells on one side of each selected object (its leftmost
// column, rightmost column, top row or bottom row) are recoloured. ConceptARC
// "AboveBelow"; common in ARC (the end of a bar marked in another colour).
//
// args: side   left | right | top | bottom | cw | ccw (the side met by turning
//              clockwise / anticlockwise from the direction towards the largest
//              object, a separator line: above it cw marks the left end, below it
//              the right end; objects that overlap its rows and columns are left
//              alone; the selected cells are taken in 4-connected
//              single-colour parts, whatever the parse)
//              | marker_up | marker_down | marker_left | marker_right (each
//              cell of the least frequent colour recolours its object from
//              itself to the object's end; concept_marker_paint.cpp)
//              | orient_fill | orient_extent | orient_stripes (formerly the op
//              orient_mark by=fill / extent / stripes: one full line along a
//              grid edge, chosen by the drawing's orientation; with h_side,
//              v_side; concept_orient_mark.cpp)
//       color  0-9, or -1 = the colour of the largest object of the scene (a
//              separator line included); not for marker_*; for orient_*, the
//              line colour

#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void paint_from_marker(Scene& scene, const DslAction& action);
DslArgs paint_from_marker_sample(ArcRandom& rng);
std::vector<DslArgs> paint_from_marker_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);
void orient_mark(Scene& scene, const DslAction& action);
DslArgs orient_mark_sample(ArcRandom& rng);
std::vector<DslArgs> orient_mark_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

DslArgs make(const std::string& side, int color) {
  DslArgs a;
  a.emplace_back("side", cdv_str(side));
  a.emplace_back("color", cdv_int(color));
  return a;
}

void apply(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action) {
  std::string side = arg_str(action.args, "side", "left");
  if (side.rfind("marker_", 0) == 0) return paint_from_marker(scene, action);
  if (side.rfind("orient_", 0) == 0) return orient_mark(scene, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int color = arg_int(action.args, "color", -1);
  // The largest object: the colour for color=-1, the reference for cw / ccw.
  size_t best = 0;
  Box ref{0, 0, -1, -1};
  int ref_color = -1;
  for (size_t i : scene.get_nodes_by_type("SubObject", true)) {
    const Node& n = scene.nodes()[i];
    if (n.color.has_value() && *n.color != concept_bg(scene) && n.pixel_list.size() > best) {
      best = n.pixel_list.size();
      ref_color = *n.color;
      ref = bbox_of(n.pixel_list);
    }
  }
  if (color < 0) color = ref_color;
  bool turn = side == "cw" || side == "ccw";
  if (color < 0 || (turn && !best)) return;
  std::set<std::string> ids(selected_ids.begin(), selected_ids.end());
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::vector<Coord>> parts;
  CGrid sel(H, std::vector<int>(W, -1));
  for (size_t i : scene.get_nodes_by_type("SubObject", false)) {
    const Node& n = scene.nodes()[i];
    if (!ids.count(n.node_id) || n.pixel_list.empty()) continue;
    if (!turn) parts.push_back(n.pixel_list);
    for (const auto& p : n.pixel_list)
      if (p.first >= 0 && p.first < H && p.second >= 0 && p.second < W) sel[p.first][p.second] = g[p.first][p.second];
  }
  // cw / ccw: the selected cells in 4-connected single-colour parts.
  if (turn) parts = concept_components(sel, -1, false, true);
  for (const auto& part : parts) {
    Box b = bbox_of(part);
    std::string s = side;
    if (turn) {
      // Towards the reference: down, up, right, left; clockwise of each.
      bool cw = side == "cw";
      if (b.y1 < ref.y0) s = cw ? "left" : "right";
      else if (b.y0 > ref.y1) s = cw ? "right" : "left";
      else if (b.x1 < ref.x0) s = cw ? "bottom" : "top";
      else if (b.x0 > ref.x1) s = cw ? "top" : "bottom";
      else continue;
    }
    for (const auto& p : part) {
      bool on = s == "left" ? p.second == b.x0 : s == "right" ? p.second == b.x1
              : s == "top" ? p.first == b.y0 : p.first == b.y1;
      if (on && p.first >= 0 && p.first < H && p.second >= 0 && p.second < W) g[p.first][p.second] = color;
    }
  }
  concept_install(scene, g);
}

// A third of the draws are orient_* (the weight of the former ORIENT_MARK op);
// of the others, half are marker_* (the former PAINT_FROM_MARKER).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 0, 1})) return orient_mark_sample(rng);
  if (pick_int(rng, {0, 1})) return paint_from_marker_sample(rng);
  return make(pick_str(rng, {"left", "right", "top", "bottom", "cw", "ccw"}),
              pick_int(rng, {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Seeds when every train output keeps the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = orient_mark_seeds(ins, outs);
  for (DslArgs& a : paint_from_marker_seeds(ins, outs)) res.push_back(std::move(a));
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const char* s : {"left", "right", "top", "bottom"}) res.push_back(make(s, -1));
  return res;
}

}  // namespace

ConceptOp mark_side_op() { return {ActionKind::MARK_SIDE, 6, apply, sample, seeds}; }

}  // namespace arc
