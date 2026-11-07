// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// CONCAT_OBJECTS: the clean shapes of the grid (solid rectangles and one-cell
// frames, at least 2x2, single colour) are cut out and laid side by side in
// reading order; everything else is noise. ConceptARC "ExtractObjects".
//
// args: axis  auto (side by side when all share a height, else stacked when all
//             share a width) | h | v | row, column (every object, clean or
//             not, gives one cell of its colour: a 1xN or Nx1 colour list)
//       order x (left to right, then top to bottom) | y (top to bottom first)

#include <algorithm>
#include <tuple>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(const std::string& axis, const std::string& order) {
  DslArgs a;
  a.emplace_back("axis", cdv_str(axis));
  a.emplace_back("order", cdv_str(order));
  return a;
}

// Solid rectangle, or a frame whose inside is all background, at least 2x2.
bool clean(const CGrid& g, int bg, const std::vector<Coord>& comp, const Box& b) {
  int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1, n = static_cast<int>(comp.size());
  if (h < 2 || w < 2) return false;
  if (n == h * w) return true;
  if (h < 3 || w < 3 || n != 2 * (h + w) - 4) return false;
  for (int y = b.y0 + 1; y < b.y1; ++y)
    for (int x = b.x0 + 1; x < b.x1; ++x)
      if (g[y][x] != bg) return false;
  return true;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  std::string axis = arg_str(action.args, "axis", "auto");
  bool by_y = arg_str(action.args, "order", "x") == "y";
  auto comps = concept_components(g, bg, false, true);
  if (axis == "row" || axis == "column") {
    if (comps.empty() || comps.size() > 30) return;
    std::vector<std::tuple<int, int, int>> cells;  // (first key, second key, colour)
    for (const auto& comp : comps) {
      Box b = bbox_of(comp);
      cells.emplace_back(by_y ? b.y0 : b.x0, by_y ? b.x0 : b.y0, g[comp[0].first][comp[0].second]);
    }
    std::sort(cells.begin(), cells.end());
    CGrid out;
    for (const auto& c : cells) {
      if (axis == "column") out.push_back({std::get<2>(c)});
      else if (out.empty()) out.push_back({std::get<2>(c)});
      else out[0].push_back(std::get<2>(c));
    }
    concept_install(scene, out);
    return;
  }
  std::vector<Box> keep;
  for (const auto& comp : comps) {
    Box b = bbox_of(comp);
    if (clean(g, bg, comp, b)) keep.push_back(b);
  }
  if (keep.empty()) return;
  std::sort(keep.begin(), keep.end(), [by_y](const Box& a, const Box& b) {
    return by_y ? std::tie(a.y0, a.x0) < std::tie(b.y0, b.x0) : std::tie(a.x0, a.y0) < std::tie(b.x0, b.y0);
  });
  auto all_same = [&](bool height) {
    for (const Box& b : keep)
      if ((height ? b.y1 - b.y0 : b.x1 - b.x0) != (height ? keep[0].y1 - keep[0].y0 : keep[0].x1 - keep[0].x0))
        return false;
    return true;
  };
  bool horiz;
  if (axis == "h") horiz = true;
  else if (axis == "v") horiz = false;
  else if (all_same(true)) horiz = true;
  else if (all_same(false)) horiz = false;
  else return;
  if (!all_same(horiz)) return;
  CGrid out;
  if (horiz) {
    out.assign(keep[0].y1 - keep[0].y0 + 1, {});
    for (const Box& b : keep)
      for (int y = b.y0; y <= b.y1; ++y)
        out[y - b.y0].insert(out[y - b.y0].end(), g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  } else {
    for (const Box& b : keep)
      for (int y = b.y0; y <= b.y1; ++y) out.emplace_back(g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  }
  if (out.size() > 30 || out[0].size() > 30) return;
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) {
  return make(pick_str(rng, {"auto", "h", "v", "row", "column"}), pick_str(rng, {"x", "y"}));
}

// Seeds only when every train output is smaller than its input.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (outs[i].height() * outs[i].width() >= ins[i].height() * ins[i].width()) return res;
  for (const char* o : {"x", "y"}) res.push_back(make("auto", o));
  return res;
}

}  // namespace

ConceptOp concat_objects_op() { return {ActionKind::CONCAT_OBJECTS, 2, apply, sample, seeds}; }

}  // namespace arc
