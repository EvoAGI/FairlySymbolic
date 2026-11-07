// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// TILE_GAP: the grid is repeated twice, side by side or one above the other,
// with `gap` background rows / columns between the copies. ConceptARC "Copy".
//
// args: axis  h (side by side) | v (stacked) | auto (across the way the objects
//             line up: stacked when a background column splits the drawing and no
//             background row does, side by side when a row does and no column;
//             else stacked when the grid is wider than tall)
//       gap   1 | 2 (gap 0 was dropped: the global transforms tile_h / tile_v
//             double the grid with no gap; a program with gap 0 changes nothing)

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

// A background line strictly between non-background cells.
bool splits(const CGrid& g, int bg, bool rows) {
  int n = static_cast<int>(rows ? g.size() : g[0].size()), m = static_cast<int>(rows ? g[0].size() : g.size());
  int first = -1, last = -1;
  std::vector<char> empty(n, 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j)
      if ((rows ? g[i][j] : g[j][i]) != bg) empty[i] = 0;
    if (!empty[i]) { if (first < 0) first = i; last = i; }
  }
  for (int i = first + 1; i < last; ++i)
    if (empty[i]) return true;
  return false;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int gap = std::min(2, arg_int(action.args, "gap", 1));
  if (gap < 1) return;
  std::string axis = arg_str(action.args, "axis", "auto");
  if (axis == "auto") {
    bool col = splits(g, bg, false), row = splits(g, bg, true);
    axis = col && !row ? "v" : row && !col ? "h" : W > H ? "v" : "h";
  }
  CGrid out;
  if (axis == "v") {
    if (2 * H + gap > 30) return;
    out = g;
    for (int i = 0; i < gap; ++i) out.push_back(std::vector<int>(W, bg));
    out.insert(out.end(), g.begin(), g.end());
  } else {
    if (2 * W + gap > 30) return;
    for (const auto& row : g) {
      std::vector<int> r = row;
      r.insert(r.end(), gap, bg);
      r.insert(r.end(), row.begin(), row.end());
      out.push_back(r);
    }
  }
  concept_install(scene, out);
}

DslArgs make(const std::string& axis, int gap) { return {{"axis", cdv_str(axis)}, {"gap", cdv_int(gap)}}; }

DslArgs sample(ArcRandom& rng) {
  return make(pick_str(rng, {"auto", "auto", "h", "v"}), pick_int(rng, {1, 1, 2}));
}

// Seeds only when every train output is larger than its input.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (outs[i].height() * outs[i].width() <= ins[i].height() * ins[i].width()) return res;
  for (const char* a : {"auto", "h", "v"})
    for (int gp : {1, 2}) res.push_back(make(a, gp));
  return res;
}

}  // namespace

ConceptOp tile_gap_op() { return {ActionKind::TILE_GAP, 2, apply, sample, seeds}; }

}  // namespace arc
