// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// BRING_TO_FRONT keep=weave_<top> (formerly the op WEAVE top=<top>): vertical
// lines cross horizontal bars, and at every crossing one of the two directions
// is on top. ConceptARC "TopBottom3D". Reached from concept_front.cpp.
//
// <top>:     vertical (crossings take the column line's colour) | horizontal |
//            swap (each crossing takes the colour that is not on top in the input) |
//            drop (a line under the other line at every crossing it has is removed;
//            the crossings keep the line on top)
// A column (row) is a line of colour c when c is cut by another colour into runs,
// fills half of it or more, or sits at both of its ends; a crossing is a cell of a line column and a line row of two
// different colours.

#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(const std::string& top) {
  DslArgs a;
  a.emplace_back("keep", cdv_str("weave_" + top));
  return a;
}

// Line colour of a run of cells, or -1: first a colour cut by other colours into
// two runs or more (the one with most cells), then one filling half the cells,
// then one at both ends.
int line_color(const std::vector<int>& v, int bg) {
  std::map<int, int> n;
  for (int c : v)
    if (c != bg) n[c]++;
  int best = -1;
  for (const auto& kv : n) {
    int first = -1, last = -1, runs = 0;
    for (int i = 0; i < static_cast<int>(v.size()); ++i)
      if (v[i] == kv.first) {
        if (first < 0) first = i;
        if (i == 0 || v[i - 1] != kv.first) ++runs;
        last = i;
      }
    bool joined = runs >= 2;
    for (int i = first; joined && i <= last; ++i) joined = v[i] != bg;
    if (joined && (best < 0 || kv.second > n[best])) best = kv.first;
  }
  if (best >= 0) return best;
  for (const auto& kv : n)
    if (2 * kv.second >= static_cast<int>(v.size())) return kv.first;
  if (v.size() >= 2 && v.front() != bg && v.front() == v.back()) return v.front();
  return -1;
}

CGrid weave(const CGrid& g, int bg, const std::string& top) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<int> rc(H), cc(W);
  for (int y = 0; y < H; ++y) rc[y] = line_color(g[y], bg);
  for (int x = 0; x < W; ++x) {
    std::vector<int> col(H);
    for (int y = 0; y < H; ++y) col[y] = g[y][x];
    cc[x] = line_color(col, bg);
  }
  CGrid out = g;
  if (top == "drop") {
    // A line is kept when it is on top at one of its crossings or has none.
    std::vector<char> row_top(H, 0), col_top(W, 0), row_cross(H, 0), col_cross(W, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (rc[y] >= 0 && cc[x] >= 0 && rc[y] != cc[x] && (g[y][x] == rc[y] || g[y][x] == cc[x])) {
          row_cross[y] = col_cross[x] = 1;
          row_top[y] |= g[y][x] == rc[y];
          col_top[x] |= g[y][x] == cc[x];
        }
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        bool cross = rc[y] >= 0 && cc[x] >= 0 && rc[y] != cc[x];
        if (cross && (g[y][x] == rc[y] || g[y][x] == cc[x])) continue;
        if ((row_cross[y] && !row_top[y] && g[y][x] == rc[y]) || (col_cross[x] && !col_top[x] && g[y][x] == cc[x]))
          out[y][x] = bg;
      }
    return out;
  }
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (rc[y] >= 0 && cc[x] >= 0 && rc[y] != cc[x] && (g[y][x] == rc[y] || g[y][x] == cc[x]))
        out[y][x] = top == "vertical" ? cc[x] : top == "horizontal" ? rc[y] : g[y][x] == cc[x] ? rc[y] : cc[x];
  return out;
}

}  // namespace

// BRING_TO_FRONT keep=weave_* (concept_front.cpp).
void weave_apply(Scene& scene, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  concept_install(scene, weave(g, concept_bg(scene), arg_str(action.args, "keep", "weave_vertical").substr(6)));
}

DslArgs weave_sample(ArcRandom& rng) { return make(pick_str(rng, {"vertical", "horizontal", "swap", "drop"})); }

// Seeds only when every train output has the input's size.
std::vector<DslArgs> weave_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  res.push_back(make("vertical"));
  res.push_back(make("horizontal"));
  res.push_back(make("swap"));
  res.push_back(make("drop"));
  return res;
}

}  // namespace arc
