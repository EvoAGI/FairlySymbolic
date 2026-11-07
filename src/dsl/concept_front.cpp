// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// BRING_TO_FRONT: an object cut in pieces by another one in front of it is made
// whole across it, and the object in front disappears. ConceptARC "TopBottom3D".
//
// args: keep  none (the front object is removed) | outside (the front object keeps
//             its cells outside the back object's bbox) | swap (the back object,
//             completed, is drawn over the front one, which keeps its other cells)
//             | shift (the front object steps one cell towards `dir`, and the back
//             object, completed, shows where it was)
//             | weave_vertical | weave_horizontal | weave_swap | weave_drop
//             (formerly the op weave top=...: lines crossing bars, one direction
//             on top at every crossing; concept_weave.cpp)
//       dir   left | right | up | down (keep=shift only)
// none / outside: the back colour is one whose cells form two or more pieces; a
// cell of the front colour is hidden back when its run along a row, column or
// diagonal has the back colour at both ends (so lines and rectangles both
// continue across the front).
// swap / shift: the back colour is the one with the most cells among those whose
// bbox holds another colour; the front is the objects of other colours with a
// cell in that bbox. The back is completed under them by the smallest tiling
// period of its bbox that no two visible cells contradict (no period: under
// swap the front stays, under shift the cells are background).

#include <algorithm>
#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void weave_apply(Scene& scene, const DslAction& action);
DslArgs weave_sample(ArcRandom& rng);
std::vector<DslArgs> weave_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

DslArgs make(const std::string& keep, const std::string& dir = "left") {
  DslArgs a;
  a.emplace_back("keep", cdv_str(keep));
  a.emplace_back("dir", cdv_str(dir));
  return a;
}

// keep=swap / shift.
void lift(CGrid& g, int bg, bool shift, const std::string& dir) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::map<int, std::vector<Coord>> by_colour;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] != bg) by_colour[g[y][x]].push_back({y, x});
  int back = -1;
  Box bb{0, 0, 0, 0};
  for (const auto& kv : by_colour) {
    Box b = bbox_of(kv.second);
    bool other = false;
    for (int y = b.y0; y <= b.y1 && !other; ++y)
      for (int x = b.x0; x <= b.x1 && !other; ++x) other = g[y][x] != bg && g[y][x] != kv.first;
    if (other && (back < 0 || kv.second.size() > by_colour[back].size())) { back = kv.first; bb = b; }
  }
  if (back < 0) return;
  // Front: the objects of the other colours with a cell in the back's bbox.
  CGrid rest = g;
  for (auto& row : rest)
    for (int& v : row)
      if (v == back) v = bg;
  std::vector<Coord> front;
  for (const auto& comp : concept_components(rest, bg, true, false))
    if (std::any_of(comp.begin(), comp.end(), [&](const Coord& c) {
          return c.first >= bb.y0 && c.first <= bb.y1 && c.second >= bb.x0 && c.second <= bb.x1;
        }))
      front.insert(front.end(), comp.begin(), comp.end());
  // The back's bbox as back (1) / not (0), hidden (-1) under the front.
  int h = bb.y1 - bb.y0 + 1, w = bb.x1 - bb.x0 + 1;
  std::vector<std::vector<int>> v(h, std::vector<int>(w));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) v[y][x] = g[bb.y0 + y][bb.x0 + x] == back;
  for (const Coord& c : front)
    if (c.first >= bb.y0 && c.first <= bb.y1 && c.second >= bb.x0 && c.second <= bb.x1)
      v[c.first - bb.y0][c.second - bb.x0] = -1;
  std::vector<std::pair<int, int>> periods;
  for (int py = 1; py <= h; ++py)
    for (int px = 1; px <= w; ++px)
      if (py < h || px < w) periods.push_back({py, px});
  std::stable_sort(periods.begin(), periods.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    return a.first * a.second < b.first * b.second;
  });
  std::vector<int> cls;
  int py = 0, px = 0;
  for (const auto& pp : periods) {
    py = pp.first;
    px = pp.second;
    cls.assign(py * px, -1);
    bool bad = false;
    for (int y = 0; y < h && !bad; ++y)
      for (int x = 0; x < w && !bad; ++x) {
        if (v[y][x] < 0) continue;
        int& c = cls[(y % py) * px + x % px];
        bad = c >= 0 && c != v[y][x];
        c = v[y][x];
      }
    for (int y = 0; y < h && !bad; ++y)
      for (int x = 0; x < w && !bad; ++x) bad = v[y][x] < 0 && cls[(y % py) * px + x % px] < 0;
    if (!bad) break;
    cls.clear();
  }
  CGrid out = g;
  if (shift)
    for (const Coord& c : front) out[c.first][c.second] = bg;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (v[y][x] < 0 && !cls.empty() && cls[(y % py) * px + x % px] == 1) out[bb.y0 + y][bb.x0 + x] = back;
  if (shift) {
    int dy = dir == "up" ? -1 : dir == "down" ? 1 : 0, dx = dir == "left" ? -1 : dir == "right" ? 1 : 0;
    for (const Coord& c : front) {
      int y = c.first + dy, x = c.second + dx;
      if (y >= 0 && y < H && x >= 0 && x < W) out[y][x] = g[c.first][c.second];
    }
  }
  g = out;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::string keep = arg_str(action.args, "keep", "none");
  if (keep.rfind("weave_", 0) == 0) return weave_apply(scene, action);
  if (keep == "swap" || keep == "shift") {
    lift(g, bg, keep == "shift", arg_str(action.args, "dir", "left"));
    concept_install(scene, g);
    return;
  }
  std::map<int, int> pieces;
  for (const auto& comp : concept_components(g, bg, true, true)) pieces[g[comp[0].first][comp[0].second]]++;
  bool keep_outside = keep == "outside";
  // A hidden cell is a run of one front colour with the back colour at both
  // ends, along a row, a column or a diagonal.
  const int DY[4] = {0, 1, 1, 1}, DX[4] = {1, 0, 1, -1};
  auto in = [&](int y, int x) { return y >= 0 && y < H && x >= 0 && x < W; };
  for (const auto& kv : pieces) {
    if (kv.second < 2) continue;
    int back = kv.first;
    std::set<int> fronts;
    std::vector<Coord> hidden;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int f = g[y][x];
        if (f == bg || f == back) continue;
        for (int d = 0; d < 4; ++d) {
          int ends = 0;
          for (int sgn : {1, -1}) {
            int p = y + sgn * DY[d], q = x + sgn * DX[d];
            while (in(p, q) && g[p][q] == f) { p += sgn * DY[d]; q += sgn * DX[d]; }
            ends += in(p, q) && g[p][q] == back;
          }
          if (ends == 2) {
            hidden.push_back({y, x});
            fronts.insert(f);
            break;
          }
        }
      }
    if (fronts.size() != 1) continue;
    int front = *fronts.begin();
    CGrid out = g;
    if (!keep_outside)
      for (auto& row : out)
        for (int& v : row)
          if (v == front) v = bg;
    for (const Coord& c : hidden) out[c.first][c.second] = back;
    concept_install(scene, out);
    return;
  }
}

// Half the draws are weave_* (the weight of the former WEAVE op).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 1})) return weave_sample(rng);
  std::string keep = pick_str(rng, {"none", "outside", "swap", "shift"});
  return make(keep, pick_str(rng, {"left", "right", "up", "down"}));
}

// Seeds only when every train output has the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  res.push_back(make("none"));
  res.push_back(make("outside"));
  res.push_back(make("swap"));
  for (const char* d : {"left", "right", "up", "down"}) res.push_back(make("shift", d));
  for (DslArgs& a : weave_seeds(ins, outs)) res.push_back(std::move(a));
  return res;
}

}  // namespace

ConceptOp bring_to_front_op() { return {ActionKind::BRING_TO_FRONT, 4, apply, sample, seeds}; }

}  // namespace arc
