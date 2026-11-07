// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SLIDE_UNTIL_BLOCKED dir=jump_<d> (formerly the op JUMP_OVER dir=<d>): an
// object facing a bigger object in one direction jumps to that object's other
// side, keeping the gap between them. ConceptARC "AboveBelow". Reached from
// concept_slide.cpp; `who` is not read, nor the selection.
//
// <d>: down | up | left | right
// Objects are 8-connected single-colour groups; the object faced is the nearest
// one in dir whose span across dir overlaps the mover's.

#include <algorithm>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kDirs = {"down", "up", "left", "right"};

DslArgs make(const std::string& dir) {
  DslArgs a;
  a.emplace_back("dir", cdv_str("jump_" + dir));
  return a;
}

}  // namespace

// SLIDE_UNTIL_BLOCKED dir=jump_* (concept_slide.cpp).
void jump_over(Scene& scene, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::string dir = arg_str(action.args, "dir", "jump_down").substr(5);
  if (std::find(kDirs.begin(), kDirs.end(), dir) == kDirs.end()) return;
  bool vert = dir == "down" || dir == "up", fwd = dir == "down" || dir == "right";
  auto comps = concept_components(g, bg, true, true);
  std::vector<Box> box;
  for (const auto& c : comps) box.push_back(bbox_of(c));
  // Along-dir extent [a0, a1] and across-dir extent [c0, c1] of a box.
  auto a0 = [&](const Box& b) { return vert ? b.y0 : b.x0; };
  auto a1 = [&](const Box& b) { return vert ? b.y1 : b.x1; };
  auto c0 = [&](const Box& b) { return vert ? b.x0 : b.y0; };
  auto c1 = [&](const Box& b) { return vert ? b.x1 : b.y1; };
  std::vector<std::pair<size_t, int>> moves;
  for (size_t i = 0; i < comps.size(); ++i) {
    const Box& m = box[i];
    int best = -1;
    for (size_t j = 0; j < comps.size(); ++j) {
      const Box& b = box[j];
      if (j == i || c1(b) < c0(m) || c0(b) > c1(m)) continue;
      if (fwd ? a0(b) <= a1(m) : a1(b) >= a0(m)) continue;
      if (best < 0 || (fwd ? a0(b) < a0(box[best]) : a1(b) > a1(box[best]))) best = static_cast<int>(j);
    }
    if (best < 0 || comps[best].size() <= comps[i].size()) continue;
    const Box& b = box[best];
    int d = fwd ? a1(b) + 1 + (a0(b) - a1(m) - 1) - a0(m) : a0(b) - 1 - (a0(m) - a1(b) - 1) - a1(m);
    int lim = vert ? H : W;
    if (a0(m) + d < 0 || a1(m) + d >= lim) return;
    moves.push_back({i, d});
  }
  if (moves.empty()) return;
  CGrid out = g;
  for (const auto& mv : moves)
    for (const Coord& c : comps[mv.first]) out[c.first][c.second] = bg;
  for (const auto& mv : moves)
    for (const Coord& c : comps[mv.first]) {
      int y = c.first + (vert ? mv.second : 0), x = c.second + (vert ? 0 : mv.second);
      out[y][x] = g[c.first][c.second];
    }
  concept_install(scene, out);
}

DslArgs jump_over_sample(ArcRandom& rng) { return make(pick_str(rng, kDirs)); }

// Seeds only when every train output has the input's size.
std::vector<DslArgs> jump_over_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const std::string& d : kDirs) res.push_back(make(d));
  return res;
}

}  // namespace arc
