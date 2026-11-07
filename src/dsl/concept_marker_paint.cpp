// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// MARK_SIDE side=marker_up / marker_down / marker_left / marker_right: each
// marker cell (a cell of the least frequent colour) recolours the object it sits
// in, from the marker to the object's end in that direction. ConceptARC
// "AboveBelow", "TopBottom2D". (Formerly paint_from_marker dir=up / ...; the
// selection is not read.)

#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kDirs = {"up", "down", "left", "right"};

DslArgs make(const std::string& dir) {
  DslArgs a;
  a.emplace_back("side", cdv_str("marker_" + dir));
  return a;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::map<int, int> n;
  for (const auto& row : g)
    for (int v : row)
      if (v != bg) n[v]++;
  if (n.size() < 2) return;
  int marker = n.begin()->first;
  for (const auto& kv : n)
    if (kv.second < n[marker]) marker = kv.first;
  std::string dir = arg_str(action.args, "side", "marker_up").substr(7);
  int k = 0;
  while (k < 4 && kDirs[k] != dir) ++k;
  if (k == 4) return;
  const int DY[4] = {-1, 1, 0, 0}, DX[4] = {0, 0, -1, 1};
  CGrid out = g;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] != marker) continue;
      for (int t = 1;; ++t) {
        int p = y + t * DY[k], q = x + t * DX[k];
        if (p < 0 || p >= H || q < 0 || q >= W || g[p][q] == bg || g[p][q] == marker) break;
        out[p][q] = marker;
      }
    }
  concept_install(scene, out);
}

}  // namespace

// MARK_SIDE side=marker_* (concept_mark_side.cpp).
void paint_from_marker(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

DslArgs paint_from_marker_sample(ArcRandom& rng) { return make(pick_str(rng, kDirs)); }

// Seeds only when every train output has the input's size.
std::vector<DslArgs> paint_from_marker_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const std::string& d : kDirs) res.push_back(make(d));
  return res;
}

}  // namespace arc
