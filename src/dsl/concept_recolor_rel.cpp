// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// ATTR_PAINT attr=rel_<source> (formerly the op RECOLOR_REL source=<source>;
// reached from concept_attr_paint.cpp): each selected object takes the colour of
// an object related to it; no colour map to learn. ConceptARC "FilledNotFilled", "InsideOutside",
// "Copy"; common in ARC (a frame takes its marker's colour, a shape the colour
// of what it touches).
//
// <source>:     touching (the other colour with most cells next to it) |
//               nearest (the closest other-colour object) | largest (the largest
//               object's colour) | bottom / top (the colour of the object reaching
//               lowest / highest; ConceptARC "TopBottom2D"). inside, container and
//               rare were dropped (no solved program used them); a program naming
//               one changes nothing.
//       target  object (its cells) | box_bg (the background inside its bbox)

#include <algorithm>
#include <climits>
#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kSources = {"touching", "nearest", "largest", "bottom", "top"};

DslArgs make(const std::string& source, const std::string& target) {
  DslArgs a;
  a.emplace_back("attr", cdv_str("rel_" + source));
  a.emplace_back("target", cdv_str(target));
  return a;
}

}  // namespace

// ATTR_PAINT attr=rel_* (concept_attr_paint.cpp).
void recolor_rel(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  auto comps = concept_components(g, bg, true, true);
  if (comps.size() < 2) return;
  std::vector<int> owner(H * W, -1), col(comps.size());
  std::vector<Box> box(comps.size());
  for (size_t k = 0; k < comps.size(); ++k) {
    for (const Coord& c : comps[k]) owner[c.first * W + c.second] = static_cast<int>(k);
    col[k] = g[comps[k][0].first][comps[k][0].second];
    box[k] = bbox_of(comps[k]);
  }
  std::string source = arg_str(action.args, "attr", "rel_nearest").substr(4);
  bool box_bg = arg_str(action.args, "target", "object") == "box_bg";
  int largest = 0;
  for (size_t k = 1; k < comps.size(); ++k)
    if (comps[k].size() > comps[largest].size()) largest = static_cast<int>(k);
  int lowest = 0, highest = 0;
  for (size_t k = 1; k < comps.size(); ++k) {
    if (box[k].y1 > box[lowest].y1) lowest = static_cast<int>(k);
    if (box[k].y0 < box[highest].y0) highest = static_cast<int>(k);
  }
  std::set<int> targets;
  std::set<std::string> ids(selected_ids.begin(), selected_ids.end());
  for (const Node& n : scene.nodes())
    if (ids.count(n.node_id))
      for (const auto& p : n.pixel_list)
        if (p.first >= 0 && p.first < H && p.second >= 0 && p.second < W && owner[p.first * W + p.second] >= 0)
          targets.insert(owner[p.first * W + p.second]);

  CGrid out = g;
  for (int k : targets) {
    int c = -1;
    if (source == "largest") {
      c = col[largest];
    } else if (source == "bottom" || source == "top") {
      c = col[source == "bottom" ? lowest : highest];
    } else if (source == "touching") {
      std::map<int, int> n;
      for (const Coord& e : comps[k])
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            int y = e.first + dy, x = e.second + dx;
            if (y < 0 || y >= H || x < 0 || x >= W || g[y][x] == bg || g[y][x] == col[k]) continue;
            n[g[y][x]]++;
          }
      int bn = 0;
      for (const auto& kv : n)
        if (kv.second > bn) { bn = kv.second; c = kv.first; }
    } else if (source == "nearest") {
      int best = INT_MAX;
      for (size_t q = 0; q < comps.size(); ++q) {
        if (static_cast<int>(q) == k || col[q] == col[k]) continue;
        int gy = std::max(0, std::max(box[q].y0 - box[k].y1, box[k].y0 - box[q].y1));
        int gx = std::max(0, std::max(box[q].x0 - box[k].x1, box[k].x0 - box[q].x1));
        if (std::max(gy, gx) < best) { best = std::max(gy, gx); c = col[q]; }
      }
    }
    if (c < 0) continue;
    if (box_bg) {
      for (int y = box[k].y0; y <= box[k].y1; ++y)
        for (int x = box[k].x0; x <= box[k].x1; ++x)
          if (g[y][x] == bg) out[y][x] = c;
    } else {
      for (const Coord& e : comps[k]) out[e.first][e.second] = c;
    }
  }
  concept_install(scene, out);
}

DslArgs recolor_rel_sample(ArcRandom& rng) { return make(pick_str(rng, kSources), pick_str(rng, {"object", "object", "box_bg"})); }

// Seeds when every train output keeps the input's size.
std::vector<DslArgs> recolor_rel_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const std::string& s : kSources)
    for (const char* t : {"object", "box_bg"}) res.push_back(make(s, t));
  return res;
}

}  // namespace arc
