// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// EXTRACT what=extreme_top / extreme_bottom / extreme_left / extreme_right:
// only the object that reaches farthest towards that side of the grid stays
// (all of them on a tie); the rest becomes background. Objects are 8-connected
// one-colour groups, each joined with the groups whose bbox lies inside its
// bbox (a frame with its content). ConceptARC "TopBottom2D" (keep the object at
// the bottom); in ARC, "extract the top-left / lowest object". (Formerly
// keep_extreme side=top / ...; the selection is not read.)
//
// args: out  grid (same size) | crop (the kept objects' bounding box)

#include <algorithm>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kSides = {"top", "bottom", "left", "right"};

DslArgs make(const std::string& side, const std::string& out) {
  DslArgs a;
  a.emplace_back("what", cdv_str("extreme_" + side));
  a.emplace_back("out", cdv_str(out));
  return a;
}

// How far the box reaches towards the side (larger = farther).
int reach(const Box& b, const std::string& side) {
  if (side == "top") return -b.y0;
  if (side == "left") return -b.x0;
  if (side == "right") return b.x1;
  return b.y1;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  auto comps = concept_components(g, bg, true, true);
  // Each component's box, kept along: merging j into i leaves i's box as it
  // was (j's box lies inside it).
  std::vector<Box> boxes;
  boxes.reserve(comps.size());
  for (const auto& c : comps) boxes.push_back(bbox_of(c));
  for (bool merged = true; merged;) {
    merged = false;
    for (size_t i = 0; i < comps.size() && !merged; ++i)
      for (size_t j = 0; j < comps.size() && !merged; ++j) {
        if (i == j) continue;
        const Box& a = boxes[i];
        const Box& b = boxes[j];
        if (b.y0 >= a.y0 && b.y1 <= a.y1 && b.x0 >= a.x0 && b.x1 <= a.x1) {
          comps[i].insert(comps[i].end(), comps[j].begin(), comps[j].end());
          comps.erase(comps.begin() + j);
          boxes.erase(boxes.begin() + j);
          merged = true;
        }
      }
  }
  if (comps.size() < 2) return;
  std::string side = arg_str(action.args, "what", "extreme_bottom").substr(8);
  int best = reach(bbox_of(comps[0]), side);
  for (const auto& c : comps) best = std::max(best, reach(bbox_of(c), side));
  CGrid out(g.size(), std::vector<int>(g[0].size(), bg));
  std::vector<Coord> kept;
  for (const auto& c : comps)
    if (reach(bbox_of(c), side) == best)
      for (const Coord& p : c) {
        out[p.first][p.second] = g[p.first][p.second];
        kept.push_back(p);
      }
  if (arg_str(action.args, "out", "grid") == "crop") {
    Box b = bbox_of(kept);
    CGrid c;
    for (int y = b.y0; y <= b.y1; ++y) c.emplace_back(out[y].begin() + b.x0, out[y].begin() + b.x1 + 1);
    out = c;
  }
  concept_install(scene, out);
}

}  // namespace

// EXTRACT what=extreme_* (concept_extract.cpp).
void extract_extreme(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

DslArgs extract_extreme_sample(ArcRandom& rng) { return make(pick_str(rng, kSides), pick_str(rng, {"grid", "crop"})); }

// Seeded on every task (keep_extreme was).
std::vector<DslArgs> extract_extreme_seeds() {
  std::vector<DslArgs> res;
  for (const std::string& s : kSides)
    for (const char* o : {"grid", "crop"}) res.push_back(make(s, o));
  return res;
}

}  // namespace arc
