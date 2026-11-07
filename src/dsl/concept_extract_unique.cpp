// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// EXTRACT what=unique: crops the grid to the one colour that stands apart: the only
// colour forming a single connected object (8-connected), or, when no colour or
// several do, the strictly rarest colour. Other colours inside the crop become
// background. ConceptARC "ExtractObjects".

#include <algorithm>
#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

void apply(Scene& scene, const std::vector<std::string>&, const DslAction&) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  std::map<int, int> count, objects;
  for (const auto& row : g)
    for (int v : row)
      if (v != bg) ++count[v];
  if (count.size() < 2) return;
  for (const auto& comp : concept_components(g, bg, true, true)) ++objects[g[comp[0].first][comp[0].second]];
  int pick = -1, singles = 0;
  for (const auto& kv : objects)
    if (kv.second == 1) { pick = kv.first; ++singles; }
  if (singles != 1) {
    std::vector<std::pair<int, int>> by_count;
    for (const auto& kv : count) by_count.emplace_back(kv.second, kv.first);
    std::sort(by_count.begin(), by_count.end());
    if (by_count[0].first == by_count[1].first) return;
    pick = by_count[0].second;
  }
  std::vector<Coord> cells;
  for (int y = 0; y < static_cast<int>(g.size()); ++y)
    for (int x = 0; x < static_cast<int>(g[0].size()); ++x)
      if (g[y][x] == pick) cells.emplace_back(y, x);
  Box b = bbox_of(cells);
  CGrid out(b.y1 - b.y0 + 1, std::vector<int>(b.x1 - b.x0 + 1, bg));
  for (const Coord& c : cells) out[c.first - b.y0][c.second - b.x0] = pick;
  concept_install(scene, out);
}

}  // namespace

// EXTRACT what=unique (concept_extract.cpp).
void extract_unique(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

}  // namespace arc
