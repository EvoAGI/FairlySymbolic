// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// EXTRACT what=tile: the grid is a mosaic of equal tiles of exactly two kinds, one
// plain (a single colour) and one holding several colours; the mixed tile is
// cut out (cropped to its non-zero cells). The largest such tile size wins.
// ConceptARC "ExtractObjects".

#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

void apply(Scene& scene, const std::vector<std::string>&, const DslAction&) {
  CGrid g = concept_render_full(scene);
  if (g.empty() || g[0].empty()) return;
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid best;
  int best_area = 0;
  for (int th = 1; th <= H; ++th) {
    if (H % th) continue;
    for (int tw = 1; tw <= W; ++tw) {
      if (W % tw || th * tw <= best_area || (th == H && tw == W) || th * tw < 4) continue;
      std::set<CGrid> mixed, plain;
      for (int y0 = 0; y0 < H; y0 += th)
        for (int x0 = 0; x0 < W; x0 += tw) {
          CGrid t(th, std::vector<int>(tw));
          std::set<int> colours;
          for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
              t[y][x] = g[y0 + y][x0 + x];
              colours.insert(t[y][x]);
            }
          (colours.size() > 1 ? mixed : plain).insert(std::move(t));
        }
      if (mixed.size() != 1 || plain.size() != 1) continue;
      best = *mixed.begin();
      best_area = th * tw;
    }
  }
  if (best.empty()) return;
  std::vector<Coord> cells;
  for (int y = 0; y < static_cast<int>(best.size()); ++y)
    for (int x = 0; x < static_cast<int>(best[0].size()); ++x)
      if (best[y][x] != 0) cells.emplace_back(y, x);
  if (cells.empty()) return;
  Box b = bbox_of(cells);
  CGrid out;
  for (int y = b.y0; y <= b.y1; ++y) out.emplace_back(best[y].begin() + b.x0, best[y].begin() + b.x1 + 1);
  concept_install(scene, out);
}

}  // namespace

// EXTRACT what=tile (concept_extract.cpp).
void extract_tile(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

}  // namespace arc
