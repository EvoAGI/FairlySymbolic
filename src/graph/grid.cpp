// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Grid container + neighbor helpers. See grid.hpp for the neighbor order.
#include "arc/grid.hpp"

namespace arc {

Grid Grid::from_rows(const std::vector<std::vector<int>>& rows) {
  int h = static_cast<int>(rows.size());
  int w = h > 0 ? static_cast<int>(rows[0].size()) : 0;
  Grid g(h, w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) g.set(y, x, rows[static_cast<size_t>(y)][static_cast<size_t>(x)]);
  return g;
}

std::vector<Coord> get_orthogonal_neighbors(int y, int x, int height, int width) {
  static const int deltas[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  std::vector<Coord> out;
  for (const auto& d : deltas) {
    int ny = y + d[0];
    int nx = x + d[1];
    if (0 <= ny && ny < height && 0 <= nx && nx < width) out.emplace_back(ny, nx);
  }
  return out;
}

std::vector<Coord> get_8_neighbors(int y, int x, int height, int width) {
  std::vector<Coord> out;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dy == 0 && dx == 0) continue;
      int ny = y + dy;
      int nx = x + dx;
      if (0 <= ny && ny < height && 0 <= nx && nx < width) out.emplace_back(ny, nx);
    }
  }
  return out;
}

}  // namespace arc
