// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_GRID_HPP
#define ARC_GRID_HPP

// Grid container + neighbor helpers for the encoder.
//
// Coordinates are (y, x) / (row, col) throughout. Neighbor lists come in the
// FIXED order given below, because the BFS queue order in step-3
// connected-component extraction depends on it.

#include <cstdint>
#include <utility>
#include <vector>

namespace arc {

using Coord = std::pair<int, int>;  // (y, x)

// Row-major grid of small integers (ARC colors 0..9, values fit in int).
class Grid {
 public:
  Grid() : height_(0), width_(0) {}
  Grid(int height, int width) : height_(height), width_(width), cells_(static_cast<size_t>(height) * width, 0) {}

  int height() const { return height_; }
  int width() const { return width_; }
  int size() const { return height_ * width_; }

  int at(int y, int x) const { return cells_[static_cast<size_t>(y) * width_ + x]; }
  void set(int y, int x, int v) { cells_[static_cast<size_t>(y) * width_ + x] = v; }

  const std::vector<int>& cells() const { return cells_; }
  std::vector<int>& cells() { return cells_; }

  // Build from a nested row-major vector (rows of columns).
  static Grid from_rows(const std::vector<std::vector<int>>& rows);

 private:
  int height_;
  int width_;
  std::vector<int> cells_;
};

// get_orthogonal_neighbors: order (-1,0),(1,0),(0,-1),(0,1), bounds-filtered.
std::vector<Coord> get_orthogonal_neighbors(int y, int x, int height, int width);

// get_8_neighbors: dy in [-1,0,1], dx in [-1,0,1] (skip (0,0)), bounds-filtered.
std::vector<Coord> get_8_neighbors(int y, int x, int height, int width);

}  // namespace arc

#endif  // ARC_GRID_HPP
