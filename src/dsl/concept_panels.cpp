// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// EXTRACT what=panels: the grid is cut by separator lines (full rows or full columns
// of one colour that appears nowhere else). Each panel's own background (the
// majority colour of its border) is cleared, its content is cropped, and the crops are
// joined across the separators: stacked for horizontal separators, side by side
// for vertical ones, centred on the other axis. ConceptARC "ExtractObjects".
//
// args: mode  join (the default)
//             same  the separators are full rows or columns of the scene
//                   background instead (one way of cutting, into two panels or
//                   more); when every panel holds the same crop, the output is
//                   that crop, else a single background cell. ConceptARC
//                   "SameDifferent"

#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

// Line colours along rows (rows == true) or columns: every cell of the colour
// lies on a full line of it. line_colour[i] is line i's colour, or -1.
std::vector<int> separators(const CGrid& g, bool rows, std::vector<int>* line_colour) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int n = rows ? H : W, len = rows ? W : H;
  auto at = [&](int i, int j) { return rows ? g[i][j] : g[j][i]; };
  std::map<int, int> total, on_lines;
  for (const auto& row : g)
    for (int v : row) ++total[v];
  line_colour->assign(n, -1);
  for (int i = 0; i < n; ++i) {
    int c = at(i, 0);
    bool full = true;
    for (int j = 1; j < len && full; ++j) full = at(i, j) == c;
    if (full) (*line_colour)[i] = c;
  }
  for (int i = 0; i < n; ++i)
    if ((*line_colour)[i] >= 0) on_lines[(*line_colour)[i]] += len;
  std::vector<int> res;
  for (const auto& kv : on_lines)
    if (kv.second == total[kv.first]) res.push_back(kv.first);
  return res;
}

// Crop of one panel's content (cells that differ from the panel's background,
// the majority colour of its border cells), with the background cleared to 0.
// Empty when it has none.
CGrid crop_panel(const CGrid& g, int y0, int y1, int x0, int x1) {
  std::map<int, int> count;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
      if (y == y0 || y == y1 || x == x0 || x == x1) ++count[g[y][x]];
  int bg = 0, best = -1;
  for (const auto& kv : count)
    if (kv.second > best) { best = kv.second; bg = kv.first; }
  std::vector<Coord> cells;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
      if (g[y][x] != bg) cells.emplace_back(y, x);
  if (cells.empty()) return {};
  Box b = bbox_of(cells);
  CGrid out(b.y1 - b.y0 + 1, std::vector<int>(b.x1 - b.x0 + 1, 0));
  for (const Coord& c : cells) out[c.first - b.y0][c.second - b.x0] = g[c.first][c.second];
  return out;
}

// Panel crops when the lines of colour `sep` cut the grid.
std::vector<CGrid> panels(const CGrid& g, bool rows, const std::vector<int>& line_colour, int sep) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int n = static_cast<int>(line_colour.size());
  std::vector<CGrid> crops;
  for (int i = 0; i < n;) {
    if (line_colour[i] == sep) { ++i; continue; }
    int j = i;
    while (j + 1 < n && line_colour[j + 1] != sep) ++j;
    CGrid c = rows ? crop_panel(g, i, j, 0, W - 1) : crop_panel(g, 0, H - 1, i, j);
    if (!c.empty()) crops.push_back(std::move(c));
    i = j + 1;
  }
  return crops;
}

// mode=same: the panel crops when background lines cut the grid one way only.
void same(Scene& scene, const CGrid& g) {
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<CGrid> crops;
  int found = 0;
  for (bool rows : {true, false}) {
    int n = rows ? H : W, len = rows ? W : H;
    std::vector<int> line_colour(n, -1);
    for (int i = 0; i < n; ++i) {
      bool full = true;
      for (int j = 0; j < len && full; ++j) full = (rows ? g[i][j] : g[j][i]) == bg;
      if (full) line_colour[i] = bg;
    }
    std::vector<CGrid> c = panels(g, rows, line_colour, bg);
    if (c.size() < 2) continue;
    ++found;
    crops = std::move(c);
  }
  if (found != 1) return;
  bool equal = true;
  for (const CGrid& c : crops) equal = equal && c == crops[0];
  concept_install(scene, equal ? crops[0] : CGrid(1, std::vector<int>(1, bg)));
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render_full(scene);
  if (g.empty() || g[0].empty()) return;
  if (arg_str(action.args, "mode", "join") == "same") {
    same(scene, g);
    return;
  }
  // The cut must be unambiguous: one (orientation, colour) giving two or more
  // non-empty panels.
  std::vector<CGrid> crops;
  bool rows = true;
  int found = 0;
  for (bool r : {true, false}) {
    std::vector<int> line_colour;
    for (int sep : separators(g, r, &line_colour)) {
      std::vector<CGrid> c = panels(g, r, line_colour, sep);
      if (c.size() < 2) continue;
      ++found;
      crops = std::move(c);
      rows = r;
    }
  }
  if (found != 1) return;
  int span = 0, total = 0;
  for (const CGrid& c : crops) {
    int across = rows ? static_cast<int>(c[0].size()) : static_cast<int>(c.size());
    span = std::max(span, across);
    total += rows ? static_cast<int>(c.size()) : static_cast<int>(c[0].size());
  }
  if (span > 30 || total > 30) return;
  CGrid out = rows ? CGrid(total, std::vector<int>(span, 0)) : CGrid(span, std::vector<int>(total, 0));
  int pos = 0;
  for (const CGrid& c : crops) {
    int h = static_cast<int>(c.size()), w = static_cast<int>(c[0].size());
    int off = ((rows ? span - w : span - h)) / 2;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (rows) out[pos + y][off + x] = c[y][x];
        else out[off + y][pos + x] = c[y][x];
      }
    pos += rows ? h : w;
  }
  concept_install(scene, out);
}

}  // namespace

// EXTRACT what=panels (concept_extract.cpp).
void extract_panels(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

}  // namespace arc
