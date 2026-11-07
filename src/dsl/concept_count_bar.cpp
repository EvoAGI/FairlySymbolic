// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COUNT_BAR: the output is a 1xK bar (or Kx1 column) whose length K is measured
// on the objects of the scene: their number, their colours, their cells, or the
// background cells they enclose. ConceptARC "Count".
//
// args: measure  objects | colors | mass | holes | minority (cells not of the most
//                frequent colour) | sides (straight strokes) | row_lengths / col_lengths
//                (distinct numbers of filled cells per row / column) | inside / outside
//                (objects without a hole inside / outside the bbox of one with a hole) |
//                per_frame (each object with a hole, left to right or top to bottom by
//                axis, gives its own colour once per cell of the objects inside its bbox)
//                | layers (objects, where the pieces of one colour that other colours
//                cut apart are one object: a shape seen through or under another)
//       color    bar colour, -1 = background
//       width    0 = exactly K cells, else padded with background to `width`
//       axis     h | v
//       diag     8-connected objects

#include <algorithm>
#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

// Background cells inside the object's bbox that cannot reach the bbox border.
int enclosed_cells(const std::vector<Coord>& comp) {
  Box b = bbox_of(comp);
  int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
  std::vector<std::vector<char>> obj(h, std::vector<char>(w, 0)), reach(h, std::vector<char>(w, 0));
  for (const Coord& c : comp) obj[c.first - b.y0][c.second - b.x0] = 1;
  std::vector<Coord> st;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !obj[y][x]) {
        reach[y][x] = 1;
        st.push_back({y, x});
      }
  const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
  while (!st.empty()) {
    Coord c = st.back();
    st.pop_back();
    for (int k = 0; k < 4; ++k) {
      int p = c.first + dy[k], q = c.second + dx[k];
      if (p < 0 || p >= h || q < 0 || q >= w || obj[p][q] || reach[p][q]) continue;
      reach[p][q] = 1;
      st.push_back({p, q});
    }
  }
  int n = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) n += !obj[y][x] && !reach[y][x];
  return n;
}

// Straight strokes of the drawing: maximal horizontal, vertical or diagonal runs
// of non-background cells at least 3 long (the sides of outlined polygons).
int straight_sides(const CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size()), n = 0;
  auto on = [&](int y, int x) { return y >= 0 && y < H && x >= 0 && x < W && g[y][x] != bg; };
  const int DY[4] = {0, 1, 1, 1}, DX[4] = {1, 0, 1, -1};
  for (int d = 0; d < 4; ++d)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (!on(y, x) || on(y - DY[d], x - DX[d])) continue;
        int len = 1;
        while (on(y + len * DY[d], x + len * DX[d])) ++len;
        n += len >= 3;
      }
  return n;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  std::string measure = arg_str(action.args, "measure", "objects");
  bool diag = as_bool(args_get(action.args, "diag"), false);
  auto comps = concept_components(g, bg, diag, true);
  int k = 0;
  std::vector<int> cells;  // per_frame: the bar's cells, colour by colour
  if (measure == "per_frame") {
    bool vert = arg_str(action.args, "axis", "h") == "v";
    std::vector<std::pair<Box, int>> frames;
    for (const auto& c : comps)
      if (enclosed_cells(c) > 0) frames.push_back({bbox_of(c), g[c[0].first][c[0].second]});
    std::sort(frames.begin(), frames.end(), [vert](const std::pair<Box, int>& a, const std::pair<Box, int>& b) {
      return vert ? std::make_pair(a.first.y0, a.first.x0) < std::make_pair(b.first.y0, b.first.x0)
                  : std::make_pair(a.first.x0, a.first.y0) < std::make_pair(b.first.x0, b.first.y0);
    });
    for (const auto& f : frames)
      for (const auto& c : comps) {
        if (enclosed_cells(c) > 0) continue;
        Box b = bbox_of(c);
        if (b.y0 > f.first.y0 && b.y1 < f.first.y1 && b.x0 > f.first.x0 && b.x1 < f.first.x1)
          cells.insert(cells.end(), c.size(), f.second);
      }
    k = static_cast<int>(cells.size());
  } else if (measure == "objects") {
    k = static_cast<int>(comps.size());
  } else if (measure == "colors") {
    std::set<int> cols;
    for (const auto& c : comps) cols.insert(g[c[0].first][c[0].second]);
    k = static_cast<int>(cols.size());
  } else if (measure == "mass") {
    for (const auto& c : comps) k += static_cast<int>(c.size());
  } else if (measure == "holes") {
    for (const auto& c : comps) k += enclosed_cells(c);
  } else if (measure == "minority") {
    std::map<int, int> n;
    for (const auto& row : g)
      for (int v : row)
        if (v != bg) n[v]++;
    int top = 0;
    for (const auto& kv : n) top = std::max(top, kv.second);
    for (const auto& kv : n) k += kv.second;
    k -= top;
  } else if (measure == "inside" || measure == "outside") {
    std::vector<Box> frames;
    for (const auto& c : comps)
      if (enclosed_cells(c) > 0) frames.push_back(bbox_of(c));
    if (frames.empty()) return;
    for (const auto& c : comps) {
      if (enclosed_cells(c) > 0) continue;
      Box b = bbox_of(c);
      bool in = std::any_of(frames.begin(), frames.end(), [&](const Box& f) {
        return b.y0 > f.y0 && b.y1 < f.y1 && b.x0 > f.x0 && b.x1 < f.x1;
      });
      k += in == (measure == "inside");
    }
  } else if (measure == "layers") {
    // Per colour, a flood from each of its cells through cells of any colour;
    // the pieces it joins count once.
    int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
    std::set<int> cols;
    for (const auto& c : comps) cols.insert(g[c[0].first][c[0].second]);
    for (int col : cols) {
      std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          if (g[y][x] != col || seen[y][x]) continue;
          ++k;
          std::vector<Coord> st{{y, x}};
          seen[y][x] = 1;
          while (!st.empty()) {
            Coord c = st.back();
            st.pop_back();
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if ((dy == 0 && dx == 0) || (!diag && dy != 0 && dx != 0)) continue;
                int p = c.first + dy, q = c.second + dx;
                if (p < 0 || p >= H || q < 0 || q >= W || seen[p][q] || g[p][q] == bg) continue;
                seen[p][q] = 1;
                st.push_back({p, q});
              }
          }
        }
    }
  } else if (measure == "sides") {
    k = straight_sides(g, bg);
  } else if (measure == "row_lengths" || measure == "col_lengths") {
    bool rows = measure == "row_lengths";
    std::set<int> lens;
    for (size_t i = 0; i < (rows ? g.size() : g[0].size()); ++i) {
      int len = 0;
      for (size_t j = 0; j < (rows ? g[0].size() : g.size()); ++j) len += (rows ? g[i][j] : g[j][i]) != bg;
      if (len) lens.insert(len);
    }
    k = static_cast<int>(lens.size());
  } else {
    return;
  }
  if (k < 1 || k > 30) return;
  int width = arg_int(action.args, "width", 0);
  int n = std::max(k, std::min(width, 30));
  int color = arg_int(action.args, "color", -1);
  if (color < 0 || color > 9) color = bg;
  std::vector<int> bar(n, bg);
  for (int i = 0; i < k; ++i) bar[i] = cells.empty() ? color : cells[i];
  CGrid out;
  if (arg_str(action.args, "axis", "h") == "v") {
    for (int v : bar) out.push_back({v});
  } else {
    out.push_back(bar);
  }
  concept_install(scene, out);
}

DslArgs make(const std::string& measure, int color, int width, const std::string& axis, bool diag) {
  return {{"measure", cdv_str(measure)}, {"color", cdv_int(color)}, {"width", cdv_int(width)},
          {"axis", cdv_str(axis)}, {"diag", cdv_bool(diag)}};
}

DslArgs sample(ArcRandom& rng) {
  std::string measure = pick_str(rng, {"objects", "colors", "mass", "holes", "minority", "sides", "row_lengths", "col_lengths", "inside", "outside", "per_frame", "layers"});
  int color = pick_int(rng, {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  int width = pick_int(rng, {0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::string axis = pick_str(rng, {"h", "v"});
  bool diag = pick_int(rng, {0, 1}) == 1;
  return make(measure, color, width, axis, diag);
}

// Seeds only when every train output is a single row (or column) with at most
// one non-background colour: colour and padding come from the outputs.
std::vector<DslArgs> seeds(const std::vector<Grid>&, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (outs.empty()) return res;
  bool rows = true, cols = true;
  std::set<int> colors, widths;
  for (const Grid& o : outs) {
    rows = rows && o.height() == 1;
    cols = cols && o.width() == 1;
    widths.insert(o.height() == 1 ? o.width() : o.height());
    for (int v : o.cells())
      if (v != 0) colors.insert(v);
  }
  if ((!rows && !cols) || colors.size() > 1) return res;
  std::vector<int> cs = {-1};
  if (!colors.empty()) cs = {*colors.begin()};
  std::vector<int> ws = {0};
  if (widths.size() == 1) ws.push_back(*widths.begin());
  for (const char* m : {"objects", "colors", "mass", "holes", "minority", "sides", "row_lengths", "col_lengths", "inside", "outside", "per_frame", "layers"})
    for (int c : cs)
      for (int w : ws)
        for (bool d : {false, true}) res.push_back(make(m, c, w, rows ? "h" : "v", d));
  return res;
}

}  // namespace

ConceptOp count_bar_op() { return {ActionKind::COUNT_BAR, 2, apply, sample, seeds}; }

}  // namespace arc
