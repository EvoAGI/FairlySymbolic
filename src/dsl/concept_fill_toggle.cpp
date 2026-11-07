// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// TOGGLE_FILL: hollow objects are filled and solid objects are hollowed.
// ConceptARC "FilledNotFilled".
//
// args: conn   4 | 8  connectivity of the single-colour objects
//       shape  box    an object is hollow when its bbox holds background cells
//                     that cannot reach the bbox border (4-connected); they take
//                     the object's colour. A solid object keeps only the cells
//                     with a 4-neighbour outside it, when that leaves an interior.
//              lines  a colour drawn as two parallel straight lines (rows,
//                     columns or diagonals) of one length is filled into the
//                     band they bound (the lines, the two segments joining their
//                     ends, and what those enclose); a colour drawn as one such
//                     solid band keeps only its two long sides

#include <algorithm>
#include <cstdlib>
#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(int conn, const std::string& shape = "box") {
  DslArgs a;
  a.emplace_back("conn", cdv_int(conn));
  if (shape != "box") a.emplace_back("shape", cdv_str(shape));
  return a;
}

const int kDy[4] = {-1, 1, 0, 0}, kDx[4] = {0, 0, -1, 1};

// A straight line: n cells from `start` (its first cell in reading order) in
// step (dy, dx), one of the four directions that go down or right.
struct Line { Coord start; int dy, dx, n; };

const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

// cells (sorted) as a line along direction k, when they are one.
bool as_line(const std::vector<Coord>& cells, int k, Line* l) {
  int n = static_cast<int>(cells.size());
  if (n < 2) return false;
  *l = {cells[0], kDirs[k][0], kDirs[k][1], n};
  for (int i = 0; i < n; ++i)
    if (cells[i] != Coord(cells[0].first + i * l->dy, cells[0].second + i * l->dx)) return false;
  return true;
}

int sgn(int v) { return (v > 0) - (v < 0); }

// The band between two parallel lines of one length: the lines, the straight
// segments joining their first and their last cells, and the cells those
// enclose (4-connected). Empty when the ends cannot be joined straight.
std::vector<Coord> band(const Line& a, const Line& b) {
  int oy = b.start.first - a.start.first, ox = b.start.second - a.start.second;
  if ((oy && ox && std::abs(oy) != std::abs(ox)) || (oy == 0 && ox == 0)) return {};
  if (sgn(oy) * a.dx == sgn(ox) * a.dy) return {};  // collinear
  std::vector<Coord> wall;
  for (int i = 0; i < a.n; ++i) {
    wall.push_back({a.start.first + i * a.dy, a.start.second + i * a.dx});
    wall.push_back({b.start.first + i * a.dy, b.start.second + i * a.dx});
  }
  int steps = std::max(std::abs(oy), std::abs(ox));
  for (int e : {0, a.n - 1})
    for (int s = 1; s < steps; ++s)
      wall.push_back({a.start.first + e * a.dy + s * sgn(oy), a.start.second + e * a.dx + s * sgn(ox)});
  Box bb = bbox_of(wall);
  int h = bb.y1 - bb.y0 + 1, w = bb.x1 - bb.x0 + 1;
  std::vector<std::vector<char>> on(h, std::vector<char>(w, 0)), reach(h, std::vector<char>(w, 0));
  for (const Coord& c : wall) on[c.first - bb.y0][c.second - bb.x0] = 1;
  std::vector<Coord> st;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !on[y][x]) {
        reach[y][x] = 1;
        st.push_back({y, x});
      }
  while (!st.empty()) {
    Coord c = st.back();
    st.pop_back();
    for (int k = 0; k < 4; ++k) {
      int p = c.first + kDy[k], q = c.second + kDx[k];
      if (p < 0 || p >= h || q < 0 || q >= w || on[p][q] || reach[p][q]) continue;
      reach[p][q] = 1;
      st.push_back({p, q});
    }
  }
  std::vector<Coord> res;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (!reach[y][x]) res.push_back({bb.y0 + y, bb.x0 + x});
  return res;
}

// The two long sides of a solid band: the first and last lines of the cells
// along the direction that makes them longest, when filling them back gives
// the cells again. False when no direction does.
bool long_sides(const std::vector<Coord>& comp, Line* a, Line* b) {
  std::vector<Coord> sorted = comp;
  std::sort(sorted.begin(), sorted.end());
  int best = 0;
  for (int k = 0; k < 4; ++k) {
    std::map<int, std::vector<Coord>> by_key;  // cells on each line along k
    for (const Coord& c : sorted) {
      int key = k == 0 ? c.first : k == 1 ? c.second : k == 2 ? c.first - c.second : c.first + c.second;
      by_key[key].push_back(c);
    }
    Line la, lb;
    if (by_key.size() < 3 || !as_line(by_key.begin()->second, k, &la) || !as_line(by_key.rbegin()->second, k, &lb) ||
        la.n != lb.n || la.n <= best)
      continue;
    std::vector<Coord> f = band(la, lb);
    std::sort(f.begin(), f.end());
    if (f != sorted) continue;
    best = la.n;
    *a = la;
    *b = lb;
  }
  return best > 0;
}

void toggle_lines(const CGrid& g, int bg, bool diag, CGrid* out) {
  std::map<int, std::vector<std::vector<Coord>>> by_colour;
  for (auto& comp : concept_components(g, bg, diag, true)) {
    std::sort(comp.begin(), comp.end());
    by_colour[g[comp[0].first][comp[0].second]].push_back(std::move(comp));
  }
  for (const auto& kv : by_colour) {
    const auto& comps = kv.second;
    if (comps.size() == 2) {
      for (int k = 0; k < 4; ++k) {
        Line a, b;
        if (!as_line(comps[0], k, &a) || !as_line(comps[1], k, &b) || a.n != b.n) continue;
        for (const Coord& c : band(a, b))
          if (g[c.first][c.second] == bg) (*out)[c.first][c.second] = kv.first;
        break;
      }
    } else if (comps.size() == 1) {
      Line a, b;
      if (!long_sides(comps[0], &a, &b)) continue;
      for (const Coord& c : comps[0]) (*out)[c.first][c.second] = bg;
      for (int i = 0; i < a.n; ++i) {
        (*out)[a.start.first + i * a.dy][a.start.second + i * a.dx] = kv.first;
        (*out)[b.start.first + i * a.dy][b.start.second + i * a.dx] = kv.first;
      }
    }
  }
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  bool diag = arg_int(action.args, "conn", 8) == 8;
  CGrid out = g;
  if (arg_str(action.args, "shape", "box") == "lines") {
    toggle_lines(g, bg, diag, &out);
    concept_install(scene, out);
    return;
  }
  const int (&dy)[4] = kDy, (&dx)[4] = kDx;
  for (const auto& comp : concept_components(g, bg, diag, true)) {
    Box b = bbox_of(comp);
    int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1, color = g[comp[0].first][comp[0].second];
    std::vector<std::vector<char>> obj(h, std::vector<char>(w, 0)), reach(h, std::vector<char>(w, 0));
    for (const Coord& c : comp) obj[c.first - b.y0][c.second - b.x0] = 1;
    std::vector<Coord> st;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !obj[y][x]) {
          reach[y][x] = 1;
          st.push_back({y, x});
        }
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
    bool hollow = false;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (!obj[y][x] && !reach[y][x]) {
          hollow = true;
          if (g[b.y0 + y][b.x0 + x] == bg) out[b.y0 + y][b.x0 + x] = color;
        }
    if (hollow) continue;
    std::vector<Coord> inner;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (!obj[y][x]) continue;
        bool in = true;
        for (int k = 0; k < 4 && in; ++k) {
          int p = y + dy[k], q = x + dx[k];
          in = p >= 0 && p < h && q >= 0 && q < w && obj[p][q];
        }
        if (in) inner.push_back({b.y0 + y, b.x0 + x});
      }
    for (const Coord& c : inner) out[c.first][c.second] = bg;
  }
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) {
  int conn = pick_int(rng, {4, 8});
  return make(conn, pick_str(rng, {"box", "lines"}));
}

// Seeds only when every train output has the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  res.push_back(make(8));
  res.push_back(make(4));
  res.push_back(make(8, "lines"));
  return res;
}

}  // namespace

ConceptOp toggle_fill_op() { return {ActionKind::TOGGLE_FILL, 2, apply, sample, seeds}; }

}  // namespace arc
