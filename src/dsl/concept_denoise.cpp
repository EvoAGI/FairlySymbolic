// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DENOISE: noise cells are put back to what surrounds them. ConceptARC "CleanUp".
//
// args: mode  row      a cell takes the majority colour of its row, when that
//                      colour holds at least 60 % of the row
//             col      the same along the column
//             line     row majority if the row is the more uniform line, else column
//             isolated cells with no 8-neighbour of their colour, of a colour whose
//                      cells are mostly isolated (>= 80 %), take the majority colour
//                      of their 4-neighbours (background when there is none)
//             periodic the grid is a tiling with period (py, px), chosen as the one
//                      that changes the fewest cells among those where at least
//                      half the cells have a repeat; each cell takes the
//                      majority of its position (ties go to the background)
//             fill     noise pieces (1-2 cells of a colour that comes mostly in
//                      such pieces) take the commonest colour around them, then a
//                      background cell with 3 of its 4 neighbours of one colour
//                      takes that colour (holes close)
//             frames   only rectangular frames are kept, redrawn whole: a frame is
//                      a single-colour group (at least 3x3) with nearly all its
//                      cells on its bbox border, covering 3/4 of that border
//             lines    only full lines are kept: rows and columns with no
//                      background cell (grid lines the encoder set apart count);
//                      when there is none, diagonals of 3 cells or more, edge to
//                      edge, with no background cell
//             copies   the shapes are copies of one another (also mirrored or
//                      turned) with noise over some cells: a shape is a one-colour
//                      8-connected group of 4 cells or more and at least a quarter
//                      of the biggest, every other cell is noise. A copy laid on a
//                      shape (2/3 of each on the other, and all of it on the shape
//                      or on noise; a shape is also a copy of itself) gives the
//                      shape the noise cells it covers; the remaining noise clears
//             rings    frames painted over by noise are kept, redrawn whole: the
//                      cells of a colour at most two apart form a group; an edge
//                      row or column of its bbox holding under a third of its
//                      length in the group's cells is cut off (noise of the frame's
//                      colour beside it), and what is left is a frame when it is
//                      3x3 or more, every group cell in it lies on its border and
//                      covers half of that border or more
//             squares  only solid squares are kept, redrawn whole: a square is a
//                      single-colour 8-connected group whose bbox, with end rows
//                      and columns holding one of its cells cut off (down to 2),
//                      then cut down to a square by dropping the emptier end rows
//                      (or columns) of its longer side, is 2x2 or more and 2/3
//                      covered by it
// The grid keeps its size; background is the scene's.

#include <algorithm>
#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kModes = {"row",    "col",   "line",  "isolated", "periodic", "frames",
                                         "fill",   "lines", "copies", "rings",   "squares"};

DslArgs make(const std::string& mode) {
  DslArgs a;
  a.emplace_back("mode", cdv_str(mode));
  return a;
}

// Majority value of v and its share; -1 when v is empty.
// The least value among the most frequent ones (ascending scan, strict >).
std::pair<int, double> majority(const std::vector<int>& v) {
  int best = -1, cnt = 0;
  bool small = true;  // all values in [0, 16): count in an array
  for (int c : v) small = small && c >= 0 && c < 16;
  if (small) {
    int n[16] = {0};
    for (int c : v) n[c]++;
    for (int c = 0; c < 16; ++c)
      if (n[c] > cnt) { best = c; cnt = n[c]; }
  } else {
    std::map<int, int> n;
    for (int c : v) n[c]++;
    for (const auto& kv : n)
      if (kv.second > cnt) { best = kv.first; cnt = kv.second; }
  }
  return {best, v.empty() ? 0.0 : static_cast<double>(cnt) / v.size()};
}

CGrid full_lines(const CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid out(H, std::vector<int>(W, bg));
  const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  bool any = false;
  for (int k = 0; k < 4; ++k) {
    const int* d = dirs[k];
    if (k == 2 && any) break;  // diagonals only when no row or column is full
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int py = y - d[0], px = x - d[1];
        if (py >= 0 && py < H && px >= 0 && px < W) continue;  // not a line start
        std::vector<Coord> line;
        int p = y, q = x;
        for (; p >= 0 && p < H && q >= 0 && q < W && g[p][q] != bg; p += d[0], q += d[1]) line.push_back({p, q});
        if (p >= 0 && p < H && q >= 0 && q < W) continue;  // stopped on a background cell
        if (line.size() < 3) continue;
        for (const Coord& c : line) out[c.first][c.second] = g[c.first][c.second];
        any = true;
      }
  }
  return out;
}

CGrid restore_copies(const CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  auto comps = concept_components(g, bg, true, true);
  size_t big = 0;
  for (const auto& c : comps) big = std::max(big, c.size());
  std::vector<std::vector<Coord>> shapes;
  std::vector<std::vector<int>> owner(H, std::vector<int>(W, -1));  // shape index, -2 noise
  for (const auto& c : comps) {
    bool shape = c.size() >= 4 && 4 * c.size() >= big;
    for (const Coord& e : c) owner[e.first][e.second] = shape ? static_cast<int>(shapes.size()) : -2;
    if (shape) shapes.push_back(c);
  }
  CGrid out(H, std::vector<int>(W, bg));
  for (const auto& sh : shapes)
    for (const Coord& e : sh) out[e.first][e.second] = g[e.first][e.second];
  if (shapes.size() > 12) return out;
  // Offsets s - t(c) lie in [-2 * 30, 2 * 30]; votes is a dense table over them.
  const int R = 64, N = 2 * R + 1;
  std::vector<int> votes(N * N, 0);
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& S = shapes[i];
    int col = g[S[0].first][S[0].second];
    for (const auto& T : shapes) {
      if (S.size() * T.size() > 40000) continue;
      for (int t = 0; t < 8; ++t) {
        // Turn t: bit 0 negates x, bit 1 negates y, bit 2 swaps them first.
        std::vector<Coord> tc;
        for (const Coord& e : T) {
          int a = (t & 4) ? e.second : e.first, b = (t & 4) ? e.first : e.second;
          tc.push_back({(t & 2) ? -a : a, (t & 1) ? -b : b});
        }
        std::vector<int> used;
        int top = 0;
        for (const Coord& e : S)
          for (const Coord& c : tc) {
            int k = (e.first - c.first + R) * N + (e.second - c.second + R);
            if (votes[k]++ == 0) used.push_back(k);
            top = std::max(top, votes[k]);
          }
        std::vector<Coord> best;
        for (int k : used) {
          if (votes[k] == top) best.push_back({k / N - R, k % N - R});
          votes[k] = 0;
        }
        if (3 * top < 2 * static_cast<int>(S.size()) || 3 * top < 2 * static_cast<int>(T.size())) continue;
        std::sort(best.begin(), best.end());
        for (const Coord& o : best) {
          std::vector<Coord> add;
          bool ok = true;
          for (const Coord& c : tc) {
            int y = c.first + o.first, x = c.second + o.second;
            if (y < 0 || y >= H || x < 0 || x >= W || (owner[y][x] != static_cast<int>(i) && owner[y][x] != -2)) {
              ok = false;
              break;
            }
            if (owner[y][x] == -2) add.push_back({y, x});
          }
          if (!ok) continue;
          for (const Coord& e : add)
            if (out[e.first][e.second] == bg) out[e.first][e.second] = col;
        }
      }
    }
  }
  return out;
}

// mode=squares; false when no square is found.
bool squares(const CGrid& g, int bg, CGrid& out) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  out.assign(H, std::vector<int>(W, bg));
  bool any = false;
  for (const auto& comp : concept_components(g, bg, true, true)) {
    Box b = bbox_of(comp);
    auto count = [&](bool row, int i) {
      int n = 0;
      for (const Coord& c : comp)
        n += row ? c.first == i && c.second >= b.x0 && c.second <= b.x1
                 : c.second == i && c.first >= b.y0 && c.first <= b.y1;
      return n;
    };
    for (;;) {
      bool tall = b.y1 - b.y0 >= 2, wide = b.x1 - b.x0 >= 2;
      if (tall && count(true, b.y0) <= 1) ++b.y0;
      else if (tall && count(true, b.y1) <= 1) --b.y1;
      else if (wide && count(false, b.x0) <= 1) ++b.x0;
      else if (wide && count(false, b.x1) <= 1) --b.x1;
      else break;
    }
    while (b.y1 - b.y0 != b.x1 - b.x0) {
      if (b.y1 - b.y0 > b.x1 - b.x0) {
        if (count(true, b.y0) < count(true, b.y1)) ++b.y0;
        else --b.y1;
      } else {
        if (count(false, b.x0) < count(false, b.x1)) ++b.x0;
        else --b.x1;
      }
    }
    int k = b.y1 - b.y0 + 1, in = 0;
    for (const Coord& c : comp) in += c.first >= b.y0 && c.first <= b.y1 && c.second >= b.x0 && c.second <= b.x1;
    if (k < 2 || 3 * in < 2 * k * k) continue;
    int col = g[comp[0].first][comp[0].second];
    for (int y = b.y0; y <= b.y1; ++y)
      for (int x = b.x0; x <= b.x1; ++x) out[y][x] = col;
    any = true;
  }
  return any;
}

// mode=rings; false when no frame is found.
bool rings(const CGrid& g, int bg, CGrid& out) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  out.assign(H, std::vector<int>(W, bg));
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  bool any = false;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int col = g[y][x];
      if (col == bg || seen[y][x]) continue;
      std::vector<Coord> st{{y, x}}, grp;
      seen[y][x] = 1;
      while (!st.empty()) {
        Coord c = st.back();
        st.pop_back();
        grp.push_back(c);
        for (int p = std::max(0, c.first - 2); p <= std::min(H - 1, c.first + 2); ++p)
          for (int q = std::max(0, c.second - 2); q <= std::min(W - 1, c.second + 2); ++q)
            if (!seen[p][q] && g[p][q] == col) {
              seen[p][q] = 1;
              st.push_back({p, q});
            }
      }
      Box b = bbox_of(grp);
      auto count = [&](bool row, int i) {
        int n = 0;
        for (const Coord& c : grp)
          n += (row ? c.first == i && c.second >= b.x0 && c.second <= b.x1
                    : c.second == i && c.first >= b.y0 && c.first <= b.y1);
        return n;
      };
      for (bool cut = true; cut && b.y1 - b.y0 >= 2 && b.x1 - b.x0 >= 2;) {
        int w = b.x1 - b.x0 + 1, h = b.y1 - b.y0 + 1;
        if (3 * count(true, b.y0) < w) ++b.y0;
        else if (3 * count(true, b.y1) < w) --b.y1;
        else if (3 * count(false, b.x0) < h) ++b.x0;
        else if (3 * count(false, b.x1) < h) --b.x1;
        else cut = false;
      }
      int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
      if (h < 3 || w < 3) continue;
      int on = 0;
      bool inner = false;
      for (const Coord& c : grp) {
        if (c.first < b.y0 || c.first > b.y1 || c.second < b.x0 || c.second > b.x1) continue;
        bool edge = c.first == b.y0 || c.first == b.y1 || c.second == b.x0 || c.second == b.x1;
        on += edge;
        inner = inner || !edge;
      }
      if (inner || 2 * on < 2 * (h + w) - 4) continue;
      for (int p = b.y0; p <= b.y1; ++p)
        for (int q = b.x0; q <= b.x1; ++q)
          if (p == b.y0 || p == b.y1 || q == b.x0 || q == b.x1) out[p][q] = col;
      any = true;
    }
  return any;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  std::string mode = arg_str(action.args, "mode", "row");
  CGrid g = mode == "lines" ? concept_render_full(scene) : concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid out = g;
  if (mode == "lines") {
    out = full_lines(g, bg);
  } else if (mode == "copies") {
    out = restore_copies(g, bg);
  } else if (mode == "rings") {
    if (!rings(g, bg, out)) return;
  } else if (mode == "squares") {
    if (!squares(g, bg, out)) return;
  } else if (mode == "frames") {
    CGrid res(H, std::vector<int>(W, bg));
    bool any = false;
    for (const auto& comp : concept_components(g, bg, true, true)) {
      Box b = bbox_of(comp);
      int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
      if (h < 3 || w < 3) continue;
      int on = 0;
      for (const Coord& c : comp) on += c.first == b.y0 || c.first == b.y1 || c.second == b.x0 || c.second == b.x1;
      int border = 2 * (h + w) - 4, off = static_cast<int>(comp.size()) - on;
      if (4 * on < 3 * border || 10 * off > static_cast<int>(comp.size())) continue;
      int col = g[comp[0].first][comp[0].second];
      for (int y = b.y0; y <= b.y1; ++y)
        for (int x = b.x0; x <= b.x1; ++x)
          if (y == b.y0 || y == b.y1 || x == b.x0 || x == b.x1) res[y][x] = col;
      any = true;
    }
    if (!any) return;
    out = res;
  } else if (mode == "periodic") {
    long best = -1;
    for (int py = 1; py <= H; ++py)
      for (int px = 1; px <= W; ++px) {
        // Cells whose position has no repeat: y + py >= H and x + px >= W.
        long lone = static_cast<long>(std::max(0, 2 * py - H)) * std::max(0, 2 * px - W);
        if (2 * lone > static_cast<long>(H) * W) continue;
        CGrid cand = g;
        long changes = 0;
        for (int ry = 0; ry < py; ++ry)
          for (int rx = 0; rx < px; ++rx) {
            std::map<int, int> n;
            for (int y = ry; y < H; y += py)
              for (int x = rx; x < W; x += px) n[g[y][x]]++;
            int top = bg, cnt = n.count(bg) ? n[bg] : 0;
            for (const auto& kv : n)
              if (kv.second > cnt) { top = kv.first; cnt = kv.second; }
            for (int y = ry; y < H; y += py)
              for (int x = rx; x < W; x += px)
                if (g[y][x] != top) { cand[y][x] = top; ++changes; }
          }
        if (best < 0 || changes < best) { best = changes; out = cand; }
      }
  } else if (mode == "fill") {
    // Noise: 4-connected pieces of at most 2 cells, of a colour that comes mostly
    // (>= 80 %) in such pieces. Each piece takes the commonest colour around it
    // (a tie with the background goes to an object whose bbox holds the piece).
    std::vector<std::vector<int>> piece(H, std::vector<int>(W, -1));
    std::vector<std::vector<Coord>> pieces;
    const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (g[y][x] == bg || piece[y][x] >= 0) continue;
        std::vector<Coord> st{{y, x}}, cells;
        piece[y][x] = static_cast<int>(pieces.size());
        while (!st.empty()) {
          Coord c = st.back();
          st.pop_back();
          cells.push_back(c);
          for (int k = 0; k < 4; ++k) {
            int p = c.first + dy[k], q = c.second + dx[k];
            if (p < 0 || p >= H || q < 0 || q >= W || piece[p][q] >= 0 || g[p][q] != g[y][x]) continue;
            piece[p][q] = piece[y][x];
            st.push_back({p, q});
          }
        }
        pieces.push_back(cells);
      }
    std::map<int, std::pair<int, int>> stat;  // colour -> (cells in small pieces, cells)
    for (const auto& pc : pieces) {
      auto& st2 = stat[g[pc[0].first][pc[0].second]];
      if (pc.size() <= 2) st2.first += static_cast<int>(pc.size());
      st2.second += static_cast<int>(pc.size());
    }
    for (const auto& pc : pieces) {
      int c = g[pc[0].first][pc[0].second];
      if (pc.size() > 2 || stat[c].first * 5 < stat[c].second * 4) continue;
      std::map<int, int> n;
      std::set<int> holds;  // colours of a neighbouring piece whose bbox holds this piece
      for (const Coord& e : pc)
        for (int k = 0; k < 4; ++k) {
          int p = e.first + dy[k], q = e.second + dx[k];
          if (p < 0 || p >= H || q < 0 || q >= W || g[p][q] == c) continue;
          n[g[p][q]]++;
          if (g[p][q] == bg) continue;
          Box b = bbox_of(pieces[piece[p][q]]);
          if (e.first >= b.y0 && e.first <= b.y1 && e.second >= b.x0 && e.second <= b.x1) holds.insert(g[p][q]);
        }
      int best = bg, bn = n.count(bg) ? n[bg] : 0;
      for (const auto& kv : n)
        if (kv.first != bg && (kv.second > bn || (kv.second == bn && holds.count(kv.first)))) { best = kv.first; bn = kv.second; }
      for (const Coord& e : pc) out[e.first][e.second] = best;
    }
    // Then a background cell with at least 3 of its 4 neighbours of one colour
    // takes it (holes and nicks close; concave corners stay).
    CGrid g2 = out;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (g2[y][x] != bg) continue;
        std::map<int, int> n;
        for (int k = 0; k < 4; ++k) {
          int p = y + dy[k], q = x + dx[k];
          if (p >= 0 && p < H && q >= 0 && q < W && g2[p][q] != bg) n[g2[p][q]]++;
        }
        for (const auto& kv : n)
          if (kv.second >= 3) out[y][x] = kv.first;
      }
  } else if (mode == "isolated") {
    auto isolated = [&](int y, int x) {
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          int p = y + dy, q = x + dx;
          if ((dy || dx) && p >= 0 && p < H && q >= 0 && q < W && g[p][q] == g[y][x]) return false;
        }
      return true;
    };
    std::map<int, std::pair<int, int>> stat;  // colour -> (isolated, total)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (g[y][x] == bg) continue;
        auto& s = stat[g[y][x]];
        s.first += isolated(y, x);
        s.second++;
      }
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int c = g[y][x];
        if (c == bg || !isolated(y, x) || stat[c].first * 5 < stat[c].second * 4) continue;
        std::vector<int> nb;
        const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
          int p = y + dy[k], q = x + dx[k];
          if (p >= 0 && p < H && q >= 0 && q < W && g[p][q] != c) nb.push_back(g[p][q]);
        }
        out[y][x] = nb.empty() ? bg : majority(nb).first;
      }
  } else {
    std::vector<std::pair<int, double>> rmaj(H), cmaj(W);
    for (int y = 0; y < H; ++y) rmaj[y] = majority(g[y]);
    for (int x = 0; x < W; ++x) {
      std::vector<int> col(H);
      for (int y = 0; y < H; ++y) col[y] = g[y][x];
      cmaj[x] = majority(col);
    }
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        bool use_row = mode == "row" || (mode == "line" && rmaj[y].second >= cmaj[x].second);
        const auto& m = use_row ? rmaj[y] : cmaj[x];
        if (m.second >= 0.6) out[y][x] = m.first;
      }
  }
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) { return make(pick_str(rng, kModes)); }

// Seeds only when every train output has the input's size and differs from it
// in fewer than a fifth of the cells (noise is sparse).
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    const auto& a = ins[i].cells();
    const auto& b = outs[i].cells();
    size_t diff = 0;
    for (size_t k = 0; k < a.size(); ++k) diff += a[k] != b[k];
    if (diff == 0 || diff * 5 >= a.size()) return res;
  }
  for (const std::string& m : kModes) res.push_back(make(m));
  return res;
}

}  // namespace

ConceptOp denoise_op() { return {ActionKind::DENOISE, 2, apply, sample, seeds}; }

}  // namespace arc
