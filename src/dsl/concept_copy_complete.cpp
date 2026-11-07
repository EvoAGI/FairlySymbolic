// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COMPLETE_COPIES: one group of cells is a template, and the other cells are
// pieces of copies of it; every copy is drawn whole. ConceptARC "Copy".
//
// args: gap   0 | 1  (cells closer than gap + 2 in both axes are one group; with
//                     gap 1 a template may hold blank rows or columns)
//       paint template (default): a copy is placed where template cells land on
//                     piece cells of the same colour (two at least) and every
//                     piece cell it covers has the template's colour
//             piece   only the template's shape counts: groups are one colour
//                     each, a copy is placed where it covers two piece cells at
//                     least, all of one colour, and is drawn in that colour; the
//                     template itself is erased. Among equally good places,
//                     those that put the template's first cell on a piece cell
//                     go first
//       pick  biggest (default): the template is the biggest group (no tie)
//             best    the group whose copies explain the pieces best: each copy
//                     scores the piece cells it is the first to cover, less the
//                     two that place it (ties go to the bigger group, then the
//                     first)
//       hide  none (default) | rect: a colour whose objects (8-connected) are all
//                     solid rectangles, 2x2 or more, hides parts of the copies;
//                     it is cleared first (CleanUp: blocks painted over a
//                     pattern)

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(int gap, const std::string& paint = "template", const std::string& pick = "biggest",
             const std::string& hide = "none") {
  DslArgs a;
  a.emplace_back("gap", cdv_int(gap));
  a.emplace_back("paint", cdv_str(paint));
  a.emplace_back("pick", cdv_str(pick));
  if (hide != "none") a.emplace_back("hide", cdv_str(hide));
  return a;
}

// hide=rect: clears every colour whose objects are all solid rectangles of 2x2
// or more, unless that would clear everything; false when nothing is cleared.
bool clear_rects(CGrid& g, int bg) {
  std::map<int, bool> solid;
  for (const auto& comp : concept_components(g, bg, true, true)) {
    Box b = bbox_of(comp);
    int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
    bool r = h >= 2 && w >= 2 && static_cast<int>(comp.size()) == h * w;
    int c = g[comp[0].first][comp[0].second];
    solid[c] = (solid.count(c) ? solid[c] : true) && r;
  }
  bool any = false, rest = false;
  for (const auto& kv : solid) (kv.second ? any : rest) = true;
  if (!any || !rest) return false;
  for (auto& row : g)
    for (int& v : row)
      if (v != bg && solid[v]) v = bg;
  return true;
}

// mono: a group holds one colour only.
std::vector<std::vector<Coord>> groups(const CGrid& g, int bg, int gap, bool mono) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  std::vector<std::vector<Coord>> res;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] == bg || seen[y][x]) continue;
      std::vector<Coord> st{{y, x}}, cs;
      seen[y][x] = 1;
      while (!st.empty()) {
        Coord c = st.back();
        st.pop_back();
        cs.push_back(c);
        for (int dy = -gap - 1; dy <= gap + 1; ++dy)
          for (int dx = -gap - 1; dx <= gap + 1; ++dx) {
            int p = c.first + dy, q = c.second + dx;
            if (p < 0 || p >= H || q < 0 || q >= W || seen[p][q] || g[p][q] == bg) continue;
            if (mono && g[p][q] != g[y][x]) continue;
            seen[p][q] = 1;
            st.push_back({p, q});
          }
      }
      res.push_back(cs);
    }
  return res;
}

// Draws the copies of group t into out (a copy of g); returns the piece cells
// covered. score: the sum over the copies of the piece cells each covers first, less 2.
size_t place(const CGrid& g, int bg, const std::vector<std::vector<Coord>>& gs, size_t t, bool piece,
             CGrid& out, int* score = nullptr) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  const std::vector<Coord>& tpl = gs[t];
  Coord first = *std::min_element(tpl.begin(), tpl.end());
  // rest: the cells of the other groups (a set: flags over the grid; the
  // group cells lie in it). votes per offset (dy, dx) in a dense table over
  // [-(H-1), H-1] x [-(W-1), W-1], read back in ascending (dy, dx) order.
  std::vector<char> rest(static_cast<size_t>(H) * W, 0);
  auto in_rest = [&](int y, int x) { return rest[static_cast<size_t>(y) * W + x] != 0; };
  for (size_t i = 0; i < gs.size(); ++i)
    if (i != t)
      for (const Coord& c : gs[i]) rest[static_cast<size_t>(c.first) * W + c.second] = 1;
  const int VW = 2 * W - 1;
  std::vector<int> votes(static_cast<size_t>(2 * H - 1) * VW, 0);
  for (int ry = 0; ry < H; ++ry)
    for (int rx = 0; rx < W; ++rx) {
      if (!in_rest(ry, rx)) continue;
      for (const Coord& c : tpl)
        if (piece || g[c.first][c.second] == g[ry][rx])
          votes[static_cast<size_t>(ry - c.first + H - 1) * VW + (rx - c.second + W - 1)]++;
    }
  // (-votes, first cell off the pieces, offset); the middle key only in piece mode.
  std::vector<std::tuple<int, int, Coord>> order;
  for (int vy = 0; vy < 2 * H - 1; ++vy)
    for (int vx = 0; vx < VW; ++vx) {
      const int n = votes[static_cast<size_t>(vy) * VW + vx];
      if (n < 2) continue;
      const Coord off{vy - (H - 1), vx - (W - 1)};
      const int fy = first.first + off.first, fx = first.second + off.second;
      const bool f_in_rest = fy >= 0 && fy < H && fx >= 0 && fx < W && in_rest(fy, fx);
      order.emplace_back(-n, piece && !f_in_rest ? 1 : 0, off);
    }
  std::sort(order.begin(), order.end());
  out = g;
  if (piece)
    for (const Coord& c : tpl) out[c.first][c.second] = bg;
  std::vector<char> covered(static_cast<size_t>(H) * W, 0);
  size_t n_covered = 0;
  if (score) *score = 0;
  for (const auto& o : order) {
    int dy = std::get<2>(o).first, dx = std::get<2>(o).second;
    bool ok = true, fresh = false;
    int colour = -1;
    std::vector<Coord> hit;
    for (const Coord& c : tpl) {
      int y = c.first + dy, x = c.second + dx;
      if (y < 0 || y >= H || x < 0 || x >= W || !in_rest(y, x)) continue;
      int want = !piece ? g[c.first][c.second] : colour < 0 ? g[y][x] : colour;
      if (g[y][x] != want) { ok = false; break; }
      colour = want;
      hit.push_back({y, x});
      fresh = fresh || !covered[static_cast<size_t>(y) * W + x];
    }
    if (!ok || !fresh) continue;
    size_t before = n_covered;
    for (const Coord& h : hit) {
      char& cv = covered[static_cast<size_t>(h.first) * W + h.second];
      if (!cv) { cv = 1; ++n_covered; }
    }
    if (score) *score += static_cast<int>(n_covered - before) - 2;
    for (const Coord& c : tpl) {
      int y = c.first + dy, x = c.second + dx;
      if (y >= 0 && y < H && x >= 0 && x < W && out[y][x] == bg) out[y][x] = piece ? colour : g[c.first][c.second];
    }
  }
  return n_covered;
}

bool complete(const CGrid& g, int bg, int gap, bool piece, bool best, CGrid& out) {
  auto gs = groups(g, bg, gap, piece);
  if (gs.size() < 2) return false;
  if (best) {
    if (gs.size() > 40) return false;
    int top = -1;
    size_t top_size = 0;
    for (size_t t = 0; t < gs.size(); ++t) {
      CGrid cand;
      int sc = 0;
      if (place(g, bg, gs, t, piece, cand, &sc) == 0) continue;
      if (sc > top || (sc == top && gs[t].size() > top_size)) {
        top = sc;
        top_size = gs[t].size();
        out = cand;
      }
    }
    return top >= 0;
  }
  size_t t = 0;
  for (size_t i = 1; i < gs.size(); ++i)
    if (gs[i].size() > gs[t].size()) t = i;
  for (size_t i = 0; i < gs.size(); ++i)
    if (i != t && gs[i].size() == gs[t].size()) return false;
  return place(g, bg, gs, t, piece, out) > 0;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int gap = std::max(0, std::min(1, arg_int(action.args, "gap", 0)));
  bool piece = arg_str(action.args, "paint", "template") == "piece";
  bool best = arg_str(action.args, "pick", "biggest") == "best";
  int bg = concept_bg(scene);
  bool cleared = arg_str(action.args, "hide", "none") == "rect" && clear_rects(g, bg);
  CGrid out;
  if (complete(g, bg, gap, piece, best, out)) concept_install(scene, out);
  else if (cleared) concept_install(scene, g);
}

DslArgs sample(ArcRandom& rng) {
  int gap = pick_int(rng, {0, 1});
  std::string paint = pick_str(rng, {"template", "piece"});
  std::string pick = pick_str(rng, {"biggest", "best"});
  return make(gap, paint, pick, pick_str(rng, {"none", "rect"}));
}

// Seeds only when every train output has the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  res.push_back(make(0));
  res.push_back(make(1));
  return res;
}

}  // namespace

ConceptOp complete_copies_op() { return {ActionKind::COMPLETE_COPIES, 2, apply, sample, seeds}; }

}  // namespace arc
