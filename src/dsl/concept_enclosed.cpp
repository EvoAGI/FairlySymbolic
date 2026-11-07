// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// KEEP_ENCLOSED: the most frequent colour draws an outline (gaps allowed) and
// the other colours are markers. Markers on the kept side stay, the others
// become background. ConceptARC "InsideOutside", "SameDifferent".
//
// Three inside tests:
//   flood    the outline is thickened by one cell (closing small gaps) and the
//            background is flooded from the grid border; a marker is outside
//            when it lies on the border, is flooded, or touches a flooded cell
//   between  rays from the marker hit the outline both ways along a row or a
//            column (the marker sits between two lines)
//   rays     straight rays from the marker hit the outline in at least `hits`
//            of the four directions
//   bbox     every marker lies inside the outline's bbox (concept_containment.cpp)
//
// args: side   flood_inside | flood_outside | between | beyond (= not between) |
//              rays_inside | rays_outside | bbox_markers | bbox_frame
//       edge   in | out (bbox only)
//       hits   3 | 4 (rays only)
//       frame  drop (the outline becomes background) | keep | mend (lines drawn
//              across the outline: the outline is the colour that encloses the
//              most cells once mended, not the most frequent one; a marker cell
//              whose mirror image across the outline's axis of symmetry is
//              outline becomes outline again, and the markers are then judged
//              by a plain flood through everything but the outline; side
//              flood_* / between keep the inside, flood_outside / beyond the
//              outside)

#include <algorithm>
#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void keep_by_bbox(Scene& scene, const DslAction& action);
DslArgs keep_by_bbox_sample(ArcRandom& rng);
std::vector<DslArgs> keep_by_bbox_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

const std::vector<std::string> kSides = {"flood_inside", "flood_outside", "between", "beyond"};
const int DY[4] = {1, -1, 0, 0}, DX[4] = {0, 0, 1, -1};

// frame=mend: the cells of colour c are mirrored across their best vertical and
// horizontal axes, each used when at least 90% of the cells whose image is not
// another colour land on c; other non-background cells whose image is c become c. Returns the cells the background flood from the border
// reaches through non-c cells; `enclosed` counts the others.
std::vector<std::vector<char>> mend(CGrid& g, int bg, int c, int& enclosed) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<Coord> cells;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] == c) cells.push_back({y, x});
  Box b = bbox_of(cells);
  CGrid src = g;
  for (bool vert : {true, false}) {
    int lo = vert ? b.x0 : b.y0, hi = vert ? b.x1 : b.y1, n = vert ? W : H;
    int best = -1, best_hit = 0, best_miss = 0;
    // Axis s = doubled coordinate of the mirror line; the image of t is s - t.
    for (int s2 = 2 * lo; s2 <= 2 * hi; ++s2) {
      int hit = 0, miss = 0;
      for (const Coord& p : cells) {
        int t = vert ? p.second : p.first, m = s2 - t;
        int v = m < 0 || m >= n ? bg : vert ? src[p.first][m] : src[m][p.second];
        if (v == c) ++hit;
        else if (v == bg) ++miss;
      }
      int d = std::abs(s2 - lo - hi), bd = std::abs(best - lo - hi);
      if (best < 0 || hit > best_hit || (hit == best_hit && d < bd)) {
        best = s2;
        best_hit = hit;
        best_miss = miss;
      }
    }
    if (best_hit == 0 || 10 * best_hit < 9 * (best_hit + best_miss)) continue;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (src[y][x] == bg || src[y][x] == c) continue;
        int m = best - (vert ? x : y);
        if (m >= 0 && m < n && (vert ? src[y][m] : src[m][x]) == c) g[y][x] = c;
      }
  }
  std::vector<std::vector<char>> reached(H, std::vector<char>(W, 0));
  std::vector<Coord> st;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if ((y == 0 || y == H - 1 || x == 0 || x == W - 1) && g[y][x] != c) {
        reached[y][x] = 1;
        st.emplace_back(y, x);
      }
  while (!st.empty()) {
    Coord p = st.back();
    st.pop_back();
    for (int k = 0; k < 4; ++k) {
      int y = p.first + DY[k], x = p.second + DX[k];
      if (y < 0 || y >= H || x < 0 || x >= W || reached[y][x] || g[y][x] == c) continue;
      reached[y][x] = 1;
      st.emplace_back(y, x);
    }
  }
  enclosed = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) enclosed += !reached[y][x] && g[y][x] != c;
  return reached;
}

DslArgs make(const std::string& side, const std::string& frame, int hits = 0) {
  DslArgs a;
  a.emplace_back("side", cdv_str(side));
  a.emplace_back("frame", cdv_str(frame));
  if (hits) a.emplace_back("hits", cdv_int(hits));
  return a;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  if (arg_str(action.args, "side", "flood_outside").rfind("bbox_", 0) == 0) return keep_by_bbox(scene, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::map<int, int> n;
  for (const auto& row : g)
    for (int v : row)
      if (v != bg) n[v]++;
  if (n.size() < 2) return;
  int major = n.begin()->first;
  for (const auto& kv : n)
    if (kv.second > n[major]) major = kv.first;
  std::string side = arg_str(action.args, "side", "flood_outside");
  std::string frame = arg_str(action.args, "frame", "drop");
  bool keep_frame = frame == "keep";
  bool flood = side.rfind("flood", 0) == 0;
  bool rays = side.rfind("rays", 0) == 0;
  bool keep_inside = side == "flood_inside" || side == "between" || side == "rays_inside";
  int need = std::max(1, std::min(4, arg_int(action.args, "hits", 4)));
  if (frame == "mend") {
    CGrid best_g;
    std::vector<std::vector<char>> best_r;
    int best_n = 0, best_c = -1;
    for (const auto& kv : n) {
      CGrid m = g;
      int e = 0;
      auto r = mend(m, bg, kv.first, e);
      if (e > best_n) {
        best_n = e;
        best_c = kv.first;
        best_g = std::move(m);
        best_r = std::move(r);
      }
    }
    if (best_c < 0) return;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (best_g[y][x] != bg && best_g[y][x] != best_c && !best_r[y][x] != keep_inside) best_g[y][x] = bg;
    concept_install(scene, best_g);
    return;
  }

  std::vector<std::vector<char>> reached(H, std::vector<char>(W, 0));
  if (flood) {
    std::vector<std::vector<char>> wall(H, std::vector<char>(W, 0));
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int a = -1; a <= 1 && !wall[y][x]; ++a)
          for (int b = -1; b <= 1; ++b) {
            int p = y + a, q = x + b;
            if (p >= 0 && p < H && q >= 0 && q < W && g[p][q] == major) { wall[y][x] = 1; break; }
          }
    std::vector<Coord> st;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if ((y == 0 || y == H - 1 || x == 0 || x == W - 1) && !wall[y][x]) {
          reached[y][x] = 1;
          st.emplace_back(y, x);
        }
    while (!st.empty()) {
      Coord c = st.back();
      st.pop_back();
      for (int k = 0; k < 4; ++k) {
        int p = c.first + DY[k], q = c.second + DX[k];
        if (p < 0 || p >= H || q < 0 || q >= W || reached[p][q] || wall[p][q]) continue;
        reached[p][q] = 1;
        st.emplace_back(p, q);
      }
    }
  }

  CGrid out(H, std::vector<int>(W, bg));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int v = g[y][x];
      if (v == bg) continue;
      if (v == major) {
        if (keep_frame) out[y][x] = v;
        continue;
      }
      bool inside;
      if (flood) {
        bool outside = y == 0 || y == H - 1 || x == 0 || x == W - 1 || reached[y][x];
        for (int k = 0; k < 4 && !outside; ++k) {
          int p = y + DY[k], q = x + DX[k];
          outside = p >= 0 && p < H && q >= 0 && q < W && reached[p][q];
        }
        inside = !outside;
      } else {
        bool hit[4] = {false, false, false, false};
        for (int k = 0; k < 4; ++k)
          for (int p = y + DY[k], q = x + DX[k]; p >= 0 && p < H && q >= 0 && q < W; p += DY[k], q += DX[k])
            if (g[p][q] == major) { hit[k] = true; break; }
        inside = rays ? hit[0] + hit[1] + hit[2] + hit[3] >= need : (hit[0] && hit[1]) || (hit[2] && hit[3]);
      }
      if (inside == keep_inside) out[y][x] = v;
    }
  concept_install(scene, out);
}

// A third of the draws each: bbox settings, rays settings (the weights of the
// former KEEP_BY_CONTAINMENT and KEEP_BY_RAYS ops), the other sides.
DslArgs sample(ArcRandom& rng) {
  int branch = pick_int(rng, {0, 1, 2});
  if (branch == 0) return keep_by_bbox_sample(rng);
  if (branch == 1)
    return make(pick_str(rng, {"rays_inside", "rays_outside"}), pick_str(rng, {"drop", "keep"}), pick_int(rng, {3, 4}));
  return make(pick_str(rng, kSides), pick_str(rng, {"drop", "keep", "mend"}));
}

// Seeds only when every train output has the input's size (bbox: and only
// removes cells).
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = keep_by_bbox_seeds(ins, outs);
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const std::string& s : kSides)
    for (const char* f : {"drop", "keep"}) res.push_back(make(s, f));
  for (const char* s : {"rays_inside", "rays_outside"})
    for (const char* f : {"drop", "keep"})
      for (int h : {4, 3}) res.push_back(make(s, f, h));
  return res;
}

}  // namespace

ConceptOp keep_enclosed_op() { return {ActionKind::KEEP_ENCLOSED, 6, apply, sample, seeds}; }

}  // namespace arc
