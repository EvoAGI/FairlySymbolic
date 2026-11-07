// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SLIDE_UNTIL_BLOCKED: objects move in a direction, one cell at a time, until
// the next step would leave the grid or overlap another object. ConceptARC
// "MoveToBoundary" (and objects pushed against a reference, "AboveBelow").
//
// args: who  selected (objects holding a selected cell) | non_largest | smallest | all
//       dir  up | down | left | right | toward_big (towards the largest object)
//            | toward_line
//            (every object but a full-width or full-height line moves towards it)
//            | axis_dr, axis_dl, axis_ur, axis_ul (each object moves along its own
//            long axis: tall ones down / up, wide ones right / left; square ones
//            follow the orientation most movers have) | from_base (each object
//            moves away from the side of its bbox holding most of its cells, its
//            base; no move when two sides tie) | apart (the first group along the
//            axis the groups line up on goes to the start edge, the last one to the
//            end edge; a group is the same-colour cells at most two apart, plus the
//            objects whose bbox lies inside its bbox; when that makes one group, the
//            8-connected objects are the groups; `who` is not read) | along_line
//            (each group of same-colour cells at most two apart that lies on one
//            row, column or diagonal slides along it, the way another object stops
//            it; it stays when both ways reach the grid edge; `who` is not read)
//            | jump_down, jump_up, jump_left, jump_right (formerly the op
//            jump_over: an object facing a bigger one jumps to its other side;
//            concept_jump.cpp; `who` and the selection are not read)
// Objects are 8-connected groups of non-background cells of any colour. They
// move one after the other, the one nearest its destination first, so movers
// stack against each other.

#include <algorithm>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void jump_over(Scene& scene, const DslAction& action);
DslArgs jump_over_sample(ArcRandom& rng);
std::vector<DslArgs> jump_over_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

// nearest_border (towards the closest grid edge) was dropped: no solved program
// used it; a program naming it moves nothing.
const std::vector<std::string> kDirs = {"up", "down", "left", "right", "toward_big", "toward_line",
                                          "axis_dr", "axis_dl", "axis_ur", "axis_ul", "from_base", "apart",
                                          "along_line"};

DslArgs make(const std::string& who, const std::string& dir) {
  DslArgs a;
  a.emplace_back("who", cdv_str(who));
  a.emplace_back("dir", cdv_str(dir));
  return a;
}

// Unit step away from the bbox side holding most of the cells; (0, 0) on a tie.
std::pair<int, int> from_base(const std::vector<Coord>& cells) {
  Box b = bbox_of(cells);
  int n[4] = {0, 0, 0, 0};  // top, bottom, left, right
  for (const Coord& c : cells) {
    n[0] += c.first == b.y0;
    n[1] += c.first == b.y1;
    n[2] += c.second == b.x0;
    n[3] += c.second == b.x1;
  }
  int k = static_cast<int>(std::max_element(n, n + 4) - n);
  for (int j = 0; j < 4; ++j)
    if (j != k && n[j] == n[k]) return {0, 0};
  const int dy[4] = {1, -1, 0, 0}, dx[4] = {0, 0, 1, -1};
  return {dy[k], dx[k]};
}

// Groups of same-colour cells at most two apart (Chebyshev).
std::vector<std::vector<Coord>> colour_groups(const CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<int> id(H * W, -1);
  std::vector<std::vector<Coord>> groups;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] == bg || id[y * W + x] >= 0) continue;
      int gi = static_cast<int>(groups.size());
      groups.push_back({});
      std::vector<Coord> st = {{y, x}};
      id[y * W + x] = gi;
      while (!st.empty()) {
        Coord c = st.back();
        st.pop_back();
        groups[gi].push_back(c);
        for (int dy = -2; dy <= 2; ++dy)
          for (int dx = -2; dx <= 2; ++dx) {
            int p = c.first + dy, q = c.second + dx;
            if (p < 0 || p >= H || q < 0 || q >= W || id[p * W + q] >= 0 || g[p][q] != g[c.first][c.second]) continue;
            id[p * W + q] = gi;
            st.push_back({p, q});
          }
      }
    }
  return groups;
}

// dir=along_line: each colour group on one straight line slides along it, in
// the direction where it ends against another object.
void along_line(CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  for (const auto& grp : colour_groups(g, bg)) {
    if (grp.size() < 2) continue;
    bool hor = true, ver = true, dia = true, anti = true;
    for (const Coord& c : grp) {
      hor = hor && c.first == grp[0].first;
      ver = ver && c.second == grp[0].second;
      dia = dia && c.first - c.second == grp[0].first - grp[0].second;
      anti = anti && c.first + c.second == grp[0].first + grp[0].second;
    }
    int ay = hor ? 0 : 1, ax = hor ? 1 : ver ? 0 : dia ? 1 : -1;
    if (!hor && !ver && !dia && !anti) continue;
    int col = g[grp[0].first][grp[0].second];
    std::set<Coord> own(grp.begin(), grp.end());
    // Steps the group can take towards s before the next one leaves the grid
    // (-1) or meets another object (the count).
    auto reach = [&](int s) {
      for (int t = 0;; ++t)
        for (const Coord& c : grp) {
          int y = c.first + (t + 1) * s * ay, x = c.second + (t + 1) * s * ax;
          if (y < 0 || y >= H || x < 0 || x >= W) return -1;
          if (g[y][x] != bg && !own.count({y, x})) return t;
        }
    };
    int fwd = reach(1), back = reach(-1);
    if ((fwd >= 0) == (back >= 0)) continue;
    int best_t = fwd >= 0 ? fwd : back, best_s = fwd >= 0 ? 1 : -1;
    if (best_t == 0) continue;
    for (const Coord& c : grp) g[c.first][c.second] = bg;
    for (const Coord& c : grp) g[c.first + best_t * best_s * ay][c.second + best_t * best_s * ax] = col;
  }
}

// dir=apart: groups of same-colour cells at most two apart (Chebyshev), joined
// with the groups whose bbox lies inside theirs (or, when that leaves one group,
// the 8-connected objects); first / last along the axis they line up on go to
// the start / end edge.
void apart(CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::vector<Coord>> groups = colour_groups(g, bg);
  for (bool merged = true; merged;) {
    merged = false;
    for (size_t i = 0; i < groups.size() && !merged; ++i)
      for (size_t j = 0; j < groups.size() && !merged; ++j) {
        if (i == j) continue;
        Box a = bbox_of(groups[i]), b = bbox_of(groups[j]);
        if (b.y0 >= a.y0 && b.y1 <= a.y1 && b.x0 >= a.x0 && b.x1 <= a.x1) {
          groups[i].insert(groups[i].end(), groups[j].begin(), groups[j].end());
          groups.erase(groups.begin() + j);
          merged = true;
        }
      }
  }
  if (groups.size() < 2) groups = concept_components(g, bg, true, false);
  if (groups.size() < 2) return;
  std::vector<Box> bx;
  for (const auto& gr : groups) bx.push_back(bbox_of(gr));
  bool sep_x = true, sep_y = true;
  for (size_t i = 0; i < bx.size(); ++i)
    for (size_t j = i + 1; j < bx.size(); ++j) {
      sep_x = sep_x && (bx[i].x1 < bx[j].x0 || bx[j].x1 < bx[i].x0);
      sep_y = sep_y && (bx[i].y1 < bx[j].y0 || bx[j].y1 < bx[i].y0);
    }
  if (sep_x == sep_y) return;
  size_t lo = 0, hi = 0;
  for (size_t i = 1; i < bx.size(); ++i) {
    if ((sep_x ? bx[i].x0 : bx[i].y0) < (sep_x ? bx[lo].x0 : bx[lo].y0)) lo = i;
    if ((sep_x ? bx[i].x0 : bx[i].y0) > (sep_x ? bx[hi].x0 : bx[hi].y0)) hi = i;
  }
  CGrid out = g;
  std::vector<std::pair<Coord, int>> paint;
  for (size_t i : {lo, hi}) {
    int d = i == lo ? -(sep_x ? bx[i].x0 : bx[i].y0) : (sep_x ? W - 1 - bx[i].x1 : H - 1 - bx[i].y1);
    for (const Coord& c : groups[i]) {
      out[c.first][c.second] = bg;
      paint.push_back({{c.first + (sep_x ? 0 : d), c.second + (sep_x ? d : 0)}, g[c.first][c.second]});
    }
  }
  for (const auto& p : paint) out[p.first.first][p.first.second] = p.second;
  g = out;
}

// Unit step (dy, dx) for `dir` from box b; (0, 0) when undefined.
std::pair<int, int> step_of(const std::string& dir, const Box& b, const Box& big, bool has_big, bool most_tall) {
  if (dir.rfind("axis_", 0) == 0 && dir.size() == 7) {
    int h = b.y1 - b.y0, w = b.x1 - b.x0;
    bool tall = h > w || (h == w && most_tall);
    if (tall) return {dir[5] == 'd' ? 1 : -1, 0};
    return {0, dir[6] == 'r' ? 1 : -1};
  }
  if (dir == "up") return {-1, 0};
  if (dir == "down") return {1, 0};
  if (dir == "left") return {0, -1};
  if (dir == "right") return {0, 1};
  if (dir == "toward_big" || dir == "toward_line") {
    if (!has_big) return {0, 0};
    if (b.y1 < big.y0) return {1, 0};
    if (b.y0 > big.y1) return {-1, 0};
    if (b.x1 < big.x0) return {0, 1};
    if (b.x0 > big.x1) return {0, -1};
    return {0, 0};
  }
  return {0, 0};
}

void apply(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action) {
  if (arg_str(action.args, "dir", "down").rfind("jump_", 0) == 0) return jump_over(scene, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  auto comps = concept_components(g, bg, true, false);
  std::string who0 = arg_str(action.args, "who", "selected");
  if (comps.empty() || (comps.size() < 2 && who0 != "all")) return;
  std::string who = arg_str(action.args, "who", "selected");
  std::string dir = arg_str(action.args, "dir", "down");
  if (dir == "apart" || dir == "along_line") {
    if (dir == "apart") apart(g, bg);
    else along_line(g, bg);
    concept_install(scene, g);
    return;
  }

  size_t big = 0;
  for (size_t k = 1; k < comps.size(); ++k)
    if (comps[k].size() > comps[big].size()) big = k;
  if (dir == "toward_line") {
    bool found = false;
    for (size_t k = 0; k < comps.size() && !found; ++k) {
      Box b = bbox_of(comps[k]);
      int n = static_cast<int>(comps[k].size());
      if ((b.y0 == 0 && b.y1 == H - 1 && b.x0 == b.x1 && n == H) ||
          (b.x0 == 0 && b.x1 == W - 1 && b.y0 == b.y1 && n == W)) {
        big = k;
        found = true;
      }
    }
    if (!found) return;
    if (who != "selected") who = "non_largest";
  }
  size_t small = comps[0].size();
  for (const auto& c : comps) small = std::min(small, c.size());

  std::set<Coord> sel;
  if (who == "selected") {
    std::set<std::string> ids(selected_ids.begin(), selected_ids.end());
    for (const Node& n : scene.nodes())
      if (ids.count(n.node_id))
        for (const auto& p : n.pixel_list) sel.insert({p.first, p.second});
  }
  std::vector<size_t> movers;
  for (size_t k = 0; k < comps.size(); ++k) {
    bool m = who == "all"         ? true
           : who == "non_largest" ? k != big
           : who == "smallest"    ? comps[k].size() == small && k != big
           : std::any_of(comps[k].begin(), comps[k].end(), [&](const Coord& c) { return sel.count(c) > 0; });
    if (m) movers.push_back(k);
  }
  if (movers.empty()) return;

  Box bigbox = bbox_of(comps[big]);
  int tall = 0, wide = 0;
  for (size_t k : movers) {
    Box b = bbox_of(comps[k]);
    tall += b.y1 - b.y0 > b.x1 - b.x0;
    wide += b.y1 - b.y0 < b.x1 - b.x0;
  }
  bool has_big = std::find(movers.begin(), movers.end(), big) == movers.end();
  struct Mv { size_t k; int dy, dx, lead; };
  std::vector<Mv> mv;
  for (size_t k : movers) {
    Box b = bbox_of(comps[k]);
    auto s = dir == "from_base" ? from_base(comps[k]) : step_of(dir, b, bigbox, has_big, tall >= wide);
    if (s.first == 0 && s.second == 0) continue;
    // Distance of the leading edge to the grid edge it moves towards.
    int lead = s.first < 0 ? b.y0 : s.first > 0 ? H - 1 - b.y1 : s.second < 0 ? b.x0 : W - 1 - b.x1;
    mv.push_back({k, s.first, s.second, lead});
  }
  std::stable_sort(mv.begin(), mv.end(), [](const Mv& a, const Mv& b) { return a.lead < b.lead; });

  for (const Mv& m : mv) {
    const auto& cells = comps[m.k];
    std::vector<int> col;
    for (const Coord& c : cells) col.push_back(g[c.first][c.second]);
    for (const Coord& c : cells) g[c.first][c.second] = bg;
    int t = 0;
    for (;; ++t) {
      bool ok = true;
      for (const Coord& c : cells) {
        int y = c.first + (t + 1) * m.dy, x = c.second + (t + 1) * m.dx;
        if (y < 0 || y >= H || x < 0 || x >= W || g[y][x] != bg) { ok = false; break; }
      }
      if (!ok) break;
    }
    for (size_t i = 0; i < cells.size(); ++i) g[cells[i].first + t * m.dy][cells[i].second + t * m.dx] = col[i];
  }
  concept_install(scene, g);
}

// Half the draws are jump_* (the weight of the former JUMP_OVER op).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 1})) return jump_over_sample(rng);
  return make(pick_str(rng, {"selected", "selected", "non_largest", "smallest", "all"}), pick_str(rng, kDirs));
}

// Seeds only when every train output has the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  for (size_t i = 0; i < ins.size() && i < outs.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const char* who : {"all", "non_largest", "smallest"})
    for (const std::string& d : kDirs) res.push_back(make(who, d));
  for (DslArgs& a : jump_over_seeds(ins, outs)) res.push_back(std::move(a));
  return res;
}

}  // namespace

ConceptOp slide_until_blocked_op() { return {ActionKind::SLIDE_UNTIL_BLOCKED, 4, apply, sample, seeds}; }

}  // namespace arc
