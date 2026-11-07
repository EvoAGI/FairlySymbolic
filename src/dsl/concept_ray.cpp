// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// RAY_UNTIL_BLOCKED: objects grow in a straight line, in their own colour, from
// their leading face until the next cell is another object or the grid edge.
// ConceptARC "ExtendToBoundary".
//
// args: who   selected | smallest | non_largest | lone (not a dash of a dashed line) |
//             axis (every object, along its own axis: a line along its length,
//             any other shape in every direction; with dir=toward_line or
//             toward_line_diag)
//       dir   up | down | left | right | toward_other (towards the nearest other
//             object that shares a row or a column with the mover) |
//             toward_line (the nearest object or dashed line in 8 directions;
//             the ray stops before it) | toward_line_diag (toward_line, and a
//             diagonal ray also stops where it would pass between two cells of
//             one colour, as through a diagonal line) |
//             open (from the middle of the side that faces the widest free
//             margin of the grid, to the edge) | cross (all
//             four directions; each face moves out as one block and stops at the
//             first cell any of its rays would hit)
//       stop  object (draw only rays that hit an object) | any (also rays that
//             reach the grid edge)
// Objects are 8-connected groups of any colours; with who=smallest, when there
// is only one, they are 8-connected one-colour groups.

#include <algorithm>
#include <climits>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kDirs = {"up", "down", "left", "right", "toward_other", "toward_line", "open", "cross",
                                          "toward_line_diag"};

DslArgs make(const std::string& who, const std::string& dir, const std::string& stop) {
  DslArgs a;
  a.emplace_back("who", cdv_str(who));
  a.emplace_back("dir", cdv_str(dir));
  a.emplace_back("stop", cdv_str(stop));
  return a;
}

void apply(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::string dir0 = arg_str(action.args, "dir", "toward_other");
  auto comps = concept_components(g, bg, true, false);
  if (comps.size() == 1 && dir0 != "open" && arg_str(action.args, "who", "smallest") == "smallest")
    comps = concept_components(g, bg, true, true);
  if (comps.empty() || (comps.size() < 2 && dir0 != "open")) return;
  std::string who = arg_str(action.args, "who", "smallest");
  std::string dir = arg_str(action.args, "dir", "toward_other");
  bool need_hit = arg_str(action.args, "stop", "object") == "object";

  size_t big = 0, small = comps[0].size();
  for (size_t k = 0; k < comps.size(); ++k) {
    if (comps[k].size() > comps[big].size()) big = k;
    small = std::min(small, comps[k].size());
  }
  std::set<Coord> sel;
  if (who == "selected") {
    std::set<std::string> ids(selected_ids.begin(), selected_ids.end());
    for (const Node& n : scene.nodes())
      if (ids.count(n.node_id))
        for (const auto& p : n.pixel_list) sel.insert({p.first, p.second});
  }
  std::vector<int> owner(H * W, -1);
  for (size_t k = 0; k < comps.size(); ++k)
    for (const Coord& c : comps[k]) owner[c.first * W + c.second] = static_cast<int>(k);

  CGrid out = g;
  const int DY[4] = {-1, 1, 0, 0}, DX[4] = {0, 0, -1, 1};
  const int Y8[8] = {-1, 1, 0, 0, -1, -1, 1, 1}, X8[8] = {0, 0, -1, 1, -1, 1, -1, 1};
  auto at = [&](int y, int x) { return y >= 0 && y < H && x >= 0 && x < W ? g[y][x] : bg; };
  // who=lone: objects that are not a dash of a dashed line (no cell of their
  // colour two steps away, over one background cell, in any of 8 directions).
  auto lone = [&](size_t k) {
    for (const Coord& c : comps[k])
      for (int d = 0; d < 8; ++d)
        if (at(c.first + Y8[d], c.second + X8[d]) == bg &&
            at(c.first + 2 * Y8[d], c.second + 2 * X8[d]) == g[c.first][c.second] &&
            g[c.first][c.second] != bg)
          return false;
    return true;
  };
  if (dir == "open") {
    for (size_t k = 0; k < comps.size(); ++k) {
      bool m = who == "lone" ? lone(k) : who == "non_largest" ? k != big
             : who == "smallest" ? comps[k].size() == small && k != big
             : who == "selected" ? std::any_of(comps[k].begin(), comps[k].end(), [&](const Coord& c) { return sel.count(c) > 0; })
             : true;
      if (!m) continue;
      Box b = bbox_of(comps[k]);
      int room[4] = {b.y0, H - 1 - b.y1, b.x0, W - 1 - b.x1};
      int d = 0;
      for (int e = 1; e < 4; ++e)
        if (room[e] > room[d]) d = e;
      int ties = 0;
      for (int e = 0; e < 4; ++e) ties += room[e] == room[d];
      if (ties > 1 || room[d] == 0) continue;
      int col = g[comps[k][0].first][comps[k][0].second];
      std::vector<Coord> starts;
      if (d < 2) {
        int y = d == 0 ? b.y0 : b.y1;
        for (int x : {(b.x0 + b.x1) / 2, (b.x0 + b.x1 + 1) / 2}) starts.push_back({y, x});
      } else {
        int x = d == 2 ? b.x0 : b.x1;
        for (int y : {(b.y0 + b.y1) / 2, (b.y0 + b.y1 + 1) / 2}) starts.push_back({y, x});
      }
      for (const Coord& c : starts)
        for (int t = 1;; ++t) {
          int y = c.first + t * DY[d], x = c.second + t * DX[d];
          if (y < 0 || y >= H || x < 0 || x >= W || g[y][x] != bg) break;
          out[y][x] = col;
        }
    }
    concept_install(scene, out);
    return;
  }
  if (dir == "toward_line" || dir == "toward_line_diag") {
    bool diag_cross = dir == "toward_line_diag";
    // Eight directions; a background cell flanked across the ray by two cells of
    // one colour (the gap of a dashed line) blocks it like an object cell.
    for (size_t k = 0; k < comps.size(); ++k) {
      int col = g[comps[k][0].first][comps[k][0].second];
      bool m = who == "axis"         ? true
             : who == "lone"         ? lone(k)
             : who == "non_largest"  ? k != big
             : who == "smallest"     ? comps[k].size() == small && k != big
             : std::any_of(comps[k].begin(), comps[k].end(), [&](const Coord& c) { return sel.count(c) > 0; });
      if (!m) continue;
      int best = INT_MAX;
      Coord from{0, 0};
      int bd = -1;
      // who=axis: a straight line only grows along itself.
      std::vector<char> dir_ok(8, 1);
      if (who == "axis" && comps[k].size() > 1) {
        bool hor = true, ver = true, dia = true, anti = true;
        for (const Coord& c : comps[k]) {
          hor = hor && c.first == comps[k][0].first;
          ver = ver && c.second == comps[k][0].second;
          dia = dia && c.first - c.second == comps[k][0].first - comps[k][0].second;
          anti = anti && c.first + c.second == comps[k][0].first + comps[k][0].second;
        }
        if (hor || ver || dia || anti)
          for (int d = 0; d < 8; ++d)
            dir_ok[d] = hor ? d == 2 || d == 3 : ver ? d == 0 || d == 1 : dia ? d == 4 || d == 7 : d == 5 || d == 6;
      }
      for (int d = 0; d < 8; ++d)
        for (const Coord& c : comps[k])
          for (int t = 1; t < best && dir_ok[d]; ++t) {
            int y = c.first + t * Y8[d], x = c.second + t * X8[d];
            if (y < 0 || y >= H || x < 0 || x >= W) break;
            int o = owner[y * W + x];
            if (o == static_cast<int>(k)) break;
            int a = at(y + X8[d], x - Y8[d]);
            // A diagonal step between two cells of one colour crosses their line.
            int cy = at(y - Y8[d], x), cx = at(y, x - X8[d]);
            bool corner = diag_cross && d >= 4 && cy != bg && cy == cx && owner[(y - Y8[d]) * W + x] != static_cast<int>(k);
            if (o >= 0 || corner || (a != bg && a == at(y - X8[d], x + Y8[d]))) {
              best = t; bd = d; from = c;
              break;
            }
          }
      if (bd < 0) continue;
      for (int t = 1; t < best; ++t) out[from.first + t * Y8[bd]][from.second + t * X8[bd]] = col;
    }
    concept_install(scene, out);
    return;
  }
  for (size_t k = 0; k < comps.size(); ++k) {
    bool m = who == "lone" ? lone(k)
           : who == "non_largest" ? k != big
           : who == "smallest"    ? comps[k].size() == small && k != big
           : std::any_of(comps[k].begin(), comps[k].end(), [&](const Coord& c) { return sel.count(c) > 0; });
    if (!m) continue;
    std::vector<int> dirs;
    if (dir == "toward_other") {
      // Nearest other object hit by a ray from any cell, over the four directions.
      int best = INT_MAX, bd = -1;
      for (int d = 0; d < 4; ++d)
        for (const Coord& c : comps[k]) {
          for (int t = 1;; ++t) {
            int y = c.first + t * DY[d], x = c.second + t * DX[d];
            if (y < 0 || y >= H || x < 0 || x >= W) break;
            int o = owner[y * W + x];
            if (o == static_cast<int>(k)) break;
            if (o >= 0) { if (t < best) { best = t; bd = d; } break; }
          }
        }
      if (bd < 0) continue;
      dirs.push_back(bd);
    } else if (dir == "cross") {
      dirs = {0, 1, 2, 3};
    } else {
      for (int d = 0; d < 4; ++d)
        if (dir == std::vector<std::string>{"up", "down", "left", "right"}[d]) dirs.push_back(d);
    }
    if (dir == "cross") {
      for (int d : dirs) {
        std::vector<Coord> face;
        for (const Coord& c : comps[k]) {
          int ny = c.first + DY[d], nx = c.second + DX[d];
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && owner[ny * W + nx] == static_cast<int>(k)) continue;
          face.push_back(c);
        }
        int reach = INT_MAX;
        bool edge = false;
        for (const Coord& c : face) {
          int t = 1;
          for (;; ++t) {
            int y = c.first + t * DY[d], x = c.second + t * DX[d];
            if (y < 0 || y >= H || x < 0 || x >= W) { edge = true; break; }
            if (g[y][x] != bg) break;
          }
          reach = std::min(reach, t - 1);
        }
        if (face.empty() || (edge && need_hit)) continue;
        for (const Coord& c : face)
          for (int t = 1; t <= reach; ++t) out[c.first + t * DY[d]][c.second + t * DX[d]] = g[c.first][c.second];
      }
      continue;
    }
    for (int d : dirs)
      for (const Coord& c : comps[k]) {
        int ny = c.first + DY[d], nx = c.second + DX[d];
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && owner[ny * W + nx] == static_cast<int>(k)) continue;
        std::vector<Coord> ray;
        bool hit = false;
        for (int t = 1;; ++t) {
          int y = c.first + t * DY[d], x = c.second + t * DX[d];
          if (y < 0 || y >= H || x < 0 || x >= W) break;
          if (g[y][x] != bg) { hit = true; break; }
          ray.push_back({y, x});
        }
        if (hit || !need_hit)
          for (const Coord& r : ray) out[r.first][r.second] = g[c.first][c.second];
      }
  }
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) {
  return make(pick_str(rng, {"selected", "selected", "smallest", "non_largest", "lone", "axis"}), pick_str(rng, kDirs),
              pick_str(rng, {"object", "any"}));
}

// Seeds only when every train output has the input's size and only adds cells.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    const auto& a = ins[i].cells();
    const auto& b = outs[i].cells();
    for (size_t k = 0; k < a.size(); ++k)
      if (a[k] != 0 && a[k] != b[k]) return res;
  }
  for (const char* who : {"smallest", "non_largest"})
    for (const char* stop : {"object", "any"}) res.push_back(make(who, "toward_other", stop));
  return res;
}

}  // namespace

ConceptOp ray_until_blocked_op() { return {ActionKind::RAY_UNTIL_BLOCKED, 2, apply, sample, seeds}; }

}  // namespace arc
