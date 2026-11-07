// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// CENTER_ON: objects are moved so that their bbox centre is the centre of a
// reference, or the grid is cut down to its central cells, or the objects on
// them are deleted. ConceptARC "Center".
//
// args: mode  along  every object but the largest slides along the largest
//                    object's long axis to the middle of it (a marker moved to
//                    the middle of a line)
//             both   every object but the largest moves to the largest
//                    object's centre on both axes
//             grid   every object but the largest moves to the grid centre
//             between  every object but the two largest moves, across the gap
//                    that separates those two, to the middle of it
//             crop   the output is the central cell of the grid (two cells on
//                    an axis of even length)
//             mid_row  the output is the middle row of the grid (two rows when
//                    the height is even), cut to the columns from its first to
//                    its last non-background cell (the middle of a stack)
//             mid_col  the same with the middle column
//             pick   the output is the object (not the largest) whose bbox centre
//                    is nearest the largest object's bbox centre, cropped to its
//                    bbox (ties: the bigger object); the shape in the middle of
//                    a frame (objects 8-connected here)
//             cross  each 8-connected group of any colours holds a line (its
//                    largest single-colour part) and markers (the rest); a
//                    marker slides to the middle of its line, as with along,
//                    and is drawn through the line onto its mirror image (a
//                    bar across the line's middle)
//             mark   the bbox centre of every object is painted (a centre dot);
//                    objects here are 8-connected, and lone cells of a colour
//                    at most two apart are one object (a dotted outline)
//             mark_all  the centre of the bbox of all non-background cells is
//                    painted
//             delete every object (8-connected, one colour, colour 0 as
//                    background) covering a central cell of the grid (one or
//                    two middle rows / columns) is painted 0
//       color  mark, mark_all: 0-9, or -1 = the object's colour (mark_all: the
//              most frequent colour)
// Objects are 4-connected single-colour groups. A move that would need half a
// cell is skipped, and so is a centre that falls between cells.

#include <algorithm>
#include <map>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

const std::vector<std::string> kModes = {"along", "both", "grid", "crop", "between", "cross", "mark", "mark_all",
                                         "mid_row", "mid_col", "pick"};

DslArgs make(const std::string& mode, int color = -1) {
  DslArgs a;
  a.emplace_back("mode", cdv_str(mode));
  if (mode.rfind("mark", 0) == 0) a.emplace_back("color", cdv_int(color));
  return a;
}

// mode=mark / mark_all: paint bbox centres.
void mark(CGrid& g, int bg, bool all, int color) {
  std::vector<std::vector<Coord>> groups;
  if (all) {
    groups.push_back({});
    for (int y = 0; y < static_cast<int>(g.size()); ++y)
      for (int x = 0; x < static_cast<int>(g[0].size()); ++x)
        if (g[y][x] != bg) groups[0].push_back({y, x});
    if (groups[0].empty()) return;
  } else {
    groups = concept_dotted_groups(g, bg, true);
  }
  std::vector<std::pair<Coord, int>> paint;
  for (const auto& gr : groups) {
    Box b = bbox_of(gr);
    if ((b.y0 + b.y1) % 2 || (b.x0 + b.x1) % 2) continue;
    int c = color;
    if (c < 0) {
      std::map<int, int> n;
      for (const Coord& p : gr) n[g[p.first][p.second]]++;
      for (const auto& kv : n)
        if (c < 0 || kv.second > n[c]) c = kv.first;
    }
    paint.push_back({{(b.y0 + b.y1) / 2, (b.x0 + b.x1) / 2}, c});
  }
  for (const auto& p : paint) g[p.first.first][p.first.second] = p.second;
}

// mode=delete: objects on the grid's central cells are painted 0.
void delete_center(Scene& scene) {
  CGrid g = concept_render(scene);
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::set<Coord> centre;
  for (int y : {(H - 1) / 2, H / 2})
    for (int x : {(W - 1) / 2, W / 2}) centre.insert({y, x});
  bool hit = false;
  for (const auto& comp : concept_components(g, 0, true, true)) {
    bool covers = false;
    for (const Coord& c : comp) covers = covers || centre.count(c);
    if (!covers) continue;
    hit = true;
    for (const Coord& c : comp) g[c.first][c.second] = 0;
  }
  if (hit) concept_install(scene, g);
}

// mode=cross: markers slide to the middle of the line they touch and are drawn
// through it onto their mirror image.
void cross(CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  auto groups = concept_components(g, bg, true, false);
  auto parts = concept_components(g, bg, false, true);
  std::vector<int> gid(H * W, -1);
  for (size_t i = 0; i < groups.size(); ++i)
    for (const Coord& c : groups[i]) gid[c.first * W + c.second] = static_cast<int>(i);
  std::vector<int> line(groups.size(), -1);
  for (size_t k = 0; k < parts.size(); ++k) {
    int i = gid[parts[k][0].first * W + parts[k][0].second];
    if (line[i] < 0 || parts[k].size() > parts[line[i]].size()) line[i] = static_cast<int>(k);
  }
  CGrid out = g;
  std::vector<std::pair<Coord, int>> paint;
  for (size_t k = 0; k < parts.size(); ++k) {
    int i = gid[parts[k][0].first * W + parts[k][0].second];
    if (line[i] == static_cast<int>(k)) continue;
    Box r = bbox_of(parts[line[i]]), b = bbox_of(parts[k]);
    bool wide = r.x1 - r.x0 > r.y1 - r.y0;
    int ey = wide ? 0 : r.y0 + r.y1 - (b.y0 + b.y1), ex = wide ? r.x0 + r.x1 - (b.x0 + b.x1) : 0;
    if (ey % 2 || ex % 2) continue;
    for (const Coord& c : parts[k]) {
      int col = g[c.first][c.second], y = c.first + ey / 2, x = c.second + ex / 2;
      out[c.first][c.second] = bg;
      // Mirror across the line's middle row (wide) or column (tall), and the cells between.
      int y2 = wide ? r.y0 + r.y1 - y : y, x2 = wide ? x : r.x0 + r.x1 - x;
      for (int p = std::min(y, y2); p <= std::max(y, y2); ++p)
        for (int q = std::min(x, x2); q <= std::max(x, x2); ++q) paint.push_back({{p, q}, col});
    }
  }
  for (const auto& p : paint) {
    int y = p.first.first, x = p.first.second;
    if (y >= 0 && y < H && x >= 0 && x < W) out[y][x] = p.second;
  }
  g = out;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::string mode = arg_str(action.args, "mode", "along");
  if (mode == "delete") {
    delete_center(scene);
    return;
  }
  if (mode == "mark" || mode == "mark_all" || mode == "cross") {
    if (mode == "cross") cross(g, bg);
    else mark(g, bg, mode == "mark_all", std::min(9, arg_int(action.args, "color", -1)));
    concept_install(scene, g);
    return;
  }
  if (mode == "crop") {
    int y0 = (H - 1) / 2, x0 = (W - 1) / 2;
    CGrid out;
    for (int y = y0; y <= H / 2; ++y) out.emplace_back(g[y].begin() + x0, g[y].begin() + W / 2 + 1);
    concept_install(scene, out);
    return;
  }
  if (mode == "mid_row" || mode == "mid_col") {
    bool rows = mode == "mid_row";
    int n = rows ? H : W, len = rows ? W : H, lo = len, hi = -1;
    auto at = [&](int i, int j) { return rows ? g[i][j] : g[j][i]; };
    for (int i = (n - 1) / 2; i <= n / 2; ++i)
      for (int j = 0; j < len; ++j)
        if (at(i, j) != bg) { lo = std::min(lo, j); hi = std::max(hi, j); }
    if (hi < 0) return;
    CGrid out;
    for (int i = (n - 1) / 2; i <= n / 2; ++i) {
      std::vector<int> line;
      for (int j = lo; j <= hi; ++j) line.push_back(at(i, j));
      out.push_back(line);
    }
    if (!rows) {
      CGrid t(out[0].size(), std::vector<int>(out.size()));
      for (size_t a = 0; a < out.size(); ++a)
        for (size_t b = 0; b < out[0].size(); ++b) t[b][a] = out[a][b];
      out = t;
    }
    concept_install(scene, out);
    return;
  }
  auto comps = concept_components(g, bg, mode == "pick", true);
  if (comps.size() < 2) return;
  size_t big = 0;
  for (size_t k = 1; k < comps.size(); ++k)
    if (comps[k].size() > comps[big].size()) big = k;
  if (mode == "pick") {
    Box r = bbox_of(comps[big]);
    long best_d = -1;
    size_t pick = big;
    for (size_t k = 0; k < comps.size(); ++k) {
      if (k == big) continue;
      Box b = bbox_of(comps[k]);
      long dy = (b.y0 + b.y1) - (r.y0 + r.y1), dx = (b.x0 + b.x1) - (r.x0 + r.x1), d = dy * dy + dx * dx;
      if (best_d < 0 || d < best_d || (d == best_d && comps[k].size() > comps[pick].size())) {
        best_d = d;
        pick = k;
      }
    }
    Box b = bbox_of(comps[pick]);
    CGrid out(b.y1 - b.y0 + 1, std::vector<int>(b.x1 - b.x0 + 1, bg));
    for (const Coord& c : comps[pick]) out[c.first - b.y0][c.second - b.x0] = g[c.first][c.second];
    concept_install(scene, out);
    return;
  }
  size_t big2 = big;
  if (mode == "between") {
    if (comps.size() < 3) return;
    big2 = big == 0 ? 1 : 0;
    for (size_t k = 0; k < comps.size(); ++k)
      if (k != big && comps[k].size() > comps[big2].size()) big2 = k;
  }
  Box r = bbox_of(comps[big]);
  // Doubled reference centre.
  int ry = r.y0 + r.y1, rx = r.x0 + r.x1;
  bool move_y = true, move_x = true;
  if (mode == "grid") {
    ry = H - 1;
    rx = W - 1;
  } else if (mode == "between") {
    Box a = r, b = bbox_of(comps[big2]);
    if (a.y1 < b.y0 || b.y1 < a.y0) {
      ry = a.y1 < b.y0 ? a.y1 + b.y0 : b.y1 + a.y0;
      move_x = false;
    } else if (a.x1 < b.x0 || b.x1 < a.x0) {
      rx = a.x1 < b.x0 ? a.x1 + b.x0 : b.x1 + a.x0;
      move_y = false;
    } else {
      return;
    }
  } else if (mode == "along") {
    bool wide = r.x1 - r.x0 > r.y1 - r.y0;
    move_y = !wide;
    move_x = wide;
  }
  CGrid out = g;
  std::vector<std::pair<Coord, int>> paint;
  for (size_t k = 0; k < comps.size(); ++k) {
    if (k == big || k == big2) continue;
    Box b = bbox_of(comps[k]);
    int ey = move_y ? ry - (b.y0 + b.y1) : 0, ex = move_x ? rx - (b.x0 + b.x1) : 0;
    if (ey % 2 || ex % 2) continue;
    for (const Coord& c : comps[k]) {
      out[c.first][c.second] = bg;
      paint.push_back({{c.first + ey / 2, c.second + ex / 2}, g[c.first][c.second]});
    }
  }
  for (const auto& p : paint) {
    int y = p.first.first, x = p.first.second;
    if (y >= 0 && y < H && x >= 0 && x < W) out[y][x] = p.second;
  }
  concept_install(scene, out);
}

// Half the draws are mode=delete (the weight of the former DELETE_CENTER op).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 1})) return make("delete");
  std::string mode = pick_str(rng, kModes);
  if (mode.rfind("mark", 0) != 0) return make(mode);
  return make(mode, pick_int(rng, {-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// "crop" seeds when every train output is the central cells of its input;
// the moves seed when every train output has the input's size.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  bool same = true, crop = true;
  for (size_t i = 0; i < ins.size(); ++i) {
    int h = ins[i].height(), w = ins[i].width();
    same = same && outs[i].height() == h && outs[i].width() == w;
    crop = crop && outs[i].height() == 2 - h % 2 && outs[i].width() == 2 - w % 2 && (h > 2 || w > 2);
  }
  if (crop) res.push_back(make("crop"));
  if (same)
    for (const char* m : {"along", "both", "grid", "delete"}) res.push_back(make(m));
  return res;
}

}  // namespace

ConceptOp center_on_op() { return {ActionKind::CENTER_ON, 4, apply, sample, seeds}; }

}  // namespace arc
