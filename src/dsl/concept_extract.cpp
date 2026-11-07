// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// EXTRACT: the output is the part of the grid that stands apart, cropped.
// ConceptARC "ExtractObjects", "FilledNotFilled", "InsideOutside", "TopBottom2D".
//
// args: what  unique  the one colour that stands apart (concept_extract_unique.cpp)
//             tile    the mixed tile of a two-kind mosaic (concept_tile.cpp)
//             panels  the panels' contents, joined across their separators
//                     (concept_panels.cpp; mode=same: background separators,
//                     the shared crop when every panel holds the same one)
//             hole    what lies inside the hollow object: the bbox of the cells
//                     it encloses, cropped from the grid (background included)
//             extreme_top | extreme_bottom | extreme_left | extreme_right
//                     only the object reaching farthest to that side stays
//                     (concept_keep_extreme.cpp)
//       pick  (hole) largest (the object enclosing the most cells) | smallest |
//             all (every hole, a 4-connected piece of enclosed cells, cropped
//             to its bbox; left to right, joined side by side when they share a
//             height, else stacked when they share a width; nothing changes
//             when they share neither)
//       out   (extreme_*) grid (same size, the rest cleared) | crop
// Objects are 8-connected single-colour groups; a cell is enclosed when it
// cannot reach the object's bbox border without crossing the object.

#include <algorithm>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void extract_unique(Scene& scene, const DslAction& action);
void extract_tile(Scene& scene, const DslAction& action);
void extract_panels(Scene& scene, const DslAction& action);
void extract_extreme(Scene& scene, const DslAction& action);
DslArgs extract_extreme_sample(ArcRandom& rng);
std::vector<DslArgs> extract_extreme_seeds();

namespace {

DslArgs make(const std::string& what, const std::string& pick = "") {
  DslArgs a;
  a.emplace_back("what", cdv_str(what));
  if (!pick.empty()) a.emplace_back("pick", cdv_str(pick));
  return a;
}

// Cells of comp's bbox that cannot reach the bbox border (4-connected).
std::vector<Coord> enclosed(const std::vector<Coord>& comp) {
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
  std::vector<Coord> res;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (!obj[y][x] && !reach[y][x]) res.push_back({b.y0 + y, b.x0 + x});
  return res;
}

CGrid crop(const CGrid& g, const Box& b) {
  CGrid out;
  for (int y = b.y0; y <= b.y1; ++y) out.emplace_back(g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  return out;
}

// Bboxes of the 4-connected pieces of the enclosed cells (one object, such as
// two shapes touching at a corner, can hold several holes).
std::vector<Box> hole_boxes(const std::vector<Coord>& in) {
  std::set<Coord> left(in.begin(), in.end());
  std::vector<Box> res;
  while (!left.empty()) {
    std::vector<Coord> piece, st = {*left.begin()};
    left.erase(left.begin());
    while (!st.empty()) {
      Coord c = st.back();
      st.pop_back();
      piece.push_back(c);
      for (const Coord& n : {Coord(c.first - 1, c.second), Coord(c.first + 1, c.second),
                             Coord(c.first, c.second - 1), Coord(c.first, c.second + 1)})
        if (left.erase(n)) st.push_back(n);
    }
    res.push_back(bbox_of(piece));
  }
  return res;
}

// The holes of every hollow object joined into one grid; empty when they share
// neither a height nor a width.
CGrid join_all(const CGrid& g, std::vector<Box> boxes) {
  std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
    return a.x0 != b.x0 ? a.x0 < b.x0 : a.y0 < b.y0;
  });
  bool same_h = true, same_w = true;
  for (const Box& b : boxes) {
    same_h = same_h && b.y1 - b.y0 == boxes[0].y1 - boxes[0].y0;
    same_w = same_w && b.x1 - b.x0 == boxes[0].x1 - boxes[0].x0;
  }
  CGrid out;
  if (same_h) {
    out.assign(boxes[0].y1 - boxes[0].y0 + 1, {});
    for (const Box& b : boxes)
      for (int y = b.y0; y <= b.y1; ++y)
        out[y - b.y0].insert(out[y - b.y0].end(), g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  } else if (same_w) {
    for (const Box& b : boxes)
      for (int y = b.y0; y <= b.y1; ++y) out.emplace_back(g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  }
  if (out.size() > 30 || (!out.empty() && out[0].size() > 30)) return {};
  return out;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  std::string what = arg_str(action.args, "what", "hole");
  if (what == "unique") return extract_unique(scene, action);
  if (what == "tile") return extract_tile(scene, action);
  if (what == "panels") return extract_panels(scene, action);
  if (what.rfind("extreme_", 0) == 0) return extract_extreme(scene, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  std::string pick = arg_str(action.args, "pick", "largest");
  bool largest = pick == "largest";
  std::vector<Coord> best;
  std::vector<Box> all;
  for (const auto& comp : concept_components(g, concept_bg(scene), true, true)) {
    std::vector<Coord> in = enclosed(comp);
    if (in.empty()) continue;
    for (const Box& b : hole_boxes(in)) all.push_back(b);
    if (best.empty() || (largest ? in.size() > best.size() : in.size() < best.size())) best = std::move(in);
  }
  if (best.empty()) return;
  CGrid out = pick == "all" ? join_all(g, all) : crop(g, bbox_of(best));
  if (out.empty()) return;
  concept_install(scene, out);
}

// The five kinds of `what` are drawn alike (each was an op of weight 2;
// extreme stands for the four extreme_* settings, the former keep_extreme).
DslArgs sample(ArcRandom& rng) {
  std::string what = pick_str(rng, {"unique", "tile", "panels", "hole", "extreme"});
  if (what == "extreme") return extract_extreme_sample(rng);
  if (what == "panels" && pick_str(rng, {"join", "same"}) == "same") {
    DslArgs a = make(what);
    a.emplace_back("mode", cdv_str("same"));
    return a;
  }
  if (what != "hole") return make(what);
  return make(what, pick_str(rng, {"largest", "smallest", "all"}));
}

// extreme_* seeds on every task; the others only when every train output is
// smaller than its input.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = extract_extreme_seeds();
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (outs[i].height() * outs[i].width() >= ins[i].height() * ins[i].width()) return res;
  for (const char* w : {"unique", "tile", "panels"}) res.push_back(make(w));
  for (const char* p : {"largest", "smallest", "all"}) res.push_back(make("hole", p));
  return res;
}

}  // namespace

ConceptOp extract_op() { return {ActionKind::EXTRACT, 10, apply, sample, seeds}; }

}  // namespace arc
