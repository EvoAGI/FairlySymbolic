// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// REPEAT_BY_MARKERS: N marker cells beside the template ask for N+1 copies of
// it, side by side when the markers are to its left or right, stacked
// otherwise; the markers are dropped. ConceptARC "Copy".
//
// args: marker  marker colour, -1 = the one colour whose cells lie on a single
//               row or column outside the bounding box of everything else, else
//               the cells outside the largest solid rectangle (the template)
//       gap     background cells between copies

#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

struct Split {
  std::vector<Coord> marks, rest;
  Box rb;
};

bool split(const CGrid& g, int bg, int marker, Split* s) {
  s->marks.clear();
  s->rest.clear();
  for (int y = 0; y < static_cast<int>(g.size()); ++y)
    for (int x = 0; x < static_cast<int>(g[0].size()); ++x) {
      if (g[y][x] == bg) continue;
      (g[y][x] == marker ? s->marks : s->rest).push_back({y, x});
    }
  if (s->marks.empty() || s->rest.empty()) return false;
  s->rb = bbox_of(s->rest);
  return true;
}

// The template is the largest rectangle with no background cell; the marks are
// the other non-background cells (for a marker of the template's own colour).
bool split_by_rect(const CGrid& g, int bg, Split* s) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::vector<int>> pre(H + 1, std::vector<int>(W + 1, 0));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) pre[y + 1][x + 1] = pre[y][x + 1] + pre[y + 1][x] - pre[y][x] + (g[y][x] != bg);
  int best = 0;
  Box r{0, 0, -1, -1};
  for (int y0 = 0; y0 < H; ++y0)
    for (int y1 = y0; y1 < H; ++y1)
      for (int x0 = 0; x0 < W; ++x0)
        for (int x1 = x0; x1 < W; ++x1) {
          int area = (y1 - y0 + 1) * (x1 - x0 + 1);
          if (area <= best) continue;
          if (pre[y1 + 1][x1 + 1] - pre[y0][x1 + 1] - pre[y1 + 1][x0] + pre[y0][x0] != area) break;
          best = area;
          r = Box{y0, x0, y1, x1};
        }
  if (best < 4) return false;
  s->marks.clear();
  s->rest.clear();
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] == bg) continue;
      bool in = y >= r.y0 && y <= r.y1 && x >= r.x0 && x <= r.x1;
      (in ? s->rest : s->marks).push_back({y, x});
    }
  s->rb = r;
  return !s->marks.empty();
}

bool usable(const Split& s) {
  std::set<int> ys, xs;
  for (const Coord& m : s.marks) {
    ys.insert(m.first);
    xs.insert(m.second);
    if (m.first >= s.rb.y0 && m.first <= s.rb.y1 && m.second >= s.rb.x0 && m.second <= s.rb.x1) return false;
  }
  return ys.size() == 1 || xs.size() == 1;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), marker = arg_int(action.args, "marker", -1);
  Split s;
  if (marker < 0) {
    int found = -1;
    std::set<int> colors;
    for (const auto& r : g)
      for (int v : r)
        if (v != bg) colors.insert(v);
    for (int c : colors) {
      Split t;
      if (split(g, bg, c, &t) && usable(t)) {
        if (found >= 0) return;  // ambiguous
        found = c;
      }
    }
    if (found >= 0) {
      marker = found;
    } else if (split_by_rect(g, bg, &s) && usable(s)) {
      marker = -1;  // the marks share the template's colours
    } else {
      return;
    }
  }
  if (marker >= 0 && !split(g, bg, marker, &s)) return;
  int n = static_cast<int>(s.marks.size()) + 1, gap = std::max(0, arg_int(action.args, "gap", 1));
  int th = s.rb.y1 - s.rb.y0 + 1, tw = s.rb.x1 - s.rb.x0 + 1;
  bool horiz = true;
  for (const Coord& m : s.marks) horiz = horiz && (m.second > s.rb.x1 || m.second < s.rb.x0);
  int oh = horiz ? th : n * th + (n - 1) * gap, ow = horiz ? n * tw + (n - 1) * gap : tw;
  if (oh > 30 || ow > 30) return;
  CGrid out(oh, std::vector<int>(ow, bg));
  for (int i = 0; i < n; ++i)
    for (int y = 0; y < th; ++y)
      for (int x = 0; x < tw; ++x) {
        int v = g[s.rb.y0 + y][s.rb.x0 + x];
        if (v == marker) v = bg;
        int p = horiz ? y : i * (th + gap) + y, q = horiz ? i * (tw + gap) + x : x;
        out[p][q] = v;
      }
  concept_install(scene, out);
}

DslArgs make(int marker, int gap) { return {{"marker", cdv_int(marker)}, {"gap", cdv_int(gap)}}; }

DslArgs sample(ArcRandom& rng) {
  return make(pick_int(rng, {-1, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9}), pick_int(rng, {0, 1}));
}

// Seeds when the output is larger than the input in some pair (copies added).
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  bool grows = false;
  for (size_t i = 0; i < ins.size(); ++i)
    grows = grows || outs[i].height() > ins[i].height() || outs[i].width() > ins[i].width();
  if (!grows) return {};
  return {make(-1, 0), make(-1, 1)};
}

}  // namespace

ConceptOp repeat_by_markers_op() { return {ActionKind::REPEAT_BY_MARKERS, 1, apply, sample, seeds}; }

}  // namespace arc
