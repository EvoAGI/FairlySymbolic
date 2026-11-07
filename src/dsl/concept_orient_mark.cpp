// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// MARK_SIDE side=orient_<by> (formerly the op ORIENT_MARK by=<by>; reached from
// concept_mark_side.cpp; the selection is not read): the drawing is horizontal when the top and bottom rows of its
// bbox are fuller than its left and right columns (by=fill), or when its bbox is
// wider than tall (by=extent: a row of copies), or when its colour changes
// more often along rows than along columns (by=stripes: side-by-side vertical
// stripes, a sequence running across), vertical otherwise; a full
// line of one colour is drawn along a grid edge chosen by that orientation.
// ConceptARC "HorizontalVertical".
//
// args: side   orient_fill | orient_extent | orient_stripes (<by>)
//       color  line colour
//       h_side top | bottom (edge used for a horizontal drawing)
//       v_side left | right (edge used for a vertical drawing)

#include <algorithm>
#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(int color, const std::string& h, const std::string& v, const std::string& by = "fill") {
  DslArgs a;
  a.emplace_back("side", cdv_str("orient_" + by));
  a.emplace_back("color", cdv_int(color));
  a.emplace_back("h_side", cdv_str(h));
  a.emplace_back("v_side", cdv_str(v));
  return a;
}

}  // namespace

// MARK_SIDE side=orient_* (concept_mark_side.cpp).
void orient_mark(Scene& scene, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int y0 = H, y1 = -1, x0 = W, x1 = -1;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] != bg) {
        y0 = std::min(y0, y); y1 = std::max(y1, y);
        x0 = std::min(x0, x); x1 = std::max(x1, x);
      }
  if (y1 < 0 || (y1 == y0 && x1 == x0)) return;
  int hn = 0, vn = 0;
  for (int x = x0; x <= x1; ++x) hn += (g[y0][x] != bg) + (g[y1][x] != bg);
  for (int y = y0; y <= y1; ++y) vn += (g[y][x0] != bg) + (g[y][x1] != bg);
  // Coverage fractions hn / (2 w) against vn / (2 h), cross-multiplied.
  long long hc = static_cast<long long>(hn) * (y1 - y0 + 1), vc = static_cast<long long>(vn) * (x1 - x0 + 1);
  std::string by = arg_str(action.args, "side", "orient_fill").substr(7);
  if (by == "extent") {
    hc = x1 - x0;
    vc = y1 - y0;
  } else if (by == "stripes") {
    hc = vc = 0;
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        int v = g[y][x];
        if (v == bg) continue;
        if (x < x1 && g[y][x + 1] != bg && g[y][x + 1] != v) ++hc;
        if (y < y1 && g[y + 1][x] != bg && g[y + 1][x] != v) ++vc;
      }
  }
  if (hc == vc) return;
  int color = arg_int(action.args, "color", 1);
  if (hc > vc) {
    int r = arg_str(action.args, "h_side", "top") == "top" ? 0 : H - 1;
    for (int x = 0; x < W; ++x) g[r][x] = color;
  } else {
    int c = arg_str(action.args, "v_side", "right") == "left" ? 0 : W - 1;
    for (int y = 0; y < H; ++y) g[y][c] = color;
  }
  concept_install(scene, g);
}

DslArgs orient_mark_sample(ArcRandom& rng) {
  return make(pick_int(rng, {1, 2, 3, 4, 5, 6, 7, 8, 9}), pick_str(rng, {"top", "bottom"}),
              pick_str(rng, {"left", "right"}), pick_str(rng, {"fill", "extent", "stripes"}));
}

// Seeds only when every train output has the input's size; one colour per
// colour that appears in an output and not in its input.
std::vector<DslArgs> orient_mark_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  std::set<int> fresh;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    std::set<int> in(ins[i].cells().begin(), ins[i].cells().end());
    for (int c : outs[i].cells())
      if (!in.count(c)) fresh.insert(c);
  }
  for (int c : fresh)
    for (const char* h : {"top", "bottom"})
      for (const char* v : {"left", "right"})
        for (const char* by : {"fill", "extent", "stripes"}) res.push_back(make(c, h, v, by));
  return res;
}

}  // namespace arc
