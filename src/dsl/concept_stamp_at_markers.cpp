// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// STAMP_AT_MARKERS: for each colour with single-cell objects (markers), the
// largest object of that colour is the template; its bounding-box content is
// copied so that its anchor cell lands on every marker. ConceptARC "Copy".
//
// args: anchor  top_left | top_right | bottom_left | bottom_right | center (of the
//               template's bounding box) | middle (the centre too, which may be
//               background: a ring is drawn around the marker, which clears)
//       diag    8-connected objects
// The anchor cell must hold the template's colour, except with middle.

#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  auto comps = concept_components(g, bg, as_bool(args_get(action.args, "diag"), false), true);
  std::string anchor = arg_str(action.args, "anchor", "top_left");
  bool middle = anchor == "middle", center = middle || anchor == "center";
  std::vector<Coord> singles;
  std::set<int> marker_colors;
  for (const auto& c : comps)
    if (c.size() == 1) {
      singles.push_back(c[0]);
      marker_colors.insert(g[c[0].first][c[0].second]);
    }
  CGrid out = g;
  bool hit = false;
  for (int col : marker_colors) {
    const std::vector<Coord>* tpl = nullptr;
    for (const auto& c : comps)
      if (c.size() > 1 && g[c[0].first][c[0].second] == col && (!tpl || c.size() > tpl->size())) tpl = &c;
    if (!tpl) continue;
    Box b = bbox_of(*tpl);
    int ay = center ? (b.y1 - b.y0) / 2 : anchor.rfind("bottom", 0) == 0 ? b.y1 - b.y0 : 0;
    int ax = center ? (b.x1 - b.x0) / 2 : anchor.size() > 5 && anchor.substr(anchor.size() - 5) == "right" ? b.x1 - b.x0 : 0;
    if (!middle && g[b.y0 + ay][b.x0 + ax] != col) continue;
    for (const Coord& m : singles) {
      if (g[m.first][m.second] != col) continue;
      hit = true;
      for (int y = b.y0; y <= b.y1; ++y)
        for (int x = b.x0; x <= b.x1; ++x) {
          int v = g[y][x], p = m.first - ay + (y - b.y0), q = m.second - ax + (x - b.x0);
          if (v != bg && p >= 0 && p < H && q >= 0 && q < W) out[p][q] = v;
        }
      if (middle) out[m.first][m.second] = g[b.y0 + ay][b.x0 + ax];
    }
  }
  if (hit) concept_install(scene, out);
}

DslArgs make(const std::string& anchor, bool diag) {
  return {{"anchor", cdv_str(anchor)}, {"diag", cdv_bool(diag)}};
}

DslArgs sample(ArcRandom& rng) {
  return make(pick_str(rng, {"top_left", "top_right", "bottom_left", "bottom_right", "center", "middle"}),
              pick_int(rng, {0, 1}) == 1);
}

std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return {};
  std::vector<DslArgs> res;
  for (const char* a : {"top_left", "top_right", "bottom_left", "bottom_right", "center", "middle"})
    for (bool d : {false, true}) res.push_back(make(a, d));
  return res;
}

}  // namespace

ConceptOp stamp_at_markers_op() { return {ActionKind::STAMP_AT_MARKERS, 2, apply, sample, seeds}; }

}  // namespace arc
