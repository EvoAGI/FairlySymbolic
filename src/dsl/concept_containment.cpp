// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// KEEP_ENCLOSED side=bbox_markers / bbox_frame: the cells of the other colours
// (markers) are kept when they all lie inside the bbox (edges included) of the
// most frequent colour, and the most frequent colour is kept otherwise; the rest
// becomes background. bbox_frame is the reverse: the frame colour is kept when
// the markers are inside. ConceptARC "InsideOutside". (Formerly
// keep_by_containment when_inside=markers / frame.)
//
// args: edge  in (a marker on the bbox edge is inside; the default) | out

#include <algorithm>
#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(const std::string& when_inside, const std::string& edge = "in") {
  DslArgs a;
  a.emplace_back("side", cdv_str("bbox_" + when_inside));
  if (edge != "in") a.emplace_back("edge", cdv_str(edge));
  return a;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
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
  int y0 = H, y1 = -1, x0 = W, x1 = -1;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] == major) {
        y0 = std::min(y0, y); y1 = std::max(y1, y);
        x0 = std::min(x0, x); x1 = std::max(x1, x);
      }
  bool inside = true, strict = arg_str(action.args, "edge", "in") == "out";
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] != bg && g[y][x] != major)
        inside = inside && (strict ? y > y0 && y < y1 && x > x0 && x < x1 : y >= y0 && y <= y1 && x >= x0 && x <= x1);
  bool keep_markers = inside == (arg_str(action.args, "side", "bbox_markers") == "bbox_markers");
  CGrid out = g;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] != bg && (g[y][x] == major) == keep_markers) out[y][x] = bg;
  concept_install(scene, out);
}

}  // namespace

// KEEP_ENCLOSED side=bbox_* (concept_enclosed.cpp).
void keep_by_bbox(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

DslArgs keep_by_bbox_sample(ArcRandom& rng) {
  return make(pick_str(rng, {"markers", "frame"}), pick_str(rng, {"in", "out"}));
}

// Seeds only when every train output has the input's size and only removes cells.
std::vector<DslArgs> keep_by_bbox_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    const auto& a = ins[i].cells();
    const auto& b = outs[i].cells();
    for (size_t k = 0; k < a.size(); ++k)
      if (b[k] != 0 && a[k] != b[k]) return res;
  }
  for (const char* e : {"in", "out"}) {
    res.push_back(make("markers", e));
    res.push_back(make("frame", e));
  }
  return res;
}

}  // namespace arc
