// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COPY_INTO_FRAMES: every empty rectangular frame (its inside holds only
// background, or one dot of its own colour) receives a copy of the rest of the
// scene's content, centred; the inside is cleared first. ConceptARC "Copy".
//
// args: diag  8-connected frames

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
  std::vector<Box> frames;
  std::vector<std::vector<char>> used(H, std::vector<char>(W, 0));
  for (const auto& c : comps) {
    Box b = bbox_of(c);
    if (b.y1 - b.y0 < 2 || b.x1 - b.x0 < 2) continue;
    std::set<Coord> cells(c.begin(), c.end());
    int fc = g[c[0].first][c[0].second], dots = 0;
    bool ok = true;
    for (int y = b.y0; y <= b.y1 && ok; ++y)
      for (int x = b.x0; x <= b.x1 && ok; ++x) {
        bool border = y == b.y0 || y == b.y1 || x == b.x0 || x == b.x1;
        if (border) ok = cells.count({y, x}) > 0;
        else if (g[y][x] != bg) ok = g[y][x] == fc && ++dots <= 1;
      }
    if (!ok) continue;
    frames.push_back(b);
    for (int y = b.y0; y <= b.y1; ++y)
      for (int x = b.x0; x <= b.x1; ++x) used[y][x] = 1;
  }
  if (frames.empty()) return;
  int ty0 = H, tx0 = W, ty1 = -1, tx1 = -1;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] != bg && !used[y][x]) {
        ty0 = std::min(ty0, y); tx0 = std::min(tx0, x);
        ty1 = std::max(ty1, y); tx1 = std::max(tx1, x);
      }
  if (ty1 < 0) return;
  int th = ty1 - ty0 + 1, tw = tx1 - tx0 + 1;
  CGrid out = g;
  for (const Box& f : frames) {
    int ih = f.y1 - f.y0 - 1, iw = f.x1 - f.x0 - 1;
    if (ih < th || iw < tw) continue;
    for (int y = f.y0 + 1; y < f.y1; ++y)
      for (int x = f.x0 + 1; x < f.x1; ++x) out[y][x] = bg;
    int oy = f.y0 + 1 + (ih - th) / 2, ox = f.x0 + 1 + (iw - tw) / 2;
    for (int y = 0; y < th; ++y)
      for (int x = 0; x < tw; ++x) {
        int v = g[ty0 + y][tx0 + x];
        if (v != bg && !used[ty0 + y][tx0 + x]) out[oy + y][ox + x] = v;
      }
  }
  concept_install(scene, out);
}

DslArgs make(bool diag) { return {{"diag", cdv_bool(diag)}}; }

DslArgs sample(ArcRandom& rng) { return make(pick_int(rng, {0, 1}) == 1); }

std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return {};
  return {make(false), make(true)};
}

}  // namespace

ConceptOp copy_into_frames_op() { return {ActionKind::COPY_INTO_FRAMES, 2, apply, sample, seeds}; }

}  // namespace arc
