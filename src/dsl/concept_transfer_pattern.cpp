// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// TRANSFER_PATTERN: one object carries a pattern (cells of other colours on a
// body of one colour); every one-colour object receives the pattern, fitted to
// its own bbox. ConceptARC "Copy".
//
// args: turn  0 | 1  (1: the source may also be mirrored or rotated, the way
//                    whose cells best match the target's shape; ties go to the
//                    unturned one)
// Objects are 8-connected groups of any colour; the source is the biggest one
// with two colours or more, its body the commonest colour. The source bbox is
// stretched or shrunk to the target bbox (each target cell reads the nearest
// source cell); target cells under a pattern cell take its colour.

#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(bool turn) { return {{"turn", cdv_bool(turn)}}; }

// Box picture: -1 background, -2 body, else the pattern colour.
using Pic = std::vector<std::vector<int>>;

// The 8 turns of p; t bit 0 mirrors columns, bit 1 mirrors rows, bit 2 transposes first.
Pic turned(const Pic& p, int t) {
  int h = static_cast<int>(p.size()), w = static_cast<int>(p[0].size());
  bool tr = t & 4;
  int H = tr ? w : h, W = tr ? h : w;
  Pic r(H, std::vector<int>(W));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int sy = (t & 2) ? H - 1 - y : y, sx = (t & 1) ? W - 1 - x : x;
      r[y][x] = tr ? p[sx][sy] : p[sy][sx];
    }
  return r;
}

// Nearest source index for target index i of n, source length m.
int near(int i, int n, int m) { return n <= 1 ? 0 : (2 * i * (m - 1) + (n - 1)) / (2 * (n - 1)); }

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  bool turn = as_bool(args_get(action.args, "turn"), false);
  auto comps = concept_components(g, bg, true, false);
  int src = -1, body = -1;
  std::vector<int> mono;
  for (size_t i = 0; i < comps.size(); ++i) {
    std::map<int, int> n;
    for (const Coord& c : comps[i]) n[g[c.first][c.second]]++;
    if (n.size() == 1) {
      if (comps[i].size() >= 4) mono.push_back(static_cast<int>(i));
      continue;
    }
    if (src >= 0 && comps[i].size() <= comps[src].size()) continue;
    src = static_cast<int>(i);
    body = -1;
    for (const auto& kv : n)
      if (body < 0 || kv.second > n[body]) body = kv.first;
  }
  if (src < 0 || mono.empty()) return;
  Box sb = bbox_of(comps[src]);
  Pic sp(sb.y1 - sb.y0 + 1, std::vector<int>(sb.x1 - sb.x0 + 1, -1));
  for (const Coord& c : comps[src]) {
    int v = g[c.first][c.second];
    sp[c.first - sb.y0][c.second - sb.x0] = v == body ? -2 : v;
  }
  CGrid out = g;
  for (int m : mono) {
    Box tb = bbox_of(comps[m]);
    int h = tb.y1 - tb.y0 + 1, w = tb.x1 - tb.x0 + 1;
    Pic best;
    int best_score = -1;
    for (int t = 0; t < (turn ? 8 : 1); ++t) {
      Pic p = turned(sp, t);
      int ph = static_cast<int>(p.size()), pw = static_cast<int>(p[0].size()), score = 0;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          score += (p[near(y, h, ph)][near(x, w, pw)] != -1) == (g[tb.y0 + y][tb.x0 + x] != bg);
      if (score > best_score) { best_score = score; best = p; }
    }
    int ph = static_cast<int>(best.size()), pw = static_cast<int>(best[0].size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int v = best[near(y, h, ph)][near(x, w, pw)];
        if (v >= 0) out[tb.y0 + y][tb.x0 + x] = v;
      }
  }
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) { return make(pick_int(rng, {0, 1}) == 1); }

std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return {};
  return {make(false), make(true)};
}

}  // namespace

ConceptOp transfer_pattern_op() { return {ActionKind::TRANSFER_PATTERN, 2, apply, sample, seeds}; }

}  // namespace arc
