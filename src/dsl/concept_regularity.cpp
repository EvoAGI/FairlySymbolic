// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COMPLETE_REGULARITY: a picture that is regular (mirror symmetric about some
// axes, or a periodic tiling) has some cells hidden; each hidden cell takes the
// colour its regular partners show. ConceptARC "CompleteShape"; common in ARC
// (a symmetric mosaic with a painted-over patch, a tiling with holes).
//
// args: mask  bg    the hidden cells are background
//             rect  the hidden cells are the colour that forms one solid rectangle
//             rare  the hidden cells are the least frequent colour
//             0-9   the hidden cells are that colour
//       rule  mirror  every mirror (left-right, up-down, the two diagonals; any
//                     axis position) that no pair of visible cells contradicts
//             period  the smallest tiling period no pair of visible cells contradicts
//             any     mirrors, then the period for what is still hidden
//             sym_lr | sym_ud | sym_both | sym_auto | sym_all_lr | sym_all_ud |
//             sym_all_both | sym_all_auto  each object (or the whole picture)
//                     completed to mirror symmetry about its best axis; mask
//                     and out are not read (concept_symmetry.cpp)
//             dashes_h_last | dashes_v_last  a colour seen twice or more on a
//                     row (column) with gaps between becomes a full row
//                     (column); h_last: rows win at crossings. Formerly the op
//                     extend_dashed_lines; mask and out are not read
//                     (concept_extend_dashed_lines.cpp)
//             frame  each colour's cells are the fragments of one rectangle
//                     frame, drawn in full on their bbox (cells of that colour
//                     inside are cleared). Formerly the op complete_frame;
//                     mask and out are not read (concept_complete_frame.cpp)
//       out   grid  the completed grid
//             crop  only the hidden region (its bounding box), completed

#include <algorithm>
#include <map>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void symmetry_complete(Scene& scene, const DslAction& action);
DslArgs symmetry_complete_sample(ArcRandom& rng);
void extend_dashed_lines(Scene& scene, const DslAction& action);
DslArgs extend_dashed_lines_sample(ArcRandom& rng);
std::vector<DslArgs> extend_dashed_lines_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);
void complete_frame(Scene& scene);
DslArgs complete_frame_args();
std::vector<DslArgs> complete_frame_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);
std::vector<DslArgs> symmetry_complete_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

DslArgs make(const std::string& mask, const std::string& rule, const std::string& out) {
  DslArgs a;
  a.emplace_back("mask", cdv_str(mask));
  a.emplace_back("rule", cdv_str(rule));
  a.emplace_back("out", cdv_str(out));
  return a;
}

// Colour of the hidden cells, or -1.
int mask_colour(const CGrid& g, int bg, const std::string& mask) {
  if (mask == "bg") return bg;
  std::map<int, int> n;
  for (const auto& row : g)
    for (int v : row) n[v]++;
  if (mask.size() == 1 && mask[0] >= '0' && mask[0] <= '9') return n.count(mask[0] - '0') ? mask[0] - '0' : -1;
  if (mask == "rare") {
    int c = -1, best = 1 << 30;
    for (const auto& kv : n)
      if (kv.second < best) { best = kv.second; c = kv.first; }
    return n.size() > 1 ? c : -1;
  }
  // rect: a colour all of whose cells fill their bounding box, at least 2x2; the largest.
  int c = -1, best = 0;
  for (const auto& kv : n) {
    std::vector<Coord> cells;
    for (int y = 0; y < static_cast<int>(g.size()); ++y)
      for (int x = 0; x < static_cast<int>(g[0].size()); ++x)
        if (g[y][x] == kv.first) cells.push_back({y, x});
    Box b = bbox_of(cells);
    int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
    if (h >= 2 && w >= 2 && h * w == kv.second && kv.second > best && kv.second * 2 < static_cast<int>(g.size() * g[0].size())) {
      best = kv.second;
      c = kv.first;
    }
  }
  return c;
}

// A mirror: kind 0 left-right (x -> p - x), 1 up-down (y -> p - y),
// 2 main diagonal ((y, x) -> (x + p, y - p)), 3 anti-diagonal ((y, x) -> (p - x, p - y)).
struct Mirror { int kind, p; };

bool partner(const Mirror& m, int y, int x, int H, int W, int& py, int& px) {
  switch (m.kind) {
    case 0: py = y; px = m.p - x; break;
    case 1: py = m.p - y; px = x; break;
    case 2: py = x + m.p; px = y - m.p; break;
    default: py = m.p - x; px = m.p - y; break;
  }
  return py >= 0 && py < H && px >= 0 && px < W;
}

// Mirrors that no two visible cells contradict and that pair up at least half
// of the visible cells with a visible partner.
std::vector<Mirror> find_mirrors(const CGrid& g, const std::vector<std::vector<char>>& hid) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int visible = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) visible += !hid[y][x];
  std::vector<Mirror> res;
  for (int kind = 0; kind < 4; ++kind) {
    int lo = kind == 2 ? -(W - 1) : 0, hi = kind == 0 ? 2 * W - 2 : kind == 1 ? 2 * H - 2 : kind == 2 ? H - 1 : H + W - 2;
    Mirror best{kind, 0};
    int best_agree = 0;
    for (int p = lo; p <= hi; ++p) {
      Mirror m{kind, p};
      int agree = 0;
      bool bad = false;
      for (int y = 0; y < H && !bad; ++y)
        for (int x = 0; x < W; ++x) {
          int py, px;
          if (hid[y][x] || !partner(m, y, x, H, W, py, px) || hid[py][px]) continue;
          if (g[py][px] != g[y][x]) { bad = true; break; }
          if (py != y || px != x) ++agree;
        }
      if (!bad && agree > best_agree) { best_agree = agree; best = m; }
    }
    if (best_agree * 2 >= visible && best_agree >= 8) res.push_back(best);
  }
  return res;
}

void fill_mirrors(CGrid& g, std::vector<std::vector<char>>& hid, const std::vector<Mirror>& ms) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  for (bool changed = true; changed;) {
    changed = false;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (!hid[y][x]) continue;
        for (const Mirror& m : ms) {
          int py, px;
          if (partner(m, y, x, H, W, py, px) && !hid[py][px]) {
            g[y][x] = g[py][px];
            hid[y][x] = 0;
            changed = true;
            break;
          }
        }
      }
  }
}

// Smallest tiling period (by area) that no two visible cells contradict and
// that gives every hidden cell a visible cell of its class.
void fill_period(CGrid& g, std::vector<std::vector<char>>& hid) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::pair<int, int>> periods;
  for (int py = 1; py <= H; ++py)
    for (int px = 1; px <= W; ++px)
      if (py < H || px < W) periods.push_back({py, px});
  std::stable_sort(periods.begin(), periods.end(),
                   [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first * a.second < b.first * b.second; });
  for (const auto& pp : periods) {
    int py = pp.first, px = pp.second;
    std::vector<int> cls(py * px, -1);
    bool bad = false;
    for (int y = 0; y < H && !bad; ++y)
      for (int x = 0; x < W; ++x) {
        if (hid[y][x]) continue;
        int& c = cls[(y % py) * px + x % px];
        if (c >= 0 && c != g[y][x]) { bad = true; break; }
        c = g[y][x];
      }
    if (bad) continue;
    for (int y = 0; y < H && !bad; ++y)
      for (int x = 0; x < W; ++x)
        if (hid[y][x] && cls[(y % py) * px + x % px] < 0) { bad = true; break; }
    if (bad) continue;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (hid[y][x]) { g[y][x] = cls[(y % py) * px + x % px]; hid[y][x] = 0; }
    return;
  }
}

CGrid complete(const CGrid& in, int bg, const std::string& mask, const std::string& rule, const std::string& out) {
  int H = static_cast<int>(in.size()), W = static_cast<int>(in[0].size());
  int mc = mask_colour(in, bg, mask);
  if (mc < 0) return in;
  std::vector<std::vector<char>> hid(H, std::vector<char>(W, 0));
  std::vector<Coord> cells;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (in[y][x] == mc) { hid[y][x] = 1; cells.push_back({y, x}); }
  if (cells.empty()) return in;
  CGrid g = in;
  if (rule != "period") fill_mirrors(g, hid, find_mirrors(g, hid));
  if (rule != "mirror") fill_period(g, hid);
  if (out != "crop") return g;
  Box b = bbox_of(cells);
  CGrid c;
  for (int y = b.y0; y <= b.y1; ++y) c.emplace_back(g[y].begin() + b.x0, g[y].begin() + b.x1 + 1);
  return c;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  if (arg_str(action.args, "rule", "any").rfind("sym_", 0) == 0) return symmetry_complete(scene, action);
  if (arg_str(action.args, "rule", "any").rfind("dashes_", 0) == 0) return extend_dashed_lines(scene, action);
  if (arg_str(action.args, "rule", "any") == "frame") return complete_frame(scene);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  concept_install(scene, complete(g, concept_bg(scene), arg_str(action.args, "mask", "rect"),
                                  arg_str(action.args, "rule", "any"), arg_str(action.args, "out", "grid")));
}

// The draws by the weights of the ops folded in: dashes_* 2/7 (the former
// EXTEND_DASHED_LINES), frame 1/7 (COMPLETE_FRAME), the rest 4/7, half of it
// sym_* (SYMMETRY_COMPLETE).
DslArgs sample(ArcRandom& rng) {
  int branch = pick_int(rng, {0, 0, 0, 0, 1, 1, 2});
  if (branch == 1) return extend_dashed_lines_sample(rng);
  if (branch == 2) return complete_frame_args();
  if (pick_int(rng, {0, 1})) return symmetry_complete_sample(rng);
  return make(pick_str(rng, {"bg", "rect", "rare", "bg", "rect", "rare", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}),
              pick_str(rng, {"mirror", "period", "any"}),
              pick_str(rng, {"grid", "crop"}));
}

CGrid to_cgrid(const Grid& gr) {
  CGrid g(gr.height(), std::vector<int>(gr.width()));
  const auto& c = gr.cells();
  for (int y = 0; y < gr.height(); ++y)
    for (int x = 0; x < gr.width(); ++x) g[y][x] = c[y * gr.width() + x];
  return g;
}

// Seeds: the argument settings that reproduce every train pair (sym_*: the
// settings of concept_symmetry.cpp's own seed rule).
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = symmetry_complete_seeds(ins, outs);
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (const char* mask : {"rect", "bg", "rare", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
    for (const char* rule : {"any", "mirror", "period"})
      for (const char* out : {"grid", "crop"}) {
        bool ok = true;
        for (size_t i = 0; i < ins.size() && ok; ++i) ok = complete(to_cgrid(ins[i]), 0, mask, rule, out) == to_cgrid(outs[i]);
        if (ok) res.push_back(make(mask, rule, out));
      }
  for (DslArgs& a : extend_dashed_lines_seeds(ins, outs)) res.push_back(std::move(a));
  for (DslArgs& a : complete_frame_seeds(ins, outs)) res.push_back(std::move(a));
  return res;
}

}  // namespace

ConceptOp complete_regularity_op() { return {ActionKind::COMPLETE_REGULARITY, 7, apply, sample, seeds}; }

}  // namespace arc
