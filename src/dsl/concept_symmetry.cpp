// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COMPLETE_REGULARITY rule=sym_<axis>: each object is completed to be mirror
// symmetric about its best axis, the one where the most cells have a filled
// mirror image (not the bbox centre, which a missing cell shifts). ConceptARC
// "CompleteShape". (Formerly symmetry_complete axis=<axis>.)
//
// <axis>: lr (a vertical axis) | ud (a horizontal one) | both (lr, then ud) |
//         auto (per object, whichever of lr and ud misses fewer mirror
//         cells per present one, among the axes worth completing);
//         all_lr | all_ud | all_both | all_auto treat the whole picture as one
//         object (for shapes made of separate parts), and also complete it
//         when up to half of it is missing
// Objects are 8-connected groups of any colour. Missing cells take their
// mirror's colour; mirrors that fall outside the grid are ignored.

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

DslArgs make(const std::string& axis) {
  DslArgs a;
  a.emplace_back("rule", cdv_str("sym_" + axis));
  return a;
}

struct Axis {
  int s = -1, match = 0, miss = 0;  // doubled position: a cell at c mirrors to s - c
};

// The best axis of comp: the most cells with a filled mirror image, then the fewest
// without. lr mirrors x, otherwise y.
Axis best_axis(const CGrid& g, int bg, const std::vector<Coord>& comp, bool lr) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  Box b = bbox_of(comp);
  int lo = lr ? 2 * b.x0 : 2 * b.y0, hi = lr ? 2 * b.x1 : 2 * b.y1;
  Axis best;
  for (int s = lo; s <= hi; ++s) {
    int match = 0, miss = 0;
    for (const Coord& c : comp) {
      int y = lr ? c.first : s - c.first, x = lr ? s - c.second : c.second;
      if (y < 0 || y >= H || x < 0 || x >= W) continue;
      if (y == c.first && x == c.second) continue;
      if (g[y][x] != bg) ++match; else ++miss;
    }
    if (match > best.match || (match == best.match && miss < best.miss)) best = {s, match, miss};
  }
  return best;
}

// Only when the shape is mostly symmetric already: few cells to add.
bool worth(const Axis& a, bool whole) { return a.s >= 0 && a.miss > 0 && a.miss * (whole ? 1 : 3) <= a.match; }

// One pass over every object; mode 0 lr, 1 ud, 2 auto.
CGrid complete(const CGrid& g, int bg, int mode, bool whole) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid out = g;
  std::vector<std::vector<Coord>> comps = concept_components(g, bg, true, false);
  if (whole && comps.size() > 1) {
    std::vector<Coord> all;
    for (const auto& c : comps) all.insert(all.end(), c.begin(), c.end());
    comps = {all};
  }
  for (const auto& comp : comps) {
    if (comp.size() < 3) continue;
    bool lr = mode != 1;
    Axis a = best_axis(g, bg, comp, lr);
    if (mode == 2) {
      Axis v = best_axis(g, bg, comp, false);
      bool ok_a = worth(a, whole), ok_v = worth(v, whole);
      if (ok_v && (!ok_a || static_cast<long>(v.miss) * a.match < static_cast<long>(a.miss) * v.match)) {
        a = v;
        lr = false;
      }
    }
    if (!worth(a, whole)) continue;
    int best = a.s;
    for (const Coord& c : comp) {
      int y = lr ? c.first : best - c.first, x = lr ? best - c.second : c.second;
      if (y >= 0 && y < H && x >= 0 && x < W && out[y][x] == bg) out[y][x] = g[c.first][c.second];
    }
  }
  return out;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  std::string axis = arg_str(action.args, "rule", "sym_both").substr(4);
  bool whole = axis.rfind("all_", 0) == 0;
  if (whole) axis = axis.substr(4);
  if (axis == "auto") {
    g = complete(g, bg, 2, whole);
  } else {
    if (axis != "ud") g = complete(g, bg, 0, whole);
    if (axis != "lr") g = complete(g, bg, 1, whole);
  }
  concept_install(scene, g);
}

}  // namespace

// COMPLETE_REGULARITY rule=sym_* (concept_regularity.cpp).
void symmetry_complete(Scene& scene, const DslAction& action) { apply(scene, {}, action); }

// auto comes up twice as often: it covers what lr and ud do on one-axis shapes.
DslArgs symmetry_complete_sample(ArcRandom& rng) {
  return make(pick_str(rng, {"lr", "ud", "both", "auto", "auto", "all_lr", "all_ud", "all_both", "all_auto",
                             "all_auto"}));
}

// Seeds only when every train output has the input's size and keeps every
// non-background input cell.
std::vector<DslArgs> symmetry_complete_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    const auto& a = ins[i].cells();
    const auto& b = outs[i].cells();
    for (size_t k = 0; k < a.size(); ++k)
      if (a[k] != 0 && a[k] != b[k]) return res;
  }
  for (const char* a : {"both", "lr", "ud", "auto", "all_both", "all_lr", "all_ud", "all_auto"}) res.push_back(make(a));
  return res;
}

}  // namespace arc
