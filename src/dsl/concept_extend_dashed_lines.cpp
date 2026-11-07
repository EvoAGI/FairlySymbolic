// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COMPLETE_REGULARITY rule=dashes_h_last | dashes_v_last (formerly the op
// EXTEND_DASHED_LINES h_last=true | false): a colour that appears at least
// twice on a row (column), with gaps between, becomes a full row (column).
// ConceptARC "CompleteShape". Reached from concept_regularity.cpp.
//
// dashes_h_last: rows are drawn after columns (rows win at crossings);
// dashes_v_last: columns after rows.

#include <set>
#include <tuple>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

// True when the positions hold >= 2 cells that are not contiguous.
bool dashed(const std::vector<int>& pos) {
  return pos.size() >= 2 && pos.back() - pos.front() + 1 > static_cast<int>(pos.size());
}

}  // namespace

// COMPLETE_REGULARITY rule=dashes_* (concept_regularity.cpp).
void extend_dashed_lines(Scene& scene, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::pair<int, int>> rows, cols;  // (colour, index)
  for (int y = 0; y < H; ++y) {
    std::set<int> cs(g[y].begin(), g[y].end());
    cs.erase(bg);
    for (int c : cs) {
      std::vector<int> pos;
      for (int x = 0; x < W; ++x)
        if (g[y][x] == c) pos.push_back(x);
      if (dashed(pos)) rows.emplace_back(c, y);
    }
  }
  for (int x = 0; x < W; ++x) {
    std::set<int> cs;
    for (int y = 0; y < H; ++y) cs.insert(g[y][x]);
    cs.erase(bg);
    for (int c : cs) {
      std::vector<int> pos;
      for (int y = 0; y < H; ++y)
        if (g[y][x] == c) pos.push_back(y);
      if (dashed(pos)) cols.emplace_back(c, x);
    }
  }
  if (rows.empty() && cols.empty()) return;
  auto draw_rows = [&] { for (auto& r : rows) for (int x = 0; x < W; ++x) g[r.second][x] = r.first; };
  auto draw_cols = [&] { for (auto& c : cols) for (int y = 0; y < H; ++y) g[y][c.second] = c.first; };
  if (arg_str(action.args, "rule", "dashes_h_last") == "dashes_h_last") {
    draw_cols();
    draw_rows();
  } else {
    draw_rows();
    draw_cols();
  }
  concept_install(scene, g);
}

static DslArgs make(bool h_last) { return {{"rule", cdv_str(h_last ? "dashes_h_last" : "dashes_v_last")}}; }

DslArgs extend_dashed_lines_sample(ArcRandom& rng) { return make(pick_int(rng, {0, 1}) == 1); }

std::vector<DslArgs> extend_dashed_lines_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return {};
  return {make(true), make(false)};
}

}  // namespace arc
