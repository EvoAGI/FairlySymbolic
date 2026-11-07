// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// COMPLETE_REGULARITY rule=frame (formerly the op COMPLETE_FRAME, no args): for
// each colour, the cells of that colour are taken as the fragments of one
// rectangle frame, which is drawn in full on their bounding box; cells of that
// colour inside the frame are cleared. ConceptARC "CompleteShape". Reached
// from concept_regularity.cpp.

#include <set>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

// COMPLETE_REGULARITY rule=frame (concept_regularity.cpp).
void complete_frame(Scene& scene) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::set<int> colors;
  for (const auto& r : g)
    for (int v : r)
      if (v != bg) colors.insert(v);
  if (colors.empty()) return;
  CGrid out = g;
  for (int c : colors) {
    int y0 = H, x0 = W, y1 = -1, x1 = -1;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (g[y][x] == c) {
          y0 = std::min(y0, y); x0 = std::min(x0, x);
          y1 = std::max(y1, y); x1 = std::max(x1, x);
        }
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        if (y == y0 || y == y1 || x == x0 || x == x1) out[y][x] = c;
        else if (g[y][x] == c) out[y][x] = bg;
      }
  }
  concept_install(scene, out);
}

DslArgs complete_frame_args() { return {{"rule", cdv_str("frame")}}; }

std::vector<DslArgs> complete_frame_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return {};
  return {complete_frame_args()};
}

}  // namespace arc
