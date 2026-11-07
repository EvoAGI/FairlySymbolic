// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_DEPLOY_HPP
#define ARC_DEPLOY_HPP

// Deployment helpers for the arc_solver binary (src/runtime/main.cpp): view
// inversion, the color prior, and the attempt_1 / attempt_2 construction.

#include <string>
#include <vector>

#include "arc/grid.hpp"
#include "arc/search.hpp"
#include "arc/taskio.hpp"

namespace arc {

// Map a view-space grid back into task space.
// transpose/fliplr/flipud/identity self-inverse; rot90 <-> rot270 swap.
Grid invert_view_grid(const Grid& g, const std::string& view);

// Parse an ArcConfig.view_continue value ("0"/""/"1"/comma list) into the view
// name list.
std::vector<std::string> parse_view_continue(const std::string& raw);

// The observed train-output colors (sorted, deduplicated) -- run_task's color
// prior.
std::vector<int> observed_output_colors(const std::vector<TrainPair>& pairs);

// attempt_1 / attempt_2 for one test input.
struct Attempts {
  Grid attempt_1;
  Grid attempt_2;
  // ARC_DUMP_CANDS=1 only: every candidate prediction with a label
  // ("primary", "view:<view>:loo=<n>:c=<complexity>", "fin:<i>:solved=<0|1>:c=<complexity>").
  std::vector<std::pair<std::string, Grid>> candidates;
};

// Build the per-test-input attempts from a run_task_full result, using a diverse
// multi-guess pool:
//   attempt_1 = primary genome executed on each test input (predict_grid).
//   attempt_2 = the top DISTINCT view solver's back-mapped prediction, else the
//               guess pool's diverse guess #2 (winner re-paired with other seed-
//               bank encoders that stay train-exact, then farthest-point
//               object-distance selection), else a copy of attempt_1.
// `train_pairs` gate the encoder re-pairing (a re-pairing is kept only if it
// stays train-exact); `test_inputs` are the task's test-input grids.
// `scene_cache` (optional, non-owning) serves the encodes of every prediction.
std::vector<Attempts> build_attempts(const TaskResultFull& res,
                                     const std::vector<TrainPair>& train_pairs,
                                     const std::vector<Grid>& test_inputs,
                                     SceneCache* scene_cache = nullptr);

}  // namespace arc

#endif  // ARC_DEPLOY_HPP
