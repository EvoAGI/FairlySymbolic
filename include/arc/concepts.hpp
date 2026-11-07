// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Concept primitives: DSL actions added for the ConceptARC concept study.
//
// Each primitive registers itself in concept_ops() (src/dsl/concepts.cpp). The
// hooks in the action dispatcher, the codec and the seed enumerator only fire
// for a registered kind, so with none registered they change nothing.
#pragma once

#include <string>
#include <vector>

#include "arc/dsl.hpp"
#include "arc/graph.hpp"
#include "arc/grid.hpp"
#include "arc/rng.hpp"

namespace arc {

struct ConceptOp {
  ActionKind kind;
  int weight;  // weight in the GA action table
  void (*apply)(Scene& scene, const std::vector<std::string>& selected_ids,
                const DslAction& action);
  DslArgs (*sample_args)(ArcRandom& rng);
  // Argument settings tried in the seed population, from the train pairs.
  std::vector<DslArgs> (*seed_args)(const std::vector<Grid>& inputs,
                                    const std::vector<Grid>& outputs);
};

const std::vector<ConceptOp>& concept_ops();
const ConceptOp* find_concept_op(ActionKind kind);

// Names of ops since folded into others (keep_by_rays, keep_by_containment,
// paint_from_marker, symmetry_complete, keep_extreme, delete_center, extract_unique, extract_tile, concat_panels, crop_hole,
// weave, jump_over, extend_dashed_lines, complete_frame, sort_panels, orient_mark, recolor_rel): rewrites the name and
// args to the op and mode that now hold them. False for any other name.
bool concept_alias(const std::string& name, ActionKind* kind, DslArgs* args);

// Runs a concept op. With ARC_CONCEPT_SCOPE=1 and a selection that leaves some
// objects out, its changes are kept only inside the bounding boxes of the
// selected objects (ops that read the selection themselves, and ops that
// change the grid size, are left as they are).
void apply_concept_op(const ConceptOp& op, Scene& scene, const std::vector<std::string>& selected_ids,
                      const DslAction& action);

// ---- helpers shared by the primitives --------------------------------------
// The primitives work on the rendered grid. Background is the scene's (from
// the evolved parser config), else colour 0.
using CGrid = std::vector<std::vector<int>>;
int concept_bg(const Scene& scene);
CGrid concept_render(const Scene& scene);
// concept_render plus the grid-line cells the encoder keeps out of the objects
// (full one-colour rows and columns), read from the scene grid.
CGrid concept_render_full(const Scene& scene);
void concept_install(Scene& scene, const CGrid& g);

// ARC_TASK_OTHER=1: the colours of the task's train inputs, most frequent
// first (set per task by the runner, empty otherwise). patch objbox
// fill=other falls back to them when the grid has no other colour.
extern thread_local std::vector<int> g_task_colors;

DslValue cdv_int(long long v);
DslValue cdv_str(const std::string& v);
DslValue cdv_bool(bool v);

// Uniform pick from a list, consuming the GA random stream.
int pick_int(ArcRandom& rng, const std::vector<int>& xs);
std::string pick_str(ArcRandom& rng, const std::vector<std::string>& xs);

// Connected components of non-background cells, row-major discovery order.
// mono: same colour only. diag: 8-connectivity.
std::vector<std::vector<Coord>> concept_components(const CGrid& g, int bg, bool diag, bool mono);
// Single-colour components (8-connected with diag), with the lone cells of a
// colour joined when at most two apart (Chebyshev): a dotted outline is one group.
std::vector<std::vector<Coord>> concept_dotted_groups(const CGrid& g, int bg, bool diag);

struct Box { int y0, x0, y1, x1; };
Box bbox_of(const std::vector<Coord>& cells);

}  // namespace arc
