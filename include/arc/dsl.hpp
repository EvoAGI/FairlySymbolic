// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_DSL_HPP
#define ARC_DSL_HPP

// DSL executor, selectors and decoder:
//
//   * select_node_ids(scene, selector, depth) -> ordered node-id list, for
//     every one of the 19 SelectorKinds. The order is deterministic and ties
//     always resolve the same way.
//   * guards_pass(ids, guards) -> MIN/MAX/EXACT_COUNT.
//   * execute_program_gout(...) -> the executor rule-loop: for each rule
//     select -> guard -> apply_action -> re-encode-if-next-selector-not-SAFE.
//     Returns the mutated G_out scene (no decode).
//
// The AST here does NOT reuse arc::Selector/ArgMap (ast.hpp): selector args
// carry NESTED DICTS (e.g. left_selector_args) rather than nested Selector
// objects, so a small json-tree value type (DslValue / DslArgs) models the arg
// dicts. A minimal JSON reader parses compact program/selector JSON into these
// structs.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "arc/ast.hpp"      // SelectorKind / ActionKind / GuardKind / GlobalTransform enums
#include "arc/config.hpp"
#include "arc/graph.hpp"
#include "arc/grid.hpp"

namespace arc {

// Per-task encoder scene cache (scene_cache.hpp). Non-owning pointer only, so a
// forward declaration suffices for every header that threads it through.
class SceneCache;

// ---------------------------------------------------------------------------
// DSL arg value tree: int / float / str / bool, plus a nested dict (used by
// *_selector_args). Absent lookups return nullptr.
// ---------------------------------------------------------------------------

struct DslValue;
using DslArgs = std::vector<std::pair<std::string, DslValue>>;  // insertion-ordered

struct DslValue {
  enum class Tag { Int, Float, Str, Bool, Dict, Null };
  Tag tag = Tag::Null;
  long long i = 0;
  double f = 0.0;
  std::string s;
  bool b = false;
  std::shared_ptr<DslArgs> dict;  // non-null iff tag == Dict
};

// Insertion-ordered lookup over a DslArgs map. Returns nullptr when absent.
const DslValue* args_get(const DslArgs& args, const std::string& key);

struct DslSelector {
  SelectorKind kind = SelectorKind::ALL_SUBOBJECTS;
  DslArgs args;
};

struct DslGuard {
  GuardKind kind = GuardKind::MIN_COUNT;
  DslArgs args;
};

struct DslAction {
  ActionKind kind = ActionKind::RECOLOR;
  DslArgs args;
};

struct DslRule {
  DslSelector selector;
  DslAction action;
  std::vector<DslGuard> guards;
};

struct DslProgram {
  std::vector<DslRule> rules;
  GlobalTransform global_transform = GlobalTransform::IDENTITY;
};

// ---------------------------------------------------------------------------
// JSON readers for compact program JSON. Return false on a
// structural parse error (no exceptions). The selector/program JSON shapes:
//   selector: {"kind":"by_color","args":{"color":1}}
//   guard:    {"kind":"min_count","args":{"value":2}}
//   action:   {"kind":"recolor","args":{"color":3}}
//   program:  {"rules":[{"selector":{...},"guards":[...],"action":{...}}],
//              "global_transform":"identity"}
// ---------------------------------------------------------------------------

bool parse_selector_json(const std::string& json, DslSelector* out);

// Selector name -> kind, for a program's selectors and for the selector names
// nested in args (other_selector_kind, reference_selector_kind, ...). An old
// name of a folded kind (selector_aliases) gives the merged kind and puts its
// variant arg first in *args. False for an unknown name (*kind, *args kept).
bool selector_from_name(const std::string& name, SelectorKind* kind, DslArgs* args);
// The same for a core action name (action_aliases); a concept-op name goes
// through concept_alias in the program parser.
bool action_from_name(const std::string& name, ActionKind* kind, DslArgs* args);
// The variant value of a merged core action (its first value when the arg is
// absent), "" for the other kinds.
std::string action_variant(const DslAction& a);
bool parse_program_json(const std::string& json, DslProgram* out);

// ---------------------------------------------------------------------------
// Selection / guards / executor.
// ---------------------------------------------------------------------------

// select_node_ids: ordered node ids. depth caps recursion (>=4 -> empty).
std::vector<std::string> select_node_ids(const Scene& scene,
                                          const DslSelector& selector,
                                          int depth = 0);

// guards_pass: MIN/MAX/EXACT_COUNT over the selected-id count.
bool guards_pass(const std::vector<std::string>& selected_ids,
                 const std::vector<DslGuard>& guards);

// apply_action_to_scene: apply one action to the selected nodes. Mutates the
// scene.
void apply_action_to_scene(Scene& scene,
                           const std::vector<std::string>& selected_ids,
                           const DslAction& action);

// execute_program_gout: the executor rule-loop over a freshly encoded scene.
// Runs select -> guard -> apply_action for each rule; between rules, re-encodes
// (render G_out to grid, encode_minimal) only when the NEXT rule's selector is
// NOT a safe kind (one that reads only attributes the actions keep in sync:
// color/mass/bbox/centroid). Returns the mutated G_out scene (no decode, no
// global transform). encoder_cfg drives the initial + re-encode.
// scene_cache (optional): when non-null, the INITIAL encode of the
// raw input grid is served from a per-task SceneCache (the re-encode of a
// re-rendered intermediate grid is always fresh). nullptr => byte-identical to
// the no-cache path.
Scene execute_program_gout(const Grid& grid, const DslProgram& program,
                           const GraphConfig& encoder_cfg,
                           SceneCache* scene_cache = nullptr);

// The 26 GlobalTransforms in the executor path.
//
// apply_full_grid_global: whole-grid op (every cell, bg included) for the
// permutation + mirror-tiling / tile / dedup / block-reduce family. Out-of-
// family transforms return the grid unchanged.
Grid apply_full_grid_global(const Grid& g, GlobalTransform t);

// GlobalTransform family predicates. Exposed so the fitness executor can
// reproduce execute_program_final_grid's branch selection with a non-default
// DecoderConfig.
bool is_permutation_transform(GlobalTransform t);
bool is_full_grid_transform(GlobalTransform t);
bool is_fast_path_transform(GlobalTransform t);

// apply_global_transform_to_scene: the FOREGROUND-only path (remap SubObject
// pixels, rebuild). IDENTITY + full-grid transforms are no-ops here.
void apply_global_transform_to_scene(Scene& scene, GlobalTransform t);

// execute_program_final_grid: the FULL executor incl. the global transform,
// returning the final DECODED grid. It has three branches (empty-rule fast
// path / with-rules full-grid / scene-space + render).
Grid execute_program_final_grid(const Grid& grid, const DslProgram& program,
                                const GraphConfig& encoder_cfg);

// Render the scene to a dense grid using the DEFAULT DecoderConfig with NO
// input grid (the exact call the executor's between-rule re-encode makes):
// background = scene bg or 0, paint SubObjects in lexicographic id order,
// overwrite. Exposed for the re-encode path.
Grid render_scene_default(const Scene& scene);

// Decoder policy. Strings are the canonical policy names; the defaults
// are the first entry of each policy list (the default DecoderConfig).
struct DecoderConfig {
  std::string background_policy = "scene_background_or_zero";
  std::string paint_order = "as_is";
  std::string overlap_policy = "overwrite";
  std::string postprocess = "none";
};

// render_scene_to_grid(scene, cfg, input_grid): the FULL decoder honoring all
// four policy axes (background fill, paint order, overlap, postprocess).
// `input_grid` is optional (may be nullptr); it is only consulted by the
// most_frequent_input / input_copy
// background policies. render_scene_default(scene) == render_scene_to_grid with
// the default DecoderConfig and no input grid.
Grid render_scene_to_grid(const Scene& scene, const DecoderConfig& cfg,
                          const Grid* input_grid);

}  // namespace arc

#endif  // ARC_DSL_HPP
