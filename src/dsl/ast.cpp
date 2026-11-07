// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL AST enum tables and GA active-operator tables.
//
// The (name, value) tables below are in declaration order; the enumerator
// integer value equals the row index. The ACTIVE_* tables list the operators
// the GA draws, in a fixed order with a per-entry weight.

#include "arc/ast.hpp"
#include "arc/concepts.hpp"

namespace arc {

namespace {

// (enumerator name, serialized value) rows, in declaration order.
const std::vector<std::pair<std::string, std::string>> kSelectorTable = {
    {"ALL_SUBOBJECTS", "all_subobjects"},
    {"BY_COLOR", "by_color"},
    {"NON_BACKGROUND", "non_background"},
    {"BY_SHAPE", "by_shape"},
    {"BY_MASS", "by_mass"},
    {"BY_BOUNDARY_TOUCH", "by_boundary_touch"},
    {"BY_COLOR_FREQUENCY", "by_color_frequency"},
    {"TOUCHING_COLOR", "touching_color"},
    {"EDGE_COUNT", "edge_count"},
    {"EXTREME", "extreme"},
    {"INSIDE_LARGEST_BBOX", "inside_largest_bbox"},
    {"SAME_SHAPE_AS_REFERENCE", "same_shape_as_reference"},
    {"REFERENCE_SINGLETONS", "reference_singletons"},
    {"BY_NOISE_SCORE", "by_noise_score"},
    {"DISTANCE_TO_SELECTOR", "distance_to_selector"},
    {"ADJACENT_TO_COLOR", "adjacent_to_color"},
    {"SELECTOR_SET_OP", "selector_set_op"},
    {"BY_HOLE_COUNT", "by_hole_count"},
    {"BY_ATTR", "by_attr"},
};

const std::vector<std::pair<std::string, std::string>> kActionTable = {
    {"RECOLOR", "recolor"},
    {"RECOLOR_TO", "recolor_to"},
    {"RECOLOR_FROM_EDGE_MAJORITY", "recolor_from_edge_majority"},
    {"RECOLOR_FROM_REFERENCE", "recolor_from_reference"},
    {"RECOLOR_SEQUENCE", "recolor_sequence"},
    {"SWAP_COLORS", "swap_colors"},
    {"PRUNE_ENDPOINTS", "prune_endpoints"},
    {"SHRINK_TO_CORE", "shrink_to_core"},
    {"DILATE_SELECTED", "dilate_selected"},
    {"FILL_BBOX", "fill_bbox"},
    {"OUTLINE_OBJECT", "outline_object"},
    {"TRANSLATE_BY_OWN_EXTENT", "translate_by_own_extent"},
    {"CROP_TO_OBJECT", "crop_to_object"},
    {"PANEL_SPLIT_BOOLEAN", "panel_split_boolean"},
    {"SCALE_BY_COUNT", "scale_by_count"},
    {"COMPLETE_D4_SYMMETRY", "complete_d4_symmetry"},
    {"SYMMETRIZE_OBJECT", "symmetrize_object"},
    {"PROJECT_INNER_QUAD_TO_OUTER_CORNERS", "project_inner_quad_to_outer_corners"},
    {"PROJECT_SINGLETONS_TO_FULL_LINES", "project_singletons_to_full_lines"},
    {"STAMP_LOCAL_PATTERN", "stamp_local_pattern"},
    {"EXTEND_LINE_UNTIL", "extend_line_until"},
    {"PROJECT_MARKER", "project_marker"},
    {"ATTACH_SAME_COLOR_SINGLETONS", "attach_same_color_singletons"},
    {"DELETE_REFERENCE_AFTER_TRANSFER", "delete_reference_after_transfer"},
    {"DELETE", "delete"},
    {"TRANSLATE", "translate"},
    {"MIRROR", "mirror"},
    {"KEEP_SELECTED", "keep_selected"},
    {"FILL_CAVITY", "fill_cavity"},
    {"FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS",
     "fill_between_collinear_same_color_pairs"},
    {"GRAVITY_TO_EDGE", "gravity_to_edge"},
    {"FILL_ENCLOSED_BG", "fill_enclosed_bg"},
    {"FILL_DIAGONAL_PERIODIC", "fill_diagonal_periodic"},
    {"TILE_FOLD_OR_NONBG", "tile_fold_or_nonbg"},
    {"COUNT_BAR", "count_bar"},
    {"SORT_OBJECTS", "sort_objects"},
    {"COPY_INTO_FRAMES", "copy_into_frames"},
    {"STAMP_AT_MARKERS", "stamp_at_markers"},
    {"REPEAT_BY_MARKERS", "repeat_by_markers"},
    {"SLIDE_UNTIL_BLOCKED", "slide_until_blocked"},
    {"DENOISE", "denoise"},
    {"RAY_UNTIL_BLOCKED", "ray_until_blocked"},
    {"CENTER_ON", "center_on"},
    {"TOGGLE_FILL", "toggle_fill"},
    {"ATTR_PAINT", "attr_paint"},
    {"CONCAT_OBJECTS", "concat_objects"},
    {"PATCH", "patch"},
    {"EXTRACT", "extract"},
    {"BRING_TO_FRONT", "bring_to_front"},
    {"COMPLETE_COPIES", "complete_copies"},
    {"KEEP_ENCLOSED", "keep_enclosed"},
    {"COMPLETE_REGULARITY", "complete_regularity"},
    {"MARK_SIDE", "mark_side"},
    {"TILE_GAP", "tile_gap"},
    {"TRANSFER_PATTERN", "transfer_pattern"},
};

const std::vector<std::pair<std::string, std::string>> kGlobalTransformTable = {
    {"IDENTITY", "identity"},
    {"ROT90", "rot90"},
    {"ROT180", "rot180"},
    {"ROT270", "rot270"},
    {"FLIP_H", "flip_h"},
    {"FLIP_V", "flip_v"},
    {"TRANSPOSE", "transpose"},
    {"ANTI_TRANSPOSE", "anti_transpose"},
    {"KRON_SELF_MASK", "kron_self_mask"},
    {"KRON_MASK_ANCHOR_COLOR", "kron_mask_anchor_color"},
    {"CROP_FOREGROUND", "crop_foreground"},
    {"SCALE2_NEAREST", "scale2_nearest"},
    {"SCALE3_NEAREST", "scale3_nearest"},
    {"FILL_PERIODIC_BACKGROUND", "fill_periodic_background"},
    {"MIRROR_CONCAT_H", "mirror_concat_h"},
    {"MIRROR_CONCAT_H_REV", "mirror_concat_h_rev"},
    {"MIRROR_CONCAT_V", "mirror_concat_v"},
    {"MIRROR_CONCAT_V_REV", "mirror_concat_v_rev"},
    {"MIRROR_QUAD_TL", "mirror_quad_tl"},
    {"MIRROR_QUAD_TR", "mirror_quad_tr"},
    {"MIRROR_QUAD_BL", "mirror_quad_bl"},
    {"MIRROR_QUAD_BR", "mirror_quad_br"},
    {"TILE_H", "tile_h"},
    {"TILE_V", "tile_v"},
    {"DEDUP_RUNS", "dedup_runs"},
    {"BLOCK_REDUCE2", "block_reduce2"},
};

const std::vector<std::pair<std::string, std::string>> kGuardTable = {
    {"MIN_COUNT", "min_count"},
    {"MAX_COUNT", "max_count"},
    {"EXACT_COUNT", "exact_count"},
};

// GA active-action table: (kind, weight), in a fixed order (sum 94). A merged
// kind has one entry at the place of the first kind it replaces, weighing what
// the folded kinds weighed together; its variant is drawn with their weights
// (kActionVariants), so the GA draws each old kind as often as before.
struct ActiveAction {
  ActionKind kind;
  int weight;
};

const std::vector<ActiveAction> kActiveActionTable = {
    {ActionKind::RECOLOR, 8},
    {ActionKind::RECOLOR_TO, 5},
    {ActionKind::RECOLOR_FROM_EDGE_MAJORITY, 1},
    {ActionKind::RECOLOR_FROM_REFERENCE, 4},
    {ActionKind::RECOLOR_SEQUENCE, 4},
    {ActionKind::SWAP_COLORS, 2},
    {ActionKind::PRUNE_ENDPOINTS, 3},
    {ActionKind::SHRINK_TO_CORE, 3},
    {ActionKind::DILATE_SELECTED, 1},
    {ActionKind::FILL_BBOX, 4},
    {ActionKind::COMPLETE_D4_SYMMETRY, 2},
    {ActionKind::SYMMETRIZE_OBJECT, 1},
    {ActionKind::PROJECT_INNER_QUAD_TO_OUTER_CORNERS, 2},
    {ActionKind::PROJECT_SINGLETONS_TO_FULL_LINES, 2},
    {ActionKind::STAMP_LOCAL_PATTERN, 3},
    {ActionKind::EXTEND_LINE_UNTIL, 1},
    {ActionKind::PROJECT_MARKER, 1},
    {ActionKind::ATTACH_SAME_COLOR_SINGLETONS, 2},
    {ActionKind::DELETE_REFERENCE_AFTER_TRANSFER, 3},
    {ActionKind::DELETE, 4},
    {ActionKind::TRANSLATE, 4},
    {ActionKind::MIRROR, 4},
    {ActionKind::KEEP_SELECTED, 13},
    {ActionKind::FILL_CAVITY, 1},
    {ActionKind::OUTLINE_OBJECT, 2},
    {ActionKind::TRANSLATE_BY_OWN_EXTENT, 2},
    {ActionKind::GRAVITY_TO_EDGE, 2},
    {ActionKind::FILL_ENCLOSED_BG, 2},
    {ActionKind::FILL_DIAGONAL_PERIODIC, 1},
    {ActionKind::TILE_FOLD_OR_NONBG, 1},
    {ActionKind::FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS, 1},
    {ActionKind::CROP_TO_OBJECT, 2},
    {ActionKind::PANEL_SPLIT_BOOLEAN, 2},
    {ActionKind::SCALE_BY_COUNT, 1},
};

// Merged core actions: variant key and (value, GA weight of the old kind).
const std::vector<ActionVariants> kActionVariants = {
    {ActionKind::RECOLOR_TO, "to", {{"background", 2}, {"most", 1}, {"least", 2}}},
    {ActionKind::MIRROR, "axis", {{"h", 2}, {"v", 2}}},
    {ActionKind::FILL_BBOX, "part", {{"gaps", 2}, {"interior", 2}}},
    {ActionKind::KEEP_SELECTED, "out", {{"grid", 1}, {"crop", 12}}},
    {ActionKind::TRANSLATE, "mode", {{"move", 1}, {"wrap", 2}, {"copy", 1}, {"origin", 0}}},
};

// Old names of folded core actions.
const std::vector<ActionAlias> kActionAliases = {
    {"recolor_to_background", ActionKind::RECOLOR_TO, "background"},
    {"recolor_to_most_frequent", ActionKind::RECOLOR_TO, "most"},
    {"recolor_to_least_frequent", ActionKind::RECOLOR_TO, "least"},
    {"hollow", ActionKind::FILL_BBOX, "interior"},
    {"mirror_h", ActionKind::MIRROR, "h"},
    {"mirror_v", ActionKind::MIRROR, "v"},
    {"keep_selected_normalized", ActionKind::KEEP_SELECTED, "crop"},
    {"translate_wrap", ActionKind::TRANSLATE, "wrap"},
    {"copy_translate", ActionKind::TRANSLATE, "copy"},
    {"normalize_to_origin", ActionKind::TRANSLATE, "origin"},
};

// GA selector table: one (kind, weight) entry per kind. A merged kind weighs
// what its folded kinds weighed together; its variant is then drawn with their
// weights (kSelectorVariants), so the GA draws each old kind as often as
// before (sum 81).
struct ActiveSelector {
  SelectorKind kind;
  int weight;
};

const std::vector<ActiveSelector> kActiveSelectorTable = {
    {SelectorKind::ALL_SUBOBJECTS, 8},
    {SelectorKind::BY_COLOR, 6},
    {SelectorKind::NON_BACKGROUND, 2},
    {SelectorKind::BY_SHAPE, 1},
    {SelectorKind::BY_MASS, 6},
    {SelectorKind::BY_BOUNDARY_TOUCH, 3},
    {SelectorKind::BY_COLOR_FREQUENCY, 14},
    {SelectorKind::TOUCHING_COLOR, 1},
    {SelectorKind::EDGE_COUNT, 5},
    {SelectorKind::EXTREME, 16},
    {SelectorKind::SAME_SHAPE_AS_REFERENCE, 2},
    {SelectorKind::REFERENCE_SINGLETONS, 3},
    {SelectorKind::BY_NOISE_SCORE, 1},
    {SelectorKind::DISTANCE_TO_SELECTOR, 4},
    {SelectorKind::ADJACENT_TO_COLOR, 1},
    {SelectorKind::SELECTOR_SET_OP, 1},
    {SelectorKind::BY_HOLE_COUNT, 1},
    {SelectorKind::BY_ATTR, 6},
};

// Merged selector kinds: variant key and (value, GA weight of the old kind),
// in the order of the old table entries.
const std::vector<SelectorVariants> kSelectorVariants = {
    {SelectorKind::BY_MASS, "op", {{"ge", 1}, {"le", 2}, {"eq", 3}}},
    {SelectorKind::BY_COLOR_FREQUENCY, "rank", {{"most", 11}, {"least", 3}}},
    {SelectorKind::EDGE_COUNT, "op", {{"any", 1}, {"none", 3}, {"ge", 1}}},
    {SelectorKind::EXTREME, "pick", {{"largest", 3}, {"smallest", 5}, {"leftmost", 1}, {"rightmost", 2},
                                     {"topmost", 1}, {"bottommost", 1}, {"top_left", 1},
                                     {"top_left_nondominant", 1}, {"left_nondominant", 1}}},
    {SelectorKind::DISTANCE_TO_SELECTOR, "dir", {{"nearest", 2}, {"farthest", 2}}},
};

// Old names of folded selector kinds.
const std::vector<SelectorAlias> kSelectorAliases = {
    {"edge_to_color", SelectorKind::TOUCHING_COLOR, ""},
    {"has_edge_type", SelectorKind::EDGE_COUNT, "any"},
    {"has_no_edge_type", SelectorKind::EDGE_COUNT, "none"},
    {"edge_count_at_least", SelectorKind::EDGE_COUNT, "ge"},
    {"by_mass_at_least", SelectorKind::BY_MASS, "ge"},
    {"by_mass_at_most", SelectorKind::BY_MASS, "le"},
    {"by_mass_exact", SelectorKind::BY_MASS, "eq"},
    {"largest_subobject", SelectorKind::EXTREME, "largest"},
    {"smallest_subobject", SelectorKind::EXTREME, "smallest"},
    {"leftmost_subobject", SelectorKind::EXTREME, "leftmost"},
    {"rightmost_subobject", SelectorKind::EXTREME, "rightmost"},
    {"topmost_subobject", SelectorKind::EXTREME, "topmost"},
    {"bottommost_subobject", SelectorKind::EXTREME, "bottommost"},
    {"top_leftmost_subobject", SelectorKind::EXTREME, "top_left"},
    {"top_leftmost_non_most_frequent", SelectorKind::EXTREME, "top_left_nondominant"},
    {"leftmost_non_most_frequent", SelectorKind::EXTREME, "left_nondominant"},
    {"most_frequent_color_objects", SelectorKind::BY_COLOR_FREQUENCY, "most"},
    {"least_frequent_color_objects", SelectorKind::BY_COLOR_FREQUENCY, "least"},
    {"nearest_to_selector", SelectorKind::DISTANCE_TO_SELECTOR, "nearest"},
    {"farthest_from_selector", SelectorKind::DISTANCE_TO_SELECTOR, "farthest"},
};

template <typename K>
const std::string& value_at(
    const std::vector<std::pair<std::string, std::string>>& table, K k) {
  return table[static_cast<size_t>(static_cast<int>(k))].second;
}

template <typename K>
const std::string& name_at(
    const std::vector<std::pair<std::string, std::string>>& table, K k) {
  return table[static_cast<size_t>(static_cast<int>(k))].first;
}

template <typename K>
bool from_value(const std::vector<std::pair<std::string, std::string>>& table,
                const std::string& value, K* out) {
  for (size_t i = 0; i < table.size(); ++i) {
    if (table[i].second == value) {
      *out = static_cast<K>(static_cast<int>(i));
      return true;
    }
  }
  return false;
}

}  // namespace

const std::string& to_string(SelectorKind k) { return value_at(kSelectorTable, k); }
const std::string& to_string(ActionKind k) { return value_at(kActionTable, k); }
const std::string& to_string(GlobalTransform k) {
  return value_at(kGlobalTransformTable, k);
}
const std::string& to_string(GuardKind k) { return value_at(kGuardTable, k); }

const std::string& to_name(SelectorKind k) { return name_at(kSelectorTable, k); }
const std::string& to_name(ActionKind k) { return name_at(kActionTable, k); }
const std::string& to_name(GlobalTransform k) {
  return name_at(kGlobalTransformTable, k);
}
const std::string& to_name(GuardKind k) { return name_at(kGuardTable, k); }

bool selector_kind_from_string(const std::string& value, SelectorKind* out) {
  return from_value(kSelectorTable, value, out);
}
bool action_kind_from_string(const std::string& value, ActionKind* out) {
  return from_value(kActionTable, value, out);
}
bool global_transform_from_string(const std::string& value, GlobalTransform* out) {
  return from_value(kGlobalTransformTable, value, out);
}
bool guard_kind_from_string(const std::string& value, GuardKind* out) {
  return from_value(kGuardTable, value, out);
}

const std::vector<std::pair<std::string, std::string>>& selector_kind_table() {
  return kSelectorTable;
}
const std::vector<std::pair<std::string, std::string>>& action_kind_table() {
  return kActionTable;
}
const std::vector<std::pair<std::string, std::string>>& global_transform_table() {
  return kGlobalTransformTable;
}
const std::vector<std::pair<std::string, std::string>>& guard_kind_table() {
  return kGuardTable;
}

// With concept ops registered (arc/concepts.hpp) they are appended to the GA
// table; with none, the table is the base one.
const std::vector<ActionKind>& active_actions() {
  static const std::vector<ActionKind> t = [] {
    std::vector<ActionKind> v;
    for (const ActiveAction& e : kActiveActionTable) v.push_back(e.kind);
    for (const ConceptOp& op : concept_ops()) v.push_back(op.kind);
    return v;
  }();
  return t;
}
const std::vector<int>& active_action_weights() {
  static const std::vector<int> t = [] {
    std::vector<int> v;
    for (const ActiveAction& e : kActiveActionTable) v.push_back(e.weight);
    for (const ConceptOp& op : concept_ops()) v.push_back(op.weight);
    return v;
  }();
  return t;
}
const std::vector<SelectorKind>& active_selectors() {
  static const std::vector<SelectorKind> t = [] {
    std::vector<SelectorKind> v;
    for (const ActiveSelector& e : kActiveSelectorTable) v.push_back(e.kind);
    return v;
  }();
  return t;
}
const std::vector<int>& active_selector_weights() {
  static const std::vector<int> t = [] {
    std::vector<int> v;
    for (const ActiveSelector& e : kActiveSelectorTable) v.push_back(e.weight);
    return v;
  }();
  return t;
}
const std::vector<std::pair<SelectorKind, std::string>>& active_selector_leaves() {
  static const std::vector<std::pair<SelectorKind, std::string>> t = [] {
    std::vector<std::pair<SelectorKind, std::string>> v;
    for (const ActiveSelector& e : kActiveSelectorTable) {
      const SelectorVariants* sv = selector_variants(e.kind);
      if (!sv) {
        v.emplace_back(e.kind, "");
        continue;
      }
      for (const auto& kv : sv->values) v.emplace_back(e.kind, kv.first);
    }
    return v;
  }();
  return t;
}

const std::vector<SelectorAlias>& selector_aliases() { return kSelectorAliases; }

const std::vector<ActionAlias>& action_aliases() { return kActionAliases; }

const ActionVariants* action_variants(ActionKind k) {
  for (const ActionVariants& v : kActionVariants)
    if (v.kind == k) return &v;
  return nullptr;
}

const std::string& action_variant_key(ActionKind k) {
  static const std::string none;
  const ActionVariants* v = action_variants(k);
  return v ? v->key : none;
}

const SelectorVariants* selector_variants(SelectorKind k) {
  for (const SelectorVariants& v : kSelectorVariants)
    if (v.kind == k) return &v;
  return nullptr;
}

const std::string& selector_variant_key(SelectorKind k) {
  static const std::string none;
  const SelectorVariants* v = selector_variants(k);
  return v ? v->key : none;
}

}  // namespace arc
