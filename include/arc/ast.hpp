// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_AST_HPP
#define ARC_AST_HPP

// Typed AST primitives for the ARC graph DSL. Some selector and core-action
// kinds were merged into others; their old names still parse (see the alias
// tables below).
//
//   * SelectorKind (19), ActionKind (core + concept ops; see kActionKindCount),
//     GlobalTransform (26), GuardKind (3): each enumerator has a member NAME
//     (to_name) and a `.value` STRING (to_string). The `.value` string is what
//     the program hash and serialization use, so to_string / from_string MUST
//     reproduce it byte-for-byte.
//   * Selector / Guard / Action / Rule / Program structs. `args` is an
//     insertion-ordered, string-keyed heterogeneous value map (see ArgValue),
//     so a program always serializes to the same string.
//
// Also exposes the GA active-operator tables (active_actions /
// active_action_weights, active_selectors / active_selector_weights). GA
// sampling depends on them: order and weights must not change.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arc {

// ---------------------------------------------------------------------------
// Enums. The integer value is the declaration index (a stable mapping);
// to_string / from_string translate to and from the `.value` string.
// ---------------------------------------------------------------------------

enum class SelectorKind : int {
  ALL_SUBOBJECTS = 0,
  BY_COLOR,
  NON_BACKGROUND,
  BY_SHAPE,
  BY_MASS,
  BY_BOUNDARY_TOUCH,
  BY_COLOR_FREQUENCY,
  TOUCHING_COLOR,
  EDGE_COUNT,
  EXTREME,
  INSIDE_LARGEST_BBOX,
  SAME_SHAPE_AS_REFERENCE,
  REFERENCE_SINGLETONS,
  BY_NOISE_SCORE,
  DISTANCE_TO_SELECTOR,
  ADJACENT_TO_COLOR,
  SELECTOR_SET_OP,
  BY_HOLE_COUNT,
  BY_ATTR,
};

enum class GuardKind : int {
  MIN_COUNT = 0,
  MAX_COUNT,
  EXACT_COUNT,
};

enum class ActionKind : int {
  RECOLOR = 0,
  RECOLOR_TO,
  RECOLOR_FROM_EDGE_MAJORITY,
  RECOLOR_FROM_REFERENCE,
  RECOLOR_SEQUENCE,
  SWAP_COLORS,
  PRUNE_ENDPOINTS,
  SHRINK_TO_CORE,
  DILATE_SELECTED,
  FILL_BBOX,
  OUTLINE_OBJECT,
  TRANSLATE_BY_OWN_EXTENT,
  CROP_TO_OBJECT,
  PANEL_SPLIT_BOOLEAN,
  SCALE_BY_COUNT,
  COMPLETE_D4_SYMMETRY,
  SYMMETRIZE_OBJECT,
  PROJECT_INNER_QUAD_TO_OUTER_CORNERS,
  PROJECT_SINGLETONS_TO_FULL_LINES,
  STAMP_LOCAL_PATTERN,
  EXTEND_LINE_UNTIL,
  PROJECT_MARKER,
  ATTACH_SAME_COLOR_SINGLETONS,
  DELETE_REFERENCE_AFTER_TRANSFER,
  DELETE,
  TRANSLATE,
  MIRROR,
  KEEP_SELECTED,
  FILL_CAVITY,
  FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS,
  GRAVITY_TO_EDGE,
  FILL_ENCLOSED_BG,
  FILL_DIAGONAL_PERIODIC,
  TILE_FOLD_OR_NONBG,
  // Concept primitives (arc/concepts.hpp); inert unless registered.
  COUNT_BAR,
  SORT_OBJECTS,
  COPY_INTO_FRAMES,
  STAMP_AT_MARKERS,
  REPEAT_BY_MARKERS,
  SLIDE_UNTIL_BLOCKED,
  DENOISE,
  RAY_UNTIL_BLOCKED,
  CENTER_ON,
  TOGGLE_FILL,
  ATTR_PAINT,
  CONCAT_OBJECTS,
  PATCH,
  EXTRACT,
  BRING_TO_FRONT,
  COMPLETE_COPIES,
  KEEP_ENCLOSED,
  COMPLETE_REGULARITY,
  MARK_SIDE,
  TILE_GAP,
  TRANSFER_PATTERN,
};

enum class GlobalTransform : int {
  IDENTITY = 0,
  ROT90,
  ROT180,
  ROT270,
  FLIP_H,
  FLIP_V,
  TRANSPOSE,
  ANTI_TRANSPOSE,
  KRON_SELF_MASK,
  KRON_MASK_ANCHOR_COLOR,
  CROP_FOREGROUND,
  SCALE2_NEAREST,
  SCALE3_NEAREST,
  FILL_PERIODIC_BACKGROUND,
  MIRROR_CONCAT_H,
  MIRROR_CONCAT_H_REV,
  MIRROR_CONCAT_V,
  MIRROR_CONCAT_V_REV,
  MIRROR_QUAD_TL,
  MIRROR_QUAD_TR,
  MIRROR_QUAD_BL,
  MIRROR_QUAD_BR,
  TILE_H,
  TILE_V,
  DEDUP_RUNS,
  BLOCK_REDUCE2,
};

// Member counts.
constexpr int kSelectorKindCount = 19;
constexpr int kActionKindCount = 34;    // core actions only
constexpr int kGlobalTransformCount = 26;
constexpr int kGuardKindCount = 3;

// enum <-> `.value` string. from_string returns true on a hit and
// writes the enumerator into *out; false leaves *out untouched.
const std::string& to_string(SelectorKind k);
const std::string& to_string(ActionKind k);
const std::string& to_string(GlobalTransform k);
const std::string& to_string(GuardKind k);

bool selector_kind_from_string(const std::string& value, SelectorKind* out);
bool action_kind_from_string(const std::string& value, ActionKind* out);
bool global_transform_from_string(const std::string& value, GlobalTransform* out);
bool guard_kind_from_string(const std::string& value, GuardKind* out);

// Selector kinds folded into another kind. The old name still parses: as
// `kind`, with the kind's variant arg set to `variant` (an empty variant: the
// old kind is `kind` under another name).
struct SelectorAlias {
  const char* name;
  SelectorKind kind;
  const char* variant;
};
const std::vector<SelectorAlias>& selector_aliases();

// The variant arg of a merged selector kind ("" for the other kinds). It picks
// one of the folded kinds, so it is part of the kind's name: the MDL arg count
// leaves it out, and the GA draws each variant as often as the old kind.
const std::string& selector_variant_key(SelectorKind k);
// Its values, each with the GA weight of the kind it replaces (the GA selector
// table has one entry per kind, and the variant is drawn with these weights).
// The first value is the one a missing arg means.
struct SelectorVariants {
  SelectorKind kind;
  std::string key;
  std::vector<std::pair<std::string, int>> values;
};
const SelectorVariants* selector_variants(SelectorKind k);

// Core actions folded into another kind, as for the selectors: the old name
// parses as `kind` with the variant arg set.
struct ActionAlias {
  const char* name;
  ActionKind kind;
  const char* variant;
};
const std::vector<ActionAlias>& action_aliases();

// A merged core action's variant arg: its key and values, each with the GA
// weight of the kind it replaces (0: parsed but not drawn). The first value is
// the one a missing arg means. Empty key for the other kinds.
struct ActionVariants {
  ActionKind kind;
  std::string key;
  std::vector<std::pair<std::string, int>> values;
};
const ActionVariants* action_variants(ActionKind k);
const std::string& action_variant_key(ActionKind k);

// enum -> member NAME (for debugging).
const std::string& to_name(SelectorKind k);
const std::string& to_name(ActionKind k);
const std::string& to_name(GlobalTransform k);
const std::string& to_name(GuardKind k);

// Full ordered (name, value) tables.
const std::vector<std::pair<std::string, std::string>>& selector_kind_table();
const std::vector<std::pair<std::string, std::string>>& action_kind_table();
const std::vector<std::pair<std::string, std::string>>& global_transform_table();
const std::vector<std::pair<std::string, std::string>>& guard_kind_table();

// ---------------------------------------------------------------------------
// Heterogeneous arg value: int / float / str / bool, or (for
// SELECTOR_SET_OP-style nesting) a nested Selector. The args are a tagged union
// keyed by string in insertion order, so serialization is byte-stable. bool is
// a distinct tag from int because the serialized form writes True/False, not
// 1/0.
// ---------------------------------------------------------------------------

struct Selector;  // fwd

struct ArgValue {
  enum class Tag { Int, Float, Str, Bool, Selector };

  Tag tag = Tag::Int;
  int64_t i = 0;
  double f = 0.0;
  std::string s;
  bool b = false;
  std::shared_ptr<Selector> sel;  // non-null iff tag == Selector

  ArgValue() = default;

  static ArgValue make_int(int64_t v) {
    ArgValue a;
    a.tag = Tag::Int;
    a.i = v;
    return a;
  }
  static ArgValue make_float(double v) {
    ArgValue a;
    a.tag = Tag::Float;
    a.f = v;
    return a;
  }
  static ArgValue make_str(std::string v) {
    ArgValue a;
    a.tag = Tag::Str;
    a.s = std::move(v);
    return a;
  }
  static ArgValue make_bool(bool v) {
    ArgValue a;
    a.tag = Tag::Bool;
    a.b = v;
    return a;
  }
  static ArgValue make_selector(std::shared_ptr<Selector> v) {
    ArgValue a;
    a.tag = Tag::Selector;
    a.sel = std::move(v);
    return a;
  }
};

// Insertion-ordered string->ArgValue map. A vector of pairs keeps insertion
// order, which serialization relies on; set() replaces in place.
class ArgMap {
 public:
  bool empty() const { return items_.empty(); }
  size_t size() const { return items_.size(); }

  const std::vector<std::pair<std::string, ArgValue>>& items() const {
    return items_;
  }

  // Insert, or replace in place (an existing key keeps its position).
  void set(const std::string& key, ArgValue value) {
    for (auto& kv : items_) {
      if (kv.first == key) {
        kv.second = std::move(value);
        return;
      }
    }
    items_.emplace_back(key, std::move(value));
  }

  // Lookup: returns nullptr when the key is absent.
  const ArgValue* get(const std::string& key) const {
    for (const auto& kv : items_) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

 private:
  std::vector<std::pair<std::string, ArgValue>> items_;
};

// ---------------------------------------------------------------------------
// Program / Rule / Selector / Guard / Action.
// ---------------------------------------------------------------------------

struct Selector {
  SelectorKind kind = SelectorKind::ALL_SUBOBJECTS;
  ArgMap args;
};

struct Guard {
  GuardKind kind = GuardKind::MIN_COUNT;
  ArgMap args;
};

struct Action {
  ActionKind kind = ActionKind::RECOLOR;
  ArgMap args;
};

struct Rule {
  Selector selector;
  Action action;
  std::vector<Guard> guards;
};

struct Program {
  std::vector<Rule> rules;
  GlobalTransform global_transform = GlobalTransform::IDENTITY;
};

// ---------------------------------------------------------------------------
// GA active-operator tables. GA sampling depends on their ORDER and WEIGHTS.
// Parallel arrays (kind[i] has weight[i]).
// ---------------------------------------------------------------------------

const std::vector<ActionKind>& active_actions();
const std::vector<int>& active_action_weights();
const std::vector<SelectorKind>& active_selectors();
const std::vector<int>& active_selector_weights();
// Every (kind, variant) the GA selector draw can produce, in table order (""
// for a kind without a variant).
const std::vector<std::pair<SelectorKind, std::string>>& active_selector_leaves();

}  // namespace arc

#endif  // ARC_AST_HPP
