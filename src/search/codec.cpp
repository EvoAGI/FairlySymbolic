// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// TransformCodec: genome construction and argument samplers.
//
// See codec.hpp for the draw-order contract: every sampler makes its RNG draws
// in a fixed order, so a seed always yields the same genomes. Arg insertion
// order does not affect comparisons (canonical_repr sorts by key).

#include "arc/codec.hpp"
#include "arc/concepts.hpp"
#include "arc/object_attrs.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "arc/ast.hpp"
#include "arc/macro.hpp"
#include "arc/pyrepr.hpp"

namespace arc {

namespace {

// ---- DslValue builders ----------------------------------------------------
DslValue dv_int(long long v) {
  DslValue d;
  d.tag = DslValue::Tag::Int;
  d.i = v;
  return d;
}
DslValue dv_float(double v) {
  DslValue d;
  d.tag = DslValue::Tag::Float;
  d.f = v;
  return d;
}
DslValue dv_str(const std::string& v) {
  DslValue d;
  d.tag = DslValue::Tag::Str;
  d.s = v;
  return d;
}
DslValue dv_bool(bool v) {
  DslValue d;
  d.tag = DslValue::Tag::Bool;
  d.b = v;
  return d;
}
// One weighted draw (rng.choices) over a merged kind's variant values; values
// of weight 0 are parsed but never drawn.
std::string draw_variant(const std::vector<std::pair<std::string, int>>& values, ArcRandom& rng) {
  std::vector<int64_t> w;
  std::vector<const std::string*> vals;
  for (const auto& kv : values)
    if (kv.second > 0) {
      w.push_back(kv.second);
      vals.push_back(&kv.first);
    }
  return *vals[rng.choices_indices(w, 1)[0]];
}

DslValue dv_dict(DslArgs inner) {
  DslValue d;
  d.tag = DslValue::Tag::Dict;
  d.dict = std::make_shared<DslArgs>(std::move(inner));
  return d;
}

// rng.choice(list) over a concrete int / str list -> the chosen element.
int choice_int(ArcRandom& rng, const std::vector<int>& seq) {
  return seq[rng.choice_index(seq.size())];
}
std::string choice_str(ArcRandom& rng, const std::vector<std::string>& seq) {
  return seq[rng.choice_index(seq.size())];
}
bool choice_bool(ArcRandom& rng, const std::vector<bool>& seq) {
  return seq[rng.choice_index(seq.size())];
}
double choice_dbl(ArcRandom& rng, const std::vector<double>& seq) {
  return seq[rng.choice_index(seq.size())];
}

// One weighted draw (rng.choices_indices, k = 1) -> the chosen enum.
template <typename Enum>
Enum choices_one(ArcRandom& rng, const std::vector<Enum>& pop,
                 const std::vector<int>& weights) {
  std::vector<int64_t> w(weights.begin(), weights.end());
  std::vector<size_t> idx = rng.choices_indices(w, 1);
  return pop[idx[0]];
}

// The three EDGE_CHOICES-style constant lists (their order matters: draws
// index into them).
const std::vector<std::string>& shape_choices() {
  static const std::vector<std::string> v = {
      "Point", "Line", "Square", "Rectangle", "Frame",
      "Plus", "L-Shape", "T-Shape", "Diagonal", "Irregular",
  };
  return v;
}
const std::vector<std::string>& edge_choices() {
  static const std::vector<std::string> v = {
      "TOUCHES", "OVERLAPS", "IS_IMMEDIATELY_ABOVE", "IS_IMMEDIATELY_BELOW",
      "IS_IMMEDIATELY_LEFT_OF", "IS_IMMEDIATELY_RIGHT_OF", "CONTAINS",
      "IS_PART_OF", "SAME_COLOR", "SAME_SHAPE", "IS_SYMMETRIC_TO",
      "COLLINEAR_TO", "PROJECTS_RAY_TO", "FALLS_TOWARDS",
      "SAME_CENTROID_ROW", "SAME_CENTROID_COLUMN",
  };
  return v;
}
const std::vector<std::string>& boundary_sides() {
  static const std::vector<std::string> v = {"any", "top", "bottom", "left",
                                              "right"};
  return v;
}
const std::vector<std::string>& direction3() {
  static const std::vector<std::string> v = {"outgoing", "incoming", "both"};
  return v;
}

constexpr int kTranslateMin = -10;
constexpr int kTranslateMax = 10;

// MUTATE_CUM_THRESHOLDS: cumulative cut points for the single p = random()
// guarded fall-through in mutate().
constexpr double kThRuleEdit = 0.55;
constexpr double kThAddRule = 0.70;
constexpr double kThRemoveRule = 0.80;
constexpr double kThSwapTransform = 0.85;
constexpr double kThMutateEncoder = 0.90;
constexpr double kThMutateDecoder = 0.95;

// The 21 evolvable encoder params (DEFAULT_ENCODER_PARAMS order).
// crossover_encoder draws a per-field coin ONLY for names in this set (in
// GraphConfig declaration order).
const std::vector<std::string>& evolvable_encoder_params() {
  static const std::vector<std::string> v = {
      "connectivity", "color_mode", "background_mode",
      "background_color_override", "min_object_size", "object_mode",
      "merge_distance", "noise_threshold", "multi_color_as_cluster",
      "noise_filter_order", "decompose_objects", "decomposition_mode",
      "split_by_containment", "decompose_grid_lines", "merge_by_containment",
      "grid_line_threshold", "detect_zones", "detect_cavities",
      "detect_clusters", "detect_patterns", "detect_symmetries",
  };
  return v;
}
bool is_evolvable_encoder_param(const std::string& name) {
  for (const auto& n : evolvable_encoder_params())
    if (n == name) return true;
  return false;
}

// Set a GraphConfig field by name to an int-ish value. Only the 21 evolvable
// encoder params are ever set here (in DEFAULT_ENCODER_PARAMS order); the rest
// keep their struct defaults.
void set_config_int(GraphConfig& c, const std::string& name, long long v) {
  if (name == "connectivity") c.connectivity = static_cast<int>(v);
  else if (name == "background_color_override") c.background_color_override = static_cast<int>(v);
  else if (name == "min_object_size") c.min_object_size = static_cast<int>(v);
  else if (name == "merge_distance") c.merge_distance = static_cast<int>(v);
}
void set_config_bool(GraphConfig& c, const std::string& name, bool v) {
  if (name == "multi_color_as_cluster") c.multi_color_as_cluster = v;
  else if (name == "decompose_objects") c.decompose_objects = v;
  else if (name == "split_by_containment") c.split_by_containment = v;
  else if (name == "decompose_grid_lines") c.decompose_grid_lines = v;
  else if (name == "merge_by_containment") c.merge_by_containment = v;
  else if (name == "detect_zones") c.detect_zones = v;
  else if (name == "detect_cavities") c.detect_cavities = v;
  else if (name == "detect_clusters") c.detect_clusters = v;
  else if (name == "detect_patterns") c.detect_patterns = v;
  else if (name == "detect_symmetries") c.detect_symmetries = v;
}
void set_config_float(GraphConfig& c, const std::string& name, double v) {
  if (name == "noise_threshold") c.noise_threshold = v;
  else if (name == "grid_line_threshold") c.grid_line_threshold = v;
}
void set_config_str(GraphConfig& c, const std::string& name,
                    const std::string& v) {
  if (name == "color_mode") c.color_mode = v;
  else if (name == "background_mode") c.background_mode = v;
  else if (name == "object_mode") c.object_mode = v;
  else if (name == "noise_filter_order") c.noise_filter_order = v;
  else if (name == "decomposition_mode") c.decomposition_mode = v;
}

// ---- config field GETTERS by name (for mutate_encoder_param) ---------------
// Only the 21 evolvable encoder params are ever read here.
long long get_config_int(const GraphConfig& c, const std::string& name) {
  if (name == "connectivity") return c.connectivity;
  if (name == "background_color_override") return c.background_color_override;
  if (name == "min_object_size") return c.min_object_size;
  if (name == "merge_distance") return c.merge_distance;
  return 0;
}
bool get_config_bool(const GraphConfig& c, const std::string& name) {
  if (name == "multi_color_as_cluster") return c.multi_color_as_cluster;
  if (name == "decompose_objects") return c.decompose_objects;
  if (name == "split_by_containment") return c.split_by_containment;
  if (name == "decompose_grid_lines") return c.decompose_grid_lines;
  if (name == "merge_by_containment") return c.merge_by_containment;
  if (name == "detect_zones") return c.detect_zones;
  if (name == "detect_cavities") return c.detect_cavities;
  if (name == "detect_clusters") return c.detect_clusters;
  if (name == "detect_patterns") return c.detect_patterns;
  if (name == "detect_symmetries") return c.detect_symmetries;
  return false;
}
double get_config_float(const GraphConfig& c, const std::string& name) {
  if (name == "noise_threshold") return c.noise_threshold;
  if (name == "grid_line_threshold") return c.grid_line_threshold;
  return 0.0;
}
std::string get_config_str(const GraphConfig& c, const std::string& name) {
  if (name == "color_mode") return c.color_mode;
  if (name == "background_mode") return c.background_mode;
  if (name == "object_mode") return c.object_mode;
  if (name == "noise_filter_order") return c.noise_filter_order;
  if (name == "decomposition_mode") return c.decomposition_mode;
  return "";
}

// normalize_encoder: the encoder consistency rules, applied to the struct.
// background_color_override == -1 means no override.
void normalize_encoder(GraphConfig& c) {
  if (g_force_connectivity) c.connectivity = g_force_connectivity;
  // background_mode "none": no colour override.
  if (c.background_mode == "none") c.background_color_override = -1;
  // No zone detection: no zone features.
  if (!c.detect_zones) c.compute_zone_features = false;
  // Pixel objects: no minimum object size.
  if (c.object_mode == "pixel") c.min_object_size = 0;
}

}  // namespace

thread_local int g_force_connectivity = 0;

TransformCodec::TransformCodec(int max_rules) : max_rules_(max_rules) {}

// An empty pool (the default) draws exactly as op->sample_args, so runs without
// ARC_INFORMED_ARGS are unchanged.
DslArgs TransformCodec::concept_args(ActionKind kind, ArcRandom& rng) const {
  // ARC_INFORMED_RATE: how often a pooled setting is drawn (default 0.5).
  static const double rate = [] {
    const char* e = std::getenv("ARC_INFORMED_RATE");
    return e ? std::atof(e) : 0.5;
  }();
  const ConceptOp* op = find_concept_op(kind);
  for (const auto& kv : concept_arg_pool_)
    if (kv.first == static_cast<int>(kind) && !kv.second.empty() && rng.random() < rate)
      return kv.second[rng.choice_index(kv.second.size())];
  DslArgs a = op->sample_args(rng);
  inform_colors(kind, a, rng);
  return a;
}

void TransformCodec::inform_colors(ActionKind kind, DslArgs& a, ArcRandom& rng) const {
  if (chg_from_.empty() && chg_to_.empty()) return;
  const char* key = nullptr;
  const std::vector<int>* set = nullptr;
  switch (kind) {
    case ActionKind::COMPLETE_REGULARITY: key = "mask"; set = &chg_from_; break;
    case ActionKind::MARK_SIDE: key = "color"; set = &chg_to_; break;
    case ActionKind::PATCH: key = "fill"; set = &chg_to_; break;
    default: return;
  }
  if (set->empty()) return;
  for (auto& kv : a) {
    if (kv.first != key) continue;
    DslValue& v = kv.second;
    bool is_str = v.tag == DslValue::Tag::Str && v.s.size() == 1 && v.s[0] >= '0' && v.s[0] <= '9';
    bool is_int = v.tag == DslValue::Tag::Int && v.i >= 0 && v.i <= 9;
    if (!is_str && !is_int) return;
    if (rng.random() >= 0.5) return;
    int c = (*set)[rng.choice_index(set->size())];
    v = is_str ? dv_str(std::string(1, static_cast<char>('0' + c))) : dv_int(c);
    return;
  }
}

void TransformCodec::set_color_prior(const std::vector<int>& colors) {
  color_prior_.clear();
  has_color_prior_ = false;
  if (colors.empty()) return;
  std::set<int> s;
  for (int c : colors) {
    if (c >= 0 && c <= 9) s.insert(c);
  }
  if (s.empty()) return;  // no colours: the prior stays unset
  color_prior_.assign(s.begin(), s.end());  // std::set iterates sorted
  has_color_prior_ = true;
}

// by_color's colour: with ARC_INFORMED_COLOR, half the time an input colour of
// the changed cells; otherwise (and always without the flag) sample_color.
int TransformCodec::sample_select_color(ArcRandom& rng) {
  if (!chg_from_.empty() && rng.random() < 0.5) return chg_from_[rng.choice_index(chg_from_.size())];
  return sample_color(rng);
}

int TransformCodec::sample_color(ArcRandom& rng) {
  if (has_color_prior_ && !color_prior_.empty() && rng.random() < 0.8) {
    return color_prior_[rng.choice_index(color_prior_.size())];
  }
  return static_cast<int>(rng.randint(0, 9));
}

// ---------------------------------------------------------------------------
// maybe_macro_rules. Draw-order contract:
//   no active library or no macros -> false (NO draw)
//   rate = (mutating and mutate_rate >= 0) ? mutate_rate : rate
//   rate <= 0 -> false (NO draw)
//   random() >= rate -> false (the random() draw HAPPENS)
//   rules = realize(sample(rng), codec, rng), cut to max_rules
// ---------------------------------------------------------------------------
bool TransformCodec::maybe_macro_rules(ArcRandom& rng, bool mutating,
                                       std::vector<DslRule>* out) {
  const MacroLibrary* lib = macro_lib_;
  if (lib == nullptr || lib->macros.empty()) return false;  // NO rng draw
  double rate = (mutating && lib->mutate_rate >= 0.0) ? lib->mutate_rate
                                                      : lib->rate;
  if (rate <= 0.0) return false;              // NO rng draw
  if (rng.random() >= rate) return false;     // gate draw consumed
  std::vector<DslRule> rules = macro_realize(lib->sample(rng), *this, rng);
  if (rules.empty()) return false;            // realize never returns empty; defensive
  if (static_cast<int>(rules.size()) > max_rules_) {
    rules.resize(static_cast<size_t>(max_rules_));  // keep the first max_rules
  }
  *out = std::move(rules);
  return true;
}

const std::vector<std::string>& TransformCodec::composeable_selector_names()
    const {
  // banned_selector_kinds is empty by default -> the full list. The entries
  // are the pre-merge kind names; a folded kind's old name gives the merged
  // kind with its variant (compose_leg), so the draws are unchanged.
  static const std::vector<std::string> v = {
      "all_subobjects", "non_background", "by_color", "touching_color",
      "edge_to_color", "has_edge_type", "reference_singletons", "by_mass_exact",
      "by_noise_score",
  };
  return v;
}

DslSelector TransformCodec::compose_leg(const std::string& name, ArcRandom& rng) {
  DslSelector s;
  s.args = selector_args_for_name(name, rng);
  selector_from_name(name, &s.kind, &s.args);
  return s;
}

// ---------------------------------------------------------------------------
// selector_args_for_name (reduced sampler for nested SET_OP legs).
// ---------------------------------------------------------------------------
DslArgs TransformCodec::selector_args_for_name(const std::string& kind,
                                               ArcRandom& rng) {
  DslArgs a;
  if (kind == "by_color") {
    a.emplace_back("color", dv_int(sample_color(rng)));
  } else if (kind == "touching_color") {
    a.emplace_back("color", dv_int(sample_color(rng)));
  } else if (kind == "edge_to_color") {
    a.emplace_back("color", dv_int(sample_color(rng)));
    a.emplace_back("edge_type", dv_str(choice_str(rng, edge_choices())));
    a.emplace_back("direction", dv_str(choice_str(rng, direction3())));
  } else if (kind == "has_edge_type") {
    a.emplace_back("edge_type", dv_str(choice_str(rng, edge_choices())));
    a.emplace_back("direction", dv_str(choice_str(rng, direction3())));
  } else if (kind == "reference_singletons") {
    a.emplace_back("max_mass", dv_int(choice_int(rng, {1, 1, 2, 3})));
    a.emplace_back("outside_bbox_of_largest",
                   dv_bool(choice_bool(rng, {true, false})));
  } else if (kind == "by_mass_exact") {
    a.emplace_back("mass", dv_int(rng.randint(1, 9)));
  } else if (kind == "by_hole_count") {
    a.emplace_back("holes", dv_int(choice_int(rng, {0, 1, 1, 2, 2, 3, 4})));
  } else if (kind == "by_noise_score") {
    double lo = choice_dbl(rng, {0.0, 0.1, 0.2, 0.3, 0.4});
    double hi = choice_dbl(rng, {0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    if (lo > hi) std::swap(lo, hi);
    a.emplace_back("min_score", dv_float(lo));
    a.emplace_back("max_score", dv_float(hi));
    a.emplace_back("order",
                   dv_str(choice_str(rng, {"none", "highest", "lowest"})));
    a.emplace_back("top_k", dv_int(choice_int(rng, {1, 1, 2, 3, 4})));
  }
  return a;
}

// ---------------------------------------------------------------------------
// sample_selector_args (full per-kind selector arg sampler).
// ---------------------------------------------------------------------------
DslArgs TransformCodec::sample_selector_args(SelectorKind kind, ArcRandom& rng,
                                             const std::string& variant) {
  DslArgs a;
  // A merged kind's variant comes first; it is drawn with the kind (no rng).
  if (!variant.empty()) a.emplace_back(selector_variant_key(kind), dv_str(variant));
  switch (kind) {
    case SelectorKind::BY_COLOR:
      a.emplace_back("color", dv_int(sample_select_color(rng)));
      break;
    case SelectorKind::BY_SHAPE:
      a.emplace_back("shape", dv_str(choice_str(rng, shape_choices())));
      break;
    case SelectorKind::BY_MASS:
      a.emplace_back("mass", dv_int(rng.randint(1, 9)));
      break;
    case SelectorKind::BY_HOLE_COUNT:
      a.emplace_back("holes", dv_int(choice_int(rng, {0, 1, 1, 2, 2, 3, 4})));
      break;
    case SelectorKind::BY_BOUNDARY_TOUCH:
      a.emplace_back("side", dv_str(choice_str(rng, boundary_sides())));
      break;
    case SelectorKind::TOUCHING_COLOR:
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case SelectorKind::EDGE_COUNT:
      a.emplace_back("edge_type", dv_str(choice_str(rng, edge_choices())));
      a.emplace_back("direction", dv_str(choice_str(rng, direction3())));
      if (variant == "ge") a.emplace_back("count", dv_int(rng.randint(1, 4)));
      break;
    case SelectorKind::REFERENCE_SINGLETONS:
      a.emplace_back("max_mass", dv_int(choice_int(rng, {1, 1, 2, 3})));
      a.emplace_back("outside_bbox_of_largest",
                     dv_bool(choice_bool(rng, {true, false})));
      break;
    case SelectorKind::BY_NOISE_SCORE: {
      double lo = choice_dbl(rng, {0.0, 0.1, 0.2, 0.3, 0.4});
      double hi = choice_dbl(rng, {0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
      if (lo > hi) std::swap(lo, hi);
      a.emplace_back("min_score", dv_float(lo));
      a.emplace_back("max_score", dv_float(hi));
      a.emplace_back("order",
                     dv_str(choice_str(rng, {"none", "highest", "lowest"})));
      a.emplace_back("top_k", dv_int(choice_int(rng, {1, 1, 2, 3, 4})));
      break;
    }
    case SelectorKind::DISTANCE_TO_SELECTOR: {
      std::string osk = choice_str(
          rng, {std::string("largest_subobject"),
                std::string("smallest_subobject"),
                std::string("top_leftmost_subobject"),
                to_string(SelectorKind::REFERENCE_SINGLETONS)});
      a.emplace_back("other_selector_kind", dv_str(osk));
      DslArgs osa;
      osa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 2, 3})));
      a.emplace_back("other_selector_args", dv_dict(std::move(osa)));
      a.emplace_back("distance",
                     dv_str(choice_str(rng, {"centroid", "bbox_gap"})));
      a.emplace_back("top_k", dv_int(choice_int(rng, {1, 1, 2, 3, 4})));
      a.emplace_back("include_references",
                     dv_bool(choice_bool(rng, {false, false, true})));
      break;
    }
    case SelectorKind::ADJACENT_TO_COLOR:
      a.emplace_back("color", dv_int(sample_color(rng)));
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      break;
    case SelectorKind::SELECTOR_SET_OP: {
      const std::string& left_name =
          composeable_selector_names()[rng.choice_index(
              composeable_selector_names().size())];
      const std::string& right_name =
          composeable_selector_names()[rng.choice_index(
              composeable_selector_names().size())];
      a.emplace_back(
          "op", dv_str(choice_str(
                    rng, {"intersection", "difference", "union"})));
      DslSelector left = compose_leg(left_name, rng);
      a.emplace_back("left_selector_kind", dv_str(to_string(left.kind)));
      a.emplace_back("left_selector_args", dv_dict(std::move(left.args)));
      DslSelector right = compose_leg(right_name, rng);
      a.emplace_back("right_selector_kind", dv_str(to_string(right.kind)));
      a.emplace_back("right_selector_args", dv_dict(std::move(right.args)));
      break;
    }
    case SelectorKind::INSIDE_LARGEST_BBOX:
      a.emplace_back("include_border",
                     dv_bool(choice_bool(rng, {false, false, true})));
      break;
    case SelectorKind::SAME_SHAPE_AS_REFERENCE: {
      std::string rsk = choice_str(
          rng, {std::string("top_leftmost_subobject"),
                std::string("top_leftmost_non_most_frequent"),
                std::string("leftmost_non_most_frequent"),
                to_string(SelectorKind::INSIDE_LARGEST_BBOX),
                std::string("topmost_subobject"),
                std::string("leftmost_subobject"),
                std::string("smallest_subobject"),
                std::string("largest_subobject")});
      a.emplace_back("reference_selector_kind", dv_str(rsk));
      a.emplace_back("reference_selector_args", dv_dict(DslArgs{}));
      a.emplace_back("exclude_reference",
                     dv_bool(choice_bool(rng, {true, true, false})));
      break;
    }
    case SelectorKind::BY_ATTR:
      if (!by_attr_pool_.empty() && rng.random() < 0.5) {
        a = by_attr_pool_[rng.choice_index(by_attr_pool_.size())];
        break;
      }
      a.emplace_back("attr", dv_str(object_attr_names()[rng.choice_index(object_attr_names().size())]));
      a.emplace_back("op", dv_str(choice_str(rng, {"eq", "eq", "ne", "ge", "le", "max", "max", "min", "min", "unique"})));
      a.emplace_back("value", dv_int(choice_int(rng, {0, 0, 1, 1, 1, 2, 2, 3, 4})));
      break;
    default:
      break;
  }
  return a;
}

// ---------------------------------------------------------------------------
// sample_action_args (full per-kind action arg sampler).
// ---------------------------------------------------------------------------
DslArgs TransformCodec::sample_action_args(ActionKind kind, ArcRandom& rng) {
  DslArgs a;
  // A merged core action draws its variant first, with the old kinds' weights
  // (one choices draw), so each old kind is drawn as often as before.
  if (const ActionVariants* v = action_variants(kind)) a.emplace_back(v->key, dv_str(draw_variant(v->values, rng)));
  switch (kind) {
    case ActionKind::RECOLOR:
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::RECOLOR_FROM_EDGE_MAJORITY:
      a.emplace_back("edge_type", dv_str(choice_str(rng, edge_choices())));
      a.emplace_back("direction", dv_str(choice_str(rng, direction3())));
      break;
    case ActionKind::RECOLOR_FROM_REFERENCE: {
      std::string rsk = choice_str(
          rng, {to_string(SelectorKind::NON_BACKGROUND),
                to_string(SelectorKind::ALL_SUBOBJECTS),
                to_string(SelectorKind::INSIDE_LARGEST_BBOX),
                std::string("top_leftmost_subobject"),
                std::string("top_leftmost_non_most_frequent"),
                std::string("leftmost_non_most_frequent"),
                to_string(SelectorKind::REFERENCE_SINGLETONS)});
      a.emplace_back("reference_selector_kind", dv_str(rsk));
      DslArgs rsa;
      rsa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 2, 3})));
      rsa.emplace_back("outside_bbox_of_largest",
                       dv_bool(choice_bool(rng, {true, false})));
      a.emplace_back("reference_selector_args", dv_dict(std::move(rsa)));
      a.emplace_back(
          "color_policy",
          dv_str(choice_str(rng, {"nearest", "majority", "only"})));
      a.emplace_back(
          "match_policy",
          dv_str(choice_str(rng,
                            {"same_shape", "same_shape_strict", "any"})));
      a.emplace_back("exclude_selected",
                     dv_bool(choice_bool(rng, {true, true, false})));
      break;
    }
    case ActionKind::RECOLOR_SEQUENCE:
      a.emplace_back(
          "order",
          dv_str(choice_str(rng,
                            {"y_then_x", "x_then_y", "mass_desc", "mass_asc"})));
      a.emplace_back("start_color", dv_int(sample_color(rng)));
      a.emplace_back("step", dv_int(choice_int(rng, {1, 1, 1, 2, 3})));
      break;
    case ActionKind::SWAP_COLORS: {
      int av = sample_color(rng);
      int bv = sample_color(rng);
      if (av == bv) bv = (av + 1) % 10;
      a.emplace_back("color_a", dv_int(av));
      a.emplace_back("color_b", dv_int(bv));
      break;
    }
    case ActionKind::PRUNE_ENDPOINTS:
      a.emplace_back("iterations", dv_int(choice_int(rng, {1, 1, 1, 2})));
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      a.emplace_back("protect_junctions",
                     dv_bool(choice_bool(rng, {true, true, false})));
      break;
    case ActionKind::SHRINK_TO_CORE:
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      break;
    case ActionKind::DILATE_SELECTED:
      a.emplace_back("radius", dv_int(choice_int(rng, {1, 1, 2})));
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      a.emplace_back(
          "mask",
          dv_str(choice_str(rng, {"auto", "cross", "diamond", "square"})));
      break;
    case ActionKind::COMPLETE_D4_SYMMETRY:
      a.emplace_back("include_pivot_color",
                     dv_bool(choice_bool(rng, {false, false, true})));
      break;
    case ActionKind::SYMMETRIZE_OBJECT:
      a.emplace_back("orientation", dv_str(choice_str(rng, {"v", "h"})));
      a.emplace_back(
          "mode",
          dv_str(choice_str(rng, {"intersect", "intersect", "union"})));
      a.emplace_back("cluster_radius", dv_int(choice_int(rng, {1, 1, 2})));
      break;
    case ActionKind::PROJECT_INNER_QUAD_TO_OUTER_CORNERS:
      break;
    case ActionKind::PROJECT_SINGLETONS_TO_FULL_LINES:
      break;
    case ActionKind::STAMP_LOCAL_PATTERN:
      a.emplace_back(
          "pattern",
          dv_str(choice_str(rng, {"dot", "cross", "x", "ring", "square"})));
      a.emplace_back(
          "anchor",
          dv_str(choice_str(rng, {"centroid", "bbox_center", "min_corner"})));
      a.emplace_back(
          "color_source",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::EXTEND_LINE_UNTIL:
      a.emplace_back(
          "direction",
          dv_str(choice_str(
              rng, {"up", "down", "left", "right", "ray", "ray8", "diag",
                    "up_left", "up_right", "down_left", "down_right"})));
      a.emplace_back(
          "stop_condition",
          dv_str(choice_str(rng, {"boundary", "touch_object", "color!=bg"})));
      a.emplace_back("stop_on_same_color",
                     dv_bool(choice_bool(rng, {false, false, true})));
      a.emplace_back("intersection_color", dv_int(sample_color(rng)));
      a.emplace_back(
          "anchor",
          dv_str(choice_str(rng, {"centroid", "bbox_center", "min_corner"})));
      a.emplace_back(
          "color_source",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::OUTLINE_OBJECT:
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 8, 8})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::TRANSLATE_BY_OWN_EXTENT:
      a.emplace_back(
          "extent_source",
          dv_str(choice_str(rng, {"width", "height", "mass"})));
      a.emplace_back("mult_y", dv_int(choice_int(rng, {-1, 0, 1})));
      a.emplace_back("mult_x", dv_int(choice_int(rng, {-1, 0, 1})));
      break;
    case ActionKind::GRAVITY_TO_EDGE:
      a.emplace_back(
          "direction",
          dv_str(choice_str(rng, {"up", "down", "left", "right"})));
      break;
    case ActionKind::FILL_ENCLOSED_BG:
      a.emplace_back("fill_color", dv_int(sample_color(rng)));
      break;
    case ActionKind::FILL_DIAGONAL_PERIODIC:
      a.emplace_back("diagonal", dv_str(choice_str(rng, {"main", "anti"})));
      a.emplace_back("period", dv_int(choice_int(rng, {2, 3, 4, 5})));
      break;
    case ActionKind::TILE_FOLD_OR_NONBG:
      a.emplace_back("div_y", dv_int(choice_int(rng, {1, 2, 2, 3})));
      a.emplace_back("div_x", dv_int(choice_int(rng, {1, 2, 2, 3})));
      break;
    case ActionKind::CROP_TO_OBJECT:
      a.emplace_back(
          "which",
          dv_str(choice_str(rng, {"largest", "smallest", "unique_color",
                                  "odd_one_out", "densest"})));
      a.emplace_back(
          "recolor",
          dv_str(choice_str(rng,
                            {"none", "none", "swap_two", "to_marker_color"})));
      a.emplace_back("obj_mode", dv_str(choice_str(rng, {"multi", "mono"})));
      break;
    case ActionKind::PANEL_SPLIT_BOOLEAN:
      a.emplace_back("axis", dv_str(choice_str(rng, {"h", "v", "auto"})));
      a.emplace_back(
          "op", dv_str(choice_str(
                    rng, {"and", "or", "xor", "nand", "nor", "xnor", "diff"})));
      a.emplace_back("layout",
                     dv_str(choice_str(rng, {"2", "2", "3", "2x2"})));
      a.emplace_back("bg_policy",
                     dv_str(choice_str(rng, {"zero", "zero", "detected"})));
      a.emplace_back("out_color", dv_int(sample_color(rng)));
      break;
    case ActionKind::SCALE_BY_COUNT:
      a.emplace_back(
          "mode",
          dv_str(choice_str(rng,
                            {"scale", "scale", "tile_h", "tile_v", "tile_both"})));
      a.emplace_back("count_type",
                     dv_str(choice_str(rng, {"colors", "objects"})));
      a.emplace_back("color_strict", dv_bool(choice_bool(rng, {false, true})));
      a.emplace_back("diag", dv_bool(choice_bool(rng, {false, true})));
      break;
    case ActionKind::PROJECT_MARKER:
      a.emplace_back(
          "side", dv_str(choice_str(rng, {"bottom", "top", "left", "right"})));
      a.emplace_back("align",
                     dv_str(choice_str(rng, {"centroid", "bbox_center"})));
      a.emplace_back(
          "color_mode",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::ATTACH_SAME_COLOR_SINGLETONS:
      a.emplace_back("min_line_length", dv_int(choice_int(rng, {3, 4, 5, 6})));
      a.emplace_back("delete_source",
                     dv_bool(choice_bool(rng, {true, true, false})));
      break;
    case ActionKind::DELETE_REFERENCE_AFTER_TRANSFER: {
      a.emplace_back("reference_selector_kind",
                     dv_str(to_string(SelectorKind::REFERENCE_SINGLETONS)));
      DslArgs rsa;
      rsa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 1, 2})));
      rsa.emplace_back("outside_bbox_of_largest",
                       dv_bool(choice_bool(rng, {true, false})));
      a.emplace_back("reference_selector_args", dv_dict(std::move(rsa)));
      break;
    }
    case ActionKind::TRANSLATE:  // mode (move / wrap / copy) drawn above
      a.emplace_back("dy", dv_int(rng.randint(kTranslateMin, kTranslateMax)));
      a.emplace_back("dx", dv_int(rng.randint(kTranslateMin, kTranslateMax)));
      break;
    case ActionKind::FILL_CAVITY:
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    // FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS: only the single-arg form
    // (fill_color + axes) is sampled.
    case ActionKind::FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS:
      a.emplace_back("fill_color", dv_int(sample_color(rng)));
      a.emplace_back(
          "axes",
          dv_str(choice_str(
              rng, {"horizontal", "vertical", "diagonal", "both", "all"})));
      break;
    default:
      if (find_concept_op(kind)) a = concept_args(kind, rng);
      break;
  }
  return a;
}

// ---------------------------------------------------------------------------
// random_guards.
// ---------------------------------------------------------------------------
std::vector<DslGuard> TransformCodec::random_guards(ArcRandom& rng) {
  std::vector<DslGuard> out;
  if (rng.random() < 0.7) return out;  // 70% no guards
  // ARC_NO_GUARDS=1: rules never get count guards (guarded programs fit the
  // train pairs but generalise much worse).
  static const bool no_guards = [] {
    const char* e = std::getenv("ARC_NO_GUARDS");
    return e && std::string(e) == "1";
  }();
  if (no_guards) return out;
  // One of MIN_COUNT, MAX_COUNT, EXACT_COUNT, uniformly.
  static const std::vector<GuardKind> kinds = {
      GuardKind::MIN_COUNT, GuardKind::MAX_COUNT, GuardKind::EXACT_COUNT};
  GuardKind kind = kinds[rng.choice_index(kinds.size())];
  long long value;
  if (kind == GuardKind::MIN_COUNT) {
    value = rng.randint(1, 5);
  } else if (kind == GuardKind::MAX_COUNT) {
    value = rng.randint(1, 8);
  } else {
    value = rng.randint(1, 6);
  }
  DslGuard g;
  g.kind = kind;
  g.args.emplace_back("value", dv_int(value));
  out.push_back(std::move(g));
  return out;
}

// ---------------------------------------------------------------------------
// random_rule.
// ---------------------------------------------------------------------------
DslRule TransformCodec::random_rule(ArcRandom& rng) {
  // rng.choices over the entries (choices_one's draw); an entry is a kind and,
  // for a merged kind, the variant it draws.
  const std::vector<int>& sw = active_selector_weights();
  size_t si = rng.choices_indices(std::vector<int64_t>(sw.begin(), sw.end()), 1)[0];
  SelectorKind sk = active_selectors()[si];
  // A merged kind then draws its variant with the old kinds' weights.
  std::string variant;
  if (const SelectorVariants* v = selector_variants(sk)) variant = draw_variant(v->values, rng);
  DslRule r;
  r.selector.kind = sk;
  r.selector.args = sample_selector_args(sk, rng, variant);

  ActionKind ak = choices_one(rng, active_actions(), active_action_weights());
  r.action.kind = ak;
  r.action.args = sample_action_args(ak, rng);

  r.guards = random_guards(rng);
  return r;
}

// ---------------------------------------------------------------------------
// random_encoder_config.
// ---------------------------------------------------------------------------
void TransformCodec::sample_encoder_param(const std::string& name,
                                          const ParamMeta& meta,
                                          ArcRandom& rng, GraphConfig& cfg) {
  // Special cases.
  if (name == "background_color_override") {
    // Uniform over {none, -1, 0, 0, 0, 1}; none is stored as -1 too.
    static const std::vector<int> ch = {-1, -1, 0, 0, 0, 1};
    cfg.background_color_override = choice_int(rng, ch);
    return;
  }
  if (name == "detect_zones") {
    cfg.detect_zones = choice_bool(rng, {false, false, true});
    return;
  }
  if (name == "compute_zone_features") {  // not in the 21 list, but guarded
    cfg.compute_zone_features = choice_bool(rng, {false, false, true});
    return;
  }
  if (name == "min_object_size") {
    cfg.min_object_size = choice_int(rng, {0, 0, 0, 1, 1, 2, 3, 4});
    return;
  }
  if (name == "grid_line_threshold") {
    double u = rng.uniform(0.92, 1.0);
    double v = std::max(0.7, std::min(1.0, u));
    cfg.grid_line_threshold = v;
    return;
  }
  // Generic random value by metadata kind.
  switch (meta.kind) {
    case ParamKind::Bool:
      // randint(0, 1) as a bool
      set_config_bool(cfg, name, rng.randint(0, 1) != 0);
      break;
    case ParamKind::Categorical: {
      size_t idx = rng.choice_index(meta.choices.size());
      if (meta.choices_are_int) {
        set_config_int(cfg, name, std::stoll(meta.choices[idx]));
      } else {
        set_config_str(cfg, name, meta.choices[idx]);
      }
      break;
    }
    case ParamKind::Int: {
      long long lo = static_cast<long long>(meta.low);
      long long hi = static_cast<long long>(meta.high);
      set_config_int(cfg, name, rng.randint(lo, hi));
      break;
    }
    case ParamKind::Float: {
      double v = rng.uniform(meta.low, meta.high);
      set_config_float(cfg, name, v);
      break;
    }
  }
}

GraphConfig TransformCodec::random_encoder_config(ArcRandom& rng) {
  GraphConfig cfg;  // struct defaults
  const std::vector<ParamMeta>& table = param_metadata_table();
  // Build a name->meta lookup (declaration order table).
  for (const std::string& name : default_encoder_params()) {
    const ParamMeta* meta = nullptr;
    for (const auto& pm : table) {
      if (name == pm.name) {
        meta = &pm;
        break;
      }
    }
    if (!meta) continue;  // param not in metadata -> skipped
    sample_encoder_param(name, *meta, rng, cfg);
  }
  normalize_encoder(cfg);
  return cfg;
}

// ---------------------------------------------------------------------------
// random_decoder.
// ---------------------------------------------------------------------------
DecoderConfig TransformCodec::random_decoder(ArcRandom& rng) {
  static const std::vector<std::string> bg = {
      "scene_background_or_zero", "zero", "most_frequent_input",
      "most_frequent_predicted", "input_copy"};
  static const std::vector<std::string> po = {"as_is"};  // ACTIVE_PAINT_ORDERS
  static const std::vector<std::string> ov = {"overwrite", "keep_first"};
  static const std::vector<std::string> pp = {"none", "crop_foreground", "keep_lines"};
  DecoderConfig d;
  d.background_policy = choice_str(rng, bg);
  d.paint_order = choice_str(rng, po);
  d.overlap_policy = choice_str(rng, ov);
  d.postprocess = choice_str(rng, pp);
  return d;
}

// ---------------------------------------------------------------------------
// random_genome (top-level). Macros OFF -> the macro-realize branch draws
// nothing and is skipped entirely.
// ---------------------------------------------------------------------------
TransformGenome TransformCodec::random_genome(ArcRandom& rng) {
  // maybe_macro_rules(rng): macros OFF (nullptr) -> false with NO rng draw.
  DslProgram program;
  std::vector<DslRule> macro_rules;
  if (maybe_macro_rules(rng, /*mutating=*/false, &macro_rules)) {
    program.rules = std::move(macro_rules);
  } else {
    long long rule_count = rng.randint(0, max_rules_);
    program.rules.reserve(static_cast<size_t>(rule_count));
    for (long long i = 0; i < rule_count; ++i) {
      program.rules.push_back(random_rule(rng));
    }
  }
  // transform: one uniform draw over all 26 GlobalTransforms, in enum order.
  program.global_transform =
      static_cast<GlobalTransform>(rng.choice_index(kGlobalTransformCount));

  TransformGenome g;
  g.program = std::move(program);
  g.encoder_config = random_encoder_config(rng);
  g.decoder = random_decoder(rng);
  return g;
}

// ===========================================================================
// mutate + crossover (the GA operators).
//
// The draw order at every site is fixed. mutate() is
// a GUARDED FALL-THROUGH over a single p = random(): when a branch's guard
// fails, the SAME p is tested against the next threshold, so a failed guard
// changes how many downstream draws happen. crossover() cuts each parent's rule
// list independently and crosses encoder (per-evolvable-field coin, declaration
// order), transform (50/50), decoder (50/50).
// ===========================================================================

// ---------------------------------------------------------------------------
// mutate_guard.
// ---------------------------------------------------------------------------
DslGuard TransformCodec::mutate_guard(const DslGuard& guard, ArcRandom& rng) {
  if (rng.random() < 0.25) {
    std::vector<DslGuard> ng = random_guards(rng);
    return ng.empty() ? guard : ng[0];
  }
  const DslValue* v = args_get(guard.args, "value");
  long long value = (v && v->tag == DslValue::Tag::Int) ? v->i : 1;
  value += choice_int(rng, {-2, -1, 1, 2});
  if (value < 1) value = 1;
  if (value > 12) value = 12;
  DslGuard g;
  g.kind = guard.kind;
  g.args.emplace_back("value", dv_int(value));
  return g;
}

// ---------------------------------------------------------------------------
// mutate_selector (per-kind arg jitter; guarded chain). Args start as a COPY
// of the current selector args; each branch overwrites / adds keys in place
// (existing keys keep their position).
// ---------------------------------------------------------------------------
namespace {
// The string value of `key` in args, or dflt.
std::string variant_of(const DslArgs& a, const std::string& key, const std::string& dflt) {
  const DslValue* v = args_get(a, key);
  return v && v->tag == DslValue::Tag::Str ? v->s : dflt;
}

// Overwrite-or-append a key in a DslArgs: replace in place if present, else
// append. Existing keys keep their position.
void args_set(DslArgs& a, const std::string& key, DslValue v) {
  for (auto& kv : a) {
    if (kv.first == key) {
      kv.second = std::move(v);
      return;
    }
  }
  a.emplace_back(key, std::move(v));
}
}  // namespace

DslSelector TransformCodec::mutate_selector(const DslSelector& selector,
                                            ArcRandom& rng) {
  SelectorKind kind = selector.kind;
  DslArgs args = selector.args;  // a copy
  switch (kind) {
    case SelectorKind::BY_COLOR:
      args_set(args, "color", dv_int(sample_select_color(rng)));
      break;
    case SelectorKind::BY_SHAPE:
      args_set(args, "shape", dv_str(choice_str(rng, shape_choices())));
      break;
    case SelectorKind::BY_MASS:
      args_set(args, "mass", dv_int(rng.randint(1, 12)));
      break;
    case SelectorKind::BY_HOLE_COUNT:
      args_set(args, "holes", dv_int(choice_int(rng, {0, 1, 1, 2, 2, 3, 4})));
      break;
    case SelectorKind::BY_BOUNDARY_TOUCH:
      args_set(args, "side", dv_str(choice_str(rng, boundary_sides())));
      break;
    case SelectorKind::BY_ATTR: {
      if (!by_attr_pool_.empty() && rng.random() < 0.5) {
        args = by_attr_pool_[rng.choice_index(by_attr_pool_.size())];
        break;
      }
      double r = rng.random();
      if (r < 0.4) args_set(args, "attr", dv_str(object_attr_names()[rng.choice_index(object_attr_names().size())]));
      else if (r < 0.7) args_set(args, "op", dv_str(choice_str(rng, {"eq", "ne", "ge", "le", "max", "min", "unique"})));
      else args_set(args, "value", dv_int(choice_int(rng, {0, 1, 2, 3, 4})));
      break;
    }
    case SelectorKind::TOUCHING_COLOR:
      args_set(args, "color", dv_int(sample_color(rng)));
      // An edge_type arg: the old edge_to_color, which also jittered these.
      if (args_get(args, "edge_type")) {
        if (rng.random() < 0.5)
          args_set(args, "edge_type", dv_str(choice_str(rng, edge_choices())));
        if (rng.random() < 0.5)
          args_set(args, "direction", dv_str(choice_str(rng, direction3())));
      }
      break;
    case SelectorKind::ADJACENT_TO_COLOR:
      args_set(args, "color", dv_int(sample_color(rng)));
      if (rng.random() < 0.7)
        args_set(args, "connectivity", dv_int(choice_int(rng, {4, 8})));
      break;
    case SelectorKind::EDGE_COUNT:
      // The op is kept: a new op comes only with a new selector, as a new kind
      // did before the merge.
      if (rng.random() < 0.7)
        args_set(args, "edge_type", dv_str(choice_str(rng, edge_choices())));
      if (rng.random() < 0.7)
        args_set(args, "direction", dv_str(choice_str(rng, direction3())));
      if (variant_of(args, "op", "any") == "ge") args_set(args, "count", dv_int(rng.randint(1, 5)));
      break;
    case SelectorKind::REFERENCE_SINGLETONS:
      args_set(args, "max_mass", dv_int(choice_int(rng, {1, 1, 2, 3, 4})));
      if (rng.random() < 0.7)
        args_set(args, "outside_bbox_of_largest",
                 dv_bool(choice_bool(rng, {true, false})));
      if (rng.random() < 0.3)
        args_set(args, "exclude_color", dv_int(sample_color(rng)));
      break;
    case SelectorKind::BY_NOISE_SCORE: {
      const DslValue* pmin = args_get(args, "min_score");
      const DslValue* pmax = args_get(args, "max_score");
      double lo = pmin ? (pmin->tag == DslValue::Tag::Float ? pmin->f
                                                            : (double)pmin->i)
                       : choice_dbl(rng, {0.0, 0.1, 0.2, 0.3});
      double hi = pmax ? (pmax->tag == DslValue::Tag::Float ? pmax->f
                                                            : (double)pmax->i)
                       : choice_dbl(rng, {0.7, 0.8, 0.9, 1.0});
      if (rng.random() < 0.7) {
        double d = rng.uniform(-0.2, 0.2);
        lo = std::max(0.0, std::min(1.0, lo + d));
      }
      if (rng.random() < 0.7) {
        double d = rng.uniform(-0.2, 0.2);
        hi = std::max(0.0, std::min(1.0, hi + d));
      }
      if (lo > hi) std::swap(lo, hi);
      args_set(args, "min_score", dv_float(lo));
      args_set(args, "max_score", dv_float(hi));
      if (rng.random() < 0.7)
        args_set(args, "order",
                 dv_str(choice_str(rng, {"none", "highest", "lowest"})));
      if (rng.random() < 0.7)
        args_set(args, "top_k", dv_int(choice_int(rng, {1, 1, 2, 3, 4, 5})));
      break;
    }
    case SelectorKind::DISTANCE_TO_SELECTOR:
      if (rng.random() < 0.7)
        args_set(args, "other_selector_kind",
                 dv_str(choice_str(
                     rng, {std::string("largest_subobject"),
                           std::string("smallest_subobject"),
                           std::string("top_leftmost_subobject"),
                           to_string(SelectorKind::REFERENCE_SINGLETONS)})));
      if (rng.random() < 0.7) {
        DslArgs osa;
        osa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 2, 3, 4})));
        osa.emplace_back("outside_bbox_of_largest",
                         dv_bool(choice_bool(rng, {true, false})));
        args_set(args, "other_selector_args", dv_dict(std::move(osa)));
      }
      if (rng.random() < 0.7)
        args_set(args, "distance",
                 dv_str(choice_str(rng, {"centroid", "bbox_gap"})));
      if (rng.random() < 0.7)
        args_set(args, "top_k", dv_int(choice_int(rng, {1, 1, 2, 3, 4, 5})));
      if (rng.random() < 0.4)
        args_set(args, "include_references",
                 dv_bool(choice_bool(rng, {true, false})));
      break;
    case SelectorKind::INSIDE_LARGEST_BBOX:
      if (rng.random() < 0.7)
        args_set(args, "include_border",
                 dv_bool(choice_bool(rng, {true, false})));
      break;
    case SelectorKind::SAME_SHAPE_AS_REFERENCE:
      if (rng.random() < 0.7)
        args_set(args, "reference_selector_kind",
                 dv_str(choice_str(
                     rng,
                     {std::string("top_leftmost_subobject"),
                      std::string("top_leftmost_non_most_frequent"),
                      std::string("leftmost_non_most_frequent"),
                      to_string(SelectorKind::INSIDE_LARGEST_BBOX),
                      std::string("topmost_subobject"),
                      std::string("leftmost_subobject"),
                      std::string("smallest_subobject"),
                      std::string("largest_subobject")})));
      if (rng.random() < 0.4)
        args_set(args, "exclude_reference",
                 dv_bool(choice_bool(rng, {true, false})));
      break;
    case SelectorKind::SELECTOR_SET_OP:
      if (rng.random() < 0.7)
        args_set(args, "op",
                 dv_str(choice_str(rng,
                                   {"intersection", "difference", "union"})));
      if (rng.random() < 0.7) {
        const std::string& lk = composeable_selector_names()[rng.choice_index(
            composeable_selector_names().size())];
        DslSelector leg = compose_leg(lk, rng);
        args_set(args, "left_selector_kind", dv_str(to_string(leg.kind)));
        args_set(args, "left_selector_args", dv_dict(std::move(leg.args)));
      }
      if (rng.random() < 0.7) {
        const std::string& rk = composeable_selector_names()[rng.choice_index(
            composeable_selector_names().size())];
        DslSelector leg = compose_leg(rk, rng);
        args_set(args, "right_selector_kind", dv_str(to_string(leg.kind)));
        args_set(args, "right_selector_args", dv_dict(std::move(leg.args)));
      }
      break;
    default:
      // Any other kind: 25% chance to replace the whole selector with a fresh
      // random rule's selector.
      if (rng.random() < 0.25) return random_rule(rng).selector;
      break;
  }
  DslSelector out;
  out.kind = kind;
  out.args = std::move(args);
  return out;
}

// ---------------------------------------------------------------------------
// mutate_rule.
// ---------------------------------------------------------------------------
DslRule TransformCodec::mutate_rule(const DslRule& rule, ArcRandom& rng) {
  if (rng.random() < 0.35) return random_rule(rng);

  DslSelector selector = rule.selector;
  DslAction action = rule.action;
  std::vector<DslGuard> guards = rule.guards;  // a copy

  if (rng.random() < 0.35) selector = mutate_selector(selector, rng);

  ActionKind ak = action.kind;
  bool matched = true;
  DslArgs a;
  switch (ak) {
    case ActionKind::RECOLOR:
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::RECOLOR_FROM_EDGE_MAJORITY:
      a.emplace_back("edge_type", dv_str(choice_str(rng, edge_choices())));
      a.emplace_back("direction", dv_str(choice_str(rng, direction3())));
      break;
    case ActionKind::RECOLOR_FROM_REFERENCE: {
      std::string rsk = choice_str(
          rng, {to_string(SelectorKind::NON_BACKGROUND),
                to_string(SelectorKind::ALL_SUBOBJECTS),
                to_string(SelectorKind::INSIDE_LARGEST_BBOX),
                std::string("top_leftmost_subobject"),
                std::string("top_leftmost_non_most_frequent"),
                std::string("leftmost_non_most_frequent"),
                to_string(SelectorKind::REFERENCE_SINGLETONS)});
      a.emplace_back("reference_selector_kind", dv_str(rsk));
      DslArgs rsa;
      rsa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 2, 3})));
      rsa.emplace_back("outside_bbox_of_largest",
                       dv_bool(choice_bool(rng, {true, false})));
      a.emplace_back("reference_selector_args", dv_dict(std::move(rsa)));
      a.emplace_back("color_policy",
                     dv_str(choice_str(rng, {"nearest", "majority", "only"})));
      a.emplace_back(
          "match_policy",
          dv_str(choice_str(rng, {"same_shape", "same_shape_strict", "any"})));
      a.emplace_back("exclude_selected",
                     dv_bool(choice_bool(rng, {true, false})));
      break;
    }
    case ActionKind::RECOLOR_SEQUENCE:
      a.emplace_back(
          "order",
          dv_str(choice_str(rng,
                            {"y_then_x", "x_then_y", "mass_desc", "mass_asc"})));
      a.emplace_back("start_color", dv_int(sample_color(rng)));
      a.emplace_back("step", dv_int(choice_int(rng, {1, 1, 2, 3})));
      break;
    case ActionKind::SWAP_COLORS: {
      long long av = 0, bv = 0;
      const DslValue* pa = args_get(action.args, "color_a");
      const DslValue* pb = args_get(action.args, "color_b");
      if (pa && pa->tag == DslValue::Tag::Int) av = pa->i;
      if (pb && pb->tag == DslValue::Tag::Int) bv = pb->i;
      // 50/50: jitter one leg or resample both.
      if (rng.random() < 0.5) {
        if (rng.random() < 0.5)
          av = sample_color(rng);
        else
          bv = sample_color(rng);
      } else {
        av = sample_color(rng);
        bv = sample_color(rng);
      }
      if (av == bv) bv = (av + 1) % 10;
      a.emplace_back("color_a", dv_int(av));
      a.emplace_back("color_b", dv_int(bv));
      break;
    }
    case ActionKind::PRUNE_ENDPOINTS:
      a.emplace_back("iterations", dv_int(choice_int(rng, {1, 1, 2, 3})));
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      a.emplace_back("protect_junctions",
                     dv_bool(choice_bool(rng, {true, false})));
      break;
    case ActionKind::SHRINK_TO_CORE:
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      break;
    case ActionKind::DILATE_SELECTED:
      a.emplace_back("radius", dv_int(choice_int(rng, {1, 1, 2, 3})));
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 4, 8})));
      a.emplace_back(
          "mask",
          dv_str(choice_str(rng, {"auto", "cross", "diamond", "square"})));
      break;
    case ActionKind::COMPLETE_D4_SYMMETRY:
      a.emplace_back("include_pivot_color",
                     dv_bool(choice_bool(rng, {false, false, true})));
      break;
    case ActionKind::PROJECT_INNER_QUAD_TO_OUTER_CORNERS:
      break;
    case ActionKind::PROJECT_SINGLETONS_TO_FULL_LINES:
      break;
    case ActionKind::STAMP_LOCAL_PATTERN:
      a.emplace_back(
          "pattern",
          dv_str(choice_str(rng, {"dot", "cross", "x", "ring", "square"})));
      a.emplace_back(
          "anchor",
          dv_str(choice_str(rng, {"centroid", "bbox_center", "min_corner"})));
      a.emplace_back(
          "color_source",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::EXTEND_LINE_UNTIL:
      a.emplace_back(
          "direction",
          dv_str(choice_str(
              rng, {"up", "down", "left", "right", "ray", "ray8", "diag",
                    "up_left", "up_right", "down_left", "down_right"})));
      a.emplace_back(
          "stop_condition",
          dv_str(choice_str(rng, {"boundary", "touch_object", "color!=bg"})));
      a.emplace_back("stop_on_same_color",
                     dv_bool(choice_bool(rng, {false, false, true})));
      a.emplace_back("intersection_color", dv_int(sample_color(rng)));
      a.emplace_back(
          "anchor",
          dv_str(choice_str(rng, {"centroid", "bbox_center", "min_corner"})));
      a.emplace_back(
          "color_source",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::OUTLINE_OBJECT:
      a.emplace_back("connectivity", dv_int(choice_int(rng, {4, 8, 8})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::TRANSLATE_BY_OWN_EXTENT:
      a.emplace_back("extent_source",
                     dv_str(choice_str(rng, {"width", "height", "mass"})));
      a.emplace_back("mult_y", dv_int(choice_int(rng, {-1, 0, 1})));
      a.emplace_back("mult_x", dv_int(choice_int(rng, {-1, 0, 1})));
      break;
    case ActionKind::GRAVITY_TO_EDGE:
      a.emplace_back(
          "direction",
          dv_str(choice_str(rng, {"up", "down", "left", "right"})));
      break;
    case ActionKind::FILL_ENCLOSED_BG:
      a.emplace_back("fill_color", dv_int(sample_color(rng)));
      break;
    case ActionKind::FILL_DIAGONAL_PERIODIC:
      a.emplace_back("diagonal", dv_str(choice_str(rng, {"main", "anti"})));
      a.emplace_back("period", dv_int(choice_int(rng, {2, 3, 4, 5})));
      break;
    case ActionKind::TILE_FOLD_OR_NONBG:
      a.emplace_back("div_y", dv_int(choice_int(rng, {1, 2, 2, 3})));
      a.emplace_back("div_x", dv_int(choice_int(rng, {1, 2, 2, 3})));
      break;
    // FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS: only the single-arg form
    // (fill_color + axes, 5 axis choices) is mutated.
    case ActionKind::FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS:
      a.emplace_back("fill_color", dv_int(sample_color(rng)));
      a.emplace_back(
          "axes",
          dv_str(choice_str(
              rng, {"horizontal", "vertical", "diagonal", "both", "all"})));
      break;
    case ActionKind::CROP_TO_OBJECT:
      a.emplace_back(
          "which",
          dv_str(choice_str(rng, {"largest", "smallest", "unique_color",
                                  "odd_one_out", "densest"})));
      a.emplace_back(
          "recolor",
          dv_str(choice_str(rng,
                            {"none", "none", "swap_two", "to_marker_color"})));
      a.emplace_back("obj_mode", dv_str(choice_str(rng, {"multi", "mono"})));
      break;
    case ActionKind::PANEL_SPLIT_BOOLEAN:
      a.emplace_back("axis", dv_str(choice_str(rng, {"h", "v", "auto"})));
      a.emplace_back(
          "op", dv_str(choice_str(
                    rng, {"and", "or", "xor", "nand", "nor", "xnor", "diff"})));
      a.emplace_back("layout",
                     dv_str(choice_str(rng, {"2", "2", "3", "2x2"})));
      a.emplace_back("bg_policy",
                     dv_str(choice_str(rng, {"zero", "zero", "detected"})));
      a.emplace_back("out_color", dv_int(sample_color(rng)));
      break;
    case ActionKind::SCALE_BY_COUNT:
      a.emplace_back(
          "mode",
          dv_str(choice_str(
              rng, {"scale", "scale", "tile_h", "tile_v", "tile_both"})));
      a.emplace_back("count_type",
                     dv_str(choice_str(rng, {"colors", "objects"})));
      a.emplace_back("color_strict", dv_bool(choice_bool(rng, {false, true})));
      a.emplace_back("diag", dv_bool(choice_bool(rng, {false, true})));
      break;
    case ActionKind::PROJECT_MARKER:
      a.emplace_back(
          "side", dv_str(choice_str(rng, {"bottom", "top", "left", "right"})));
      a.emplace_back("align",
                     dv_str(choice_str(rng, {"centroid", "bbox_center"})));
      a.emplace_back(
          "color_mode",
          dv_str(choice_str(rng, {"self", "constant", "background"})));
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    case ActionKind::ATTACH_SAME_COLOR_SINGLETONS:
      a.emplace_back("min_line_length",
                     dv_int(choice_int(rng, {2, 3, 4, 5, 6})));
      a.emplace_back("delete_source",
                     dv_bool(choice_bool(rng, {true, false})));
      break;
    case ActionKind::DELETE_REFERENCE_AFTER_TRANSFER: {
      a.emplace_back("reference_selector_kind",
                     dv_str(to_string(SelectorKind::REFERENCE_SINGLETONS)));
      DslArgs rsa;
      rsa.emplace_back("max_mass", dv_int(choice_int(rng, {1, 1, 2, 3})));
      rsa.emplace_back("outside_bbox_of_largest",
                       dv_bool(choice_bool(rng, {true, false})));
      a.emplace_back("reference_selector_args", dv_dict(std::move(rsa)));
      break;
    }
    case ActionKind::TRANSLATE: {
      // The mode is kept. mode=origin (normalize_to_origin) has no args, so no
      // branch matches it.
      const std::string mode = action_variant(action);
      if (mode == "origin") {
        matched = false;
        break;
      }
      a.emplace_back("mode", dv_str(mode));
      a.emplace_back("dy", dv_int(rng.randint(kTranslateMin, kTranslateMax)));
      a.emplace_back("dx", dv_int(rng.randint(kTranslateMin, kTranslateMax)));
      break;
    }
    case ActionKind::FILL_CAVITY:
      a.emplace_back("color", dv_int(sample_color(rng)));
      break;
    default:
      if (find_concept_op(ak)) a = concept_args(ak, rng);
      else matched = false;
      break;
  }
  if (matched) {
    action.kind = ak;
    action.args = std::move(a);
  } else {
    // Final branch (random() < 0.2), only reached when NONE of the action-kind
    // branches matched: replace the whole action with a fresh one.
    if (rng.random() < 0.2) action = random_rule(rng).action;
  }

  // Guard mutation tail.
  if (rng.random() < 0.25) {
    if (!guards.empty() && rng.random() < 0.5) {
      guards.erase(guards.begin() +
                   static_cast<long>(rng.randrange(
                       static_cast<long long>(guards.size()))));
    } else {
      std::vector<DslGuard> ng = random_guards(rng);
      for (auto& g : ng) guards.push_back(std::move(g));
    }
  } else if (!guards.empty() && rng.random() < 0.25) {
    std::vector<DslGuard> mg;
    mg.reserve(guards.size());
    for (const auto& g : guards) mg.push_back(mutate_guard(g, rng));
    guards = std::move(mg);
  }

  DslRule out;
  out.selector = std::move(selector);
  out.action = std::move(action);
  out.guards = std::move(guards);
  return out;
}

// ---------------------------------------------------------------------------
// mutate_encoder_param: the special-cased params first, then a generic
// per-meta.kind mutation of the current value (always set for those params).
// ---------------------------------------------------------------------------
void TransformCodec::mutate_encoder_param(const std::string& name,
                                          const ParamMeta& meta, ArcRandom& rng,
                                          GraphConfig& cfg) {
  // Special cases, IN ORDER.
  if (name == "background_color_override") {
    if (rng.random() < 0.8) {
      // Uniform over {none, -1, 0, 0, 0, 1, 2}; none is stored as -1 too.
      static const std::vector<int> ch = {-1, -1, 0, 0, 0, 1, 2};
      cfg.background_color_override = choice_int(rng, ch);
    } else {
      // Generic int mutation of the current value, with no override (-1) taken
      // as 0. meta is int kind.
      long long value = cfg.background_color_override;
      long long base = (value == -1) ? 0 : value;
      long long lo = static_cast<long long>(meta.low);
      long long hi = static_cast<long long>(meta.high);
      if (rng.random() < 0.7) {
        long long delta = choice_int(rng, {-3, -2, -1, 1, 2, 3});
        long long v = base + delta;
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        cfg.background_color_override = v;
      } else {
        cfg.background_color_override = rng.randint(lo, hi);
      }
    }
    return;
  }
  if (name == "min_object_size") {
    long long base = cfg.min_object_size;
    if (rng.random() < 0.8) {
      long long d = choice_int(rng, {-2, -1, 1, 2});
      long long v = base + d;
      if (v < 0) v = 0;
      if (v > 8) v = 8;
      cfg.min_object_size = static_cast<int>(v);
    } else {
      cfg.min_object_size = static_cast<int>(rng.randint(0, 12));
    }
    return;
  }
  if (name == "detect_zones") {
    cfg.detect_zones = choice_bool(rng, {false, false, true, true});
    return;
  }
  if (name == "compute_zone_features") {  // not in the 21, but guarded
    cfg.compute_zone_features = choice_bool(rng, {false, false, true});
    return;
  }
  if (name == "grid_line_threshold") {
    double base = cfg.grid_line_threshold;
    if (rng.random() < 0.8) {
      double g = rng.gauss(0.0, 0.03);
      double v = std::max(0.7, std::min(1.0, base + g));
      cfg.grid_line_threshold = v;
    } else {
      cfg.grid_line_threshold = rng.uniform(0.8, 1.0);
    }
    return;
  }
  // Generic mutation by metadata kind.
  switch (meta.kind) {
    case ParamKind::Bool:
      // flip the bool
      set_config_bool(cfg, name, !get_config_bool(cfg, name));
      break;
    case ParamKind::Categorical: {
      // choices minus the current value (if present and >1 choice), then choice.
      std::string cur = get_config_str(cfg, name);
      std::vector<std::string> ch;
      if (meta.choices_are_int) {
        // int-choice categorical (connectivity): current value stored as int.
        long long curi = get_config_int(cfg, name);
        std::string curs = std::to_string(curi);
        std::vector<std::string> pool = meta.choices;
        // Remove the current value if present and len > 1.
        bool present = false;
        for (const auto& c : pool)
          if (c == curs) present = true;
        if (present && pool.size() > 1) {
          std::vector<std::string> keep;
          for (const auto& c : pool)
            if (c != curs) keep.push_back(c);
          pool = std::move(keep);
        }
        if (pool.empty()) return;  // choices empty -> return value unchanged
        std::string pick = choice_str(rng, pool);
        set_config_int(cfg, name, std::stoll(pick));
      } else {
        std::vector<std::string> pool = meta.choices;
        bool present = false;
        for (const auto& c : pool)
          if (c == cur) present = true;
        if (present && pool.size() > 1) {
          std::vector<std::string> keep;
          for (const auto& c : pool)
            if (c != cur) keep.push_back(c);
          pool = std::move(keep);
        }
        if (pool.empty()) return;
        set_config_str(cfg, name, choice_str(rng, pool));
      }
      break;
    }
    case ParamKind::Int: {
      long long lo = static_cast<long long>(meta.low);
      long long hi = static_cast<long long>(meta.high);
      long long base = get_config_int(cfg, name);
      if (rng.random() < 0.7) {
        long long delta = choice_int(rng, {-3, -2, -1, 1, 2, 3});
        long long v = base + delta;
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        set_config_int(cfg, name, v);
      } else {
        set_config_int(cfg, name, rng.randint(lo, hi));
      }
      break;
    }
    case ParamKind::Float: {
      double lo = meta.low;
      double hi = meta.high;
      double base = get_config_float(cfg, name);
      if (rng.random() < 0.7) {
        double span = std::max(hi - lo, 1e-9);
        double delta = rng.gauss(0.0, span * 0.1);
        double v = std::max(lo, std::min(hi, base + delta));
        set_config_float(cfg, name, v);
      } else {
        set_config_float(cfg, name, rng.uniform(lo, hi));
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// mutate_encoder_config.
// ---------------------------------------------------------------------------
GraphConfig TransformCodec::mutate_encoder_config(const GraphConfig& config,
                                                  ArcRandom& rng) {
  GraphConfig cfg = config;  // a copy
  const std::vector<ParamMeta>& table = param_metadata_table();
  int n_changes = (rng.random() < 0.85) ? 1 : 2;
  const std::vector<std::string>& params = evolvable_encoder_params();
  for (int i = 0; i < n_changes; ++i) {
    const std::string& name = params[rng.choice_index(params.size())];
    const ParamMeta* meta = nullptr;
    for (const auto& pm : table) {
      if (name == pm.name) {
        meta = &pm;
        break;
      }
    }
    if (!meta) continue;
    mutate_encoder_param(name, *meta, rng, cfg);
  }
  normalize_encoder(cfg);
  return cfg;
}

// ---------------------------------------------------------------------------
// mutate_decoder.
// ---------------------------------------------------------------------------
DecoderConfig TransformCodec::mutate_decoder(const DecoderConfig& decoder,
                                             ArcRandom& rng) {
  static const std::vector<std::string> bg = {
      "scene_background_or_zero", "zero", "most_frequent_input",
      "most_frequent_predicted", "input_copy"};
  static const std::vector<std::string> ov = {"overwrite", "keep_first"};
  static const std::vector<std::string> pp = {"none", "crop_foreground", "keep_lines"};
  DecoderConfig d = decoder;
  // Skip paint_order mutations: one uniform draw over bg / ov / pp.
  std::string which = choice_str(rng, {"bg", "ov", "pp"});
  if (which == "bg") {
    d.background_policy = choice_str(rng, bg);
  } else if (which == "ov") {
    d.overlap_policy = choice_str(rng, ov);
  } else {
    d.postprocess = choice_str(rng, pp);
  }
  return d;
}

// ---------------------------------------------------------------------------
// mutate (the guarded fall-through). SINGLE p = random(); a failed guard falls
// through to the next threshold, changing downstream draw counts.
// ---------------------------------------------------------------------------
TransformGenome TransformCodec::mutate(const TransformGenome& genome,
                                       ArcRandom& rng) {
  GraphConfig encoder_config = genome.encoder_config;
  std::vector<DslRule> rules = genome.program.rules;  // a copy
  GlobalTransform transform = genome.program.global_transform;
  DecoderConfig decoder = genome.decoder;

  double p = rng.random();
  const int nr = static_cast<int>(rules.size());

  if (p < kThRuleEdit && nr > 0) {
    long long idx = rng.randrange(static_cast<long long>(rules.size()));
    rules[static_cast<size_t>(idx)] =
        mutate_rule(rules[static_cast<size_t>(idx)], rng);
  } else if (p < kThAddRule && nr < max_rules_) {
    // maybe_macro_rules(mutating=true): macros OFF -> false, NO draw ->
    // random_rule. ON -> append macro rules up to max_rules in total.
    std::vector<DslRule> macro_rules;
    if (maybe_macro_rules(rng, /*mutating=*/true, &macro_rules)) {
      int room = max_rules_ - nr;
      if (room < 0) room = 0;
      if (static_cast<int>(macro_rules.size()) > room) {
        macro_rules.resize(static_cast<size_t>(room));
      }
      for (auto& mr : macro_rules) rules.push_back(std::move(mr));
    } else {
      rules.push_back(random_rule(rng));
    }
  } else if (p < kThRemoveRule && nr > 0) {
    long long idx = rng.randrange(static_cast<long long>(rules.size()));
    rules.erase(rules.begin() + static_cast<long>(idx));
  } else if (p < kThSwapTransform) {
    transform =
        static_cast<GlobalTransform>(rng.choice_index(kGlobalTransformCount));
  } else if (p < kThMutateEncoder) {
    encoder_config = mutate_encoder_config(encoder_config, rng);
  } else if (p < kThMutateDecoder) {
    decoder = mutate_decoder(decoder, rng);
  } else if (nr >= 2) {
    std::vector<size_t> ij = rng.sample_indices(rules.size(), 2);
    std::swap(rules[ij[0]], rules[ij[1]]);
  } else {
    // Fall-through: mutate a rule if any, else add one.
    if (nr > 0) {
      long long idx = rng.randrange(static_cast<long long>(rules.size()));
      rules[static_cast<size_t>(idx)] =
          mutate_rule(rules[static_cast<size_t>(idx)], rng);
    } else if (nr < max_rules_) {
      rules.push_back(random_rule(rng));
    }
  }

  TransformGenome out;
  out.encoder_config = std::move(encoder_config);
  out.program.rules = std::move(rules);
  out.program.global_transform = transform;
  out.decoder = std::move(decoder);
  return out;
}

// ---------------------------------------------------------------------------
// crossover_encoder. Iterate GraphConfig declaration order; draw random() ONLY
// for the 21 evolvable fields (mom<0.5 else dad); copy-from-mom (NO draw) for
// the rest. normalize afterwards.
// ---------------------------------------------------------------------------
GraphConfig TransformCodec::crossover_encoder(const GraphConfig& mom,
                                              const GraphConfig& dad,
                                              ArcRandom& rng) {
  GraphConfig out = mom;  // start from mom (non-evolvable fields copy from mom)
  for (const std::string& name : param_metadata_order()) {
    if (!is_evolvable_encoder_param(name)) continue;  // no draw
    bool take_dad = !(rng.random() < 0.5);
    if (!take_dad) continue;  // already mom
    // Copy dad's field.
    const std::vector<ParamMeta>& table = param_metadata_table();
    const ParamMeta* meta = nullptr;
    for (const auto& pm : table)
      if (name == pm.name) { meta = &pm; break; }
    if (!meta) continue;
    switch (meta->kind) {
      case ParamKind::Bool:
        set_config_bool(out, name, get_config_bool(dad, name));
        break;
      case ParamKind::Categorical:
        if (meta->choices_are_int)
          set_config_int(out, name, get_config_int(dad, name));
        else
          set_config_str(out, name, get_config_str(dad, name));
        break;
      case ParamKind::Int:
        set_config_int(out, name, get_config_int(dad, name));
        break;
      case ParamKind::Float:
        set_config_float(out, name, get_config_float(dad, name));
        break;
    }
  }
  normalize_encoder(out);
  return out;
}

// ---------------------------------------------------------------------------
// crossover. Encoder first, then transform (50/50), then independent rule cuts,
// then decoder (50/50). Rule list capped at max_rules.
// ---------------------------------------------------------------------------
TransformGenome TransformCodec::crossover(const TransformGenome& mom,
                                          const TransformGenome& dad,
                                          ArcRandom& rng) {
  GraphConfig encoder_config =
      crossover_encoder(mom.encoder_config, dad.encoder_config, rng);
  GlobalTransform transform = (rng.random() < 0.5)
                                  ? mom.program.global_transform
                                  : dad.program.global_transform;
  const std::vector<DslRule>& left = mom.program.rules;
  const std::vector<DslRule>& right = dad.program.rules;
  long long cut_l = rng.randint(0, static_cast<long long>(left.size()));
  long long cut_r = rng.randint(0, static_cast<long long>(right.size()));
  std::vector<DslRule> rules;
  for (long long i = 0; i < cut_l; ++i)
    rules.push_back(left[static_cast<size_t>(i)]);
  for (size_t i = static_cast<size_t>(cut_r); i < right.size(); ++i)
    rules.push_back(right[i]);
  if (static_cast<int>(rules.size()) > max_rules_)
    rules.resize(static_cast<size_t>(max_rules_));
  DecoderConfig decoder = (rng.random() < 0.5) ? mom.decoder : dad.decoder;

  TransformGenome out;
  out.encoder_config = std::move(encoder_config);
  out.program.rules = std::move(rules);
  out.program.global_transform = transform;
  out.decoder = std::move(decoder);
  return out;
}

// ===========================================================================
// delta_guided_mutate (biased mutation).
//
// The hint is a plain INPUT struct (DeltaHint, computed by compute_delta_hint).
// Draw order:
//   Branch-A: one choice over the candidates (one randbelow(len)), then no
//             more draws.
//   Branch-B: no draws at all (append a fixed rule).
//   Fallback: delegate to mutate() (its full guarded fall-through draw order).
// ===========================================================================

const char* color_arg_by_actionkind(ActionKind kind) {
  // COLOR_ARG_BY_ACTIONKIND: the colour arg of 9 action kinds. Every other
  // kind -> nullptr.
  switch (kind) {
    case ActionKind::RECOLOR:
    case ActionKind::FILL_BBOX:
    case ActionKind::OUTLINE_OBJECT:
    case ActionKind::STAMP_LOCAL_PATTERN:
    case ActionKind::EXTEND_LINE_UNTIL:
    case ActionKind::PROJECT_MARKER:
    case ActionKind::FILL_CAVITY:
      return "color";
    case ActionKind::PANEL_SPLIT_BOOLEAN:
      return "out_color";
    default:
      return nullptr;
  }
}

TransformGenome delta_guided_mutate(TransformCodec& codec,
                                    const TransformGenome& genome,
                                    const DeltaHint& hint, ArcRandom& rng) {
  std::vector<DslRule> rules = genome.program.rules;  // a copy

  // Branch-A: dominant_swap + a matching rule exists.
  if (hint.has_dominant_swap) {
    const int src = hint.dominant_swap_src;
    const int dst = hint.dominant_swap_dst;
    std::vector<size_t> candidates;
    for (size_t i = 0; i < rules.size(); ++i) {
      const char* arg_name = color_arg_by_actionkind(rules[i].action.kind);
      if (arg_name == nullptr) continue;
      const DslValue* av = args_get(rules[i].action.args, arg_name);
      // Only an int arg equal to src qualifies (a missing key or a non-int
      // value does not).
      if (av && av->tag == DslValue::Tag::Int && av->i == src) {
        candidates.push_back(i);
      }
    }
    if (!candidates.empty()) {
      size_t idx = candidates[rng.choice_index(candidates.size())];
      const char* arg_name = color_arg_by_actionkind(rules[idx].action.kind);
      // new_args = a copy of the old action's args with arg_name set to dst.
      DslRule& r = rules[idx];
      DslArgs new_args = r.action.args;  // copy preserving insertion order
      // set() in-place (ArgMap semantics): replace if present, else append.
      bool replaced = false;
      for (auto& kv : new_args) {
        if (kv.first == arg_name) {
          kv.second = dv_int(dst);
          replaced = true;
          break;
        }
      }
      if (!replaced) new_args.emplace_back(arg_name, dv_int(dst));
      r.action.args = std::move(new_args);
      TransformGenome out;
      out.encoder_config = genome.encoder_config;
      out.program.rules = std::move(rules);
      out.program.global_transform = genome.program.global_transform;
      out.decoder = genome.decoder;
      return out;
    }
  }

  // Branch-B: append BY_COLOR(majority_got) -> RECOLOR(majority_expected) when
  // room remains and majority_got covers >=60% of wrong cells.
  if (static_cast<int>(rules.size()) < codec.max_rules()) {
    const size_t all_wrong = hint.wrong_got_colors.size();
    if (all_wrong > 0) {
      size_t got_mode_count = 0;
      for (int c : hint.wrong_got_colors) {
        if (c == hint.majority_got_color) ++got_mode_count;
      }
      // got_mode_count / all_wrong >= 0.60, in floating point.
      if (static_cast<double>(got_mode_count) / static_cast<double>(all_wrong) >=
          0.60) {
        DslRule new_rule;
        new_rule.selector.kind = SelectorKind::BY_COLOR;
        new_rule.selector.args.emplace_back(
            "color", dv_int(hint.majority_got_color));
        new_rule.action.kind = ActionKind::RECOLOR;
        new_rule.action.args.emplace_back(
            "color", dv_int(hint.majority_expected_color));
        rules.push_back(std::move(new_rule));
        TransformGenome out;
        out.encoder_config = genome.encoder_config;
        out.program.rules = std::move(rules);
        out.program.global_transform = genome.program.global_transform;
        out.decoder = genome.decoder;
        return out;
      }
    }
  }

  // Both branches inapplicable: fall back to uniform mutation.
  return codec.mutate(genome, rng);
}

// ===========================================================================
// The three dedup key shapes, each printed as a repr-style string
// (pyrepr.hpp).
//
// Two genomes collapse under a key iff their key strings are equal, so dedup
// is a plain string comparison.
// ===========================================================================

namespace {

// Quoted form of a plain ASCII identifier (policy names, config str values),
// via py_repr_str.
std::string q(const std::string& s) { return py_repr_str(s); }

// A bool as True / False.
std::string repr_bool(bool b) { return b ? "True" : "False"; }

// One config field's value in repr form, addressed by declaration index 0..43
// (== param_metadata_order() index). Floats via py_repr_float, bools as
// True/False, background_color_override == -1 as None (no override, the
// default), strings via py_repr_str, ints decimal.
std::string cfg_value_repr(const GraphConfig& c, int idx) {
  switch (idx) {
    case 0:  return std::to_string(c.connectivity);
    case 1:  return q(c.color_mode);
    case 2:  return q(c.background_mode);
    case 3:  return c.background_color_override < 0
                 ? std::string("None")
                 : std::to_string(c.background_color_override);
    case 4:  return std::to_string(c.min_object_size);
    case 5:  return q(c.object_mode);
    case 6:  return std::to_string(c.merge_distance);
    case 7:  return py_repr_float(c.noise_threshold);
    case 8:  return repr_bool(c.multi_color_as_cluster);
    case 9:  return repr_bool(c.cluster_decompose_two_color);
    case 10: return q(c.noise_filter_order);
    case 11: return repr_bool(c.decompose_objects);
    case 12: return q(c.decomposition_mode);
    case 13: return repr_bool(c.split_by_containment);
    case 14: return repr_bool(c.decompose_grid_lines);
    case 15: return repr_bool(c.merge_by_containment);
    case 16: return py_repr_float(c.grid_line_threshold);
    case 17: return repr_bool(c.detect_zones);
    case 18: return repr_bool(c.detect_cavities);
    case 19: return repr_bool(c.detect_clusters);
    case 20: return repr_bool(c.detect_patterns);
    case 21: return repr_bool(c.detect_symmetries);
    case 22: return repr_bool(c.compute_spatial_edges);
    case 23: return std::to_string(c.spatial_touch_connectivity);
    case 24: return repr_bool(c.compute_containment_edges);
    case 25: return repr_bool(c.compute_alignment_edges);
    case 26: return repr_bool(c.compute_ray_edges);
    case 27: return repr_bool(c.compute_gravity_edges);
    case 28: return repr_bool(c.compute_bridge_edges);
    case 29: return repr_bool(c.compute_logic_edges);
    case 30: return repr_bool(c.compute_shape_classification);
    case 31: return repr_bool(c.compute_shape_hash);
    case 32: return repr_bool(c.compute_local_coordinates);
    case 33: return repr_bool(c.compute_z_index);
    case 34: return repr_bool(c.compute_noise_scoring);
    case 35: return repr_bool(c.compute_zone_features);
    case 36: return py_repr_float(c.noise_weight_isolation);
    case 37: return py_repr_float(c.noise_weight_singleton_color);
    case 38: return py_repr_float(c.noise_weight_size_outlier);
    case 39: return py_repr_float(c.noise_weight_shape_outlier);
    case 40: return py_repr_float(c.noise_weight_position_outlier);
    case 41: return py_repr_float(c.alignment_tolerance);
    case 42: return py_repr_float(c.symmetry_tolerance);
    case 43: return std::to_string(c.pattern_min_size);
    default: return "None";
  }
}

// Tuple helper: wrap comma-joined element reprs in parens, with a trailing
// comma for 1-tuples.
std::string repr_tuple(const std::vector<std::string>& elems) {
  std::string out = "(";
  for (size_t i = 0; i < elems.size(); ++i) {
    if (i) out += ", ";
    out += elems[i];
  }
  if (elems.size() == 1) out += ",";
  out += ")";
  return out;
}

}  // namespace

std::string decoder_repr(const DecoderConfig& d) {
  // DecoderConfig(...): field=value in declaration order, str values quoted.
  std::string out = "DecoderConfig(";
  out += "background_policy=" + q(d.background_policy);
  out += ", paint_order=" + q(d.paint_order);
  out += ", overlap_policy=" + q(d.overlap_policy);
  out += ", postprocess=" + q(d.postprocess);
  out += ")";
  return out;
}

std::string encoder_signature_repr(const GraphConfig& cfg) {
  // Every config value in param_metadata order (== declaration order ==
  // cfg_value_repr index order), as a tuple.
  std::vector<std::string> elems;
  elems.reserve(44);
  for (int i = 0; i < 44; ++i) elems.push_back(cfg_value_repr(cfg, i));
  return repr_tuple(elems);
}

std::string enc_sig(const GraphConfig& cfg) {
  // A list of (key, value) 2-tuples, sorted by key (byte-wise string order).
  const std::vector<std::string>& names = param_metadata_order();
  std::vector<std::pair<std::string, int>> pairs;  // (name, decl-index)
  pairs.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i)
    pairs.emplace_back(names[i], static_cast<int>(i));
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<std::string, int>& a,
               const std::pair<std::string, int>& b) {
              return a.first < b.first;
            });
  std::string out = "[";
  for (size_t i = 0; i < pairs.size(); ++i) {
    if (i) out += ", ";
    out += "(" + q(pairs[i].first) + ", " +
           cfg_value_repr(cfg, pairs[i].second) + ")";
  }
  out += "]";
  return out;
}

std::string dedup_finalist_key(const TransformGenome& g) {
  // (hash_program(program), decoder) as a 2-tuple.
  return repr_tuple({q(hash_program(g.program)), q(decoder_repr(g.decoder))});
}

std::string dedup_ga_key(const TransformGenome& g) {
  // (encoder_signature, hash, bg, po, ov, pp) as a 6-tuple. The
  // encoder_signature element is itself a tuple, nested as is.
  return repr_tuple({
      encoder_signature_repr(g.encoder_config),
      q(hash_program(g.program)),
      q(g.decoder.background_policy),
      q(g.decoder.paint_order),
      q(g.decoder.overlap_policy),
      q(g.decoder.postprocess),
  });
}

std::string dedup_guesspool_key(const TransformGenome& g) {
  // (hash, decoder, encoder signature) as a 3-tuple.
  return repr_tuple({
      q(hash_program(g.program)),
      q(decoder_repr(g.decoder)),
      q(enc_sig(g.encoder_config)),
  });
}

}  // namespace arc
