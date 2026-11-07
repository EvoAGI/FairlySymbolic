// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_CONFIG_HPP
#define ARC_CONFIG_HPP

// The 44-field GraphConfig: every parameter of the graph encoder.
//
// Field DECLARATION ORDER is load-bearing: it is the order the GA walks at
// crossover to draw per-parameter RNG values, so reordering the fields changes
// the search. Likewise param_metadata_order() and default_encoder_params() are
// ordered constants whose order reaches those same RNG draws.
//
// background_color_override uses -1 as its "unset" sentinel (a value >= 0
// selects an explicit background color; -1 means "use the heuristic").

#include <string>
#include <vector>

namespace arc {

struct GraphConfig {
  // --- Object definition / extraction (steps 1, 3, 3b, 3c) -----------------
  int connectivity = 8;                              // 4 or 8
  std::string color_mode = "mono";                   // "mono" | "multi"
  std::string background_mode = "auto";              // "auto" | "none" | "bg_as_object"
  int background_color_override = -1;                // -1 == unset (use heuristic)
  int min_object_size = 0;
  std::string object_mode = "connected_component";   // + "pixel" | "color_global"
  int merge_distance = 0;
  double noise_threshold = 1.0;
  bool multi_color_as_cluster = false;
  bool cluster_decompose_two_color = false;
  std::string noise_filter_order = "size_first";     // "size_first" | "noise_first"
  bool decompose_objects = false;
  std::string decomposition_mode = "lines";          // "lines" | "rectangles" | "both"
  bool split_by_containment = false;
  bool decompose_grid_lines = false;
  bool merge_by_containment = false;
  double grid_line_threshold = 1.0;

  // --- Detection toggles (steps 2, 4, 5, 16, 17) ---------------------------
  bool detect_zones = true;
  bool detect_cavities = true;
  bool detect_clusters = true;
  bool detect_patterns = true;
  bool detect_symmetries = true;

  // --- Edge computation toggles (steps 6, 8, 11, 12, 13, 14, 15) -----------
  bool compute_spatial_edges = true;
  int spatial_touch_connectivity = 4;                // 4 or 8
  bool compute_containment_edges = true;
  bool compute_alignment_edges = true;
  bool compute_ray_edges = true;
  bool compute_gravity_edges = true;
  bool compute_bridge_edges = true;
  bool compute_logic_edges = true;

  // --- Node feature computation toggles (steps 7, 9, 9b, 10) ---------------
  bool compute_shape_classification = true;
  bool compute_shape_hash = true;
  bool compute_local_coordinates = true;
  bool compute_z_index = true;
  bool compute_noise_scoring = true;
  bool compute_zone_features = true;

  // --- Noise-scoring weights (step 7) --------------------------------------
  double noise_weight_isolation = 0.3;
  double noise_weight_singleton_color = 0.2;
  double noise_weight_size_outlier = 0.2;
  double noise_weight_shape_outlier = 0.15;
  double noise_weight_position_outlier = 0.15;

  // --- Tolerances (steps 11, 17) -------------------------------------------
  double alignment_tolerance = 0.5;
  double symmetry_tolerance = 0.5;
  int pattern_min_size = 3;
};

// ---------------------------------------------------------------------------
// Parameter metadata (kind/choices/low/high), the source of truth for EA
// mutation. One entry per field, in declaration order.
// ---------------------------------------------------------------------------

enum class ParamKind { Categorical, Int, Float, Bool };

struct ParamMeta {
  const char* name;
  ParamKind kind;
  // For Categorical: choices are held as strings; int-choice params
  // (connectivity, spatial_touch_connectivity) store their choices as the
  // decimal strings "4"/"8" and set choices_are_int = true so the serializer
  // renders them as JSON numbers.
  std::vector<std::string> choices;
  bool choices_are_int;
  // For Int / Float: inclusive [low, high] bounds. Unused for Categorical/Bool.
  double low;
  double high;
};

// Ordered metadata table (44 entries, declaration order). Defined in config.cpp.
const std::vector<ParamMeta>& param_metadata_table();

// Ordered field names (declaration order) -- the metadata table's names.
const std::vector<std::string>& param_metadata_order();

// The 21 evolvable encoder params, in order.
const std::vector<std::string>& default_encoder_params();

// Canonical serializers (declaration-order keys; floats as <d raw-bit hex;
// background_color_override -1 renders as JSON null).
std::string config_metadata_json();
std::string config_to_dict(const GraphConfig& cfg);

}  // namespace arc

#endif  // ARC_CONFIG_HPP
