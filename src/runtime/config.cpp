// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#include "arc/config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace arc {

namespace {

// A double as the lowercase hex of its 8 little-endian IEEE-754 bytes, with no
// separators (same as fbits in encode.cpp).
std::string fbits(double x) {
  uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  static const char* hexd = "0123456789abcdef";
  std::string out;
  out.reserve(16);
  for (int i = 0; i < 8; ++i) {
    uint8_t b = static_cast<uint8_t>((bits >> (8 * i)) & 0xff);
    out.push_back(hexd[b >> 4]);
    out.push_back(hexd[b & 0xf]);
  }
  return out;
}

// Minimal JSON string escape for the ASCII identifiers/choices we emit.
void json_str(std::ostringstream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    if (c == '"' || c == '\\') os << '\\' << c;
    else os << c;
  }
  os << '"';
}

// --- Categorical choice sets ---------------------------------------------
ParamMeta cat_int(const char* name, std::vector<std::string> choices) {
  return ParamMeta{name, ParamKind::Categorical, std::move(choices), true, 0, 0};
}
ParamMeta cat_str(const char* name, std::vector<std::string> choices) {
  return ParamMeta{name, ParamKind::Categorical, std::move(choices), false, 0, 0};
}
ParamMeta pint(const char* name, double lo, double hi) {
  return ParamMeta{name, ParamKind::Int, {}, false, lo, hi};
}
ParamMeta pflt(const char* name, double lo, double hi) {
  return ParamMeta{name, ParamKind::Float, {}, false, lo, hi};
}
ParamMeta pbool(const char* name) {
  return ParamMeta{name, ParamKind::Bool, {}, false, 0, 0};
}

// Built once, in a FIXED field order. This is the load-bearing ordering the GA
// relies on at crossover; do not reorder.
const std::vector<ParamMeta> kMeta = {
    cat_int("connectivity", {"4", "8"}),
    cat_str("color_mode", {"mono", "multi"}),
    cat_str("background_mode", {"auto", "none", "bg_as_object"}),
    pint("background_color_override", -1, 9),
    pint("min_object_size", 0, 20),
    cat_str("object_mode", {"connected_component", "pixel", "color_global"}),
    pint("merge_distance", 0, 5),
    pflt("noise_threshold", 0.0, 1.0),
    pbool("multi_color_as_cluster"),
    pbool("cluster_decompose_two_color"),
    cat_str("noise_filter_order", {"size_first", "noise_first"}),
    pbool("decompose_objects"),
    cat_str("decomposition_mode", {"lines", "rectangles", "both"}),
    pbool("split_by_containment"),
    pbool("decompose_grid_lines"),
    pbool("merge_by_containment"),
    pflt("grid_line_threshold", 0.5, 1.0),
    pbool("detect_zones"),
    pbool("detect_cavities"),
    pbool("detect_clusters"),
    pbool("detect_patterns"),
    pbool("detect_symmetries"),
    pbool("compute_spatial_edges"),
    cat_int("spatial_touch_connectivity", {"4", "8"}),
    pbool("compute_containment_edges"),
    pbool("compute_alignment_edges"),
    pbool("compute_ray_edges"),
    pbool("compute_gravity_edges"),
    pbool("compute_bridge_edges"),
    pbool("compute_logic_edges"),
    pbool("compute_shape_classification"),
    pbool("compute_shape_hash"),
    pbool("compute_local_coordinates"),
    pbool("compute_z_index"),
    pbool("compute_noise_scoring"),
    pbool("compute_zone_features"),
    pflt("noise_weight_isolation", 0.0, 1.0),
    pflt("noise_weight_singleton_color", 0.0, 1.0),
    pflt("noise_weight_size_outlier", 0.0, 1.0),
    pflt("noise_weight_shape_outlier", 0.0, 1.0),
    pflt("noise_weight_position_outlier", 0.0, 1.0),
    pflt("alignment_tolerance", 0.1, 2.0),
    pflt("symmetry_tolerance", 0.1, 2.0),
    pint("pattern_min_size", 2, 10),
};

// The 21 encoder params.
const std::vector<std::string> kEncoderParams = {
    "connectivity",
    "color_mode",
    "background_mode",
    "background_color_override",
    "min_object_size",
    "object_mode",
    "merge_distance",
    "noise_threshold",
    "multi_color_as_cluster",
    "noise_filter_order",
    "decompose_objects",
    "decomposition_mode",
    "split_by_containment",
    "decompose_grid_lines",
    "merge_by_containment",
    "grid_line_threshold",
    "detect_zones",
    "detect_cavities",
    "detect_clusters",
    "detect_patterns",
    "detect_symmetries",
};

const char* kind_name(ParamKind k) {
  switch (k) {
    case ParamKind::Categorical: return "categorical";
    case ParamKind::Int: return "int";
    case ParamKind::Float: return "float";
    case ParamKind::Bool: return "bool";
  }
  return "unknown";
}

// Render an integer-valued bound as a plain int (int-kind bounds are whole).
void emit_int_bound(std::ostringstream& os, double v) {
  os << static_cast<long long>(v);
}

// Render a float-kind bound as the shortest decimal that round-trips to the
// same IEEE-754 double: try increasing precision until the printed string
// parses back to the identical double.
void emit_float_bound(std::ostringstream& os, double v) {
  char buf[64];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    double rt = std::strtod(buf, nullptr);
    if (rt == v) break;
  }
  // Ensure a decimal point so it reads as a float (e.g. "1" -> "1.0").
  std::string s(buf);
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find('n') == std::string::npos /* nan/inf */) {
    s += ".0";
  }
  os << s;
}

}  // namespace

const std::vector<ParamMeta>& param_metadata_table() { return kMeta; }

const std::vector<std::string>& param_metadata_order() {
  static std::vector<std::string> names = [] {
    std::vector<std::string> out;
    out.reserve(kMeta.size());
    for (const auto& m : kMeta) out.emplace_back(m.name);
    return out;
  }();
  return names;
}

const std::vector<std::string>& default_encoder_params() { return kEncoderParams; }

// Canonical metadata JSON: an ordered array, one object per field. Each object
// has keys in fixed order: name, kind, then choices (categorical) OR low/high
// (int/float); bool has neither. int-choice params emit numeric choices.
std::string config_metadata_json() {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < kMeta.size(); ++i) {
    const ParamMeta& m = kMeta[i];
    if (i) os << ',';
    os << '{';
    os << "\"name\":"; json_str(os, m.name);
    os << ",\"kind\":"; json_str(os, kind_name(m.kind));
    if (m.kind == ParamKind::Categorical) {
      os << ",\"choices\":[";
      for (size_t j = 0; j < m.choices.size(); ++j) {
        if (j) os << ',';
        if (m.choices_are_int) os << m.choices[j];
        else json_str(os, m.choices[j]);
      }
      os << ']';
    } else if (m.kind == ParamKind::Int || m.kind == ParamKind::Float) {
      os << ",\"low\":";
      if (m.kind == ParamKind::Int) emit_int_bound(os, m.low);
      else emit_float_bound(os, m.low);
      os << ",\"high\":";
      if (m.kind == ParamKind::Int) emit_int_bound(os, m.high);
      else emit_float_bound(os, m.high);
    }
    os << '}';
  }
  os << ']';
  return os.str();
}

// Canonical to_dict: keys in declaration order. Categorical/string fields emit
// JSON strings; ints emit numbers; floats emit raw-bit hex strings (fbits);
// bools emit true/false; background_color_override == -1 emits null.
std::string config_to_dict(const GraphConfig& cfg) {
  std::ostringstream os;
  os << '{';
  bool first = true;
  auto comma = [&]() { if (!first) os << ','; first = false; };

  comma(); json_str(os, "connectivity"); os << ':' << cfg.connectivity;
  comma(); json_str(os, "color_mode"); os << ':'; json_str(os, cfg.color_mode);
  comma(); json_str(os, "background_mode"); os << ':'; json_str(os, cfg.background_mode);
  comma(); json_str(os, "background_color_override");
  os << ':';
  if (cfg.background_color_override < 0) os << "null";
  else os << cfg.background_color_override;
  comma(); json_str(os, "min_object_size"); os << ':' << cfg.min_object_size;
  comma(); json_str(os, "object_mode"); os << ':'; json_str(os, cfg.object_mode);
  comma(); json_str(os, "merge_distance"); os << ':' << cfg.merge_distance;
  comma(); json_str(os, "noise_threshold"); os << ':'; json_str(os, fbits(cfg.noise_threshold));
  comma(); json_str(os, "multi_color_as_cluster"); os << ':' << (cfg.multi_color_as_cluster ? "true" : "false");
  comma(); json_str(os, "cluster_decompose_two_color"); os << ':' << (cfg.cluster_decompose_two_color ? "true" : "false");
  comma(); json_str(os, "noise_filter_order"); os << ':'; json_str(os, cfg.noise_filter_order);
  comma(); json_str(os, "decompose_objects"); os << ':' << (cfg.decompose_objects ? "true" : "false");
  comma(); json_str(os, "decomposition_mode"); os << ':'; json_str(os, cfg.decomposition_mode);
  comma(); json_str(os, "split_by_containment"); os << ':' << (cfg.split_by_containment ? "true" : "false");
  comma(); json_str(os, "decompose_grid_lines"); os << ':' << (cfg.decompose_grid_lines ? "true" : "false");
  comma(); json_str(os, "merge_by_containment"); os << ':' << (cfg.merge_by_containment ? "true" : "false");
  comma(); json_str(os, "grid_line_threshold"); os << ':'; json_str(os, fbits(cfg.grid_line_threshold));
  comma(); json_str(os, "detect_zones"); os << ':' << (cfg.detect_zones ? "true" : "false");
  comma(); json_str(os, "detect_cavities"); os << ':' << (cfg.detect_cavities ? "true" : "false");
  comma(); json_str(os, "detect_clusters"); os << ':' << (cfg.detect_clusters ? "true" : "false");
  comma(); json_str(os, "detect_patterns"); os << ':' << (cfg.detect_patterns ? "true" : "false");
  comma(); json_str(os, "detect_symmetries"); os << ':' << (cfg.detect_symmetries ? "true" : "false");
  comma(); json_str(os, "compute_spatial_edges"); os << ':' << (cfg.compute_spatial_edges ? "true" : "false");
  comma(); json_str(os, "spatial_touch_connectivity"); os << ':' << cfg.spatial_touch_connectivity;
  comma(); json_str(os, "compute_containment_edges"); os << ':' << (cfg.compute_containment_edges ? "true" : "false");
  comma(); json_str(os, "compute_alignment_edges"); os << ':' << (cfg.compute_alignment_edges ? "true" : "false");
  comma(); json_str(os, "compute_ray_edges"); os << ':' << (cfg.compute_ray_edges ? "true" : "false");
  comma(); json_str(os, "compute_gravity_edges"); os << ':' << (cfg.compute_gravity_edges ? "true" : "false");
  comma(); json_str(os, "compute_bridge_edges"); os << ':' << (cfg.compute_bridge_edges ? "true" : "false");
  comma(); json_str(os, "compute_logic_edges"); os << ':' << (cfg.compute_logic_edges ? "true" : "false");
  comma(); json_str(os, "compute_shape_classification"); os << ':' << (cfg.compute_shape_classification ? "true" : "false");
  comma(); json_str(os, "compute_shape_hash"); os << ':' << (cfg.compute_shape_hash ? "true" : "false");
  comma(); json_str(os, "compute_local_coordinates"); os << ':' << (cfg.compute_local_coordinates ? "true" : "false");
  comma(); json_str(os, "compute_z_index"); os << ':' << (cfg.compute_z_index ? "true" : "false");
  comma(); json_str(os, "compute_noise_scoring"); os << ':' << (cfg.compute_noise_scoring ? "true" : "false");
  comma(); json_str(os, "compute_zone_features"); os << ':' << (cfg.compute_zone_features ? "true" : "false");
  comma(); json_str(os, "noise_weight_isolation"); os << ':'; json_str(os, fbits(cfg.noise_weight_isolation));
  comma(); json_str(os, "noise_weight_singleton_color"); os << ':'; json_str(os, fbits(cfg.noise_weight_singleton_color));
  comma(); json_str(os, "noise_weight_size_outlier"); os << ':'; json_str(os, fbits(cfg.noise_weight_size_outlier));
  comma(); json_str(os, "noise_weight_shape_outlier"); os << ':'; json_str(os, fbits(cfg.noise_weight_shape_outlier));
  comma(); json_str(os, "noise_weight_position_outlier"); os << ':'; json_str(os, fbits(cfg.noise_weight_position_outlier));
  comma(); json_str(os, "alignment_tolerance"); os << ':'; json_str(os, fbits(cfg.alignment_tolerance));
  comma(); json_str(os, "symmetry_tolerance"); os << ':'; json_str(os, fbits(cfg.symmetry_tolerance));
  comma(); json_str(os, "pattern_min_size"); os << ':' << cfg.pattern_min_size;
  os << '}';
  return os.str();
}

}  // namespace arc
