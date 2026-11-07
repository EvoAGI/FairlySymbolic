// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL argument coercion and small string helpers (as_int / as_bool).
// `inline` so every TU inlines them.
// Ordering rules: every sort is a std::stable_sort over an explicit key tuple;
// node-id comparisons are byte-wise lexicographic; no exceptions.

#ifndef ARC_DSL_ARG_COERCE_HPP
#define ARC_DSL_ARG_COERCE_HPP

#include "arc/dsl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

namespace arc {

inline int as_int(const DslValue* v, int dflt) {
  if (v == nullptr) return dflt;
  switch (v->tag) {
    case DslValue::Tag::Int: return static_cast<int>(v->i);
    case DslValue::Tag::Bool: return v->b ? 1 : 0;
    case DslValue::Tag::Float: return static_cast<int>(v->f);
    case DslValue::Tag::Str: {
      // A string must be a whole decimal integer ("3"); anything else ("abc") gives the default.
      const std::string& s = v->s;
      if (s.empty()) return dflt;
      char* endp = nullptr;
      long r = std::strtol(s.c_str(), &endp, 10);
      if (endp == s.c_str() || *endp != '\0') return dflt;
      return static_cast<int>(r);
    }
    default: return dflt;
  }
}

inline int arg_int(const DslArgs& args, const std::string& key, int dflt) {
  const DslValue* v = args_get(args, key);
  if (v == nullptr) return dflt;
  switch (v->tag) {
    case DslValue::Tag::Int: return static_cast<int>(v->i);
    case DslValue::Tag::Bool: return v->b ? 1 : 0;
    case DslValue::Tag::Float: return static_cast<int>(v->f);
    case DslValue::Tag::Str: return std::atoi(v->s.c_str());
    default: return dflt;
  }
}

inline double arg_float(const DslArgs& args, const std::string& key, double dflt) {
  const DslValue* v = args_get(args, key);
  if (v == nullptr) return dflt;
  switch (v->tag) {
    case DslValue::Tag::Float: return v->f;
    case DslValue::Tag::Int: return static_cast<double>(v->i);
    case DslValue::Tag::Bool: return v->b ? 1.0 : 0.0;
    case DslValue::Tag::Str: return std::strtod(v->s.c_str(), nullptr);
    default: return dflt;
  }
}

inline std::string arg_str(const DslArgs& args, const std::string& key, const std::string& dflt) {
  const DslValue* v = args_get(args, key);
  if (v == nullptr) return dflt;
  if (v->tag == DslValue::Tag::Str) return v->s;
  if (v->tag == DslValue::Tag::Bool) return v->b ? "True" : "False";
  if (v->tag == DslValue::Tag::Int) return std::to_string(v->i);
  return dflt;
}

inline bool as_bool(const DslValue* v, bool dflt) {
  if (v == nullptr) return dflt;
  if (v->tag == DslValue::Tag::Bool) return v->b;
  if (v->tag == DslValue::Tag::Int) return v->i != 0;
  if (v->tag == DslValue::Tag::Float) return v->f != 0.0;
  if (v->tag == DslValue::Tag::Str) {
    std::string t = v->s;
    for (auto& ch : t) ch = static_cast<char>(std::tolower(ch));
    // strip surrounding whitespace
    size_t b = t.find_first_not_of(" \t\n\r");
    size_t e = t.find_last_not_of(" \t\n\r");
    if (b == std::string::npos) return dflt;
    t = t.substr(b, e - b + 1);
    if (t == "1" || t == "true" || t == "yes" || t == "y" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no" || t == "n" || t == "off") return false;
    return dflt;
  }
  return dflt;
}

inline std::string to_lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(c));
  return s;
}

inline std::string to_upper(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(c));
  return s;
}

inline std::string strip(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\n\r");
  size_t e = s.find_last_not_of(" \t\n\r");
  if (b == std::string::npos) return "";
  return s.substr(b, e - b + 1);
}

}  // namespace arc

#endif  // ARC_DSL_ARG_COERCE_HPP
