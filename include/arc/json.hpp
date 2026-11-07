// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_JSON_HPP
#define ARC_JSON_HPP

// A small but REAL JSON parser/writer for ARC task I/O.
//
// ARC task JSON is simple: nested
// int arrays (grids), objects with string keys, arrays, strings, ints, bools,
// null. This parser covers exactly that surface plus doubles (for numeric
// config values, though config parsing does not go through here).
//
// Design notes for determinism:
//   * Object member ORDER is preserved as a std::vector of (key, JsonValue)
//     pairs (insertion order), so a round-trip re-serialises members in the
//     order they were parsed. This keeps writing deterministic and lossless.
//   * The writer emits compact JSON (no spaces). Matching any other writer's
//     formatting is NOT a goal: this code is both the reader and the writer,
//     so the only requirements are round-trip stability and well-formedness.
//
// No exceptions (the build is -fno-exceptions): parse failures set an ok flag
// and return a Null value. The ARC pipeline treats a parse failure as a
// hard-stop the caller reports.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace arc {

class JsonValue {
 public:
  enum class Type { Null, Bool, Int, Double, String, Array, Object };

  JsonValue() : type_(Type::Null) {}

  static JsonValue make_null() { return JsonValue(); }
  static JsonValue make_bool(bool b) {
    JsonValue v;
    v.type_ = Type::Bool;
    v.bool_ = b;
    return v;
  }
  static JsonValue make_int(long long n) {
    JsonValue v;
    v.type_ = Type::Int;
    v.int_ = n;
    return v;
  }
  static JsonValue make_double(double d) {
    JsonValue v;
    v.type_ = Type::Double;
    v.double_ = d;
    return v;
  }
  static JsonValue make_string(std::string s) {
    JsonValue v;
    v.type_ = Type::String;
    v.str_ = std::move(s);
    return v;
  }
  static JsonValue make_array() {
    JsonValue v;
    v.type_ = Type::Array;
    return v;
  }
  static JsonValue make_object() {
    JsonValue v;
    v.type_ = Type::Object;
    return v;
  }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_int() const { return type_ == Type::Int; }
  bool is_double() const { return type_ == Type::Double; }
  bool is_number() const { return type_ == Type::Int || type_ == Type::Double; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const { return bool_; }
  long long as_int() const {
    return type_ == Type::Double ? static_cast<long long>(double_) : int_;
  }
  double as_double() const {
    return type_ == Type::Int ? static_cast<double>(int_) : double_;
  }
  const std::string& as_string() const { return str_; }

  // Array access.
  const std::vector<JsonValue>& items() const { return arr_; }
  std::vector<JsonValue>& items() { return arr_; }
  size_t size() const { return arr_.size(); }
  const JsonValue& operator[](size_t i) const { return arr_[i]; }
  void push_back(JsonValue v) { arr_.push_back(std::move(v)); }

  // Object access (insertion-ordered members).
  const std::vector<std::pair<std::string, JsonValue>>& members() const {
    return obj_;
  }
  std::vector<std::pair<std::string, JsonValue>>& members() { return obj_; }
  // Returns nullptr if the key is absent (first match wins).
  const JsonValue* find(const std::string& key) const {
    for (const auto& kv : obj_)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool has(const std::string& key) const { return find(key) != nullptr; }
  void set(std::string key, JsonValue v) {
    obj_.emplace_back(std::move(key), std::move(v));
  }

 private:
  Type type_;
  bool bool_ = false;
  long long int_ = 0;
  double double_ = 0.0;
  std::string str_;
  std::vector<JsonValue> arr_;
  std::vector<std::pair<std::string, JsonValue>> obj_;
};

// Parse a JSON document. On success *ok is set true and the parsed value is
// returned. On failure *ok is set false (and, when err != nullptr, a short
// message is written) and a Null value is returned. Trailing whitespace after
// the top-level value is tolerated; trailing non-whitespace is an error.
JsonValue json_parse(const std::string& text, bool* ok, std::string* err = nullptr);

// Serialise a JsonValue to compact JSON (no spaces). Object members are emitted
// in stored (insertion) order. Round-trips losslessly for the ARC surface.
std::string json_dump(const JsonValue& v);

// Convenience: build a JsonValue array-of-arrays-of-int from a flat grid, and
// extract a grid (vector<vector<int>>) from such a JsonValue. grid_from_json
// returns false if the shape is not a rectangular int matrix.
JsonValue json_from_grid(const std::vector<std::vector<int>>& grid);
bool grid_from_json(const JsonValue& v, std::vector<std::vector<int>>* out);

}  // namespace arc

#endif  // ARC_JSON_HPP
