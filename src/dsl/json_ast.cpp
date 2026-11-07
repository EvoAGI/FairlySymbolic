// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL: minimal JSON reader, arg coercion, AST parsing. Shared helpers live in
// arg_coerce.hpp / scene_ops.hpp, which also state the ordering rules.

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

#include "arc/concepts.hpp"
#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

#include "arg_coerce.hpp"

namespace arc {

// ===========================================================================
// Minimal JSON reader. Parses the compact selector/program JSON the ABI
// receives into a small generic JSON tree (objects / arrays / strings /
// numbers / bool / null). No exceptions: on a structural error `ok` goes false
// and callers bail. The grammar is standard JSON restricted to what we emit.
// ===========================================================================

namespace {

struct JsonNode;
using JsonPtr = std::shared_ptr<JsonNode>;

struct JsonNode {
  enum class Tag { Null, Int, Float, Str, Bool, Array, Object };
  Tag tag = Tag::Null;
  long long i = 0;
  double f = 0.0;
  std::string s;
  bool b = false;
  std::vector<JsonPtr> arr;
  std::vector<std::pair<std::string, JsonPtr>> obj;  // insertion-ordered

  const JsonNode* get(const std::string& key) const {
    for (const auto& kv : obj)
      if (kv.first == key) return kv.second.get();
    return nullptr;
  }
};

struct JsonParser {
  const char* p;
  const char* end;
  bool ok = true;

  explicit JsonParser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }
  char peek() {
    skip_ws();
    return p < end ? *p : '\0';
  }
  bool expect(char c) {
    skip_ws();
    if (p < end && *p == c) { ++p; return true; }
    ok = false;
    return false;
  }

  bool parse_string(std::string* out) {
    skip_ws();
    if (p >= end || *p != '"') { ok = false; return false; }
    ++p;
    out->clear();
    while (p < end && *p != '"') {
      if (*p == '\\' && p + 1 < end) {
        ++p;
        char c = *p;
        if (c == 'n') out->push_back('\n');
        else if (c == 't') out->push_back('\t');
        else if (c == 'r') out->push_back('\r');
        else if (c == 'b') out->push_back('\b');
        else if (c == 'f') out->push_back('\f');
        else out->push_back(c);  // \" \\ / and any other
        ++p;
      } else {
        out->push_back(*p);
        ++p;
      }
    }
    if (p >= end) { ok = false; return false; }
    ++p;
    return true;
  }

  JsonPtr parse_value() {
    auto node = std::make_shared<JsonNode>();
    char c = peek();
    if (c == '"') {
      node->tag = JsonNode::Tag::Str;
      parse_string(&node->s);
    } else if (c == '{') {
      node->tag = JsonNode::Tag::Object;
      parse_object(node.get());
    } else if (c == '[') {
      node->tag = JsonNode::Tag::Array;
      parse_array(node.get());
    } else if (c == 't') {
      node->tag = JsonNode::Tag::Bool; node->b = true; p += 4;
    } else if (c == 'f') {
      node->tag = JsonNode::Tag::Bool; node->b = false; p += 5;
    } else if (c == 'n') {
      node->tag = JsonNode::Tag::Null; p += 4;
    } else {
      skip_ws();
      const char* start = p;
      bool is_float = false;
      if (p < end && (*p == '-' || *p == '+')) ++p;
      while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' ||
                         *p == 'E' || *p == '-' || *p == '+')) {
        if (*p == '.' || *p == 'e' || *p == 'E') is_float = true;
        ++p;
      }
      std::string num(start, p);
      if (num.empty()) { ok = false; return node; }
      if (is_float) { node->tag = JsonNode::Tag::Float; node->f = std::strtod(num.c_str(), nullptr); }
      else { node->tag = JsonNode::Tag::Int; node->i = std::strtoll(num.c_str(), nullptr, 10); }
    }
    return node;
  }

  bool parse_array(JsonNode* out) {
    if (!expect('[')) return false;
    if (peek() == ']') { ++p; return true; }
    while (ok) {
      out->arr.push_back(parse_value());
      char c = peek();
      if (c == ',') { ++p; continue; }
      if (c == ']') { ++p; return true; }
      ok = false;
      return false;
    }
    return ok;
  }

  bool parse_object(JsonNode* out) {
    if (!expect('{')) return false;
    if (peek() == '}') { ++p; return true; }
    while (ok) {
      std::string key;
      if (!parse_string(&key)) return false;
      if (!expect(':')) return false;
      JsonPtr val = parse_value();
      out->obj.emplace_back(std::move(key), std::move(val));
      char c = peek();
      if (c == ',') { ++p; continue; }
      if (c == '}') { ++p; return true; }
      ok = false;
      return false;
    }
    return ok;
  }
};

// Convert a parsed JSON node into a DslValue (int/float/str/bool/dict). Arrays
// and null map to Null (selector args never carry arrays).
DslValue to_dsl_value(const JsonNode* n) {
  DslValue v;
  if (n == nullptr) { v.tag = DslValue::Tag::Null; return v; }
  switch (n->tag) {
    case JsonNode::Tag::Int: v.tag = DslValue::Tag::Int; v.i = n->i; break;
    case JsonNode::Tag::Float: v.tag = DslValue::Tag::Float; v.f = n->f; break;
    case JsonNode::Tag::Str: v.tag = DslValue::Tag::Str; v.s = n->s; break;
    case JsonNode::Tag::Bool: v.tag = DslValue::Tag::Bool; v.b = n->b; break;
    case JsonNode::Tag::Object: {
      v.tag = DslValue::Tag::Dict;
      v.dict = std::make_shared<DslArgs>();
      for (const auto& kv : n->obj)
        v.dict->emplace_back(kv.first, to_dsl_value(kv.second.get()));
      break;
    }
    default: v.tag = DslValue::Tag::Null; break;
  }
  return v;
}

DslArgs to_dsl_args(const JsonNode* n) {
  DslArgs out;
  if (n != nullptr && n->tag == JsonNode::Tag::Object)
    for (const auto& kv : n->obj)
      out.emplace_back(kv.first, to_dsl_value(kv.second.get()));
  return out;
}

}  // namespace

const DslValue* args_get(const DslArgs& args, const std::string& key) {
  for (const auto& kv : args)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Arg coercion helpers (as_int / as_bool live in arg_coerce.hpp).
// ---------------------------------------------------------------------------

namespace {



}  // namespace

// ---------------------------------------------------------------------------
// AST parsing.
// ---------------------------------------------------------------------------

namespace {

bool selector_from_node(const JsonNode* n, DslSelector* out) {
  if (n == nullptr || n->tag != JsonNode::Tag::Object) return false;
  const JsonNode* kind = n->get("kind");
  if (kind == nullptr || kind->tag != JsonNode::Tag::Str) return false;
  DslArgs args = to_dsl_args(n->get("args"));
  SelectorKind k;
  if (!selector_from_name(kind->s, &k, &args)) return false;
  out->kind = k;
  out->args = std::move(args);
  return true;
}

bool guard_from_node(const JsonNode* n, DslGuard* out) {
  if (n == nullptr || n->tag != JsonNode::Tag::Object) return false;
  const JsonNode* kind = n->get("kind");
  if (kind == nullptr || kind->tag != JsonNode::Tag::Str) return false;
  GuardKind k;
  if (!guard_kind_from_string(kind->s, &k)) return false;
  out->kind = k;
  out->args = to_dsl_args(n->get("args"));
  return true;
}

bool action_from_node(const JsonNode* n, DslAction* out) {
  if (n == nullptr || n->tag != JsonNode::Tag::Object) return false;
  const JsonNode* kind = n->get("kind");
  if (kind == nullptr || kind->tag != JsonNode::Tag::Str) return false;
  out->args = to_dsl_args(n->get("args"));
  if (action_from_name(kind->s, &out->kind, &out->args)) return true;
  return concept_alias(kind->s, &out->kind, &out->args);
}

}  // namespace

bool selector_from_name(const std::string& name, SelectorKind* kind, DslArgs* args) {
  if (selector_kind_from_string(name, kind)) return true;
  for (const SelectorAlias& a : selector_aliases()) {
    if (name != a.name) continue;
    *kind = a.kind;
    if (*a.variant) {
      const std::string& key = selector_variant_key(a.kind);
      DslArgs out;
      DslValue v;
      v.tag = DslValue::Tag::Str;
      v.s = a.variant;
      out.emplace_back(key, v);
      for (auto& kv : *args)
        if (kv.first != key) out.push_back(std::move(kv));
      *args = std::move(out);
    }
    return true;
  }
  return false;
}

bool action_from_name(const std::string& name, ActionKind* kind, DslArgs* args) {
  if (action_kind_from_string(name, kind)) return true;
  for (const ActionAlias& a : action_aliases()) {
    if (name != a.name) continue;
    *kind = a.kind;
    const std::string& key = action_variant_key(a.kind);
    DslArgs out;
    DslValue v;
    v.tag = DslValue::Tag::Str;
    v.s = a.variant;
    out.emplace_back(key, v);
    for (auto& kv : *args)
      if (kv.first != key) out.push_back(std::move(kv));
    *args = std::move(out);
    return true;
  }
  return false;
}

std::string action_variant(const DslAction& a) {
  const ActionVariants* v = action_variants(a.kind);
  if (!v) return "";
  for (const auto& kv : a.args)
    if (kv.first == v->key && kv.second.tag == DslValue::Tag::Str) return kv.second.s;
  return v->values[0].first;
}

bool parse_selector_json(const std::string& json, DslSelector* out) {
  JsonParser jp(json);
  JsonNode root;
  root.tag = JsonNode::Tag::Object;
  if (!jp.parse_object(&root) || !jp.ok) return false;
  return selector_from_node(&root, out);
}

bool parse_program_json(const std::string& json, DslProgram* out) {
  JsonParser jp(json);
  JsonNode root;
  root.tag = JsonNode::Tag::Object;
  if (!jp.parse_object(&root) || !jp.ok) return false;

  const JsonNode* gt = root.get("global_transform");
  if (gt != nullptr && gt->tag == JsonNode::Tag::Str) {
    GlobalTransform g;
    if (global_transform_from_string(gt->s, &g)) out->global_transform = g;
  }

  const JsonNode* rules = root.get("rules");
  if (rules != nullptr && rules->tag == JsonNode::Tag::Array) {
    for (const auto& rp : rules->arr) {
      const JsonNode* rn = rp.get();
      if (rn == nullptr || rn->tag != JsonNode::Tag::Object) return false;
      DslRule rule;
      if (!selector_from_node(rn->get("selector"), &rule.selector)) return false;
      if (!action_from_node(rn->get("action"), &rule.action)) return false;
      const JsonNode* guards = rn->get("guards");
      if (guards != nullptr && guards->tag == JsonNode::Tag::Array) {
        for (const auto& gp : guards->arr) {
          DslGuard guard;
          if (!guard_from_node(gp.get(), &guard)) return false;
          rule.guards.push_back(std::move(guard));
        }
      }
      out->rules.push_back(std::move(rule));
    }
  }
  return true;
}


}  // namespace arc
