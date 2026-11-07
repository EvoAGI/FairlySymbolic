// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// JSON parser/writer for ARC task I/O. See include/arc/json.hpp.
//
// Recursive-descent parser. No exceptions (-fno-exceptions): a parse failure
// sets *ok=false and returns Null. Recursion depth is bounded by the input
// nesting; ARC grids nest at most 3 deep (object -> array -> array -> int), so
// no artificial depth cap is needed.

#include "arc/json.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>

#include "arc/pyrepr.hpp"

namespace arc {

namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;
  bool ok = true;
  std::string err;

  explicit Parser(const std::string& text) : s(text) {}

  void fail(const char* msg) {
    if (ok) {
      ok = false;
      err = msg;
    }
  }

  void skip_ws() {
    while (i < s.size()) {
      char c = s[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        ++i;
      else
        break;
    }
  }

  char peek() const { return i < s.size() ? s[i] : '\0'; }

  JsonValue parse_value() {
    skip_ws();
    if (!ok) return JsonValue::make_null();
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return JsonValue::make_string(parse_string());
      case 't':
      case 'f': return parse_bool();
      case 'n': return parse_null();
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail("unexpected character");
        return JsonValue::make_null();
    }
  }

  bool match_literal(const char* lit) {
    size_t n = 0;
    while (lit[n]) ++n;
    if (i + n > s.size()) return false;
    for (size_t k = 0; k < n; ++k)
      if (s[i + k] != lit[k]) return false;
    i += n;
    return true;
  }

  JsonValue parse_null() {
    if (match_literal("null")) return JsonValue::make_null();
    fail("expected null");
    return JsonValue::make_null();
  }

  JsonValue parse_bool() {
    if (match_literal("true")) return JsonValue::make_bool(true);
    if (match_literal("false")) return JsonValue::make_bool(false);
    fail("expected bool");
    return JsonValue::make_null();
  }

  // Parse a JSON string body (leading '"' assumed present). Handles the
  // standard escapes plus \uXXXX (encoded to UTF-8). Returns the decoded text.
  std::string parse_string() {
    std::string out;
    if (peek() != '"') {
      fail("expected string");
      return out;
    }
    ++i;  // opening quote
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return out;
      if (c == '\\') {
        if (i >= s.size()) break;
        char e = s[i++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            unsigned cp = parse_hex4();
            if (!ok) return out;
            // Surrogate pair handling.
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() &&
                s[i] == '\\' && s[i + 1] == 'u') {
              i += 2;
              unsigned lo = parse_hex4();
              if (!ok) return out;
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            append_utf8(out, cp);
            break;
          }
          default:
            fail("bad escape");
            return out;
        }
      } else {
        out.push_back(c);
      }
    }
    fail("unterminated string");
    return out;
  }

  unsigned parse_hex4() {
    if (i + 4 > s.size()) {
      fail("bad \\u escape");
      return 0;
    }
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f')
        v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        v |= static_cast<unsigned>(c - 'A' + 10);
      else {
        fail("bad hex digit");
        return 0;
      }
    }
    return v;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  JsonValue parse_number() {
    size_t start = i;
    bool is_float = false;
    if (peek() == '-') ++i;
    while (i < s.size()) {
      char c = s[i];
      if (c >= '0' && c <= '9') {
        ++i;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        is_float = true;
        ++i;
      } else {
        break;
      }
    }
    std::string tok = s.substr(start, i - start);
    if (tok.empty() || tok == "-") {
      fail("bad number");
      return JsonValue::make_null();
    }
    if (is_float) {
      return JsonValue::make_double(std::strtod(tok.c_str(), nullptr));
    }
    // Integer. strtoll handles the full 64-bit range ARC never needs, but keep
    // it robust.
    return JsonValue::make_int(std::strtoll(tok.c_str(), nullptr, 10));
  }

  JsonValue parse_array() {
    JsonValue arr = JsonValue::make_array();
    ++i;  // '['
    skip_ws();
    if (peek() == ']') {
      ++i;
      return arr;
    }
    while (ok) {
      JsonValue v = parse_value();
      if (!ok) break;
      arr.push_back(std::move(v));
      skip_ws();
      char c = peek();
      if (c == ',') {
        ++i;
        skip_ws();
        continue;
      }
      if (c == ']') {
        ++i;
        break;
      }
      fail("expected ',' or ']'");
      break;
    }
    return arr;
  }

  JsonValue parse_object() {
    JsonValue obj = JsonValue::make_object();
    ++i;  // '{'
    skip_ws();
    if (peek() == '}') {
      ++i;
      return obj;
    }
    while (ok) {
      skip_ws();
      if (peek() != '"') {
        fail("expected object key string");
        break;
      }
      std::string key = parse_string();
      if (!ok) break;
      skip_ws();
      if (peek() != ':') {
        fail("expected ':'");
        break;
      }
      ++i;
      JsonValue v = parse_value();
      if (!ok) break;
      obj.set(std::move(key), std::move(v));
      skip_ws();
      char c = peek();
      if (c == ',') {
        ++i;
        continue;
      }
      if (c == '}') {
        ++i;
        break;
      }
      fail("expected ',' or '}'");
      break;
    }
    return obj;
  }
};

void dump_string(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          static const char* hexd = "0123456789abcdef";
          out += "\\u00";
          out += hexd[(c >> 4) & 0xF];
          out += hexd[c & 0xF];
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void dump_value(std::string& out, const JsonValue& v) {
  switch (v.type()) {
    case JsonValue::Type::Null:
      out += "null";
      break;
    case JsonValue::Type::Bool:
      out += v.as_bool() ? "true" : "false";
      break;
    case JsonValue::Type::Int:
      out += std::to_string(v.as_int());
      break;
    case JsonValue::Type::Double:
      out += py_repr_float(v.as_double());
      break;
    case JsonValue::Type::String:
      dump_string(out, v.as_string());
      break;
    case JsonValue::Type::Array: {
      out.push_back('[');
      const auto& arr = v.items();
      for (size_t k = 0; k < arr.size(); ++k) {
        if (k) out.push_back(',');
        dump_value(out, arr[k]);
      }
      out.push_back(']');
      break;
    }
    case JsonValue::Type::Object: {
      out.push_back('{');
      const auto& obj = v.members();
      for (size_t k = 0; k < obj.size(); ++k) {
        if (k) out.push_back(',');
        dump_string(out, obj[k].first);
        out.push_back(':');
        dump_value(out, obj[k].second);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

JsonValue json_parse(const std::string& text, bool* ok, std::string* err) {
  Parser p(text);
  JsonValue v = p.parse_value();
  if (p.ok) {
    p.skip_ws();
    if (p.i != text.size()) p.fail("trailing content after JSON value");
  }
  if (ok) *ok = p.ok;
  if (err) *err = p.err;
  return p.ok ? v : JsonValue::make_null();
}

std::string json_dump(const JsonValue& v) {
  std::string out;
  dump_value(out, v);
  return out;
}

JsonValue json_from_grid(const std::vector<std::vector<int>>& grid) {
  JsonValue arr = JsonValue::make_array();
  for (const auto& row : grid) {
    JsonValue r = JsonValue::make_array();
    for (int c : row) r.push_back(JsonValue::make_int(c));
    arr.push_back(std::move(r));
  }
  return arr;
}

bool grid_from_json(const JsonValue& v, std::vector<std::vector<int>>* out) {
  if (!v.is_array()) return false;
  out->clear();
  for (const JsonValue& row : v.items()) {
    if (!row.is_array()) return false;
    std::vector<int> r;
    r.reserve(row.size());
    for (const JsonValue& cell : row.items()) {
      if (!cell.is_number()) return false;
      r.push_back(static_cast<int>(cell.as_int()));
    }
    out->push_back(std::move(r));
  }
  return true;
}

}  // namespace arc
