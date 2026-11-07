// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// repr formatting for float / str + the program-hash canonical form.
#include "arc/pyrepr.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "arc/blake2b.hpp"

namespace arc {

// ===========================================================================
// repr(float)
// ===========================================================================
//
// The shortest round-trip DIGITS + decimal exponent come from
// std::to_chars(scientific); the fixed-vs-scientific threshold and the
// exponent padding are applied here. std::to_chars' *own* formatting is NOT
// used (its threshold + exponent style differ).

std::string py_repr_float(double x) {
  // Non-finite: inf / -inf / nan.
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";

  // Shortest scientific round-trip: "[-]d[.ddd]e[+-]dd".
  char buf[64];
  std::to_chars_result res =
      std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
  *res.ptr = '\0';

  const char* p = buf;
  bool neg = false;
  if (*p == '-') {
    neg = true;
    ++p;
  }

  // Collect the significant digits (drop the decimal point).
  std::string digits;
  digits.reserve(20);
  while (*p && *p != 'e' && *p != 'E') {
    if (*p != '.') digits.push_back(*p);
    ++p;
  }
  // Parse the exponent (to_chars always emits 'e', a sign, and >= 2 digits).
  int exp10 = 0;
  if (*p == 'e' || *p == 'E') {
    ++p;
    int esign = 1;
    if (*p == '+') {
      ++p;
    } else if (*p == '-') {
      esign = -1;
      ++p;
    }
    while (*p >= '0' && *p <= '9') {
      exp10 = exp10 * 10 + (*p - '0');
      ++p;
    }
    exp10 *= esign;
  }

  // digits[0] is the 10^exp10 place. decpt = number of digits that belong to
  // the left of the decimal point = exp10 + 1.
  const int ndigits = static_cast<int>(digits.size());
  const int decpt = exp10 + 1;

  std::string out;
  if (neg) out.push_back('-');

  // Scientific iff decpt <= -4 or decpt > 16.
  const bool use_exp = (decpt <= -4) || (decpt > 16);

  if (use_exp) {
    // Mantissa: first digit, then '.' + the rest (if any).
    out.push_back(digits[0]);
    if (ndigits > 1) {
      out.push_back('.');
      out.append(digits, 1, std::string::npos);
    }
    // Exponent: 'e', explicit sign, >= 2 digits. The printed exponent is the
    // power for a d.ddd mantissa == exp10.
    out.push_back('e');
    int e = exp10;
    if (e < 0) {
      out.push_back('-');
      e = -e;
    } else {
      out.push_back('+');
    }
    char ebuf[16];
    int n = 0;
    if (e == 0) {
      ebuf[n++] = '0';
    } else {
      char tmp[16];
      int t = 0;
      while (e > 0) {
        tmp[t++] = static_cast<char>('0' + e % 10);
        e /= 10;
      }
      while (t > 0) ebuf[n++] = tmp[--t];
    }
    // pad to at least 2 exponent digits.
    if (n < 2) {
      out.push_back('0');
    }
    out.append(ebuf, ebuf + n);
  } else if (decpt <= 0) {
    // 0.00ddd : "0." then (-decpt) leading zeros then all digits.
    out.append("0.");
    out.append(static_cast<size_t>(-decpt), '0');
    out.append(digits);
  } else if (decpt >= ndigits) {
    // ddd000.0 : all digits, then (decpt-ndigits) trailing zeros, then ".0".
    out.append(digits);
    out.append(static_cast<size_t>(decpt - ndigits), '0');
    out.append(".0");
  } else {
    // dd.ddd : split the digit string at the decimal point.
    out.append(digits, 0, static_cast<size_t>(decpt));
    out.push_back('.');
    out.append(digits, static_cast<size_t>(decpt), std::string::npos);
  }
  return out;
}

// ===========================================================================
// repr(str)
// ===========================================================================
//
// Choose the quote (single unless the string has a single quote and no double
// quote), then escape the quote char, backslash, \t \n \r, and non-printable
// code points. The UTF-8 input is decoded to code points and non-printables are
// written as \xNN (<0x100), \uNNNN (<0x10000), \UNNNNNNNN otherwise.
// Printable code points >= 0x80 are re-encoded as UTF-8 (printable non-ASCII
// stays literal), but the arg keys/values here are ASCII; the general rule is
// still implemented and, lacking a Unicode category table, all code points
// >= 0x80 are treated as "print literally" (the DSL never carries
// non-printable non-ASCII).

namespace {

// Is this ASCII code point printed without an escape? ASCII control chars and
// DEL are escaped; 0x20..0x7e print literally.
inline bool ascii_printable(uint32_t cp) { return cp >= 0x20 && cp < 0x7f; }

void append_hex_escape(std::string& out, uint32_t cp) {
  static const char* hexd = "0123456789abcdef";
  if (cp < 0x100) {
    out += "\\x";
    out.push_back(hexd[(cp >> 4) & 0xf]);
    out.push_back(hexd[cp & 0xf]);
  } else if (cp < 0x10000) {
    out += "\\u";
    for (int i = 3; i >= 0; --i) out.push_back(hexd[(cp >> (4 * i)) & 0xf]);
  } else {
    out += "\\U";
    for (int i = 7; i >= 0; --i) out.push_back(hexd[(cp >> (4 * i)) & 0xf]);
  }
}

// Decode one UTF-8 code point starting at s[i]; advances i. Malformed bytes are
// passed through as a single byte value (the DSL never produces malformed UTF-8).
uint32_t utf8_next(const std::string& s, size_t& i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) {
    ++i;
    return c;
  }
  int extra;
  uint32_t cp;
  if ((c & 0xe0) == 0xc0) {
    extra = 1;
    cp = c & 0x1f;
  } else if ((c & 0xf0) == 0xe0) {
    extra = 2;
    cp = c & 0x0f;
  } else if ((c & 0xf8) == 0xf0) {
    extra = 3;
    cp = c & 0x07;
  } else {
    ++i;
    return c;  // stray continuation / invalid lead byte
  }
  if (i + static_cast<size_t>(extra) >= s.size() + 1 &&
      i + static_cast<size_t>(extra) >= s.size()) {
    // fall through; bounds re-checked below
  }
  size_t j = i + 1;
  for (int k = 0; k < extra; ++k) {
    if (j >= s.size()) {
      ++i;
      return c;
    }
    unsigned char cc = static_cast<unsigned char>(s[j]);
    if ((cc & 0xc0) != 0x80) {
      ++i;
      return c;
    }
    cp = (cp << 6) | (cc & 0x3f);
    ++j;
  }
  i = j;
  return cp;
}

}  // namespace

std::string py_repr_str(const std::string& s) {
  // Quote choice: single quote by default; if the string contains a single
  // quote but no double quote, use double quotes (so the single quote need not
  // be escaped).
  bool has_single = s.find('\'') != std::string::npos;
  bool has_double = s.find('"') != std::string::npos;
  char quote = '\'';
  if (has_single && !has_double) quote = '"';

  std::string out;
  out.reserve(s.size() + 2);
  out.push_back(quote);

  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = utf8_next(s, i);
    if (cp == static_cast<uint32_t>(quote) || cp == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(cp));
    } else if (cp == '\t') {
      out += "\\t";
    } else if (cp == '\n') {
      out += "\\n";
    } else if (cp == '\r') {
      out += "\\r";
    } else if (cp < 0x80) {
      if (ascii_printable(cp)) {
        out.push_back(static_cast<char>(cp));
      } else {
        append_hex_escape(out, cp);  // ASCII control / DEL -> \xNN
      }
    } else {
      // Non-ASCII: the DSL only carries printable non-ASCII (repr keeps it
      // literal). Re-encode the code point as UTF-8.
      if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
      } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
      } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
      }
    }
  }
  out.push_back(quote);
  return out;
}

// ===========================================================================
// canonical_repr(program)
// ===========================================================================
//
// The canonical form:
//   struct     -> (ClassName, (field0, field1, ...))   [declaration order]
//   arg dict   -> tuple of (key, canon(value)) pairs   [sorted by key]
//   list       -> tuple of canon(x)
//   enum       -> its .value string
//   primitive  -> itself
//
// The field orders:
//   Selector(kind, args)   Guard(kind, args)   Action(kind, args)
//   Rule(selector, action, guards)             Program(rules, global_transform)
//
// The repr string of this structure is emitted directly.

namespace {

// repr of an int: decimal, with '-' for negatives.
std::string repr_int(long long v) { return std::to_string(v); }

// Forward decl: repr of a DslValue (the arg-dict value canonical form).
std::string repr_value(const DslValue& v);

// repr of a canonical arg dict: sorted (by key) tuple of (repr(key), repr(canon
// value)) pairs, rendered as a tuple. Empty dict -> "()".
std::string repr_args(const DslArgs& args) {
  // Sort a copy by key (stable). Keys are unique, so sorting by key alone gives
  // the same order as sorting the (key, value) pairs.
  std::vector<const std::pair<std::string, DslValue>*> ptrs;
  ptrs.reserve(args.size());
  for (const auto& kv : args) ptrs.push_back(&kv);
  std::stable_sort(ptrs.begin(), ptrs.end(),
                   [](const std::pair<std::string, DslValue>* a,
                      const std::pair<std::string, DslValue>* b) {
                     return a->first < b->first;
                   });

  std::string out;
  out.push_back('(');
  for (size_t i = 0; i < ptrs.size(); ++i) {
    if (i) out += ", ";
    // Each element is a 2-tuple (key, canon(value)) -> "('key', <val>)".
    out.push_back('(');
    out += py_repr_str(ptrs[i]->first);
    out += ", ";
    out += repr_value(ptrs[i]->second);
    out.push_back(')');
  }
  // A 1-element tuple gets a trailing comma; an empty one stays "()".
  if (ptrs.size() == 1) out.push_back(',');
  out.push_back(')');
  return out;
}

std::string repr_value(const DslValue& v) {
  switch (v.tag) {
    case DslValue::Tag::Int:
      return repr_int(v.i);
    case DslValue::Tag::Float:
      return py_repr_float(v.f);
    case DslValue::Tag::Str:
      return py_repr_str(v.s);
    case DslValue::Tag::Bool:
      return v.b ? "True" : "False";
    case DslValue::Tag::Dict:
      return v.dict ? repr_args(*v.dict) : "()";
    case DslValue::Tag::Null:
    default:
      return "None";
  }
}

std::string repr_selector(const DslSelector& s) {
  std::string out = "('Selector', (";
  out += py_repr_str(to_string(s.kind));
  out += ", ";
  out += repr_args(s.args);
  out += "))";
  return out;
}

std::string repr_action(const DslAction& a) {
  std::string out = "('Action', (";
  out += py_repr_str(to_string(a.kind));
  out += ", ";
  out += repr_args(a.args);
  out += "))";
  return out;
}

std::string repr_guard(const DslGuard& g) {
  std::string out = "('Guard', (";
  out += py_repr_str(to_string(g.kind));
  out += ", ";
  out += repr_args(g.args);
  out += "))";
  return out;
}

std::string repr_guards(const std::vector<DslGuard>& guards) {
  std::string out = "(";
  for (size_t i = 0; i < guards.size(); ++i) {
    if (i) out += ", ";
    out += repr_guard(guards[i]);
  }
  if (guards.size() == 1) out.push_back(',');
  out.push_back(')');
  return out;
}

std::string repr_rule(const DslRule& r) {
  // Rule field order: (selector, action, guards).
  std::string out = "('Rule', (";
  out += repr_selector(r.selector);
  out += ", ";
  out += repr_action(r.action);
  out += ", ";
  out += repr_guards(r.guards);
  out += "))";
  return out;
}

std::string repr_rules(const std::vector<DslRule>& rules) {
  std::string out = "(";
  for (size_t i = 0; i < rules.size(); ++i) {
    if (i) out += ", ";
    out += repr_rule(rules[i]);
  }
  if (rules.size() == 1) out.push_back(',');
  out.push_back(')');
  return out;
}

}  // namespace

std::string canonical_repr(const DslProgram& program) {
  // Program field order: (rules, global_transform).
  std::string out = "('Program', (";
  out += repr_rules(program.rules);
  out += ", ";
  out += py_repr_str(to_string(program.global_transform));
  out += "))";
  return out;
}

std::string canonical_args_repr(const DslArgs& args) {
  // The same sorted (key) tuple repr_args builds.
  return repr_args(args);
}

std::string hash_program(const DslProgram& program) {
  std::string repr = canonical_repr(program);
  return blake2b16_hex(reinterpret_cast<const uint8_t*>(repr.data()), repr.size());
}

}  // namespace arc
