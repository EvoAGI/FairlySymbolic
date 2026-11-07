// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_PYREPR_HPP
#define ARC_PYREPR_HPP

// repr strings for the value types the program-hash canonical form contains
// (float / str), plus a canonical-repr builder for a DslProgram. The repr
// format is Python literal syntax: tuples in parentheses (a 1-element tuple has
// a trailing comma), quoted strings, True / False / None.
//
// The program hash is the lowercase hex of the 16-byte BLAKE2b digest of the
// UTF-8 canonical repr string, so any change to these formatters changes every
// program hash.

#include <string>

#include "arc/dsl.hpp"

namespace arc {

// repr(float): shortest round-trip decimal (David Gay 'r' mode); scientific
// iff the decimal exponent is <= -4 or > 16, with lowercase 'e', explicit
// exponent sign and >= 2 exponent digits; a trailing ".0" for integer-valued
// finite floats. Non-finite values print as inf / -inf / nan.
std::string py_repr_float(double x);

// repr(str): single quotes unless the string contains a
// single quote and no double quote (then double quotes); backslash-escape the
// quote char, backslash, \n \r \t, and non-printable / non-ASCII code points as
// \xNN / \uNNNN / \UNNNNNNNN (surrogate-free UTF-8 in, code points out).
std::string py_repr_str(const std::string& s);

// canonical_repr(program): the repr string of the program's canonical tuple
// structure, built directly (no intermediate object tree).
std::string canonical_repr(const DslProgram& program);

// canonical_args_repr(args): the canonical repr of a single (possibly nested)
// arg dict -- sorted (key) tuple of (repr(key), repr(canon(value))) pairs. This
// is the exact fragment canonical_repr writes for an arg dict. Empty dict ->
// "()".
std::string canonical_args_repr(const DslArgs& args);

// hash_program(program): blake2b16_hex(canonical_repr(program)).
std::string hash_program(const DslProgram& program);

}  // namespace arc

#endif  // ARC_PYREPR_HPP
