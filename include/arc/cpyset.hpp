// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_CPYSET_HPP
#define ARC_CPYSET_HPP

// Hash-set iteration order for sets of small non-negative ints (colors 0..9).
// The encoder iterates some color sets in this order, and a later tie-break
// (e.g. the first color with the highest score) depends on it.
//
// The layout is the one of CPython's open-addressing set (setobject.c): for
// small non-negative ints hash(v) == v, so the probe sequence alone fixes each
// key's slot. This helper reproduces that layout and returns the keys in slot
// (iteration) order. It matches CPython 3.9 for every permutation of {0..9} up
// to size 6.
// NOTE: only the insertion path is emulated (no discard/dummy handling), which
// is all the encoder's int-set uses require.

#include <vector>

namespace arc {

// Return `keys` in the iteration order of the set built by inserting them.
std::vector<int> cpython_set_order_int(const std::vector<int>& keys);

// Return the iteration order of the set intersection `set(a) & set(b)`
// for small non-negative int keys. `a`/`b` are given in INSERTION order (the
// order their keys were first added). The intersection iterates the SMALLER
// operand (on a size tie it iterates the RIGHT operand `b`) and inserts the
// shared keys into a fresh set in that iteration order; the returned vector is
// that fresh set's iteration order.
// Chain left-associatively for `a & b & c & ...`.
std::vector<int> cpython_set_intersection_int(const std::vector<int>& a,
                                              const std::vector<int>& b);

}  // namespace arc

#endif  // ARC_CPYSET_HPP
