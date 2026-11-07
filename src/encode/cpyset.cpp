// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Iteration order of a small open-addressing int hash set. See cpyset.hpp.
#include "arc/cpyset.hpp"

#include <cstdint>

namespace arc {

namespace {

constexpr int LINEAR_PROBES = 9;
constexpr int PERTURB_SHIFT = 5;

struct IntSet {
  int mask = 7;                                     // minimum table size (8) - 1
  std::vector<int> table = std::vector<int>(8, -1);  // -1 == empty slot

  void insert_clean(int k) {
    uint64_t h = static_cast<uint64_t>(k);
    uint64_t perturb = h;
    int i = static_cast<int>(h & static_cast<uint64_t>(mask));
    while (true) {
      if (table[i] < 0) {
        table[i] = k;
        return;
      }
      if (i + LINEAR_PROBES <= mask) {
        for (int j = 1; j <= LINEAR_PROBES; ++j) {
          if (table[i + j] < 0) {
            table[i + j] = k;
            return;
          }
        }
      }
      perturb >>= PERTURB_SHIFT;
      i = static_cast<int>((static_cast<uint64_t>(i) * 5 + 1 + perturb) & static_cast<uint64_t>(mask));
    }
  }

  int used() const {
    int u = 0;
    for (int v : table)
      if (v >= 0) ++u;
    return u;
  }

  void resize(int minused) {
    int ns = 8;
    while (ns <= minused) ns <<= 1;
    std::vector<int> old;
    for (int v : table)
      if (v >= 0) old.push_back(v);
    mask = ns - 1;
    table.assign(static_cast<size_t>(ns), -1);
    for (int v : old) insert_clean(v);
  }

  void add(int k) {
    uint64_t h = static_cast<uint64_t>(k);
    uint64_t perturb = h;
    int i = static_cast<int>(h & static_cast<uint64_t>(mask));
    while (true) {
      if (table[i] < 0) {
        table[i] = k;
        maybe_grow();
        return;
      }
      if (table[i] == k) return;
      if (i + LINEAR_PROBES <= mask) {
        for (int j = 1; j <= LINEAR_PROBES; ++j) {
          if (table[i + j] < 0) {
            table[i + j] = k;
            maybe_grow();
            return;
          }
          if (table[i + j] == k) return;
        }
      }
      perturb >>= PERTURB_SHIFT;
      i = static_cast<int>((static_cast<uint64_t>(i) * 5 + 1 + perturb) & static_cast<uint64_t>(mask));
    }
  }

  void maybe_grow() {
    // fill == used here (no dummies), so fill*5 >= mask*3 -> resize(used*4).
    int u = used();
    if (static_cast<long long>(u) * 5 >= static_cast<long long>(mask) * 3) resize(u * 4);
  }

  std::vector<int> order() const {
    std::vector<int> out;
    for (int v : table)
      if (v >= 0) out.push_back(v);
    return out;
  }
};

}  // namespace

std::vector<int> cpython_set_order_int(const std::vector<int>& keys) {
  IntSet s;
  for (int k : keys) s.add(k);
  return s.order();
}

std::vector<int> cpython_set_intersection_int(const std::vector<int>& a,
                                              const std::vector<int>& b) {
  // Distinct-key sets, so |set| == number of unique keys. The intersection
  // iterates the smaller operand and probes the larger; on a size tie it
  // iterates the RIGHT operand `b` (the size test below is a strict '>').
  IntSet sa;
  for (int k : a) sa.add(k);
  IntSet sb;
  for (int k : b) sb.add(k);
  std::vector<int> a_order = sa.order();
  std::vector<int> b_order = sb.order();
  int size_a = static_cast<int>(a_order.size());
  int size_b = static_cast<int>(b_order.size());

  // Membership set for the operand NOT iterated.
  auto contains = [](const std::vector<int>& keys, int v) {
    for (int k : keys)
      if (k == v) return true;
    return false;
  };

  const std::vector<int>* iter_order;
  const std::vector<int>* other;
  if (size_b > size_a) {
    iter_order = &a_order;  // iterate a
    other = &b_order;
  } else {
    iter_order = &b_order;  // iterate b (includes the size tie)
    other = &a_order;
  }

  // Insert the shared keys into a fresh set in the iteration order.
  IntSet result;
  for (int v : *iter_order)
    if (contains(*other, v)) result.add(v);
  return result.order();
}

}  // namespace arc
