// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// compute_delta_hint: the wrong-pixel signal the delta-guided hill-climb
// consumes.
//
// Runs the genome on every train pair (predict_grid == execute_program with the
// genome's decoder), diffs against the target, and aggregates the DeltaHint
// fields delta_guided_mutate reads. A "most common" tie goes to the key seen
// first, and keys are seen in the order wrong pixels are visited: pair order,
// then row-major within each pair. The counters below keep insertion order to
// give exactly that.
//
// No hint (has_hint stays false) when:
//   * any pair prediction shape != target shape
//   * the largest per-pair mismatch == 0.0 (already solved)
//   * the largest per-pair mismatch  > 0.20 (too noisy)
//   * total wrong-cell count < 2
// The executor never throws, so there is no error case.

#include "arc/search.hpp"

#include <vector>

#include "arc/fitness.hpp"

namespace arc {

namespace {

// Hint thresholds.
constexpr double kDominantSwapFraction = 0.50;
constexpr double kGateMaxPairMismatch = 0.20;
constexpr int kMinWrongCells = 2;

// Insertion-order-preserving counter over small non-negative int keys: the
// first add() of a key fixes its position; most_common_key() returns the
// max-count key, ties broken by earliest insertion.
struct OrderedCounter {
  std::vector<int> keys;     // insertion order
  std::vector<long long> counts;

  void add(int k) {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] == k) {
        ++counts[i];
        return;
      }
    }
    keys.push_back(k);
    counts.push_back(1);
  }

  // Max count, first-inserted on ties. Precondition:
  // non-empty.
  int most_common_key() const {
    size_t best = 0;
    for (size_t i = 1; i < keys.size(); ++i) {
      if (counts[i] > counts[best]) best = i;  // strict > keeps earlier on tie
    }
    return keys[best];
  }
};

// Insertion-order-preserving counter over (got, expected) pairs (Branch-A's
// dominant_swap). Same tie discipline as OrderedCounter.
struct OrderedPairCounter {
  std::vector<std::pair<int, int>> keys;
  std::vector<long long> counts;

  void add(int got, int expected) {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i].first == got && keys[i].second == expected) {
        ++counts[i];
        return;
      }
    }
    keys.emplace_back(got, expected);
    counts.push_back(1);
  }

  size_t most_common_idx() const {
    size_t best = 0;
    for (size_t i = 1; i < keys.size(); ++i) {
      if (counts[i] > counts[best]) best = i;
    }
    return best;
  }
};

}  // namespace

DeltaHint compute_delta_hint(const TransformGenome& genome,
                             const std::vector<TrainPair>& train_pairs,
                             bool* has_hint) {
  DeltaHint hint;
  if (has_hint) *has_hint = false;

  // Per-pair wrong (got, expected) lists, plus each pair's mismatch fraction.
  // all_wrong preserves cross-pair concatenation order (== insertion order for
  // the counters below).
  std::vector<std::pair<int, int>> all_wrong;  // (got, expected)
  std::vector<double> pair_mismatches;

  for (const auto& tp : train_pairs) {
    Grid pred = predict_grid(tp.input, genome.program, genome.encoder_config,
                             genome.decoder);
    // Shape mismatch -> no hint.
    if (pred.height() != tp.target.height() || pred.width() != tp.target.width()) {
      return hint;  // has_hint stays false
    }
    int h = tp.target.height();
    int w = tp.target.width();
    long long wrong = 0;
    // Row-major order.
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int got = pred.at(y, x);
        int expected = tp.target.at(y, x);
        if (got != expected) {
          all_wrong.emplace_back(got, expected);
          ++wrong;
        }
      }
    }
    long long total_cells = static_cast<long long>(h) * static_cast<long long>(w);
    double frac = total_cells ? (static_cast<double>(wrong) / static_cast<double>(total_cells))
                              : 1.0;
    pair_mismatches.push_back(frac);
  }

  if (pair_mismatches.empty()) return hint;
  // max_pair_mismatch = max(pair_mismatches).
  double max_pair = pair_mismatches[0];
  for (size_t i = 1; i < pair_mismatches.size(); ++i)
    if (pair_mismatches[i] > max_pair) max_pair = pair_mismatches[i];
  if (max_pair == 0.0) return hint;                 // solved
  if (max_pair > kGateMaxPairMismatch) return hint;  // too noisy

  if (static_cast<int>(all_wrong.size()) < kMinWrongCells) return hint;

  // got_counts / expected_counts / pair_counts (counters over all_wrong).
  OrderedCounter got_counts;
  OrderedCounter expected_counts;
  OrderedPairCounter pair_counts;
  for (const auto& wp : all_wrong) {
    got_counts.add(wp.first);
    expected_counts.add(wp.second);
    pair_counts.add(wp.first, wp.second);
  }
  int majority_got = got_counts.most_common_key();
  int majority_expected = expected_counts.most_common_key();

  size_t top_idx = pair_counts.most_common_idx();
  long long top_count = pair_counts.counts[top_idx];
  bool dominant =
      (static_cast<double>(top_count) / static_cast<double>(all_wrong.size())) >=
      kDominantSwapFraction;

  hint.majority_got_color = majority_got;
  hint.majority_expected_color = majority_expected;
  hint.has_dominant_swap = dominant;
  if (dominant) {
    hint.dominant_swap_src = pair_counts.keys[top_idx].first;
    hint.dominant_swap_dst = pair_counts.keys[top_idx].second;
  }
  // wrong_got_colors: one entry per wrong cell (its `got` color), for Branch-B's
  // >=60% coverage gate (count(==majority_got) / all_wrong).
  hint.wrong_got_colors.reserve(all_wrong.size());
  for (const auto& wp : all_wrong) hint.wrong_got_colors.push_back(wp.first);

  if (has_hint) *has_hint = true;
  return hint;
}

}  // namespace arc
