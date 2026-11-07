// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_SELECT_HPP
#define ARC_SELECT_HPP

// Parent selection: the 8-key lexicographic fitness rank + the three
// selection modes (tournament / lexicase / lexicase_inject) the GA breeds from.
//
//   * fitness_rank             -- the 8-key comparable tuple (order is
//                                 load-bearing).
//   * select_parent_tournament -- size-4 random draw, keep the rank-best.
//   * lexicase_epsilon         -- per-case MAD (median absolute deviation).
//   * lexicase_select_idx      -- epsilon-lexicase filter (per-gen shuffled
//                                 case order; optional complement bias by a
//                                 parent's error vector, DESCENDING via stable
//                                 sort).
//   * case_champions           -- the lex_champions per-case best carry.
//
// Draw order is part of the behaviour: tournament draws k randrange(len) in
// order; lexicase shuffles the case order from the SAME rng then draws
// randrange(len(pool)) to break the final tie; champions draw one
// rng.sample(range(n_cases), k). One extra/missing/reordered draw changes the
// entire search. Ranking uses ordinary double comparisons.
//
// This layer only needs the FITNESS SCALARS + CASE VECTORS, not the genomes.
// The case vectors come from arc/fitness.hpp.

#include <vector>

#include "arc/rng.hpp"

namespace arc {

// The fitness scalars fitness_rank reads. loo_fail and spec are extra fields
// set post-search / env-gated; default 0 (the GA hot loop never sets them).
struct FitnessScalars {
  bool solved = false;
  int loo_fail = 0;         // post-search generalization penalty (0 in the hot loop)
  int spec = 0;             // pinned-literal count (consulted iff rank_specificity)
  double complexity = 0.0;
  double exact_ratio = 0.0;
  double max_pair_mismatch = 0.0;
  double pixel_mismatch_mean = 0.0;
};

// The 8-key lexicographic rank tuple. EXACT order:
//   (0 if solved else 1,
//    loo_fail if solved else 0,
//    spec if (solved and rank_specificity) else 0,
//    complexity if solved else 0.0,
//    -exact_ratio,
//    max_pair_mismatch,
//    pixel_mismatch_mean,
//    complexity)                 <- complexity appears TWICE (gated + final tiebreak)
// key0/1/2 are integer-valued; key3..7 are doubles. Compared lexicographically.
struct FitnessRank {
  int k0 = 1;
  int k1 = 0;
  int k2 = 0;
  double k3 = 0.0;
  double k4 = 0.0;   // -exact_ratio
  double k5 = 0.0;   // max_pair_mismatch
  double k6 = 0.0;   // pixel_mismatch_mean
  double k7 = 0.0;   // complexity
};

// Build the rank tuple. rank_specificity is the ARC_RANK_SPECIFICITY gate
// (default off -> the spec slot stays 0).
FitnessRank fitness_rank(const FitnessScalars& fit, bool rank_specificity = false);

// Strict lexicographic '<' over the 8 keys.
bool rank_less(const FitnessRank& a, const FitnessRank& b);

// One scored population member: the fitness scalars + its case-error vector.
// case_errors is the fitness case vector (empty in tournament mode). scored is
// kept rank-sorted by the evolver; the selection functions assume that same
// index-aligned order.
struct Scored {
  FitnessScalars fit;
  std::vector<double> case_errors;
};

// Return the population indices in rank-sorted order (stable sort by
// fitness_rank). Ties keep original order.
std::vector<int> rank_sort_indices(const std::vector<FitnessScalars>& fits,
                                   bool rank_specificity = false);

// Tournament: k random draws scored[randrange(len)], keep the rank-best (min
// fitness_rank; ties -> first drawn, via stable sort). Returns the winner's
// population index.
int select_parent_tournament(const std::vector<FitnessScalars>& fits,
                             ArcRandom& rng, int k = 4,
                             bool rank_specificity = false);

// lexicase_epsilon: per-case epsilon = median-absolute-deviation of each
// case's error across the population. errors is [genome][case] (all rows same
// length). The median of an even count is the mean of the two middle values.
std::vector<double> lexicase_epsilon(const std::vector<std::vector<double>>& errors);

// lexicase_select_idx: epsilon-lexicase parent selection -> population index.
// Shuffle the case order from `rng`, then (if bias non-null) stable-sort by
// bias[c] DESCENDING, filter the pool case-by-case to those within best+eps[c],
// stop when <=1 survives, break the final tie with randrange(len(pool)).
// bias == nullptr is the unbiased draw (byte-identical to plain lexicase).
int lexicase_select_idx(const std::vector<std::vector<double>>& errors,
                        const std::vector<double>& eps, ArcRandom& rng,
                        const std::vector<double>* bias = nullptr);

// lex_champions per-case carry: sample champions cases via
// rng.sample(range(n_cases), min(champions, n_cases)) IN THAT ORDER, and for
// each sampled case c return the population index of the best (lowest-error)
// genome, ties -> lowest index. Returns the champion indices in the
// sampled-case order. Draws exactly one rng.sample (nothing when
// champions <= 0).
std::vector<int> case_champions(const std::vector<std::vector<double>>& errors,
                                int champions, ArcRandom& rng);

}  // namespace arc

#endif  // ARC_SELECT_HPP
