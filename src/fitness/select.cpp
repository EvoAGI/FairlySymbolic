// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Parent selection: fitness ranking, tournament, epsilon-lexicase and case
// champions. See arc/select.hpp for the contract.

#include "arc/select.hpp"

#include <algorithm>

namespace arc {

FitnessRank fitness_rank(const FitnessScalars& fit, bool rank_specificity) {
  FitnessRank r;
  const bool solved = fit.solved;
  r.k0 = solved ? 0 : 1;
  r.k1 = solved ? fit.loo_fail : 0;
  r.k2 = (solved && rank_specificity) ? fit.spec : 0;
  r.k3 = solved ? fit.complexity : 0.0;
  r.k4 = -fit.exact_ratio;
  r.k5 = fit.max_pair_mismatch;
  r.k6 = fit.pixel_mismatch_mean;
  r.k7 = fit.complexity;
  return r;
}

bool rank_less(const FitnessRank& a, const FitnessRank& b) {
  // Lexicographic over k0..k7. Integer keys first, then doubles.
  if (a.k0 != b.k0) return a.k0 < b.k0;
  if (a.k1 != b.k1) return a.k1 < b.k1;
  if (a.k2 != b.k2) return a.k2 < b.k2;
  if (a.k3 != b.k3) return a.k3 < b.k3;
  if (a.k4 != b.k4) return a.k4 < b.k4;
  if (a.k5 != b.k5) return a.k5 < b.k5;
  if (a.k6 != b.k6) return a.k6 < b.k6;
  return a.k7 < b.k7;
}

std::vector<int> rank_sort_indices(const std::vector<FitnessScalars>& fits,
                                   bool rank_specificity) {
  const int n = static_cast<int>(fits.size());
  std::vector<int> idx(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
  std::vector<FitnessRank> ranks;
  ranks.reserve(fits.size());
  for (const auto& f : fits) ranks.push_back(fitness_rank(f, rank_specificity));
  // Stable sort: ties keep index order.
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    return rank_less(ranks[static_cast<size_t>(a)], ranks[static_cast<size_t>(b)]);
  });
  return idx;
}

int select_parent_tournament(const std::vector<FitnessScalars>& fits,
                             ArcRandom& rng, int k, bool rank_specificity) {
  const size_t n = fits.size();
  if (n == 0) return -1;
  // Draw k indices uniformly with replacement and return the best-ranked one.
  std::vector<int> picks(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    picks[static_cast<size_t>(i)] = static_cast<int>(rng.randrange(static_cast<int64_t>(n)));
  }
  std::vector<FitnessRank> ranks(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    ranks[static_cast<size_t>(i)] =
        fitness_rank(fits[static_cast<size_t>(picks[static_cast<size_t>(i)])], rank_specificity);
  }
  // Stable sort keeps the first-drawn on a rank tie.
  std::vector<int> order(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) order[static_cast<size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return rank_less(ranks[static_cast<size_t>(a)], ranks[static_cast<size_t>(b)]);
  });
  return picks[static_cast<size_t>(order[0])];
}

namespace {
// Median of a copy of `col`: sort, then the middle element (odd count) or
// the mean of the two middle elements (even count) as exactly (lo + hi) / 2.0.
double np_median(std::vector<double> col) {
  const size_t n = col.size();
  if (n == 0) return 0.0;
  std::sort(col.begin(), col.end());
  const size_t mid = n / 2;
  if (n % 2 == 1) return col[mid];
  return (col[mid - 1] + col[mid]) / 2.0;
}
}  // namespace

std::vector<double> lexicase_epsilon(const std::vector<std::vector<double>>& errors) {
  std::vector<double> eps;
  if (errors.empty()) return eps;
  const size_t n_cases = errors[0].size();
  eps.resize(n_cases);
  // med[case] = median over genomes; mad[case] = median(|err - med|).
  for (size_t c = 0; c < n_cases; ++c) {
    std::vector<double> col;
    col.reserve(errors.size());
    for (const auto& row : errors) col.push_back(row[c]);
    const double med = np_median(col);
    std::vector<double> dev;
    dev.reserve(col.size());
    for (double v : col) dev.push_back(std::abs(v - med));
    eps[c] = np_median(std::move(dev));
  }
  return eps;
}

int lexicase_select_idx(const std::vector<std::vector<double>>& errors,
                        const std::vector<double>& eps, ArcRandom& rng,
                        const std::vector<double>* bias) {
  const size_t n = errors.size();
  if (n == 0) return -1;
  const size_t n_cases = eps.size();

  // Start with every genome in the pool.
  std::vector<int> pool(n);
  for (size_t i = 0; i < n; ++i) pool[i] = static_cast<int>(i);

  // Visit the cases in random order.
  std::vector<size_t> case_order(n_cases);
  for (size_t i = 0; i < n_cases; ++i) case_order[i] = i;
  rng.shuffle_indices(case_order);  // in-place shuffle of the identity permutation

  // With a bias, stable-sort the cases by descending bias (ties keep the
  // shuffled order).
  if (bias != nullptr) {
    std::stable_sort(case_order.begin(), case_order.end(),
                     [&](size_t a, size_t b) { return -(*bias)[a] < -(*bias)[b]; });
  }

  // Filter case-by-case.
  for (size_t ci = 0; ci < case_order.size(); ++ci) {
    if (pool.size() <= 1) break;
    const size_t c = case_order[ci];
    double best = errors[static_cast<size_t>(pool[0])][c];
    for (int i : pool) {
      const double v = errors[static_cast<size_t>(i)][c];
      if (v < best) best = v;
    }
    const double threshold = best + eps[c];
    std::vector<int> next;
    next.reserve(pool.size());
    for (int i : pool) {
      if (errors[static_cast<size_t>(i)][c] <= threshold) next.push_back(i);
    }
    pool.swap(next);
  }

  // Uniform pick among the survivors.
  const int64_t pick = rng.randrange(static_cast<int64_t>(pool.size()));
  return pool[static_cast<size_t>(pick)];
}

std::vector<int> case_champions(const std::vector<std::vector<double>>& errors,
                                int champions, ArcRandom& rng) {
  std::vector<int> out;
  if (champions <= 0 || errors.empty()) return out;
  const int n_cases = static_cast<int>(errors[0].size());
  const int k = std::min(champions, n_cases);
  if (k <= 0) return out;
  // k distinct random cases, from a single sample_indices call.
  std::vector<size_t> sample = rng.sample_indices(static_cast<size_t>(n_cases),
                                                  static_cast<size_t>(k));
  const size_t n_genomes = errors.size();
  for (size_t c : sample) {
    // The genome with the lowest error on case c; the lowest index on a tie.
    size_t best_i = 0;
    double best = errors[0][c];
    for (size_t i = 1; i < n_genomes; ++i) {
      if (errors[i][c] < best) {
        best = errors[i][c];
        best_i = i;
      }
    }
    out.push_back(static_cast<int>(best_i));
  }
  return out;
}

}  // namespace arc
