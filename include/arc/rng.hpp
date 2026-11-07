// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_RNG_HPP
#define ARC_RNG_HPP

// ArcRandom: a Mersenne Twister (MT19937) generator whose seeding and
// distribution methods follow CPython 3.9's random.Random draw for draw, so a
// seed gives the same stream as random.Random(seed). Every draw-consumption
// detail is kept (the randbelow rejection loop, the gauss cached spare,
// choices' random()*total + bisect_right, sample's two branches).
//
// Word conventions: 32-bit words, little-endian key packing for seed(),
// doubles produced from two 32-bit draws.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arc {

// A big non-negative integer as little-endian 32-bit words (word[0] = least
// significant). Used for seeds and getrandbits results that exceed 64 bits.
using u32vec = std::vector<uint32_t>;

class ArcRandom {
 public:
  // MT19937 constants.
  static constexpr int N = 624;
  static constexpr int M = 397;
  static constexpr uint32_t MATRIX_A = 0x9908b0dfUL;
  static constexpr uint32_t UPPER_MASK = 0x80000000UL;
  static constexpr uint32_t LOWER_MASK = 0x7fffffffUL;

  ArcRandom() { seed(0); }
  explicit ArcRandom(uint64_t s) { seed(s); }

  // ---- seeding -----------------------------------------------------------
  // seed(int n): takes abs(n), splits into little-endian
  // 32-bit words (min length 1), then init_by_array.
  void seed(uint64_t n);
  // seed from an arbitrary-size non-negative magnitude given as LE 32-bit
  // words (already abs()'d). Empty vector means n==0 -> key = [0].
  void seed_words(const u32vec& mag_words);
  // seed from a decimal string (handles >64-bit seeds); sign is stripped.
  void seed_decimal(const char* decimal);

  // Low-level MT init (exposed for testing / direct use).
  void init_genrand(uint32_t s);
  void init_by_array(const uint32_t* init_key, size_t key_length);

  // ---- core --------------------------------------------------------------
  uint32_t genrand_uint32();
  // random(): 53-bit double in [0, 1).
  double random();

  // ---- integer draws -----------------------------------------------------
  // getrandbits(k), k in [1, 64]. Returns the value in a uint64_t.
  uint64_t getrandbits(int k);
  // getrandbits(k) for arbitrary k > 0, as LE 32-bit words.
  u32vec getrandbits_big(int k);

  // randbelow(n) for n up to 64 bits (rejection loop). Returns 0 if n==0.
  uint64_t randbelow(uint64_t n);

  // randrange / randint (return signed since ranges may be negative).
  int64_t randrange(int64_t stop);
  int64_t randrange(int64_t start, int64_t stop);
  int64_t randrange(int64_t start, int64_t stop, int64_t step);
  int64_t randint(int64_t a, int64_t b);

  // choice over a sequence of length n -> chosen index.
  size_t choice_index(size_t n);

  // ---- float draws -------------------------------------------------------
  double uniform(double a, double b);
  double gauss(double mu, double sigma);

  // ---- sequence draws ----------------------------------------------------
  // choices with cumulative-weight path (integer weights). Returns k indices
  // into population = [0, len(weights)).
  std::vector<size_t> choices_indices(const std::vector<int64_t>& weights,
                                      size_t k);

  // shuffle: fills perm with the shuffled permutation of [0, n).
  void shuffle_indices(std::vector<size_t>& perm);

  // sample: returns k selected indices from population = [0, n). Reproduces
  // both branches (pool for n<=setsize, selection-set otherwise).
  std::vector<size_t> sample_indices(size_t n, size_t k);

 private:
  uint32_t mt_[N];
  int mti_ = N + 1;  // mti == N+1 means mt_ not initialized.

  // gauss cached spare (per-instance, persists across calls). has_gauss_spare_
  // tells whether gauss_next_ holds one.
  bool has_gauss_spare_ = false;
  double gauss_next_ = 0.0;
};

}  // namespace arc

#endif  // ARC_RNG_HPP
