// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// ArcRandom implementation: the MT19937 core plus the seeding and distribution
// methods described in rng.hpp.

#include "arc/rng.hpp"

#include <cmath>
#include <cstring>

namespace arc {

// Portable floating-point barrier. Forces the operand through memory so the
// optimizer cannot reassociate across it or fuse the surrounding cos()/sin()
// into a single __sincos routine. clang at -O2 otherwise fuses them into
// __sincos_stret, whose sin result can differ by 1 ULP, so gauss() would
// depend on the compiler. Routing the angle through this barrier before each
// call keeps them as two independent libm calls. (See gauss() below.)
static inline double fp_barrier(double x) {
  volatile double t = x;
  return t;
}


// ---------------------------------------------------------------------------
// MT19937 core (init_genrand / init_by_array / genrand_uint32), as in the
// Matsumoto-Nishimura reference code.
// ---------------------------------------------------------------------------

void ArcRandom::init_genrand(uint32_t s) {
  mt_[0] = s;
  for (mti_ = 1; mti_ < N; mti_++) {
    mt_[mti_] =
        (1812433253UL * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
         static_cast<uint32_t>(mti_));
    // mt_ is uint32_t so the assignment already masks to 32 bits.
  }
}

void ArcRandom::init_by_array(const uint32_t* init_key, size_t key_length) {
  init_genrand(19650218UL);
  size_t i = 1, j = 0;
  size_t k = (N > key_length ? static_cast<size_t>(N) : key_length);
  for (; k; k--) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525UL)) +
             init_key[j] + static_cast<uint32_t>(j);  // non linear
    i++;
    j++;
    if (i >= static_cast<size_t>(N)) {
      mt_[0] = mt_[N - 1];
      i = 1;
    }
    if (j >= key_length) j = 0;
  }
  for (k = N - 1; k; k--) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941UL)) -
             static_cast<uint32_t>(i);  // non linear
    i++;
    if (i >= static_cast<size_t>(N)) {
      mt_[0] = mt_[N - 1];
      i = 1;
    }
  }
  mt_[0] = 0x80000000UL;  // MSB is 1; assuring non-zero initial array
}

uint32_t ArcRandom::genrand_uint32() {
  uint32_t y;
  static const uint32_t mag01[2] = {0x0UL, MATRIX_A};

  if (mti_ >= N) {  // generate N words at one time
    int kk;
    if (mti_ == N + 1)     // if init_genrand() has not been called,
      init_genrand(5489UL);  // a default initial seed is used

    for (kk = 0; kk < N - M; kk++) {
      y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
      mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < N - 1; kk++) {
      y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
      mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
    mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    mti_ = 0;
  }

  y = mt_[mti_++];

  // Tempering.
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  return y;
}

// ---------------------------------------------------------------------------
// Seeding: abs(n) -> LE 32-bit key words -> init_by_array
// ---------------------------------------------------------------------------

void ArcRandom::seed_words(const u32vec& mag_words) {
  // Seeding resets the gauss cached spare.
  // All seed_* entry points funnel through here, so clear the spare once.
  has_gauss_spare_ = false;
  gauss_next_ = 0.0;
  // keylen = max(1, #words). For n==0 (empty) the key is a single [0].
  if (mag_words.empty()) {
    uint32_t zero = 0;
    init_by_array(&zero, 1);
    return;
  }
  init_by_array(mag_words.data(), mag_words.size());
}

void ArcRandom::seed(uint64_t n) {
  // n is already non-negative here (abs handled by caller / by construction).
  u32vec words;
  uint64_t v = n;
  while (v) {
    words.push_back(static_cast<uint32_t>(v & 0xffffffffULL));
    v >>= 32;
  }
  seed_words(words);
}

void ArcRandom::seed_decimal(const char* decimal) {
  // Parse a (possibly signed) decimal magnitude into LE 32-bit words by
  // repeated big-int * 10 + digit. Sign is stripped (the seed is abs(n)).
  const char* p = decimal;
  if (*p == '+' || *p == '-') ++p;
  u32vec words;  // little-endian base-2^32 magnitude, no leading zero words.
  for (; *p; ++p) {
    if (*p < '0' || *p > '9') continue;  // skip non-digits defensively
    uint32_t digit = static_cast<uint32_t>(*p - '0');
    // words = words * 10 + digit
    uint64_t carry = digit;
    for (size_t i = 0; i < words.size(); ++i) {
      uint64_t cur = static_cast<uint64_t>(words[i]) * 10ULL + carry;
      words[i] = static_cast<uint32_t>(cur & 0xffffffffULL);
      carry = cur >> 32;
    }
    while (carry) {
      words.push_back(static_cast<uint32_t>(carry & 0xffffffffULL));
      carry >>= 32;
    }
  }
  // Trim any leading-zero high words (magnitude has no leading zeros anyway).
  while (!words.empty() && words.back() == 0) words.pop_back();
  seed_words(words);
}

// ---------------------------------------------------------------------------
// random(): 53-bit double.
// ---------------------------------------------------------------------------

double ArcRandom::random() {
  uint32_t a = genrand_uint32() >> 5;  // 27 bits
  uint32_t b = genrand_uint32() >> 6;  // 26 bits
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// getrandbits.
// ---------------------------------------------------------------------------

uint64_t ArcRandom::getrandbits(int k) {
  // k in [1, 64]. For k>64 use getrandbits_big.
  if (k <= 32) {
    return static_cast<uint64_t>(genrand_uint32() >> (32 - k));
  }
  // k in (32, 64]: low word full 32 bits, high word takes remaining (k-32).
  uint64_t lo = genrand_uint32();  // full 32 bits (word 0)
  int hi_bits = k - 32;
  uint64_t hi = genrand_uint32() >> (32 - hi_bits);
  return lo | (hi << 32);
}

u32vec ArcRandom::getrandbits_big(int k) {
  // General case: assemble ceil(k/32) words, LOW word first,
  // each full 32 bits except the TOP word which takes the remaining bits.
  u32vec words;
  if (k <= 0) return words;
  int remaining = k;
  while (remaining > 0) {
    uint32_t r = genrand_uint32();
    if (remaining < 32) {
      r >>= (32 - remaining);
    }
    words.push_back(r);
    remaining -= 32;
  }
  // Trim leading-zero high words for a canonical magnitude.
  while (words.size() > 1 && words.back() == 0) words.pop_back();
  return words;
}

// ---------------------------------------------------------------------------
// randbelow: rejection sampling on getrandbits(bit_length(n)).
// ---------------------------------------------------------------------------

static int bit_length_u64(uint64_t n) {
  int b = 0;
  while (n) {
    n >>= 1;
    ++b;
  }
  return b;
}

uint64_t ArcRandom::randbelow(uint64_t n) {
  if (n == 0) return 0;
  int k = bit_length_u64(n);
  uint64_t r = getrandbits(k);
  while (r >= n) {
    r = getrandbits(k);
  }
  return r;
}

// ---------------------------------------------------------------------------
// randrange / randint.
// ---------------------------------------------------------------------------

int64_t ArcRandom::randrange(int64_t stop) {
  // 1-arg form: randrange(stop) with istart=0. Requires stop>0.
  return static_cast<int64_t>(randbelow(static_cast<uint64_t>(stop)));
}

int64_t ArcRandom::randrange(int64_t start, int64_t stop) {
  int64_t width = stop - start;
  // step == 1, width > 0.
  return start + static_cast<int64_t>(randbelow(static_cast<uint64_t>(width)));
}

int64_t ArcRandom::randrange(int64_t start, int64_t stop, int64_t step) {
  int64_t width = stop - start;
  int64_t n;
  if (step > 0) {
    n = (width + step - 1) / step;
  } else {  // step < 0
    n = (width + step + 1) / step;
  }
  return start + step * static_cast<int64_t>(randbelow(static_cast<uint64_t>(n)));
}

int64_t ArcRandom::randint(int64_t a, int64_t b) {
  return randrange(a, b + 1);
}

size_t ArcRandom::choice_index(size_t n) {
  return static_cast<size_t>(randbelow(static_cast<uint64_t>(n)));
}

// ---------------------------------------------------------------------------
// uniform / gauss.
// ---------------------------------------------------------------------------

double ArcRandom::uniform(double a, double b) {
  return a + (b - a) * random();
}

double ArcRandom::gauss(double mu, double sigma) {
  // Box-Muller: each pair of draws gives two values; the second is cached as
  // the spare for the next call.
  static const double TWOPI = 6.283185307179586;  // == 2 * pi
  double z;
  if (has_gauss_spare_) {
    z = gauss_next_;
    has_gauss_spare_ = false;
  } else {
    double x2pi = random() * TWOPI;
    double g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
    // fp_barrier keeps cos()/sin() as two separate libm calls (no __sincos
    // fusion). See fp_barrier.
    z = std::cos(fp_barrier(x2pi)) * g2rad;
    gauss_next_ = std::sin(fp_barrier(x2pi)) * g2rad;
    has_gauss_spare_ = true;
  }
  return mu + z * sigma;
}

// ---------------------------------------------------------------------------
// choices (cumulative-weight path). Integer weights.
// ---------------------------------------------------------------------------

std::vector<size_t> ArcRandom::choices_indices(
    const std::vector<int64_t>& weights, size_t k) {
  size_t n = weights.size();
  // Guard empty weights before touching cum[n-1] (n-1 wraps to SIZE_MAX -> OOB
  // with -fno-exceptions): no weights give an empty result.
  if (n == 0) return std::vector<size_t>();
  // Cumulative weights. The solver's weights fit in int64, so the sums stay
  // exact.
  std::vector<int64_t> cum(n);
  int64_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    acc += weights[i];
    cum[i] = acc;
  }
  double total = static_cast<double>(cum[n - 1]);  // last cumulative weight
  size_t hi = n - 1;
  std::vector<size_t> out;
  out.reserve(k);
  for (size_t i = 0; i < k; ++i) {
    double x = random() * total;
    // bisect_right(cum, x, 0, hi): the first index in [0, hi) whose cum is
    // > x, else hi.
    size_t lo = 0, h = hi;
    while (lo < h) {
      size_t mid = (lo + h) / 2;
      if (x < static_cast<double>(cum[mid])) {
        h = mid;
      } else {
        lo = mid + 1;
      }
    }
    out.push_back(lo);
  }
  return out;
}

// ---------------------------------------------------------------------------
// shuffle: Fisher-Yates from the end, swapping i with randbelow(i + 1).
// ---------------------------------------------------------------------------

void ArcRandom::shuffle_indices(std::vector<size_t>& perm) {
  size_t len = perm.size();
  if (len < 2) return;
  for (size_t i = len - 1; i >= 1; --i) {
    uint64_t j = randbelow(static_cast<uint64_t>(i + 1));
    size_t sj = static_cast<size_t>(j);
    size_t tmp = perm[i];
    perm[i] = perm[sj];
    perm[sj] = tmp;
    if (i == 1) break;  // avoid size_t underflow past 1
  }
}

// ---------------------------------------------------------------------------
// sample. Both branches (pool / selection set).
// ---------------------------------------------------------------------------

std::vector<size_t> ArcRandom::sample_indices(size_t n, size_t k) {
  std::vector<size_t> result(k);
  size_t setsize = 21;
  if (k > 5) {
    // setsize += 4^ceil(log4(k*3))
    double exponent = std::ceil(std::log(k * 3.0) / std::log(4.0));
    // Raise 4 to the exponent with an integer loop, not pow(4.0,...): a float
    // straddle could silently pick the wrong sample branch.
    // exponent is a small non-negative int after ceil, so this is exact.
    size_t pow4 = 1;
    for (int e = static_cast<int>(exponent); e > 0; --e) pow4 *= 4;
    setsize += pow4;
  }
  if (n <= setsize) {
    // Pool branch.
    std::vector<size_t> pool(n);
    for (size_t i = 0; i < n; ++i) pool[i] = i;
    for (size_t i = 0; i < k; ++i) {
      size_t j = static_cast<size_t>(randbelow(static_cast<uint64_t>(n - i)));
      result[i] = pool[j];
      pool[j] = pool[n - i - 1];
    }
  } else {
    // Selection-set branch.
    // Use a boolean membership array over [0, n) (n is small enough here).
    std::vector<bool> selected(n, false);
    for (size_t i = 0; i < k; ++i) {
      size_t j = static_cast<size_t>(randbelow(static_cast<uint64_t>(n)));
      while (selected[j]) {
        j = static_cast<size_t>(randbelow(static_cast<uint64_t>(n)));
      }
      selected[j] = true;
      result[i] = j;  // population[j] == j since population = [0, n)
    }
  }
  return result;
}

}  // namespace arc
