// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// BLAKE2b (RFC 7693). Reference-style single-shot implementation, unkeyed,
// with a configurable digest size. See RFC 7693 sec. 3.
#include "arc/blake2b.hpp"

#include <cstring>

namespace arc {
namespace {

// IV: the first 64 bits of the fractional parts of the square roots of the
// first 8 primes (RFC 7693 sec. 2.6).
const uint64_t kIV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

// Message-word permutation schedule sigma (RFC 7693 sec. 2.7), 12 rounds.
const uint8_t kSigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

inline uint64_t rotr64(uint64_t x, unsigned n) {
  return (x >> n) | (x << (64 - n));
}

inline uint64_t load64_le(const uint8_t* p) {
  uint64_t r = 0;
  for (int i = 0; i < 8; ++i) r |= static_cast<uint64_t>(p[i]) << (8 * i);
  return r;
}

struct State {
  uint64_t h[8];
  uint64_t t[2];  // 128-bit byte counter
};

inline void mix(uint64_t* v, int a, int b, int c, int d, uint64_t x, uint64_t y) {
  v[a] = v[a] + v[b] + x;
  v[d] = rotr64(v[d] ^ v[a], 32);
  v[c] = v[c] + v[d];
  v[b] = rotr64(v[b] ^ v[c], 24);
  v[a] = v[a] + v[b] + y;
  v[d] = rotr64(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr64(v[b] ^ v[c], 63);
}

// Compress one 128-byte block. `last` sets the final-block flag f0.
void compress(State& s, const uint8_t block[128], bool last) {
  uint64_t m[16];
  for (int i = 0; i < 16; ++i) m[i] = load64_le(block + 8 * i);

  uint64_t v[16];
  for (int i = 0; i < 8; ++i) {
    v[i] = s.h[i];
    v[i + 8] = kIV[i];
  }
  v[12] ^= s.t[0];
  v[13] ^= s.t[1];
  if (last) v[14] = ~v[14];

  for (int r = 0; r < 12; ++r) {
    const uint8_t* sig = kSigma[r];
    mix(v, 0, 4, 8, 12, m[sig[0]], m[sig[1]]);
    mix(v, 1, 5, 9, 13, m[sig[2]], m[sig[3]]);
    mix(v, 2, 6, 10, 14, m[sig[4]], m[sig[5]]);
    mix(v, 3, 7, 11, 15, m[sig[6]], m[sig[7]]);
    mix(v, 0, 5, 10, 15, m[sig[8]], m[sig[9]]);
    mix(v, 1, 6, 11, 12, m[sig[10]], m[sig[11]]);
    mix(v, 2, 7, 8, 13, m[sig[12]], m[sig[13]]);
    mix(v, 3, 4, 9, 14, m[sig[14]], m[sig[15]]);
  }

  for (int i = 0; i < 8; ++i) s.h[i] ^= v[i] ^ v[i + 8];
}

}  // namespace

void blake2b(const uint8_t* data, size_t len, uint8_t* out, size_t digest_size) {
  if (digest_size < 1 || digest_size > 64) return;

  State s;
  for (int i = 0; i < 8; ++i) s.h[i] = kIV[i];
  s.t[0] = 0;
  s.t[1] = 0;
  // Parameter block XOR: only digest_length, key_length(0), fanout=1,
  // depth=1 are non-zero for an unkeyed hash. These occupy the low bytes of
  // the first parameter word, so h[0] ^= 0x0101kknn (nn=digest, kk=key=0).
  s.h[0] ^= 0x01010000ULL ^ static_cast<uint64_t>(digest_size);

  // Process all but the final block (RFC keeps >=1 byte for the last block,
  // so an all-full input still finalizes its last 128 bytes with last=true).
  size_t off = 0;
  while (len - off > 128) {
    s.t[0] += 128;
    if (s.t[0] < 128) s.t[1] += 1;  // carry
    compress(s, data + off, false);
    off += 128;
  }

  // Final block: remaining bytes (0..128), zero-padded to 128.
  uint8_t block[128];
  std::memset(block, 0, sizeof(block));
  size_t rem = len - off;
  if (rem > 0) std::memcpy(block, data + off, rem);
  s.t[0] += rem;
  if (s.t[0] < rem) s.t[1] += 1;  // carry
  compress(s, block, true);

  uint8_t full[64];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      full[i * 8 + j] = static_cast<uint8_t>((s.h[i] >> (8 * j)) & 0xff);
  std::memcpy(out, full, digest_size);
}

std::string blake2b16_hex(const uint8_t* data, size_t len) {
  uint8_t digest[16];
  blake2b(data, len, digest, 16);
  static const char* hexd = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (int i = 0; i < 16; ++i) {
    out.push_back(hexd[digest[i] >> 4]);
    out.push_back(hexd[digest[i] & 0xf]);
  }
  return out;
}

}  // namespace arc
