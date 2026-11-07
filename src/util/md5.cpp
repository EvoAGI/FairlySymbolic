// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Minimal MD5 (RFC 1321). Public-domain-style reference implementation,
// adapted to a single-shot API that returns the lowercase hex digest.
#include "arc/md5.hpp"

#include <cstring>

namespace arc {
namespace {

inline uint32_t rotl(uint32_t x, uint32_t c) { return (x << c) | (x >> (32 - c)); }

const uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

// Message word of step i: i, (5i+1)%16, (3i+5)%16, 7i%16 in the four rounds.
const int G[64] = {0, 1, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                   1, 6, 11, 0,  5,  10, 15, 4,  9,  14, 3,  8,  13, 2,  7,  12,
                   5, 8, 11, 14, 1,  4,  7,  10, 13, 0,  3,  6,  9,  12, 15, 2,
                   0, 7, 14, 5,  12, 3,  10, 1,  8,  15, 6,  13, 4,  11, 2,  9};

const uint32_t S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

}  // namespace

std::string md5_hex(const uint8_t* data, size_t len) {
  uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;

  // Pad: append 0x80, then zeros, until length % 64 == 56, then 64-bit length.
  // The whole 64-byte blocks are read in place; the padded tail (one or two
  // blocks) is built on the stack.
  const size_t full = len / 64 * 64;
  uint8_t tail[128] = {0};
  const size_t rest = len - full;
  if (rest) std::memcpy(tail, data + full, rest);
  tail[rest] = 0x80;
  const size_t tail_len = rest < 56 ? 64 : 128;
  const uint64_t bit_len = static_cast<uint64_t>(len) * 8u;
  for (int i = 0; i < 8; ++i) tail[tail_len - 8 + i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xff);

  for (size_t off = 0; off < full + tail_len; off += 64) {
    const uint8_t* blk = off < full ? data + off : tail + (off - full);
    uint32_t M[16];
    for (int i = 0; i < 16; ++i) {
      M[i] = static_cast<uint32_t>(blk[4 * i]) |
             (static_cast<uint32_t>(blk[4 * i + 1]) << 8) |
             (static_cast<uint32_t>(blk[4 * i + 2]) << 16) |
             (static_cast<uint32_t>(blk[4 * i + 3]) << 24);
    }
    uint32_t A = a0, B = b0, C = c0, D = d0;
    // The four rounds of 16 steps, each with its own F and message order.
    auto step = [&](uint32_t F, int i) {
      F = F + A + K[i] + M[G[i]];
      A = D;
      D = C;
      C = B;
      B = B + rotl(F, S[i]);
    };
    for (int i = 0; i < 16; ++i) step((B & C) | (~B & D), i);
    for (int i = 16; i < 32; ++i) step((D & B) | (~D & C), i);
    for (int i = 32; i < 48; ++i) step(B ^ C ^ D, i);
    for (int i = 48; i < 64; ++i) step(C ^ (B | ~D), i);
    a0 += A;
    b0 += B;
    c0 += C;
    d0 += D;
  }

  uint8_t digest[16];
  uint32_t words[4] = {a0, b0, c0, d0};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) digest[i * 4 + j] = static_cast<uint8_t>((words[i] >> (8 * j)) & 0xff);

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
