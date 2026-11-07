// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SHA-1 (RFC 3174) single-shot implementation. See include/arc/sha1.hpp.

#include "arc/sha1.hpp"

#include <cstring>
#include <vector>

namespace arc {

namespace {

inline uint32_t rotl(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

}  // namespace

void sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
  uint32_t h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu,
           h3 = 0x10325476u, h4 = 0xC3D2E1F0u;

  // Pad: append 0x80, then zeros, then the 64-bit big-endian bit length, to a
  // multiple of 64 bytes.
  std::vector<uint8_t> msg(data, data + len);
  uint64_t bitlen = static_cast<uint64_t>(len) * 8u;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0x00);
  for (int i = 7; i >= 0; --i)
    msg.push_back(static_cast<uint8_t>((bitlen >> (8 * i)) & 0xFF));

  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[80];
    for (int t = 0; t < 16; ++t) {
      w[t] = (static_cast<uint32_t>(msg[off + 4 * t]) << 24) |
             (static_cast<uint32_t>(msg[off + 4 * t + 1]) << 16) |
             (static_cast<uint32_t>(msg[off + 4 * t + 2]) << 8) |
             (static_cast<uint32_t>(msg[off + 4 * t + 3]));
    }
    for (int t = 16; t < 80; ++t)
      w[t] = rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);

    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int t = 0; t < 80; ++t) {
      uint32_t f, k;
      if (t < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5A827999u;
      } else if (t < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1u;
      } else if (t < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDCu;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6u;
      }
      uint32_t tmp = rotl(a, 5) + f + e + k + w[t];
      e = d;
      d = c;
      c = rotl(b, 30);
      b = a;
      a = tmp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }

  const uint32_t hs[5] = {h0, h1, h2, h3, h4};
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = static_cast<uint8_t>((hs[i] >> 24) & 0xFF);
    out[4 * i + 1] = static_cast<uint8_t>((hs[i] >> 16) & 0xFF);
    out[4 * i + 2] = static_cast<uint8_t>((hs[i] >> 8) & 0xFF);
    out[4 * i + 3] = static_cast<uint8_t>(hs[i] & 0xFF);
  }
}

}  // namespace arc
