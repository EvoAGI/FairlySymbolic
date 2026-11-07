// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_BLAKE2B_HPP
#define ARC_BLAKE2B_HPP

// BLAKE2b (RFC 7693) with configurable digest length, single-shot API.
//
// blake2b16_hex(data, len): the 16-byte digest as lowercase hex -- the digest
// size the program hash uses.

#include <cstddef>
#include <cstdint>
#include <string>

namespace arc {

// Raw digest: writes `digest_size` bytes to `out` (out must hold digest_size,
// 1..64). No key / salt / personalization (unkeyed).
void blake2b(const uint8_t* data, size_t len, uint8_t* out, size_t digest_size);

// Lowercase hex digest of `len` bytes with digest_size=16 (128-bit).
std::string blake2b16_hex(const uint8_t* data, size_t len);

}  // namespace arc

#endif  // ARC_BLAKE2B_HPP
