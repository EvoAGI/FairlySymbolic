// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_MD5_HPP
#define ARC_MD5_HPP

// Minimal MD5 (RFC 1321), used by Node::compute_shape_hash.
// Produces the 32-char lowercase hex digest.

#include <cstddef>
#include <cstdint>
#include <string>

namespace arc {

// Hex-digest of `len` bytes at `data`.
std::string md5_hex(const uint8_t* data, size_t len);

}  // namespace arc

#endif  // ARC_MD5_HPP
