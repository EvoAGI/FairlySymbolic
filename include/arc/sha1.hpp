// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_SHA1_HPP
#define ARC_SHA1_HPP

// SHA-1 (RFC 3174), single-shot API, producing the 20-byte digest. Used by the
// per-task seed derivation (task_seed hashes the task_id with SHA-1).

#include <cstddef>
#include <cstdint>

namespace arc {

// Writes the 20-byte SHA-1 digest of `data[0..len)` into out[0..20).
void sha1(const uint8_t* data, size_t len, uint8_t out[20]);

}  // namespace arc

#endif  // ARC_SHA1_HPP
