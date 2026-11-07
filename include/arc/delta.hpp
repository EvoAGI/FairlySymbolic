// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_DELTA_HPP
#define ARC_DELTA_HPP

// Delta seed emitters: emit_delta_seeds and the delta inference helpers it
// uses.
//
// emit_delta_seeds recognises directly-observed input->output deltas and emits
// focused one-/few-rule seed genomes.
//
// The FOUR branches that emit:
//   1.  Global color map (identity): infer_global_color_map_for_all_pairs ->
//       one N-recolor program (BY_COLOR -> RECOLOR per non-identity src->dst,
//       sorted by src). 1b: if the non-identity map is a proper 2-cycle a<->b,
//       also emit a single ALL_SUBOBJECTS -> SWAP_COLORS(a,b) program.
//   2.  Global color map under a geometric transform, for each of the 7
//       non-identity transforms (rot90/180/270, flip_h/v, transpose, anti):
//       infer_global_color_map_with_transform_for_all_pairs -> the recolor
//       rules with the transform set (skipped when the rule list is empty).
//   3.  Single-object translation: every pair shares one (dy,dx) -> one
//       ALL_SUBOBJECTS -> TRANSLATE(dy,dx) program (skipped for (0,0)).
//   4.  Per-src ranked color candidate (ONLY when there is no global map):
//       infer_color_candidates(max_per_src=1) -> per src, a BY_COLOR ->
//       RECOLOR(top-1 dst) program (skip identity dst==src).
//
// The output is deduplicated by an internal structure key, then the caller
// (enumerate_seeds block 5) re-dedups by dedup_ga_key. The EMISSION ORDER
// matters: delta seeds are appended after the ordered Phase-A seed list.

#include <vector>

#include "arc/codec.hpp"
#include "arc/config.hpp"
#include "arc/dsl.hpp"
#include "arc/seed_enum.hpp"  // SeedTrainPair

namespace arc {

// emit_delta_seeds(train_pairs, default_encoder, default_decoder): the ordered,
// structure-deduplicated delta seed genomes for branches 1-4. default_encoder
// is the default GraphConfig (delta seeds always use the default encoder).
std::vector<TransformGenome> emit_delta_seeds(const std::vector<SeedTrainPair>& train_pairs,
                                              const GraphConfig& default_encoder,
                                              const DecoderConfig& default_decoder);

}  // namespace arc

#endif  // ARC_DELTA_HPP
