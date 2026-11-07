// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_SEED_ENUM_HPP
#define ARC_SEED_ENUM_HPP

// Phase-A seed enumeration.
//
// enumerate_seeds produces the ordered, deduplicated seed-genome list for
// (train_pairs, codec, encoder_seed_bank, default_decoder,
// include_delta_seeds). Without a wall budget (max_seconds <= 0) every budget
// check is dormant and the full deterministic list is emitted.
//
// Blocks 1-4 (empty-program x gated GlobalTransform x encoder bank, length-1
// default-encoder x decoder variants, length-1 encoder-bank x short lists,
// length-2 templates) are the NON-delta core. Block 5 (delta-guided seeds) is
// gated by include_delta_seeds; block 6 (mined macros) runs only with a macro
// seed library (macros are off by default).
//
// Enumeration order + the encoder seed bank contents + the dedup key matter:
// the ordered seed list feeds run_single_attempt's Phase-A quick-rank and the
// initial population, so one extra/missing/reordered seed changes the whole
// trajectory.

#include <vector>

#include "arc/codec.hpp"
#include "arc/config.hpp"
#include "arc/dsl.hpp"
#include "arc/grid.hpp"

namespace arc {

// One (input, target) training pair (raw grids). Same shape as search.hpp's
// TrainPair; declared here so seed_enum does not depend on search.hpp. (The two
// structs have the same layout.)
struct SeedTrainPair {
  Grid input;
  Grid target;
};

// build_encoder_seed_bank(): the 6 seed encoder variants, IN ORDER:
//   [base, light, pixel, edges_light, bg_obj, no_zone].
// Each is a GraphConfig::default() with a specific toggle combination.
std::vector<GraphConfig> build_encoder_seed_bank();

// Input-side gates. Determine which gated GlobalTransforms
// / hole selectors / input_copy decoder are emitted for a task.
bool looks_like_crop_case(const std::vector<SeedTrainPair>& pairs);
bool looks_like_scale_case(const std::vector<SeedTrainPair>& pairs, int factor);
bool looks_like_kron_case(const std::vector<SeedTrainPair>& pairs);
bool looks_like_hole_case(const std::vector<SeedTrainPair>& pairs);
bool looks_like_edit_in_place(const std::vector<SeedTrainPair>& pairs);

// decoder_variants(default, train_pairs): the ordered decoder policy list Phase
// A probes -- the default + two fallbacks, plus the input_copy variant when the
// edit-in-place gate fires. dec_variants[0] is `default`; blocks use [1:].
std::vector<DecoderConfig> decoder_variants(const DecoderConfig& default_decoder,
                                            const std::vector<SeedTrainPair>& train_pairs);

// observed_colors(train_pairs): sorted distinct colors across all inputs+outputs.
std::vector<int> observed_colors(const std::vector<SeedTrainPair>& train_pairs);

// enumerate_seeds: the ordered, deduplicated seed-genome list. The dedup key
// is dedup_ga_key (the 6-tuple (encoder_signature, program_hash, bg, po, ov,
// pp) repr).
//
// include_delta_seeds: when true, block 5 folds the delta-guided seeds
// (delta.hpp / emit_delta_seeds) into the ordered list AFTER blocks 1-4, each
// re-deduped by the same dedup_ga_key. The two call sites differ: Phase A
// passes false (delta seeds are scored in a separate lane), while the initial
// population (Phase B) uses the default true.
//
// seed_lib: the mined-macro seed library (typically the broader near-mined
// library). When non-null AND seed_topn>0 AND non-empty, block 6 tail-appends
// the macro seeds: the top-N macros by mined order, 3 realizations each drawn
// from a FRESH ArcRandom seeded 47, realized via macro_realize + capped to
// codec.max_rules, wrapped as an IDENTITY program under base encoder + default
// decoder, each re-deduped by dedup_ga_key. nullptr (default) keeps block 6 OFF.
// max_seconds (Phase-A wall budget): when > 0, a wall-clock budget is checked
// at the block boundaries (between blocks 2/3/4/5/6). Once elapsed, later
// blocks are skipped and the seeds found so far are returned. <= 0 (default) =
// UNBOUNDED (deterministic). The search passes 2.0 for Phase A and 5.0 for the
// Phase-B initial population when a per-task deadline is set.
std::vector<TransformGenome> enumerate_seeds(const std::vector<SeedTrainPair>& train_pairs,
                                             TransformCodec& codec,
                                             const std::vector<GraphConfig>& encoder_seed_bank,
                                             const DecoderConfig& default_decoder,
                                             bool include_delta_seeds = true,
                                             const MacroLibrary* seed_lib = nullptr,
                                             double max_seconds = 0.0);

}  // namespace arc

#endif  // ARC_SEED_ENUM_HPP
