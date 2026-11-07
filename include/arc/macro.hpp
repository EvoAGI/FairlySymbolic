// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_MACRO_HPP
#define ARC_MACRO_HPP

// Macro library realization: a learned sampling prior over rule
// compositions.
//
// A *macro* is a mined, parametric multi-rule strategy: a role skeleton with a
// mined per-slot distribution over the concrete (SelectorKind, ActionKind)
// heads that realized it, plus soft typed-hole bindings. Realization expands a
// macro into ordinary DslRule objects at genome-construction time -- the
// executor, fitness, hashing and serialization never see macros, only rules.
//
// Draw-order contract (as for the codec, the draw order is part of the
// behaviour):
//   * MacroLibrary::sample(rng): ONE rng.choices draw over the macros, weighted
//     by utility.
//   * realize(macro, codec, rng): PER SLOT, IN ORDER,
//       (sk, ak) = one rng.choices draw over slot.heads, weighted by count
//       sel_args = codec.sample_selector_args(sk, rng)
//       act_args = codec.sample_action_args(ak, rng)
//       slot.fixed_sel / slot.fixed_act then overwrite those args
//     THEN, per binding (arg_type, i, j, fraction), IN ORDER:
//       if i or j out of range -> skip (NO rng draw)
//       if rng.random() >= fraction -> skip (soft gate; the random() draw HAPPENS)
//       else copy the first typed-int arg of type `arg_type` from slot i to
//       slot j (source: sel_i preferred over act_i; dest: sel_j preferred over
//       act_j -- "preferred" == whichever has a matching typed-int arg first).
//
// Macros are OFF by default (no library configured); --macro-library turns
// them on.

#include <string>
#include <utility>
#include <vector>

#include "arc/ast.hpp"
#include "arc/dsl.hpp"
#include "arc/rng.hpp"

namespace arc {

class TransformCodec;  // fwd (codec.hpp includes are heavy; realize takes ref)

// One mined head: a concrete (SelectorKind, ActionKind) pair with its count.
struct MacroHead {
  SelectorKind sel;
  std::string sel_variant;  // variant arg of a merged kind named by an old name
  ActionKind act;
  std::string act_variant;  // the same for a merged core action
  long long count;  // mined realization count (choices weight)
};

// One rule slot: a mined head distribution + constant categorical overlays.
struct MacroSlot {
  std::pair<std::string, std::string> role;  // (SelRole, ActRole) -- descriptive
  std::vector<MacroHead> heads;              // >= 1 (empty-slot macros dropped)
  DslArgs fixed_sel;                         // sel: overlay (insertion order)
  DslArgs fixed_act;                         // act: overlay (insertion order)
};

// One binding: copy the first typed-int arg of `type` from slot i to slot j,
// applied softly with probability `fraction`.
struct MacroBinding {
  std::string type;  // arg_type() string: "color" | "size" | "scalar:<name>"
  int slot_i;
  int slot_j;
  double fraction;
};

struct Macro {
  std::string name;
  int length;
  int support;
  int utility;  // >= 1 (max(utility, 1) at load); the sample() weight
  std::vector<MacroSlot> slots;
  std::vector<MacroBinding> binding;
};

struct MacroLibrary {
  std::vector<Macro> macros;
  double rate = 0.0;          // P(realize) in random_genome (population seeding)
  int seed_topn = 0;          // top-N macros emitted as Phase A seeds
  double mutate_rate = -1.0;  // P(realize) in add_rule mutation (< 0 => == rate)

  // sample(rng): one rng.choices draw over the macros, weighted by utility.
  const Macro& sample(ArcRandom& rng) const;
};

// arg_type(name): type a scalar arg by name so holes bind within a type.
//   "color" in name          -> "color"
//   "mass" | "holes" | "count" in name -> "size"
//   else                     -> "scalar:" + name
// (the lowercase name is checked as a substring for "color"/"mass"/"count",
// and must equal "holes" for that case.)
std::string macro_arg_type(const std::string& name);

// load_macros(path): parse a mined macros JSON (via arc/json.hpp). Heads whose
// kind no longer exists in the enums are skipped; a macro with an empty slot is
// dropped; returns [] on any structural failure (library/code drift must never
// crash the solver).
std::vector<Macro> load_macros(const std::string& path);

// realize(macro, codec, rng): expand into concrete DslRules using the codec's
// own arg samplers (so every hole honours the codec's colour prior), then apply
// each typed-hole binding softly. See the draw-order contract above.
std::vector<DslRule> macro_realize(const Macro& macro, TransformCodec& codec,
                                   ArcRandom& rng);

}  // namespace arc

#endif  // ARC_MACRO_HPP
