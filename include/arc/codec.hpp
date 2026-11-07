// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_CODEC_HPP
#define ARC_CODEC_HPP

// TransformCodec: genome construction, argument samplers and GA operators.
//
// The order of RNG draws is part of the behaviour: random_genome and the arg
// samplers draw from the ArcRandom in a fixed order, and one extra, missing or
// reordered choice/randint/uniform/gauss/random draw changes the whole search
// trajectory from that point on.
//
// This layer builds a DslProgram (arc/dsl.hpp) so canonical_repr /
// hash_program can be reused as-is for the program half of a genome. The
// encoder half is a GraphConfig (arc/config.hpp, serialized via config_to_dict)
// and the decoder half is a DecoderConfig (its 4 policy strings).
//
// Macros are OFF by default (no macro library): the random_genome
// macro-realize branch then draws NOTHING and is skipped before any rng call.

#include <string>
#include <vector>

#include "arc/config.hpp"
#include "arc/dsl.hpp"
#include "arc/rng.hpp"

namespace arc {

// ARC_ENS_ENC: a search can pin the encoder connectivity (4 or 8; 0 = free).
// Set per thread by the ensemble loop before each member search.
extern thread_local int g_force_connectivity;

struct MacroLibrary;  // fwd (arc/macro.hpp); the codec holds a non-owning ptr.

// The full genome the GA evolves: encoder_config + program + decoder.
struct TransformGenome {
  GraphConfig encoder_config;
  DslProgram program;
  DecoderConfig decoder;
};

// DeltaHint: the wrong-pixel signal delta_guided_mutate consumes. It is
// computed by compute_delta_hint (arc/search.hpp) from a genome's
// prediction-vs-target diff; here it is a PLAIN INPUT struct.
// delta_guided_mutate reads only: dominant_swap (Branch-A),
// majority_got/expected_color (Branch-B recolor args), and the flattened
// wrong-pixel `got` colors (Branch-B 60% coverage gate: the share of wrong
// cells whose `got` color is majority_got_color). The (y,x) coordinates do not
// matter to the mutation, so only the per-wrong-cell `got` color is carried.
struct DeltaHint {
  bool has_dominant_swap = false;
  int dominant_swap_src = 0;
  int dominant_swap_dst = 0;
  int majority_got_color = 0;
  int majority_expected_color = 0;
  // One entry per wrong cell across all pairs (all_wrong), holding its `got`
  // color. len() == number of wrong cells; count(== majority_got_color) is the
  // Branch-B numerator. Empty means no wrong cells -> Branch-B never fires.
  std::vector<int> wrong_got_colors;
};

// color_arg_by_actionkind(kind): the color-arg key delta_guided_mutate reads/
// writes for Branch-A. Returns nullptr when the kind has none (skip), else
// "color" or "out_color".
const char* color_arg_by_actionkind(ActionKind kind);

class TransformCodec {
 public:
  explicit TransformCodec(int max_rules = 3);

  // Color prior: when set (non-empty), sample_color biases toward these
  // colors 80% of the time (else full 0..9). set_color_prior sorts, dedups,
  // and clamps to 0..9. Empty vector clears.
  void set_color_prior(const std::vector<int>& colors);

  // ARC_INFORMED_COLOR=1: the input colours of the cells that change on the
  // same-size train pairs (from) and their output colours (to). A concept op's
  // colour argument drawn as a specific colour is redrawn from the matching set
  // half the time: complete_regularity mask (from); mark_side color, patch fill
  // (to); by_color's colour (from). Empty sets (the default) draw exactly as before.
  void set_change_colors(const std::vector<int>& from, const std::vector<int>& to) {
    chg_from_ = from;
    chg_to_ = to;
  }

  // Active macro library: when non-null AND non-empty with rate>0,
  // random_genome / the add_rule mutation branch realize a mined macro (see
  // maybe_macro_rules). nullptr (default) leaves every sampling path exactly as
  // with macros OFF (the branch returns BEFORE consuming any rng draw). The
  // pointer is NON-OWNING.
  void set_macro_library(const MacroLibrary* lib) { macro_lib_ = lib; }
  // Train-informed concept-op arguments (ARC_INFORMED_ARGS=1): when mutation
  // samples arguments for a concept op, half of the time it draws one of the
  // argument sets the op derives from the train pairs instead.
  void set_concept_arg_pool(std::vector<std::pair<int, std::vector<DslArgs>>> pool) {
    concept_arg_pool_ = std::move(pool);
  }
  // ARC_INFORMED_ATTR=1: by_attr argument sets read off the train pairs; half of
  // the by_attr draws in sampling and mutation pick one of them.
  void set_by_attr_pool(std::vector<DslArgs> pool) { by_attr_pool_ = std::move(pool); }
  const MacroLibrary* macro_library() const { return macro_lib_; }

  // Top-level genome sampler. Draw order (macros OFF):
  //   rule_count = randint(0, max_rules); rule_count x random_rule;
  //   transform  = choice over the 26 GlobalTransforms [enum order];
  //   encoder    = random_encoder_config;
  //   decoder    = random_decoder.
  TransformGenome random_genome(ArcRandom& rng);

  GraphConfig random_encoder_config(ArcRandom& rng);
  DslRule random_rule(ArcRandom& rng);
  // variant: the variant arg value of a merged kind ("" for none), put first.
  DslArgs sample_selector_args(SelectorKind kind, ArcRandom& rng,
                               const std::string& variant = "");
  DslArgs sample_action_args(ActionKind kind, ArcRandom& rng);
  std::vector<DslGuard> random_guards(ArcRandom& rng);
  DecoderConfig random_decoder(ArcRandom& rng);

  // GA operators. Draw order matters: mutate is a GUARDED
  // FALL-THROUGH over a SINGLE p = random(); a failed guard falls through to the
  // next branch (changing how many draws happen). crossover cuts each parent's
  // rule list independently and crosses encoder fields (with a per-field draw
  // ONLY for evolvable fields, in GraphConfig declaration order), transform, and
  // decoder 50/50. With macros OFF (the default) the add_rule macro attempt
  // draws nothing.
  TransformGenome mutate(const TransformGenome& genome, ArcRandom& rng);
  TransformGenome crossover(const TransformGenome& mom, const TransformGenome& dad,
                            ArcRandom& rng);

  int max_rules() const { return max_rules_; }

 private:
  // Prior-biased single color draw.
  int sample_color(ArcRandom& rng);

  // With the site-appropriate probability, realize a mined macro instead of
  // uniform random rule construction. Returns true and fills `out` (capped to
  // max_rules) when a macro fired; returns false WITHOUT consuming any rng draw
  // when the library is inactive (nullptr / empty / rate<=0), so the OFF path
  // is unchanged. Sites: mutating == false uses rate (random_genome);
  // mutating == true uses mutate_rate (falls back to rate when
  // mutate_rate < 0).
  bool maybe_macro_rules(ArcRandom& rng, bool mutating, std::vector<DslRule>* out);

  // The reduced arg sampler used for the nested
  // left/right selectors inside SELECTOR_SET_OP. NOT the same as
  // sample_selector_args (different, smaller distribution).
  // Keyed by the original kind name (composeable_selector_names).
  DslArgs selector_args_for_name(const std::string& kind, ArcRandom& rng);

  // The 9 kind names SELECTOR_SET_OP draws from.
  const std::vector<std::string>& composeable_selector_names() const;
  // One SELECTOR_SET_OP leg for a composeable name: its args, then the name
  // resolved (an old name gives the merged kind and its variant arg).
  DslSelector compose_leg(const std::string& name, ArcRandom& rng);

  // One encoder param sample (special-cased params, else a generic draw from
  // the param metadata).
  // Mutates `cfg` in place (sets the named field).
  void sample_encoder_param(const std::string& name, const ParamMeta& meta,
                            ArcRandom& rng, GraphConfig& cfg);

  // --- GA operator helpers -----------------------------------------------
  // Swap selector/action/guards, change args (guarded chain).
  DslRule mutate_rule(const DslRule& rule, ArcRandom& rng);
  // Per-kind selector-arg jitter (guarded chain).
  DslSelector mutate_selector(const DslSelector& selector, ArcRandom& rng);
  // 25% resample-whole / else jitter value by +-{1,2}, clamp 1..12.
  DslGuard mutate_guard(const DslGuard& guard, ArcRandom& rng);
  // 1 or 2 param changes (85/15), each param chosen
  // uniformly from the 21 evolvable params.
  GraphConfig mutate_encoder_config(const GraphConfig& config, ArcRandom& rng);
  // Special-cased params, else a generic mutation from the param metadata.
  void mutate_encoder_param(const std::string& name, const ParamMeta& meta,
                            ArcRandom& rng, GraphConfig& cfg);
  // choice(["bg","ov","pp"]) -> resample that one axis.
  DecoderConfig mutate_decoder(const DecoderConfig& decoder, ArcRandom& rng);
  // Per-field mom/dad pick (draw ONLY for evolvable fields,
  // in GraphConfig declaration order; copy-from-mom for the rest, NO draw).
  GraphConfig crossover_encoder(const GraphConfig& mom, const GraphConfig& dad,
                                ArcRandom& rng);

  int max_rules_;
  bool has_color_prior_ = false;
  std::vector<int> color_prior_;
  const MacroLibrary* macro_lib_ = nullptr;  // NON-OWNING; nullptr == macros OFF
  std::vector<std::pair<int, std::vector<DslArgs>>> concept_arg_pool_;  // (ActionKind, arg sets)
  DslArgs concept_args(ActionKind kind, ArcRandom& rng) const;
  std::vector<int> chg_from_, chg_to_;
  void inform_colors(ActionKind kind, DslArgs& a, ArcRandom& rng) const;
  int sample_select_color(ArcRandom& rng);
  std::vector<DslArgs> by_attr_pool_;
};

// delta_guided_mutate: biased mutation. Branch-A swaps one matching rule's
// color arg src->dst (rng.choice over candidate indices). Branch-B appends
// BY_COLOR(majority_got)->RECOLOR(majority_expected) when >=60% wrong-pixel
// coverage and room remains. Else falls back to uniform mutate. Draw order is
// load-bearing (rng.choice in Branch-A; the fallback delegates to mutate()).
TransformGenome delta_guided_mutate(TransformCodec& codec,
                                    const TransformGenome& genome,
                                    const DeltaHint& hint, ArcRandom& rng);

// Dedup keys, each returned as the repr() string of a tuple, so two genomes
// share a key iff the tuples are equal:
//   finalist  = repr((hash_program(prog), repr(decoder)))          [2-tuple]
//   ga        = repr((encoder_signature, hash, bg, po, ov, pp))     [6-tuple]
//   guesspool = repr((hash, repr(decoder), enc_sig(cfg)))           [3-tuple]
// encoder_signature = the tuple of config values in param_metadata order;
// enc_sig(cfg) = repr of the config's (name, value) items sorted by name.
std::string dedup_finalist_key(const TransformGenome& genome);
std::string dedup_ga_key(const TransformGenome& genome);
std::string dedup_guesspool_key(const TransformGenome& genome);

// repr(decoder): DecoderConfig(...) with its 4 named str fields in declaration
// order. The encoder_signature tuple repr and enc_sig(cfg) are exposed too.
std::string decoder_repr(const DecoderConfig& decoder);
std::string encoder_signature_repr(const GraphConfig& cfg);
std::string enc_sig(const GraphConfig& cfg);

}  // namespace arc

#endif  // ARC_CODEC_HPP
