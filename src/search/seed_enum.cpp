// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Phase-A seed enumeration (blocks 1-6 below). See seed_enum.hpp for the
// contract.

#include "arc/seed_enum.hpp"
#include "arc/concepts.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>

#include "arc/ast.hpp"
#include "arc/delta.hpp"
#include "arc/macro.hpp"
#include "arc/pyrepr.hpp"

namespace arc {

// ---------------------------------------------------------------------------
// build_encoder_seed_bank. Six GraphConfig variants,
// IN ORDER: [base, light, pixel, edges_light, bg_obj, no_zone].
// ---------------------------------------------------------------------------
std::vector<GraphConfig> build_encoder_seed_bank() {
  GraphConfig base;  // default config

  GraphConfig light;
  light.detect_clusters = false;
  light.detect_patterns = false;
  light.detect_symmetries = false;
  light.compute_alignment_edges = false;
  light.compute_ray_edges = false;
  light.compute_gravity_edges = false;
  light.compute_bridge_edges = false;

  GraphConfig pixel;
  pixel.object_mode = "pixel";
  pixel.detect_clusters = false;
  pixel.detect_patterns = false;
  pixel.detect_symmetries = false;

  GraphConfig edges_light;
  edges_light.compute_ray_edges = false;
  edges_light.compute_gravity_edges = false;
  edges_light.compute_bridge_edges = false;
  edges_light.compute_alignment_edges = false;
  edges_light.compute_logic_edges = true;

  GraphConfig bg_obj;
  bg_obj.background_mode = "bg_as_object";
  bg_obj.object_mode = "pixel";
  bg_obj.detect_clusters = false;
  bg_obj.detect_patterns = false;
  bg_obj.detect_symmetries = false;

  GraphConfig no_zone;
  no_zone.object_mode = "pixel";
  no_zone.background_color_override = 0;
  no_zone.detect_zones = false;
  no_zone.compute_zone_features = false;
  no_zone.detect_clusters = false;
  no_zone.detect_patterns = false;
  no_zone.detect_symmetries = false;
  no_zone.compute_alignment_edges = false;
  no_zone.compute_ray_edges = false;
  no_zone.compute_gravity_edges = false;
  no_zone.compute_bridge_edges = false;

  return {base, light, pixel, edges_light, bg_obj, no_zone};
}

// ---------------------------------------------------------------------------
// Input-side gates.
// ---------------------------------------------------------------------------
bool looks_like_crop_case(const std::vector<SeedTrainPair>& pairs) {
  if (pairs.empty()) return false;
  for (const auto& p : pairs) {
    int ih = p.input.height(), iw = p.input.width();
    int oh = p.target.height(), ow = p.target.width();
    if (oh <= ih && ow <= iw && (oh < ih || ow < iw)) return true;
  }
  return false;
}

bool looks_like_scale_case(const std::vector<SeedTrainPair>& pairs, int factor) {
  if (pairs.empty()) return false;
  for (const auto& p : pairs) {
    int ih = p.input.height(), iw = p.input.width();
    int oh = p.target.height(), ow = p.target.width();
    if (oh != ih * factor || ow != iw * factor) return false;
  }
  return true;
}

bool looks_like_kron_case(const std::vector<SeedTrainPair>& pairs) {
  if (pairs.empty()) return false;
  for (const auto& p : pairs) {
    int ih = p.input.height(), iw = p.input.width();
    int oh = p.target.height(), ow = p.target.width();
    if (oh != ih * ih || ow != iw * iw) return false;
  }
  return true;
}

bool looks_like_hole_case(const std::vector<SeedTrainPair>& pairs) {
  for (const auto& p : pairs) {
    const Grid& inp = p.input;
    int h = inp.height(), w = inp.width();
    if (h == 0 || w == 0) continue;
    // bg = the value with the max count; on a tie, the smallest value.
    int counts[256];
    for (int c = 0; c < 256; ++c) counts[c] = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int c = inp.at(y, x);
        if (c >= 0 && c < 256) ++counts[c];
      }
    int bg = 0, best = -1;
    for (int c = 0; c < 256; ++c) {
      if (counts[c] > 0 && counts[c] > best) {  // strict > keeps the FIRST max
        best = counts[c];
        bg = c;
      }
    }
    // mask = inp != bg. If no non-bg cell, continue.
    bool any_mask = false;
    std::vector<char> mask(static_cast<size_t>(h * w), 0);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (inp.at(y, x) != bg) {
          mask[static_cast<size_t>(y * w + x)] = 1;
          any_mask = true;
        }
      }
    if (!any_mask) continue;
    // Flood the bg-complement (mask==0) from the grid border.
    std::vector<char> seen(static_cast<size_t>(h * w), 0);
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        bool border = (y == 0 || y == h - 1 || x == 0 || x == w - 1);
        if (border && mask[static_cast<size_t>(y * w + x)] == 0) {
          seen[static_cast<size_t>(y * w + x)] = 1;
          stack.emplace_back(y, x);
        }
      }
    static const int DY[4] = {-1, 1, 0, 0};
    static const int DX[4] = {0, 0, -1, 1};
    while (!stack.empty()) {
      auto yx = stack.back();
      stack.pop_back();
      int y = yx.first, x = yx.second;
      for (int d = 0; d < 4; ++d) {
        int ny = y + DY[d], nx = x + DX[d];
        if (ny >= 0 && ny < h && nx >= 0 && nx < w &&
            mask[static_cast<size_t>(ny * w + nx)] == 0 &&
            seen[static_cast<size_t>(ny * w + nx)] == 0) {
          seen[static_cast<size_t>(ny * w + nx)] = 1;
          stack.emplace_back(ny, nx);
        }
      }
    }
    // An unreached bg cell is enclosed => holes exist.
    for (int i = 0; i < h * w; ++i)
      if (mask[static_cast<size_t>(i)] == 0 && seen[static_cast<size_t>(i)] == 0)
        return true;
  }
  return false;
}

bool looks_like_edit_in_place(const std::vector<SeedTrainPair>& pairs) {
  if (pairs.empty()) return false;
  std::vector<double> kept_ratios;
  for (const auto& p : pairs) {
    const Grid& inp = p.input;
    const Grid& out = p.target;
    if (inp.height() != out.height() || inp.width() != out.width()) return false;
    int h = inp.height(), w = inp.width();
    long long total = 0;
    long long kept = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        bool nonbg = inp.at(y, x) != 0;
        if (nonbg) {
          ++total;
          if (inp.at(y, x) == out.at(y, x)) ++kept;
        }
      }
    if (total == 0) {
      kept_ratios.push_back(1.0);
      continue;
    }
    kept_ratios.push_back(static_cast<double>(kept) / static_cast<double>(total));
  }
  if (kept_ratios.empty()) return false;
  double mn = kept_ratios[0];
  for (double v : kept_ratios) mn = std::min(mn, v);
  return mn >= 0.6;
}

// ---------------------------------------------------------------------------
// decoder_variants / observed_colors.
// ---------------------------------------------------------------------------
std::vector<DecoderConfig> decoder_variants(const DecoderConfig& default_decoder,
                                            const std::vector<SeedTrainPair>& train_pairs) {
  std::vector<DecoderConfig> variants;
  variants.push_back(default_decoder);
  {
    DecoderConfig d;
    d.background_policy = "zero";
    d.paint_order = "as_is";
    d.overlap_policy = "keep_first";
    d.postprocess = "none";
    variants.push_back(d);
  }
  {
    DecoderConfig d;
    d.background_policy = "most_frequent_input";
    d.paint_order = "as_is";
    d.overlap_policy = "keep_first";
    d.postprocess = "none";
    variants.push_back(d);
  }
  if (!train_pairs.empty() && looks_like_edit_in_place(train_pairs)) {
    DecoderConfig d;
    d.background_policy = "input_copy";
    d.paint_order = "as_is";
    d.overlap_policy = "overwrite";
    d.postprocess = "none";
    variants.push_back(d);
  }
  return variants;
}

std::vector<int> observed_colors(const std::vector<SeedTrainPair>& train_pairs) {
  std::set<int> seen;
  for (const auto& p : train_pairs) {
    int h = p.input.height(), w = p.input.width();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) seen.insert(p.input.at(y, x));
    int oh = p.target.height(), ow = p.target.width();
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) seen.insert(p.target.at(y, x));
  }
  return std::vector<int>(seen.begin(), seen.end());
}

// ---------------------------------------------------------------------------
// enumerate_seeds.
// ---------------------------------------------------------------------------
namespace {

// DslArgs builders (args in insertion order).
DslValue dv_int(long long v) {
  DslValue d;
  d.tag = DslValue::Tag::Int;
  d.i = v;
  return d;
}
DslValue dv_str(const std::string& s) {
  DslValue d;
  d.tag = DslValue::Tag::Str;
  d.s = s;
  return d;
}
DslValue dv_empty_dict() {
  DslValue d;
  d.tag = DslValue::Tag::Dict;
  d.dict = std::make_shared<DslArgs>();
  return d;
}

DslSelector sel(SelectorKind k, DslArgs args = {}) {
  DslSelector s;
  s.kind = k;
  s.args = std::move(args);
  return s;
}
DslAction act(ActionKind k, DslArgs args = {}) {
  DslAction a;
  a.kind = k;
  a.args = std::move(args);
  return a;
}
DslRule rule(DslSelector s, DslAction a) {
  DslRule r;
  r.selector = std::move(s);
  r.action = std::move(a);
  return r;
}

}  // namespace

std::vector<TransformGenome> enumerate_seeds(const std::vector<SeedTrainPair>& train_pairs,
                                             TransformCodec& codec,
                                             const std::vector<GraphConfig>& encoder_seed_bank,
                                             const DecoderConfig& default_decoder,
                                             bool include_delta_seeds,
                                             const MacroLibrary* seed_lib,
                                             double max_seconds) {
  GraphConfig base_enc;  // default config
  std::vector<DecoderConfig> dec_variants = decoder_variants(default_decoder, train_pairs);
  std::vector<int> obs = observed_colors(train_pairs);

  // Phase-A wall-clock budget (budget_exceeded()). max_seconds <= 0 ->
  // unbounded: the guard is dormant and the full ordered list is emitted.
  const auto seed_start = std::chrono::steady_clock::now();
  auto budget_exceeded = [&]() -> bool {
    if (max_seconds <= 0.0) return false;
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - seed_start;
    return elapsed.count() > max_seconds;
  };

  std::vector<TransformGenome> seeds;
  std::set<std::string> uniq;

  auto mk = [&](const DslProgram& program, const GraphConfig* enc,
                const DecoderConfig* dec) -> TransformGenome {
    TransformGenome g;
    g.encoder_config = (enc != nullptr) ? *enc : base_enc;
    g.program = program;
    g.decoder = (dec != nullptr) ? *dec : default_decoder;
    return g;
  };

  auto add = [&](const TransformGenome& g) {
    std::string key = dedup_ga_key(g);
    if (uniq.count(key)) return;
    uniq.insert(key);
    seeds.push_back(g);
  };

  // ---- 1. Empty program + gated GlobalTransform x encoder bank -------------
  for (int gti = 0; gti < kGlobalTransformCount; ++gti) {
    GlobalTransform gt = static_cast<GlobalTransform>(gti);
    if (gt == GlobalTransform::KRON_SELF_MASK || gt == GlobalTransform::KRON_MASK_ANCHOR_COLOR) {
      if (!looks_like_kron_case(train_pairs)) continue;
    }
    if (gt == GlobalTransform::SCALE2_NEAREST && !looks_like_scale_case(train_pairs, 2)) continue;
    if (gt == GlobalTransform::SCALE3_NEAREST && !looks_like_scale_case(train_pairs, 3)) continue;
    if (gt == GlobalTransform::CROP_FOREGROUND && !looks_like_crop_case(train_pairs)) continue;
    if (gt == GlobalTransform::FILL_PERIODIC_BACKGROUND) continue;  // always skipped

    DslProgram prog;
    prog.global_transform = gt;
    for (const auto& enc : encoder_seed_bank) add(mk(prog, &enc, nullptr));
    for (size_t di = 1; di < dec_variants.size(); ++di) add(mk(prog, nullptr, &dec_variants[di]));
  }

  // ---- 2. Length-1 programs, default encoder + 3 decoder variants ----------
  std::vector<DslSelector> core_selectors = {
      sel(SelectorKind::ALL_SUBOBJECTS),
      sel(SelectorKind::EXTREME, DslArgs{{"pick", dv_str("largest")}}),
      sel(SelectorKind::EXTREME, DslArgs{{"pick", dv_str("smallest")}}),
      sel(SelectorKind::BY_COLOR_FREQUENCY, DslArgs{{"rank", dv_str("most")}}),
      sel(SelectorKind::BY_COLOR_FREQUENCY, DslArgs{{"rank", dv_str("least")}}),
      sel(SelectorKind::NON_BACKGROUND),
      sel(SelectorKind::BY_NOISE_SCORE),
      sel(SelectorKind::BY_MASS, DslArgs{{"op", dv_str("eq")}, {"mass", dv_int(1)}}),
      sel(SelectorKind::SAME_SHAPE_AS_REFERENCE,
          DslArgs{{"reference_selector_kind", dv_str("largest_subobject")},
                  {"reference_selector_args", dv_empty_dict()}}),
      sel(SelectorKind::SAME_SHAPE_AS_REFERENCE,
          DslArgs{{"reference_selector_kind", dv_str("leftmost_non_most_frequent")},
                  {"reference_selector_args", dv_empty_dict()}}),
  };
  for (int c : obs)
    core_selectors.push_back(sel(SelectorKind::BY_COLOR, DslArgs{{"color", dv_int(c)}}));
  // ARC_CONCEPT_SEEDS=0 leaves only the generic seeds: no by_attr selectors
  // and no concept-op programs at generation 0 (evolution can still use both).
  const char* cs_env = std::getenv("ARC_CONCEPT_SEEDS");
  const bool concept_seeds = !(cs_env && std::string(cs_env) == "0");
  if (concept_seeds)
  for (const auto& sa : std::vector<std::vector<std::string>>{
           {"n_same_shape", "eq", "0"}, {"n_same_color", "eq", "0"}, {"n_same_size", "eq", "0"},
           {"holes", "max", "0"}, {"holes", "ge", "1"}, {"is_frame", "eq", "1"},
           {"is_rect", "eq", "1"}, {"is_rect", "eq", "0"}, {"rank_size", "eq", "2"},
           {"n_touching", "max", "0"}, {"n_inside", "ge", "1"}, {"is_inside", "eq", "1"},
           {"sym_lr", "eq", "0"}, {"height", "unique", "0"}, {"width", "unique", "0"},
           {"rel_y_line", "eq", "0"}, {"rel_y_line", "eq", "2"}, {"rel_x_line", "eq", "0"},
           {"rel_x_line", "eq", "2"}, {"rel_y_big", "eq", "0"}, {"rel_y_big", "eq", "2"},
           {"in_big", "eq", "1"}, {"touch_big", "eq", "1"}, {"on_center", "eq", "1"},
           {"rank_center", "eq", "1"}, {"orient", "eq", "0"}, {"orient", "eq", "2"},
           {"rank_center_small", "eq", "1"}, {"rel_y_uniq", "eq", "0"}, {"rel_y_uniq", "eq", "2"},
           {"rel_x_uniq", "eq", "0"}, {"rel_x_uniq", "eq", "2"}})
    core_selectors.push_back(sel(SelectorKind::BY_ATTR,
                                 DslArgs{{"attr", dv_str(sa[0])}, {"op", dv_str(sa[1])},
                                         {"value", dv_int(std::stoi(sa[2]))}}));
  if (looks_like_hole_case(train_pairs)) {
    for (int h : {0, 1, 2, 3, 4})
      core_selectors.push_back(sel(SelectorKind::BY_HOLE_COUNT, DslArgs{{"holes", dv_int(h)}}));
  }

  std::vector<DslAction> core_actions = {
      act(ActionKind::KEEP_SELECTED, DslArgs{{"out", dv_str("grid")}}),
      act(ActionKind::KEEP_SELECTED, DslArgs{{"out", dv_str("crop")}}),
      act(ActionKind::DELETE),
      act(ActionKind::PRUNE_ENDPOINTS),
      act(ActionKind::COMPLETE_D4_SYMMETRY),
      act(ActionKind::PROJECT_INNER_QUAD_TO_OUTER_CORNERS),
      act(ActionKind::PROJECT_SINGLETONS_TO_FULL_LINES),
      act(ActionKind::PROJECT_MARKER),
      act(ActionKind::MIRROR, DslArgs{{"axis", dv_str("h")}}),
      act(ActionKind::MIRROR, DslArgs{{"axis", dv_str("v")}}),
      act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("background")}}),
      act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("most")}}),
      act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("least")}}),
      act(ActionKind::RECOLOR_FROM_REFERENCE),
      act(ActionKind::RECOLOR_SEQUENCE),
      act(ActionKind::TRANSLATE, DslArgs{{"mode", dv_str("origin")}}),
      act(ActionKind::STAMP_LOCAL_PATTERN,
          DslArgs{{"pattern", dv_str("ring")}, {"anchor", dv_str("centroid")},
                  {"color_source", dv_str("self")}}),
      act(ActionKind::STAMP_LOCAL_PATTERN,
          DslArgs{{"pattern", dv_str("cross")}, {"anchor", dv_str("centroid")},
                  {"color_source", dv_str("self")}}),
      act(ActionKind::STAMP_LOCAL_PATTERN,
          DslArgs{{"pattern", dv_str("x")}, {"anchor", dv_str("centroid")},
                  {"color_source", dv_str("self")}}),
      act(ActionKind::EXTEND_LINE_UNTIL,
          DslArgs{{"direction", dv_str("ray")}, {"stop_condition", dv_str("touch_object")},
                  {"anchor", dv_str("centroid")}, {"color_source", dv_str("self")}}),
      act(ActionKind::EXTEND_LINE_UNTIL,
          DslArgs{{"direction", dv_str("ray")}, {"stop_condition", dv_str("boundary")},
                  {"anchor", dv_str("centroid")}, {"color_source", dv_str("self")}}),
  };
  for (int c : obs)
    core_actions.push_back(act(ActionKind::RECOLOR, DslArgs{{"color", dv_int(c)}}));
  core_actions.push_back(act(ActionKind::SYMMETRIZE_OBJECT,
                             DslArgs{{"orientation", dv_str("v")}, {"mode", dv_str("intersect")},
                                     {"cluster_radius", dv_int(1)}}));

  for (const auto& s : core_selectors) {
    if (budget_exceeded()) return seeds;
    for (const auto& a : core_actions) {
      DslProgram prog;
      prog.rules.push_back(rule(s, a));
      prog.global_transform = GlobalTransform::IDENTITY;
      add(mk(prog, nullptr, nullptr));
      for (size_t di = 1; di < dec_variants.size(); ++di)
        add(mk(prog, nullptr, &dec_variants[di]));
    }
  }

  // ---- 2b. Concept primitives (arc/concepts.hpp) ---------------------------
  if (concept_seeds && !concept_ops().empty()) {
    std::vector<Grid> ins, outs;
    for (const auto& p : train_pairs) {
      ins.push_back(p.input);
      outs.push_back(p.target);
    }
    for (const ConceptOp& op : concept_ops()) {
      for (const DslArgs& args : op.seed_args(ins, outs)) {
        DslProgram prog;
        prog.rules.push_back(rule(sel(SelectorKind::ALL_SUBOBJECTS), act(op.kind, args)));
        prog.global_transform = GlobalTransform::IDENTITY;
        add(mk(prog, nullptr, nullptr));
        for (const auto& enc : encoder_seed_bank) add(mk(prog, &enc, nullptr));
      }
    }
  }

  // ---- 3. Length-1 x (encoder bank minus base) x short selector/action -----
  std::vector<DslAction> short_actions = {
      act(ActionKind::KEEP_SELECTED, DslArgs{{"out", dv_str("grid")}}),
      act(ActionKind::KEEP_SELECTED, DslArgs{{"out", dv_str("crop")}}),
      act(ActionKind::RECOLOR_FROM_REFERENCE),
      act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("least")}}),
      act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("most")}}),
  };
  std::vector<DslSelector> short_selectors = {
      sel(SelectorKind::EXTREME, DslArgs{{"pick", dv_str("largest")}}),
      sel(SelectorKind::BY_COLOR_FREQUENCY, DslArgs{{"rank", dv_str("most")}}),
      sel(SelectorKind::BY_COLOR_FREQUENCY, DslArgs{{"rank", dv_str("least")}}),
      sel(SelectorKind::BY_NOISE_SCORE),
  };
  for (int c : obs)
    short_selectors.push_back(sel(SelectorKind::BY_COLOR, DslArgs{{"color", dv_int(c)}}));
  for (size_t ei = 1; ei < encoder_seed_bank.size(); ++ei) {  // skip base
    if (budget_exceeded()) return seeds;
    const GraphConfig& enc = encoder_seed_bank[ei];
    for (const auto& s : short_selectors) {
      for (const auto& a : short_actions) {
        DslProgram prog;
        prog.rules.push_back(rule(s, a));
        prog.global_transform = GlobalTransform::IDENTITY;
        add(mk(prog, &enc, nullptr));
      }
    }
  }

  // ---- 4. Proven length-2 templates ---------------------------------------
  if (!budget_exceeded()) {  // (stop before the length-2 block)
  {
    DslProgram p1;
    p1.rules.push_back(rule(sel(SelectorKind::EXTREME, DslArgs{{"pick", dv_str("largest")}}),
                            act(ActionKind::KEEP_SELECTED, DslArgs{{"out", dv_str("grid")}})));
    p1.rules.push_back(rule(sel(SelectorKind::ALL_SUBOBJECTS), act(ActionKind::PRUNE_ENDPOINTS)));
    p1.global_transform = GlobalTransform::IDENTITY;
    add(mk(p1, nullptr, nullptr));
  }
  {
    DslProgram p2;
    p2.rules.push_back(rule(sel(SelectorKind::BY_COLOR_FREQUENCY, DslArgs{{"rank", dv_str("most")}}),
                            act(ActionKind::RECOLOR_TO, DslArgs{{"to", dv_str("least")}})));
    p2.rules.push_back(rule(sel(SelectorKind::ALL_SUBOBJECTS),
                            act(ActionKind::DELETE_REFERENCE_AFTER_TRANSFER)));
    p2.global_transform = GlobalTransform::IDENTITY;
    add(mk(p2, nullptr, nullptr));
  }
  {
    DslProgram p3;
    p3.rules.push_back(rule(sel(SelectorKind::BY_COLOR, DslArgs{{"color", dv_int(0)}}),
                            act(ActionKind::RECOLOR_FROM_REFERENCE)));
    p3.rules.push_back(rule(sel(SelectorKind::ALL_SUBOBJECTS), act(ActionKind::PRUNE_ENDPOINTS)));
    p3.global_transform = GlobalTransform::IDENTITY;
    add(mk(p3, nullptr, nullptr));
  }
  }  // end length-2 budget guard

  // ---- 5. Delta-guided seeds ----------------------------------------------
  // emit_delta_seeds recognises directly-observed input->output deltas and
  // emits focused one-/few-rule seeds (four branches). Tail-appended and
  // re-deduped by dedup_ga_key, so the block 1-4 order is preserved. base_enc =
  // GraphConfig::default() (delta seeds always use the default encoder).
  if (include_delta_seeds && !budget_exceeded()) {
    for (const auto& g : emit_delta_seeds(train_pairs, base_enc, default_decoder)) add(g);
  }

  // ---- 6. Mined macro seeds (OFF unless seed_lib && seed_topn>0) -----------
  // Top-N macros by mined (library) order, 3 realizations each, drawn from a
  // FRESH ArcRandom seeded 47 so the emitted seeds are
  // deterministic per task. Tail-appended so the block 1-5 order (and budget
  // behaviour) is preserved. base encoder + default decoder + IDENTITY program.
  if (seed_lib != nullptr && seed_lib->seed_topn > 0 && !seed_lib->macros.empty() &&
      !budget_exceeded()) {
    ArcRandom mrng;
    mrng.seed(47);
    int topn = seed_lib->seed_topn;
    int n = static_cast<int>(seed_lib->macros.size());
    if (topn > n) topn = n;
    for (int mi = 0; mi < topn; ++mi) {
      for (int rep = 0; rep < 3; ++rep) {
        std::vector<DslRule> rules = macro_realize(seed_lib->macros[mi], codec, mrng);
        if (static_cast<int>(rules.size()) > codec.max_rules()) {
          rules.resize(static_cast<size_t>(codec.max_rules()));  // keep the first max_rules
        }
        if (!rules.empty()) {
          DslProgram prog;
          prog.rules = std::move(rules);
          prog.global_transform = GlobalTransform::IDENTITY;
          add(mk(prog, nullptr, nullptr));
        }
      }
    }
  }
  return seeds;
}

}  // namespace arc
