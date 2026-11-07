// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Delta seed emitters: seeds read off the input -> output deltas the train
// pairs show directly. See delta.hpp for the contract. Four branches (global
// colour map, colour map under a transform, single-object translation,
// per-colour candidates) and the delta inference helpers they use.

#include "arc/delta.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>

#include "arc/ast.hpp"
#include "arc/pyrepr.hpp"

namespace arc {

namespace {

// Dense row-major grid view for the delta helpers.
using RawGrid = std::vector<std::vector<int>>;

RawGrid to_raw(const Grid& g) {
  int h = g.height(), w = g.width();
  RawGrid r(h, std::vector<int>(w, 0));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) r[y][x] = g.at(y, x);
  return r;
}

int rows(const RawGrid& g) { return static_cast<int>(g.size()); }
int cols(const RawGrid& g) { return g.empty() ? 0 : static_cast<int>(g[0].size()); }

// ---------------------------------------------------------------------------
// DslValue / rule builders (args in insertion order).
// ---------------------------------------------------------------------------
DslValue dv_int(long long v) {
  DslValue d; d.tag = DslValue::Tag::Int; d.i = v; return d;
}
DslValue dv_str(const std::string& v) {
  DslValue d; d.tag = DslValue::Tag::Str; d.s = v; return d;
}
DslSelector sel(SelectorKind k, DslArgs args = {}) {
  DslSelector s; s.kind = k; s.args = std::move(args); return s;
}
DslAction act(ActionKind k, DslArgs args = {}) {
  DslAction a; a.kind = k; a.args = std::move(args); return a;
}
DslRule rule_of(DslSelector s, DslAction a) {
  DslRule r; r.selector = std::move(s); r.action = std::move(a); return r;
}

// ---------------------------------------------------------------------------
// infer_global_color_map(inp, out): consistent src->dst map, or none. Requires
// equal shape. Returns success + the map (insertion order kept; callers only
// iterate it by ascending src).
// ---------------------------------------------------------------------------
struct ColorMap {
  std::vector<std::pair<int, int>> entries;  // (src, dst) insertion order
  bool get(int src, int* dst) const {
    for (const auto& e : entries) if (e.first == src) { *dst = e.second; return true; }
    return false;
  }
  void set_new(int src, int dst) { entries.emplace_back(src, dst); }
};

bool infer_global_color_map(const RawGrid& inp, const RawGrid& out, ColorMap* mapping) {
  if (rows(inp) != rows(out) || cols(inp) != cols(out)) return false;
  int h = rows(inp), w = cols(inp);
  ColorMap m;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int src = inp[y][x], dst = out[y][x];
      int old = 0;
      if (!m.get(src, &old)) m.set_new(src, dst);
      else if (old != dst) return false;
    }
  *mapping = m;
  return true;
}

// infer_global_color_map_for_all_pairs: merge per-pair maps; none on conflict.
bool infer_global_color_map_for_all_pairs(const std::vector<RawGrid>& ins,
                                          const std::vector<RawGrid>& outs,
                                          ColorMap* merged_out) {
  if (ins.empty()) return false;
  ColorMap merged;
  for (size_t i = 0; i < ins.size(); ++i) {
    ColorMap local;
    if (!infer_global_color_map(ins[i], outs[i], &local)) return false;
    for (const auto& e : local.entries) {
      int old = 0;
      if (!merged.get(e.first, &old)) merged.set_new(e.first, e.second);
      else if (old != e.second) return false;
    }
  }
  *merged_out = merged;
  return true;
}

// apply_raw_transform for the 7 non-identity permutation transforms only (the
// delta seeds never use the others).
bool apply_raw_transform(const RawGrid& g, GlobalTransform t, RawGrid* out) {
  int h = rows(g), w = cols(g);
  switch (t) {
    case GlobalTransform::ROT90: {  // counter-clockwise: out[i][j] = g[j][w-1-i]
      RawGrid r(w, std::vector<int>(h, 0));
      for (int i = 0; i < w; ++i)
        for (int j = 0; j < h; ++j) r[i][j] = g[j][w - 1 - i];
      *out = r; return true;
    }
    case GlobalTransform::ROT180: {  // half turn
      RawGrid r(h, std::vector<int>(w, 0));
      for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) r[i][j] = g[h - 1 - i][w - 1 - j];
      *out = r; return true;
    }
    case GlobalTransform::ROT270: {  // clockwise: out[i][j] = g[h-1-j][i]
      RawGrid r(w, std::vector<int>(h, 0));
      for (int i = 0; i < w; ++i)
        for (int j = 0; j < h; ++j) r[i][j] = g[h - 1 - j][i];
      *out = r; return true;
    }
    case GlobalTransform::FLIP_H: {  // reverse columns
      RawGrid r(h, std::vector<int>(w, 0));
      for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) r[i][j] = g[i][w - 1 - j];
      *out = r; return true;
    }
    case GlobalTransform::FLIP_V: {  // reverse rows
      RawGrid r(h, std::vector<int>(w, 0));
      for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) r[i][j] = g[h - 1 - i][j];
      *out = r; return true;
    }
    case GlobalTransform::TRANSPOSE: {  // out[i][j] = g[j][i]
      RawGrid r(w, std::vector<int>(h, 0));
      for (int i = 0; i < w; ++i)
        for (int j = 0; j < h; ++j) r[i][j] = g[j][i];
      *out = r; return true;
    }
    case GlobalTransform::ANTI_TRANSPOSE: {  // transpose, then reverse columns and rows
      // transpose -> (w x h); reversing its columns (h of them) and its rows
      // (w of them) gives out[i][j] = g[h-1-j][w-1-i].
      RawGrid r(w, std::vector<int>(h, 0));
      for (int i = 0; i < w; ++i)
        for (int j = 0; j < h; ++j) r[i][j] = g[h - 1 - j][w - 1 - i];
      *out = r; return true;
    }
    default: return false;  // not one of the 7
  }
}

// infer_global_color_map_with_transform_for_all_pairs.
bool infer_global_color_map_with_transform(const std::vector<RawGrid>& ins,
                                           const std::vector<RawGrid>& outs,
                                           GlobalTransform t, ColorMap* merged_out) {
  ColorMap merged;
  for (size_t i = 0; i < ins.size(); ++i) {
    RawGrid transformed;
    if (!apply_raw_transform(ins[i], t, &transformed)) return false;
    if (rows(transformed) != rows(outs[i]) || cols(transformed) != cols(outs[i])) return false;
    ColorMap local;
    if (!infer_global_color_map(transformed, outs[i], &local)) return false;
    for (const auto& e : local.entries) {
      int old = 0;
      if (!merged.get(e.first, &old)) merged.set_new(e.first, e.second);
      else if (old != e.second) return false;
    }
  }
  *merged_out = merged;
  return true;
}

// Round half to EVEN (banker's rounding); the cast of the integer-valued result
// is exact.
long long np_round_to_int(double v) {
  double r = std::nearbyint(v);  // uses current rounding mode: round-half-even
  return static_cast<long long>(r);
}

// infer_single_object_translation(inp, out): exactly one non-zero color; the
// (dy,dx) shifting every input pixel onto the output pixel set, else none.
bool infer_single_object_translation(const RawGrid& inp, const RawGrid& out,
                                     int* dy_out, int* dx_out) {
  int h = rows(inp), w = cols(inp);
  // colors = the distinct non-zero values in inp (ascending); exactly one allowed.
  std::set<int> uniq;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) if (inp[y][x] != 0) uniq.insert(inp[y][x]);
  if (uniq.size() != 1) return false;
  int color = *uniq.begin();
  // in_pts / out_pts = the cells of that colour, in row-major order.
  std::vector<Coord> in_pts, out_pts;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) if (inp[y][x] == color) in_pts.push_back({y, x});
  int oh = rows(out), ow = cols(out);
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x) if (out[y][x] == color) out_pts.push_back({y, x});
  if (in_pts.empty() || in_pts.size() != out_pts.size()) return false;
  double in_my = 0, in_mx = 0, out_my = 0, out_mx = 0;
  for (const auto& p : in_pts) { in_my += p.first; in_mx += p.second; }
  for (const auto& p : out_pts) { out_my += p.first; out_mx += p.second; }
  in_my /= in_pts.size(); in_mx /= in_pts.size();
  out_my /= out_pts.size(); out_mx /= out_pts.size();
  int dy = static_cast<int>(np_round_to_int(out_my - in_my));
  int dx = static_cast<int>(np_round_to_int(out_mx - in_mx));
  // shifted set == target set.
  std::set<std::pair<int, int>> shifted, target;
  for (const auto& p : in_pts) shifted.insert({p.first + dy, p.second + dx});
  for (const auto& p : out_pts) target.insert({p.first, p.second});
  if (shifted != target) return false;
  *dy_out = dy; *dx_out = dx;
  return true;
}

// infer_color_candidates(pairs, max_per_src): {src: [dst,...]} ranked by
// (-count, dst), top max_per_src. Counts over the min-shape overlap of each pair.
// The result is iterated by ascending src downstream, so it is a std::map.
std::map<int, std::vector<int>> infer_color_candidates(const std::vector<RawGrid>& ins,
                                                       const std::vector<RawGrid>& outs,
                                                       int max_per_src) {
  // counts[src][dst] = n. The ranking sorts by (-count, dst), so dst breaks
  // ties and a std::map<int,int> per src suffices.
  std::map<int, std::map<int, int>> counts;
  for (size_t i = 0; i < ins.size(); ++i) {
    int h = std::min(rows(ins[i]), rows(outs[i]));
    int w = std::min(cols(ins[i]), cols(outs[i]));
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) counts[ins[i][y][x]][outs[i][y][x]]++;
  }
  std::map<int, std::vector<int>> result;
  for (const auto& sc : counts) {
    std::vector<std::pair<int, int>> ranked(sc.second.begin(), sc.second.end());  // (dst,count)
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                       if (a.second != b.second) return a.second > b.second;  // -count
                       return a.first < b.first;                              // dst asc
                     });
    std::vector<int> dsts;
    for (int k = 0; k < static_cast<int>(ranked.size()) && k < max_per_src; ++k)
      dsts.push_back(ranked[k].first);
    result[sc.first] = std::move(dsts);
  }
  return result;
}

// recolor_pair_rules(mapping): per (src != dst), a BY_COLOR -> RECOLOR rule,
// sorted by src.
std::vector<DslRule> recolor_pair_rules(const ColorMap& mapping) {
  std::vector<std::pair<int, int>> sorted_entries(mapping.entries.begin(), mapping.entries.end());
  std::sort(sorted_entries.begin(), sorted_entries.end());  // by src, then dst
  std::vector<DslRule> rules;
  for (const auto& e : sorted_entries) {
    if (e.first == e.second) continue;
    rules.push_back(rule_of(
        sel(SelectorKind::BY_COLOR, DslArgs{{"color", dv_int(e.first)}}),
        act(ActionKind::RECOLOR, DslArgs{{"color", dv_int(e.second)}})));
  }
  return rules;
}

// Structure dedup key: per rule (selector kind, sorted selector args, action
// kind, sorted action args), then the global transform, as one string.
// Selector/action args here are only int-valued (color/dy/dx/color_a/color_b),
// so the numeric arg text is exact.
std::string arg_repr(const DslArgs& args) {
  std::vector<std::pair<std::string, std::string>> kv;
  for (const auto& a : args) {
    std::string vs;
    switch (a.second.tag) {
      case DslValue::Tag::Int: vs = std::to_string(a.second.i); break;
      case DslValue::Tag::Str: vs = "s:" + a.second.s; break;
      case DslValue::Tag::Bool: vs = a.second.b ? "T" : "F"; break;
      case DslValue::Tag::Float: vs = std::to_string(a.second.f); break;
      default: vs = "?"; break;
    }
    kv.emplace_back(a.first, vs);
  }
  std::sort(kv.begin(), kv.end());
  std::string s = "{";
  for (const auto& p : kv) s += p.first + "=" + p.second + ";";
  s += "}";
  return s;
}

std::string structure_key(const DslProgram& p) {
  std::string s = "[";
  for (const auto& r : p.rules) {
    s += "(" + std::to_string(static_cast<int>(r.selector.kind)) + "," + arg_repr(r.selector.args) +
         "," + std::to_string(static_cast<int>(r.action.kind)) + "," + arg_repr(r.action.args) + ")";
  }
  s += "]|" + std::to_string(static_cast<int>(p.global_transform));
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// emit_delta_seeds (the four branches).
// ---------------------------------------------------------------------------
std::vector<TransformGenome> emit_delta_seeds(const std::vector<SeedTrainPair>& train_pairs,
                                              const GraphConfig& default_encoder,
                                              const DecoderConfig& default_decoder) {
  std::vector<TransformGenome> seeds;
  std::set<std::string> seen;

  auto mk = [&](const DslProgram& program) -> TransformGenome {
    TransformGenome g;
    g.encoder_config = default_encoder;  // a copy of the default config
    g.program = program;
    g.decoder = default_decoder;
    return g;
  };

  auto add = [&](const DslProgram& program) {
    std::string key = structure_key(program);
    if (seen.count(key)) return;
    seen.insert(key);
    seeds.push_back(mk(program));
  };

  // Materialise raw grids once.
  std::vector<RawGrid> ins, outs;
  ins.reserve(train_pairs.size());
  outs.reserve(train_pairs.size());
  for (const auto& p : train_pairs) { ins.push_back(to_raw(p.input)); outs.push_back(to_raw(p.target)); }

  // ---- 1. Global color map, identity transform ----------------------------
  ColorMap global_map;
  bool have_global = infer_global_color_map_for_all_pairs(ins, outs, &global_map);
  if (have_global) {
    std::vector<DslRule> rules = recolor_pair_rules(global_map);
    if (!rules.empty()) {
      DslProgram prog; prog.rules = rules; prog.global_transform = GlobalTransform::IDENTITY;
      add(prog);
    }
    // 1b. proper 2-cycle a<->b -> single SWAP_COLORS rule.
    std::vector<std::pair<int, int>> non_id;  // (src, dst) sorted by src
    for (const auto& e : global_map.entries) if (e.first != e.second) non_id.emplace_back(e.first, e.second);
    std::sort(non_id.begin(), non_id.end());
    if (non_id.size() == 2) {
      int s1 = non_id[0].first, d1 = non_id[0].second;
      int s2 = non_id[1].first, d2 = non_id[1].second;
      if (s1 == d2 && s2 == d1) {
        DslProgram prog;
        prog.rules.push_back(rule_of(
            sel(SelectorKind::ALL_SUBOBJECTS),
            act(ActionKind::SWAP_COLORS,
                DslArgs{{"color_a", dv_int(s1)}, {"color_b", dv_int(s2)}})));
        prog.global_transform = GlobalTransform::IDENTITY;
        add(prog);
      }
    }
  }

  // ---- 2. Global color map under a geometric transform --------------------
  static const GlobalTransform kNonIdentity[] = {
      GlobalTransform::ROT90, GlobalTransform::ROT180, GlobalTransform::ROT270,
      GlobalTransform::FLIP_H, GlobalTransform::FLIP_V,
      GlobalTransform::TRANSPOSE, GlobalTransform::ANTI_TRANSPOSE};
  for (GlobalTransform gt : kNonIdentity) {
    ColorMap mapping;
    if (!infer_global_color_map_with_transform(ins, outs, gt, &mapping)) continue;
    std::vector<DslRule> rules = recolor_pair_rules(mapping);
    if (rules.empty()) continue;  // pure transform handled elsewhere
    DslProgram prog; prog.rules = rules; prog.global_transform = gt;
    add(prog);
  }

  // ---- 3. Single-object translation ---------------------------------------
  std::set<std::pair<int, int>> translations;
  bool any_single = true;
  for (size_t i = 0; i < ins.size(); ++i) {
    int dy = 0, dx = 0;
    if (!infer_single_object_translation(ins[i], outs[i], &dy, &dx)) { any_single = false; break; }
    translations.insert({dy, dx});
  }
  if (any_single && translations.size() == 1) {
    int dy = translations.begin()->first, dx = translations.begin()->second;
    if (!(dy == 0 && dx == 0)) {
      DslProgram prog;
      prog.rules.push_back(rule_of(
          sel(SelectorKind::ALL_SUBOBJECTS),
          act(ActionKind::TRANSLATE, DslArgs{{"mode", dv_str("move")}, {"dy", dv_int(dy)}, {"dx", dv_int(dx)}})));
      prog.global_transform = GlobalTransform::IDENTITY;
      add(prog);
    }
  }

  // ---- 4. Per-src ranked color candidate (only with no global map) --------
  if (!have_global) {
    std::map<int, std::vector<int>> ranked = infer_color_candidates(ins, outs, 1);
    for (const auto& sc : ranked) {  // std::map -> ascending src
      if (sc.second.empty()) continue;
      int dst = sc.second[0];
      if (dst == sc.first) continue;
      DslProgram prog;
      prog.rules.push_back(rule_of(
          sel(SelectorKind::BY_COLOR, DslArgs{{"color", dv_int(sc.first)}}),
          act(ActionKind::RECOLOR, DslArgs{{"color", dv_int(dst)}})));
      prog.global_transform = GlobalTransform::IDENTITY;
      add(prog);
    }
  }

  return seeds;
}

}  // namespace arc
