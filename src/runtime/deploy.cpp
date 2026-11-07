// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Deployment helpers (attempt_1/attempt_2 construction, view mapping, color
// prior) for the arc_solver binary. See include/arc/deploy.hpp.

#include "arc/deploy.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "arc/codec.hpp"     // encoder_signature_repr
#include "arc/concepts.hpp"  // cdv_int, cdv_str
#include "arc/object_attrs.hpp"
#include "arc/encode.hpp"    // calculate_background_heuristic
#include "arc/fitness.hpp"   // predict_grid, extract_objects, object_distance
#include "arc/pyrepr.hpp"    // hash_program
#include "arc/scene_cache.hpp"
#include "arc/seed_enum.hpp" // build_encoder_seed_bank

namespace arc {

Grid invert_view_grid(const Grid& g, const std::string& view) {
  if (view == "rot90") return apply_view_grid(g, "rot270");
  if (view == "rot270") return apply_view_grid(g, "rot90");
  return apply_view_grid(g, view);  // self-inverse
}

std::vector<std::string> parse_view_continue(const std::string& raw) {
  std::string s;
  for (char c : raw) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  size_t a = s.find_first_not_of(" \t");
  size_t b = s.find_last_not_of(" \t");
  s = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
  if (s.empty() || s == "0") return {};
  if (s == "1") return {"transpose", "fliplr", "flipud"};
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    size_t ta = tok.find_first_not_of(" \t");
    size_t tb = tok.find_last_not_of(" \t");
    if (ta != std::string::npos) tok = tok.substr(ta, tb - ta + 1);
    else tok.clear();
    if (!tok.empty()) out.push_back(tok);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

std::vector<int> observed_output_colors(const std::vector<TrainPair>& pairs) {
  std::set<int> seen;
  for (const TrainPair& p : pairs)
    for (int c : p.target.cells()) seen.insert(c);
  return std::vector<int>(seen.begin(), seen.end());
}

namespace {
bool grids_equal(const Grid& a, const Grid& b) {
  return a.height() == b.height() && a.width() == b.width() && a.cells() == b.cells();
}

// Safety bound on the candidate pool.
constexpr int GUESS_MAX_POOL = 16;

// Dedup identity: program hash + decoder + encoder signature.
std::string genome_key(const TransformGenome& g) {
  return hash_program(g.program) + "\x1f" + g.decoder.background_policy + "|" +
         g.decoder.paint_order + "|" + g.decoder.overlap_policy + "|" +
         g.decoder.postprocess + "\x1f" + encoder_signature_repr(g.encoder_config);
}

// ---- answer-free sanity checks on a test prediction -------------------------
// A prediction is suspect when it (1) copies the test input although no train
// output equals its input, (2) breaks the size relation every train pair
// follows (same size, one fixed size, or an integer scale up or down), or
// (3) uses a colour found neither in the train outputs nor in the test input.
// Measured on earlier runs: these flags hit wrong answers almost only.
bool size_rule_ok(const std::vector<TrainPair>& tp, const Grid& in, const Grid& pred) {
  if (tp.empty()) return true;
  bool same = true, fixed = true;
  for (const TrainPair& p : tp) {
    same = same && p.input.height() == p.target.height() && p.input.width() == p.target.width();
    fixed = fixed && p.target.height() == tp[0].target.height() && p.target.width() == tp[0].target.width();
  }
  if (same) return pred.height() == in.height() && pred.width() == in.width();
  if (fixed) return pred.height() == tp[0].target.height() && pred.width() == tp[0].target.width();
  for (int fy = 1; fy <= 4; ++fy)
    for (int fx = 1; fx <= 4; ++fx) {
      bool up = true, down = fy > 1 || fx > 1;
      for (const TrainPair& p : tp) {
        up = up && p.target.height() == p.input.height() * fy && p.target.width() == p.input.width() * fx;
        down = down && p.input.height() % fy == 0 && p.input.width() % fx == 0 &&
               p.target.height() == p.input.height() / fy && p.target.width() == p.input.width() / fx;
      }
      if (up) return pred.height() == in.height() * fy && pred.width() == in.width() * fx;
      if (down) return pred.height() == in.height() / fy && pred.width() == in.width() / fx;
    }
  return true;  // no shared relation: nothing to check
}

bool prediction_sane(const std::vector<TrainPair>& tp, const Grid& in, const Grid& pred) {
  if (pred.height() <= 0 || pred.width() <= 0) return false;
  bool any_identity = false;
  for (const TrainPair& p : tp) any_identity = any_identity || grids_equal(p.input, p.target);
  if (!any_identity && grids_equal(pred, in)) return false;
  if (!size_rule_ok(tp, in, pred)) return false;
  std::set<int> allowed(in.cells().begin(), in.cells().end());
  for (const TrainPair& p : tp) allowed.insert(p.target.cells().begin(), p.target.cells().end());
  for (int c : pred.cells())
    if (!allowed.count(c)) return false;
  // ARC_SANITY_PLUS=1: when every train pair keeps the input's size and only
  // adds cells (every non-background input cell stays) or only removes cells
  // (every output cell is the input's or background), the prediction must too.
  static const bool plus = [] {
    const char* e = std::getenv("ARC_SANITY_PLUS");
    return e && std::string(e) == "1";
  }();
  if (plus && pred.height() == in.height() && pred.width() == in.width()) {
    bool adds = true, removes = true;
    for (const TrainPair& p : tp) {
      if (p.target.height() != p.input.height() || p.target.width() != p.input.width()) {
        adds = removes = false;
        break;
      }
      auto h = calculate_background_heuristic(p.input);
      int bg = h ? *h : 0;
      const auto& a = p.input.cells();
      const auto& b = p.target.cells();
      for (size_t k = 0; k < a.size(); ++k) {
        if (a[k] != bg && b[k] != a[k]) adds = false;
        if (b[k] != a[k] && b[k] != bg) removes = false;
      }
    }
    auto h = calculate_background_heuristic(in);
    int bg = h ? *h : 0;
    const auto& a = in.cells();
    const auto& b = pred.cells();
    for (size_t k = 0; k < a.size(); ++k) {
      if (adds && a[k] != bg && b[k] != a[k]) return false;
      if (removes && b[k] != a[k] && b[k] != bg) return false;
    }
  }
  return true;
}

Grid predict(const TransformGenome& g, const Grid& in, SceneCache* sc) {
  return predict_grid(in, g.program, g.encoder_config, g.decoder, sc);
}

bool is_train_exact(const TransformGenome& g, const std::vector<TrainPair>& train_pairs, SceneCache* sc) {
  if (train_pairs.empty()) return false;
  for (const TrainPair& p : train_pairs)
    if (!grids_equal(predict(g, p.input, sc), p.target)) return false;
  return true;
}

// Mean over test inputs of the encoder-free object-graph distance; a shape
// mismatch on any input is maximal distance.
double behavioral_dist(const std::vector<std::vector<FitObject>>& objs_a,
                       const std::vector<std::vector<FitObject>>& objs_b,
                       const std::vector<std::pair<int, int>>& shapes_a,
                       const std::vector<std::pair<int, int>>& shapes_b) {
  size_t n = objs_a.size();
  if (n == 0) return 0.0;
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (shapes_a[i] != shapes_b[i]) {
      total += 1.0;
    } else {
      total += object_distance(objs_a[i], objs_b[i], shapes_a[i].first, shapes_a[i].second);
    }
  }
  return total / static_cast<double>(n);
}

// Greedy farthest-point (max-min) selection; cands[0] (the winner) is always
// guess #1.
std::vector<TransformGenome> diversity_select(const std::vector<TransformGenome>& cands,
                                              const std::vector<Grid>& test_inputs, int k,
                                              SceneCache* sc) {
  if (cands.size() <= 1 || test_inputs.empty()) {
    std::vector<TransformGenome> out;
    for (size_t i = 0; i < cands.size() && static_cast<int>(i) < k; ++i) out.push_back(cands[i]);
    return out;
  }
  size_t nc = cands.size(), nt = test_inputs.size();
  std::vector<std::vector<std::pair<int, int>>> shapes(nc, std::vector<std::pair<int, int>>(nt));
  std::vector<std::vector<std::vector<FitObject>>> objs(nc, std::vector<std::vector<FitObject>>(nt));
  for (size_t i = 0; i < nc; ++i)
    for (size_t t = 0; t < nt; ++t) {
      Grid p = predict(cands[i], test_inputs[t], sc);
      shapes[i][t] = {p.height(), p.width()};
      objs[i][t] = extract_objects(p);
    }
  std::vector<size_t> chosen{0};
  while (static_cast<int>(chosen.size()) < k && chosen.size() < nc) {
    long long best_i = -1;
    double best_score = -1.0;
    for (size_t i = 0; i < nc; ++i) {
      if (std::find(chosen.begin(), chosen.end(), i) != chosen.end()) continue;
      double d = -1.0;
      for (size_t j : chosen) {
        double dj = behavioral_dist(objs[i], objs[j], shapes[i], shapes[j]);
        if (d < 0.0 || dj < d) d = dj;  // min distance to any chosen guess
      }
      if (d > best_score) {  // strict > keeps the smallest index on ties
        best_score = d;
        best_i = static_cast<long long>(i);
      }
    }
    if (best_i < 0) break;
    chosen.push_back(static_cast<size_t>(best_i));
  }
  std::vector<TransformGenome> out;
  for (size_t i : chosen) out.push_back(cands[i]);
  return out;
}

// Guess pool: winner first (pass@1 invariant), distinct
// finalists, then encoder re-pairings of the winner that stay train-exact, then
// diversity-select up to k.
std::vector<TransformGenome> build_guess_pool(
    const TransformGenome& winner,
    const std::vector<std::pair<TransformGenome, TransformFitnessFull>>& finalists,
    const std::vector<TrainPair>& train_pairs, const std::vector<Grid>& test_inputs,
    int k, bool widen, SceneCache* sc) {
  std::set<std::string> seen;
  std::vector<TransformGenome> cands;
  seen.insert(genome_key(winner));
  cands.push_back(winner);
  for (const auto& gf : finalists) {
    std::string key = genome_key(gf.first);
    if (seen.count(key)) continue;
    seen.insert(key);
    cands.push_back(gf.first);
  }
  // Layer 1 -- encoder re-pairings of the winning program/decoder.
  if (widen) {
    std::vector<GraphConfig> bank = build_encoder_seed_bank();
    std::string winner_sig = encoder_signature_repr(winner.encoder_config);
    for (const GraphConfig& enc : bank) {
      if (static_cast<int>(cands.size()) >= GUESS_MAX_POOL) break;
      if (encoder_signature_repr(enc) == winner_sig) continue;
      TransformGenome variant = winner;
      variant.encoder_config = enc;
      std::string key = genome_key(variant);
      if (seen.count(key)) continue;
      if (is_train_exact(variant, train_pairs, sc)) {
        seen.insert(key);
        cands.push_back(std::move(variant));
      }
    }
  }
  // Layer 2 -- diversity-select the final guess set.
  return diversity_select(cands, test_inputs, k, sc);
}
}  // namespace

std::vector<Attempts> build_attempts(const TaskResultFull& res,
                                     const std::vector<TrainPair>& train_pairs,
                                     const std::vector<Grid>& test_inputs,
                                     SceneCache* sc) {
  std::vector<Attempts> out;
  out.reserve(test_inputs.size());

  // Train background: when the heuristic gives every train input the same
  // background b, a test input whose heuristic says otherwise is encoded with b
  // as background: one that holds b (say, a grid framed by another colour), and
  // one without b (every cell is then part of an object, as b never shows).
  int train_bg = -1;
  for (const TrainPair& p : train_pairs) {
    auto h = calculate_background_heuristic(p.input);
    int b = h ? *h : -2;
    if (train_bg == -1) train_bg = b;
    else if (train_bg != b) train_bg = -2;
  }
  auto enc_for = [&](const GraphConfig& cfg, const Grid& in) {
    GraphConfig c = cfg;
    if (train_bg < 0 || c.background_color_override >= 0 || c.background_mode != "auto") return c;
    auto h = calculate_background_heuristic(in);
    if (h && *h != train_bg) c.background_color_override = train_bg;
    return c;
  };

  // attempt_1: primary genome on each test input.
  std::vector<Grid> primary;
  primary.reserve(test_inputs.size());
  for (const Grid& in : test_inputs) {
    primary.push_back(predict_grid(in, res.primary_genome.program,
                                   enc_for(res.primary_genome.encoder_config, in),
                                   res.primary_genome.decoder, sc));
  }

  // ARC_DUMP_CANDS=1: keep every candidate prediction for offline policy replay.
  const char* dump_env = std::getenv("ARC_DUMP_CANDS");
  const bool dump = dump_env && std::string(dump_env) == "1";
  std::vector<std::pair<std::string, std::vector<Grid>>> dumped;
  if (dump) dumped.emplace_back("primary", primary);

  // attempt_2 candidate 1: VIEW-ENSEMBLE.
  // Rank view solvers by (loo_fail, complexity); execute each, back-map to task
  // space, and VOTE over full prediction tuples. attempt_2 = the most-voted tuple
  // DISTINCT from the primary (ties -> most general view = smallest ranked index).
  // CONSENSUS SWAP: if that winning alt has >=2 votes and NO view backs the
  // primary, it is promoted to attempt_1 (primary demoted to attempt_2); the
  // pass@2 set is unchanged.
  auto sig_of = [](const std::vector<Grid>& preds) {
    std::string s;
    for (const Grid& g : preds) {
      s += std::to_string(g.height());
      s += 'x';
      s += std::to_string(g.width());
      s += ':';
      for (int c : g.cells()) { s += std::to_string(c); s += ','; }
      s += ';';
    }
    return s;
  };
  std::vector<Grid> alt;
  bool swap_consensus = false;
  std::vector<std::vector<Grid>> extra;  // further candidates, in rank order
  {
    std::vector<const ViewSolver*> ranked;
    ranked.reserve(res.view_solvers.size());
    for (const ViewSolver& vs : res.view_solvers) ranked.push_back(&vs);
    std::stable_sort(ranked.begin(), ranked.end(),
        [](const ViewSolver* a, const ViewSolver* b) {
          if (a->fit.loo_fail != b->fit.loo_fail) return a->fit.loo_fail < b->fit.loo_fail;
          return a->fit.complexity < b->fit.complexity;
        });
    std::map<std::string, int> votes, first_seen;
    std::map<std::string, std::vector<Grid>> by_sig;
    int vi = 0;
    for (const ViewSolver* vs : ranked) {
      std::vector<Grid> vpreds;
      vpreds.reserve(test_inputs.size());
      for (const Grid& in : test_inputs) {
        Grid vin = apply_view_grid(in, vs->view);
        Grid vp = predict_grid(vin, vs->genome.program, enc_for(vs->genome.encoder_config, vin), vs->genome.decoder, sc);
        vpreds.push_back(invert_view_grid(vp, vs->view));
      }
      if (dump)
        dumped.emplace_back("view:" + vs->view + ":loo=" + std::to_string(vs->fit.loo_fail) +
                                ":c=" + std::to_string(vs->fit.complexity),
                            vpreds);
      std::string s = sig_of(vpreds);
      votes[s] += 1;
      if (!first_seen.count(s)) first_seen[s] = vi;
      if (!by_sig.count(s)) by_sig[s] = std::move(vpreds);
      ++vi;
    }
    std::string primary_sig = sig_of(primary);
    int primary_support = votes.count(primary_sig) ? votes[primary_sig] : 0;
    // best alt = most votes, ties -> smallest first_seen.
    bool have_best = false;
    std::string best;
    int best_votes = -1, best_first = 0;
    for (const auto& kv : votes) {
      if (kv.first == primary_sig) continue;
      int v = kv.second, fs = first_seen[kv.first];
      if (!have_best || v > best_votes || (v == best_votes && fs < best_first)) {
        have_best = true; best = kv.first; best_votes = v; best_first = fs;
      }
    }
    std::vector<std::pair<int, std::string>> order;  // (first_seen, sig)
    for (const auto& kv : first_seen) order.emplace_back(kv.second, kv.first);
    std::sort(order.begin(), order.end());
    for (const auto& o : order) extra.push_back(by_sig[o.second]);
    if (have_best) {
      alt = by_sig[best];
      swap_consensus = (best_votes >= 2 && primary_support == 0);
    }
  }

  // attempt_2 candidate 2 (fallback when NO view disagrees): the diverse guess
  // pool. pool[1] is used as attempt_2 even when it predicts the same as the
  // primary. ARC_GUESS_POOL=0 switches to the "first distinct finalist" rule.
  if (alt.empty()) {
    const char* gp_env = std::getenv("ARC_GUESS_POOL");
    bool use_pool = !(gp_env && std::string(gp_env) == "0");
    if (use_pool) {
      std::vector<TransformGenome> pool = build_guess_pool(
          res.primary_genome, res.finalists, train_pairs, test_inputs,
          /*k=*/5, /*widen=*/res.solved, sc);
      const TransformGenome& g2 = pool.size() > 1 ? pool[1] : pool[0];
      alt.reserve(test_inputs.size());
      for (const Grid& in : test_inputs) alt.push_back(predict(g2, in, sc));
    } else {
      // ARC_GUESS_POOL=0: the first finalist whose prediction differs from the
      // primary.
      for (const auto& fin : res.finalists) {
        std::vector<Grid> fpreds;
        fpreds.reserve(test_inputs.size());
        for (const Grid& in : test_inputs)
          fpreds.push_back(predict(fin.first, in, sc));
        bool distinct = false;
        for (size_t i = 0; i < fpreds.size(); ++i)
          if (!grids_equal(fpreds[i], primary[i])) { distinct = true; break; }
        if (distinct) {
          alt = std::move(fpreds);
          break;
        }
      }
    }
  }

  // ARC_SEL_VOTE=1: selector-equivalence vote. Each variant of the solved
  // primary that swaps one rule's selector for another and stays exact on the
  // train pairs predicts the test inputs. The most common prediction becomes
  // attempt_1 and, when it differs, the primary's own prediction attempt_2.
  // ARC_SEL_VOTE=dump (with ARC_DUMP_CANDS=1) only records the vote.
  const char* sv_env = std::getenv("ARC_SEL_VOTE");
  const std::string sv = sv_env ? sv_env : "";
  if ((sv == "1" || (sv == "dump" && dump)) && res.solved && !res.primary_genome.program.rules.empty()) {
    std::vector<DslSelector> sels;
    for (const auto& leaf : active_selector_leaves()) {
      SelectorKind k = leaf.first;
      const std::string& variant = leaf.second;
      if (k == SelectorKind::BY_ATTR || k == SelectorKind::BY_COLOR) continue;
      DslSelector s{k, {}};
      if (!variant.empty()) s.args.emplace_back(selector_variant_key(k), cdv_str(variant));
      sels.push_back(std::move(s));
    }
    for (int c = 0; c < 10; ++c) sels.push_back(DslSelector{SelectorKind::BY_COLOR, {{"color", cdv_int(c)}}});
    for (const std::string& a : object_attr_names()) {
      for (const char* op : {"max", "min", "unique"})
        sels.push_back(DslSelector{SelectorKind::BY_ATTR, {{"attr", cdv_str(a)}, {"op", cdv_str(op)}, {"value", cdv_int(0)}}});
      for (int v : {0, 1, 2})
        sels.push_back(DslSelector{SelectorKind::BY_ATTR, {{"attr", cdv_str(a)}, {"op", cdv_str("eq")}, {"value", cdv_int(v)}}});
      sels.push_back(DslSelector{SelectorKind::BY_ATTR, {{"attr", cdv_str(a)}, {"op", cdv_str("ne")}, {"value", cdv_int(0)}}});
    }
    std::string psig = sig_of(primary);
    std::map<std::string, int> votes{{psig, 1}};
    std::map<std::string, std::vector<Grid>> by_sig{{psig, primary}};
    std::set<std::string> seen{hash_program(res.primary_genome.program)};
    for (size_t r = 0; r < res.primary_genome.program.rules.size(); ++r)
      for (const DslSelector& s : sels) {
        TransformGenome v = res.primary_genome;
        v.program.rules[r].selector = s;
        if (!seen.insert(hash_program(v.program)).second || !is_train_exact(v, train_pairs, sc)) continue;
        std::vector<Grid> vp;
        for (const Grid& in : test_inputs)
          vp.push_back(predict_grid(in, v.program, enc_for(v.encoder_config, in), v.decoder, sc));
        std::string s2 = sig_of(vp);
        if (++votes[s2] == 1) by_sig[s2] = std::move(vp);
      }
    std::string best = psig;
    for (const auto& kv : votes)
      if (kv.second > votes[best]) best = kv.first;
    if (dump)
      for (const auto& kv : votes)
        dumped.emplace_back("selvote:" + std::to_string(kv.second) + (kv.first == psig ? ":primary" : ""), by_sig[kv.first]);
    if (sv == "1" && best != psig) {
      alt = primary;
      primary = by_sig[best];
      swap_consensus = false;
    }
  }

  // ARC_FIT_VOTE=1: vote over the finalists that fit every train pair (and the
  // view solvers).
  // attempt_1 = the test prediction most of them give (ties -> the better
  // ranked finalist), attempt_2 = the runner-up, else the former primary.
  const char* fv_env = std::getenv("ARC_FIT_VOTE");
  if (fv_env && std::string(fv_env) == "1" && res.solved) {
    std::map<std::string, int> votes, first;
    std::map<std::string, std::vector<Grid>> by_sig;
    int fi = 0;
    for (const auto& gf : res.finalists) {
      if (!gf.second.solved) continue;
      std::vector<Grid> fp;
      for (const Grid& in : test_inputs)
        fp.push_back(predict_grid(in, gf.first.program, enc_for(gf.first.encoder_config, in), gf.first.decoder, sc));
      bool sane = true;  // an answer that fails the sanity checks gets no vote
      for (size_t i = 0; i < fp.size(); ++i) sane = sane && prediction_sane(train_pairs, test_inputs[i], fp[i]);
      if (!sane) continue;
      std::string s = sig_of(fp);
      if (++votes[s] == 1) {
        first[s] = fi;
        by_sig[s] = std::move(fp);
      }
      ++fi;
    }
    // The view solvers (programs fitted on rotated / flipped train pairs) vote too.
    for (const ViewSolver& vs : res.view_solvers) {
      std::vector<Grid> vp;
      for (const Grid& in : test_inputs) {
        Grid vin = apply_view_grid(in, vs.view);
        vp.push_back(invert_view_grid(
            predict_grid(vin, vs.genome.program, enc_for(vs.genome.encoder_config, vin), vs.genome.decoder, sc), vs.view));
      }
      bool sane = true;
      for (size_t i = 0; i < vp.size(); ++i) sane = sane && prediction_sane(train_pairs, test_inputs[i], vp[i]);
      if (!sane) continue;
      std::string s = sig_of(vp);
      if (++votes[s] == 1) {
        first[s] = fi;
        by_sig[s] = std::move(vp);
      }
      ++fi;
    }
    if (!votes.empty()) {
      std::vector<std::string> order;
      for (const auto& kv : votes) order.push_back(kv.first);
      std::sort(order.begin(), order.end(), [&](const std::string& a, const std::string& b) {
        return votes[a] != votes[b] ? votes[a] > votes[b] : first[a] < first[b];
      });
      std::string old_sig = sig_of(primary);
      std::vector<Grid> old_primary = primary;
      primary = by_sig[order[0]];
      if (order.size() > 1) alt = by_sig[order[1]];
      else if (old_sig != order[0]) alt = old_primary;
      swap_consensus = false;
    }
  }

  if (dump) {
    int fi = 0;
    for (const auto& gf : res.finalists) {
      if (fi >= 30) break;
      std::vector<Grid> fp;
      for (const Grid& in : test_inputs)
        fp.push_back(predict_grid(in, gf.first.program, enc_for(gf.first.encoder_config, in), gf.first.decoder, sc));
      dumped.emplace_back("fin:" + std::to_string(fi++) + ":solved=" + std::to_string(gf.second.solved ? 1 : 0) +
                              ":c=" + std::to_string(gf.second.complexity),
                          std::move(fp));
    }
    std::vector<TransformGenome> pool = build_guess_pool(
        res.primary_genome, res.finalists, train_pairs, test_inputs, /*k=*/5, /*widen=*/res.solved, sc);
    for (size_t pi = 0; pi < pool.size(); ++pi) {
      std::vector<Grid> pp;
      for (const Grid& in : test_inputs) pp.push_back(predict(pool[pi], in, sc));
      dumped.emplace_back("pool:" + std::to_string(pi), std::move(pp));
    }
  }

  bool pool_added = false;
  for (size_t i = 0; i < test_inputs.size(); ++i) {
    Attempts at;
    Grid a1 = primary[i];
    Grid a2 = alt.empty() ? primary[i] : alt[i];  // fallback: copy attempt_1
    if (swap_consensus) std::swap(a1, a2);         // consensus theory -> attempt_1
    // Sanity filter: when an attempt fails the checks, take the first sane
    // distinct candidate in rank order (views, then the guess pool); suspect
    // candidates are kept only as a last resort. No flag -> no change.
    bool ok1 = prediction_sane(train_pairs, test_inputs[i], a1);
    bool ok2 = prediction_sane(train_pairs, test_inputs[i], a2);
    if (!ok1 || !ok2) {
      if (!pool_added) {
        pool_added = true;
        std::vector<TransformGenome> pool = build_guess_pool(
            res.primary_genome, res.finalists, train_pairs, test_inputs, /*k=*/5, /*widen=*/res.solved, sc);
        for (const TransformGenome& g : pool) {
          std::vector<Grid> preds;
          for (const Grid& in : test_inputs) preds.push_back(predict(g, in, sc));
          extra.push_back(std::move(preds));
        }
      }
      std::vector<Grid> cands{a1, a2};
      for (const auto& e : extra) cands.push_back(e[i]);
      std::vector<Grid> sane, suspect;
      for (const Grid& g : cands) {
        auto& bucket = prediction_sane(train_pairs, test_inputs[i], g) ? sane : suspect;
        bool dup = false;
        for (const Grid& h : sane) dup = dup || grids_equal(g, h);
        for (const Grid& h : suspect) dup = dup || grids_equal(g, h);
        if (!dup) bucket.push_back(g);
      }
      sane.insert(sane.end(), suspect.begin(), suspect.end());
      a1 = sane[0];
      a2 = sane.size() > 1 ? sane[1] : sane[0];
    }
    at.attempt_1 = std::move(a1);
    at.attempt_2 = std::move(a2);
    for (const auto& lp : dumped) at.candidates.emplace_back(lp.first, lp.second[i]);
    out.push_back(std::move(at));
  }
  return out;
}

}  // namespace arc
