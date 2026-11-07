// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Registry and shared helpers of the concept primitives (see arc/concepts.hpp).

#include "arc/concepts.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>

#include "arg_coerce.hpp"
#include "scene_ops.hpp"

namespace arc {

// ---- registry ---------------------------------------------------------------
// Each primitive branch declares its op here and adds it to the list.
// CONCEPT_DECLS
ConceptOp count_bar_op();
ConceptOp sort_objects_op();
ConceptOp slide_until_blocked_op();
ConceptOp copy_into_frames_op();
ConceptOp stamp_at_markers_op();
ConceptOp repeat_by_markers_op();
ConceptOp denoise_op();
ConceptOp ray_until_blocked_op();
ConceptOp center_on_op();
ConceptOp toggle_fill_op();
ConceptOp attr_paint_op();
ConceptOp concat_objects_op();
ConceptOp patch_op();
ConceptOp extract_op();
ConceptOp bring_to_front_op();
ConceptOp complete_copies_op();
ConceptOp keep_enclosed_op();
ConceptOp complete_regularity_op();
ConceptOp mark_side_op();
ConceptOp tile_gap_op();
ConceptOp transfer_pattern_op();

const std::vector<ConceptOp>& concept_ops() {
  static const std::vector<ConceptOp> ops = {
      // CONCEPT_OPS
      count_bar_op(),
      sort_objects_op(),
      slide_until_blocked_op(),
      copy_into_frames_op(),
      stamp_at_markers_op(),
      repeat_by_markers_op(),
      denoise_op(),
      ray_until_blocked_op(),
      center_on_op(),
      toggle_fill_op(),
      attr_paint_op(),
      concat_objects_op(),
      patch_op(),
      extract_op(),
      bring_to_front_op(),
      complete_copies_op(),
      keep_enclosed_op(),
      complete_regularity_op(),
      mark_side_op(),
      tile_gap_op(),
      transfer_pattern_op(),
  };
  return ops;
}

const ConceptOp* find_concept_op(ActionKind kind) {
  for (const ConceptOp& op : concept_ops())
    if (op.kind == kind) return &op;
  return nullptr;
}

bool concept_alias(const std::string& name, ActionKind* kind, DslArgs* args) {
  auto get = [&](const char* key, const char* def) { return arg_str(*args, key, def); };
  DslArgs out;
  if (name == "keep_by_rays") {
    *kind = ActionKind::KEEP_ENCLOSED;
    out.emplace_back("side", cdv_str(get("keep", "outside") == "inside" ? "rays_inside" : "rays_outside"));
    out.emplace_back("frame", cdv_str(get("frame", "drop") == "keep" ? "keep" : "drop"));
    out.emplace_back("hits", cdv_int(arg_int(*args, "hits", 4)));
  } else if (name == "keep_by_containment") {
    *kind = ActionKind::KEEP_ENCLOSED;
    out.emplace_back("side", cdv_str(get("when_inside", "markers") == "markers" ? "bbox_markers" : "bbox_frame"));
    if (get("edge", "in") == "out") out.emplace_back("edge", cdv_str("out"));
  } else if (name == "paint_from_marker") {
    *kind = ActionKind::MARK_SIDE;
    out.emplace_back("side", cdv_str("marker_" + get("dir", "up")));
  } else if (name == "symmetry_complete") {
    *kind = ActionKind::COMPLETE_REGULARITY;
    out.emplace_back("rule", cdv_str("sym_" + get("axis", "both")));
  } else if (name == "keep_extreme") {
    *kind = ActionKind::EXTRACT;
    out.emplace_back("what", cdv_str("extreme_" + get("side", "bottom")));
    out.emplace_back("out", cdv_str(get("out", "grid")));
  } else if (name == "weave") {
    *kind = ActionKind::BRING_TO_FRONT;
    out.emplace_back("keep", cdv_str("weave_" + get("top", "vertical")));
  } else if (name == "jump_over") {
    *kind = ActionKind::SLIDE_UNTIL_BLOCKED;
    out.emplace_back("dir", cdv_str("jump_" + get("dir", "down")));
  } else if (name == "extend_dashed_lines") {
    *kind = ActionKind::COMPLETE_REGULARITY;
    out.emplace_back("rule", cdv_str(as_bool(args_get(*args, "h_last"), true) ? "dashes_h_last" : "dashes_v_last"));
  } else if (name == "complete_frame") {
    *kind = ActionKind::COMPLETE_REGULARITY;
    out.emplace_back("rule", cdv_str("frame"));
  } else if (name == "sort_panels") {
    *kind = ActionKind::SORT_OBJECTS;
    out.emplace_back("key", cdv_str("panel_" + get("key", "minority")));
    for (auto& kv : *args)
      if (kv.first != "key") out.push_back(kv);
  } else if (name == "orient_mark") {
    *kind = ActionKind::MARK_SIDE;
    out.emplace_back("side", cdv_str("orient_" + get("by", "fill")));
    for (auto& kv : *args)
      if (kv.first != "by") out.push_back(kv);
  } else if (name == "recolor_rel") {
    *kind = ActionKind::ATTR_PAINT;
    out.emplace_back("attr", cdv_str("rel_" + get("source", "nearest")));
    for (auto& kv : *args)
      if (kv.first != "source") out.push_back(kv);
  } else if (name == "delete_center") {
    *kind = ActionKind::CENTER_ON;
    out.emplace_back("mode", cdv_str("delete"));
  } else if (name == "extract_unique" || name == "extract_tile" || name == "concat_panels" || name == "crop_hole") {
    *kind = ActionKind::EXTRACT;
    out.emplace_back("what", cdv_str(name == "extract_unique" ? "unique" : name == "extract_tile" ? "tile"
                                     : name == "concat_panels" ? "panels" : "hole"));
    if (name == "crop_hole") out.emplace_back("pick", cdv_str(get("pick", "largest")));
    if (name == "concat_panels" && get("mode", "join") == "same") out.emplace_back("mode", cdv_str("same"));
  } else {
    return false;
  }
  *args = std::move(out);
  return true;
}

void apply_concept_op(const ConceptOp& op, Scene& scene, const std::vector<std::string>& selected_ids,
                      const DslAction& action) {
  static const bool scope = [] {
    const char* e = std::getenv("ARC_CONCEPT_SCOPE");
    return e && std::string(e) == "1";
  }();
  bool own = op.kind == ActionKind::RAY_UNTIL_BLOCKED ||
             (op.kind == ActionKind::SLIDE_UNTIL_BLOCKED && arg_str(action.args, "dir", "down").rfind("jump_", 0) != 0) ||
             (op.kind == ActionKind::MARK_SIDE && arg_str(action.args, "side", "left").rfind("marker_", 0) != 0 &&
              arg_str(action.args, "side", "left").rfind("orient_", 0) != 0) ||
             (op.kind == ActionKind::ATTR_PAINT && arg_str(action.args, "attr", "orient").rfind("rel_", 0) == 0);
  std::vector<size_t> subs = scene.get_nodes_by_type("SubObject", false);
  if (!scope || own || selected_ids.empty() || selected_ids.size() >= subs.size()) {
    op.apply(scene, selected_ids, action);
    return;
  }
  CGrid before = concept_render(scene);
  int H = static_cast<int>(before.size()), W = H ? static_cast<int>(before[0].size()) : 0;
  std::vector<std::vector<char>> in(H, std::vector<char>(W, 0));
  std::set<std::string> ids(selected_ids.begin(), selected_ids.end());
  for (size_t i : subs) {
    const Node& n = scene.nodes()[i];
    if (!ids.count(n.node_id) || n.pixel_list.empty()) continue;
    Box b = bbox_of(n.pixel_list);
    for (int y = std::max(0, b.y0); y <= std::min(H - 1, b.y1); ++y)
      for (int x = std::max(0, b.x0); x <= std::min(W - 1, b.x1); ++x) in[y][x] = 1;
  }
  op.apply(scene, selected_ids, action);
  CGrid after = concept_render(scene);
  if (static_cast<int>(after.size()) != H || (H && static_cast<int>(after[0].size()) != W)) return;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (!in[y][x]) after[y][x] = before[y][x];
  concept_install(scene, after);
}

// ---- helpers ----------------------------------------------------------------
int concept_bg(const Scene& scene) { return scene.has_background() ? scene.background() : 0; }

CGrid concept_render(const Scene& scene) {
  int h = scene.height(), w = scene.width(), bg = concept_bg(scene);
  CGrid g(h, std::vector<int>(w, bg));
  for (size_t i : scene.get_nodes_by_type("SubObject", true)) {
    const Node& n = scene.nodes()[i];
    if (!n.color.has_value() || *n.color == bg) continue;
    for (const auto& px : n.pixel_list)
      if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
        g[px.first][px.second] = *n.color;
  }
  return g;
}

CGrid concept_render_full(const Scene& scene) {
  CGrid g = concept_render(scene);
  if (!scene.has_grid() || scene.grid().height() != scene.height() || scene.grid().width() != scene.width())
    return g;
  for (const Coord& c : scene.grid_line_pixels())
    if (c.first >= 0 && c.first < scene.height() && c.second >= 0 && c.second < scene.width())
      g[c.first][c.second] = scene.grid().at(c.first, c.second);
  return g;
}

void concept_install(Scene& scene, const CGrid& g) {
  int h = static_cast<int>(g.size()), w = h ? static_cast<int>(g[0].size()) : 0, bg = concept_bg(scene);
  std::vector<std::tuple<int, int, int>> px;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (g[y][x] != bg) px.emplace_back(y, x, g[y][x]);
  reset_scene_subobjects(scene, px, h, w);
}

DslValue cdv_int(long long v) { DslValue d; d.tag = DslValue::Tag::Int; d.i = v; return d; }
DslValue cdv_str(const std::string& v) { DslValue d; d.tag = DslValue::Tag::Str; d.s = v; return d; }
DslValue cdv_bool(bool v) { DslValue d; d.tag = DslValue::Tag::Bool; d.b = v; return d; }

int pick_int(ArcRandom& rng, const std::vector<int>& xs) { return xs[rng.choice_index(xs.size())]; }
std::string pick_str(ArcRandom& rng, const std::vector<std::string>& xs) {
  return xs[rng.choice_index(xs.size())];
}

std::vector<std::vector<Coord>> concept_components(const CGrid& g, int bg, bool diag, bool mono) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  std::vector<std::vector<Coord>> out;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] == bg || seen[y][x]) continue;
      std::vector<Coord> st{{y, x}}, comp;
      seen[y][x] = 1;
      while (!st.empty()) {
        Coord c = st.back();
        st.pop_back();
        comp.push_back(c);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if ((dy == 0 && dx == 0) || (!diag && dy != 0 && dx != 0)) continue;
            int p = c.first + dy, q = c.second + dx;
            if (p < 0 || p >= H || q < 0 || q >= W || seen[p][q] || g[p][q] == bg) continue;
            if (mono && g[p][q] != g[y][x]) continue;
            seen[p][q] = 1;
            st.push_back({p, q});
          }
      }
      std::sort(comp.begin(), comp.end());
      out.push_back(comp);
    }
  return out;
}

std::vector<std::vector<Coord>> concept_dotted_groups(const CGrid& g, int bg, bool diag) {
  std::vector<std::vector<Coord>> comps = concept_components(g, bg, diag, true), out, lone;
  for (auto& c : comps) (c.size() == 1 ? lone : out).push_back(std::move(c));
  std::vector<char> used(lone.size(), 0);
  for (size_t i = 0; i < lone.size(); ++i) {
    if (used[i]) continue;
    used[i] = 1;
    std::vector<Coord> grp = lone[i];
    for (size_t k = 0; k < grp.size(); ++k)
      for (size_t j = 0; j < lone.size(); ++j) {
        const Coord &a = grp[k], &b = lone[j][0];
        if (used[j] || g[a.first][a.second] != g[b.first][b.second] || std::abs(a.first - b.first) > 2 ||
            std::abs(a.second - b.second) > 2)
          continue;
        used[j] = 1;
        grp.push_back(b);
      }
    std::sort(grp.begin(), grp.end());
    out.push_back(std::move(grp));
  }
  std::sort(out.begin(), out.end());
  return out;
}

Box bbox_of(const std::vector<Coord>& cells) {
  Box b{cells[0].first, cells[0].second, cells[0].first, cells[0].second};
  for (const Coord& c : cells) {
    b.y0 = std::min(b.y0, c.first);
    b.x0 = std::min(b.x0, c.second);
    b.y1 = std::max(b.y1, c.first);
    b.x1 = std::max(b.x1, c.second);
  }
  return b;
}

}  // namespace arc
