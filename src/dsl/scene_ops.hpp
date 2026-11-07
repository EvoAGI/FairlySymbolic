// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL scene/node access and subobject rebuild helpers. `inline` so every TU
// inlines them.
// Ordering rules: every sort is a std::stable_sort over an explicit key tuple;
// node-id comparisons are byte-wise lexicographic; no exceptions.

#ifndef ARC_DSL_SCENE_OPS_HPP
#define ARC_DSL_SCENE_OPS_HPP

#include "arc/dsl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

namespace arc {

inline int node_color(const Node& n, int d) { return n.color.has_value() ? *n.color : d; }

// A node's neighbours via edges of the named type in the direction
// (outgoing edges first in creation order, then incoming), for every node at
// once in one pass over the edges. A clique
// adds the rest of its members, in member order (the order its edges give),
// without listing them per node. Unknown type names match no edge. Valid
// while the scene's edges do not change; node fields are read by the caller.
class EdgeNeighbors {
 public:
  EdgeNeighbors(const Scene& scene, const std::string& edge_type_name, const std::string& direction)
      : scene_(scene) {
    const EdgeType t = edge_type_of(edge_type_name);
    const bool out = direction == "outgoing" || direction == "both";
    const bool in = direction == "incoming" || direction == "both";
    if (t == EdgeType::NONE || (!out && !in)) return;
    out_.resize(scene.edge_node_count());
    in_.resize(scene.edge_node_count());
    scene.for_each_edge_block(
        [&](const EdgeRec& e) {
          if (e.type != t) return;
          if (out) out_[e.src].push_back(Entry{e.dst, false});
          if (in) in_[e.dst].push_back(Entry{e.src, false});
        },
        [&](EdgeType ct, const std::vector<uint32_t>& members) {
          if (ct != t) return;
          const uint32_t c = static_cast<uint32_t>(cliques_.size());
          cliques_.push_back(members);
          for (uint32_t m : members) {
            if (out) out_[m].push_back(Entry{c, true});
            if (in) in_[m].push_back(Entry{c, true});
          }
        });
  }
  int count(const std::string& node_id) const {
    const uint32_t h = handle(node_id);
    if (h == Scene::kNoEdgeNode) return 0;
    int n = 0;
    for (const auto* list : {&out_[h], &in_[h]})
      for (const Entry& e : *list) n += e.clique ? static_cast<int>(cliques_[e.v].size()) - 1 : 1;
    return n;
  }
  // f(const std::string& neighbour_id) in list order; f returns true to stop.
  template <typename F>
  void for_each(const std::string& node_id, F&& f) const {
    const uint32_t h = handle(node_id);
    if (h == Scene::kNoEdgeNode) return;
    for (const auto* list : {&out_[h], &in_[h]})
      for (const Entry& e : *list) {
        if (!e.clique) {
          if (f(scene_.edge_node_id(e.v))) return;
          continue;
        }
        for (uint32_t m : cliques_[e.v])
          if (m != h && f(scene_.edge_node_id(m))) return;
      }
  }

 private:
  struct Entry {
    uint32_t v;   // neighbour handle, or clique index
    bool clique;
  };
  uint32_t handle(const std::string& node_id) const {
    const uint32_t h = scene_.find_edge_node(node_id);
    return h < out_.size() ? h : Scene::kNoEdgeNode;
  }
  const Scene& scene_;
  std::vector<std::vector<Entry>> out_, in_;
  std::vector<std::vector<uint32_t>> cliques_;
};

inline std::vector<Coord> neighbor_offsets(int connectivity) {
  if (connectivity == 4) return {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  return {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
}

inline const Node* find_node(const Scene& scene, const std::string& nid) {
  if (!scene.has_node(nid)) return nullptr;
  return &scene.nodes()[scene.index_of(nid)];
}

inline std::vector<const Node*> iter_subobjects(const Scene& scene) {
  std::vector<size_t> idxs = scene.get_nodes_by_type("SubObject", false);
  std::vector<const Node*> out;
  out.reserve(idxs.size());
  for (size_t i : idxs) out.push_back(&scene.nodes()[i]);
  return out;
}

inline std::vector<std::tuple<int, int, int>> collect_colored_pixels(const Scene& scene) {
  std::vector<std::tuple<int, int, int>> out;
  int bg = scene.has_background() ? scene.background() : 0;
  for (const Node* n : iter_subobjects(scene)) {
    int color = node_color(*n, bg);
    if (color == bg) continue;
    for (const auto& px : n->pixel_list) out.emplace_back(px.first, px.second, color);
  }
  return out;
}

inline void reset_scene_subobjects(Scene& scene,
                            const std::vector<std::tuple<int, int, int>>& colored_pixels,
                            int new_height, int new_width) {
  // Last-write-wins on color conflicts for identical coordinates; pixels
  // outside the new dims are dropped.
  const int H = std::max(new_height, 0), W = std::max(new_width, 0);
  std::vector<int> cell(static_cast<size_t>(H) * W, 0);
  std::vector<char> has(static_cast<size_t>(H) * W, 0);
  for (const auto& t : colored_pixels) {
    int y = std::get<0>(t), x = std::get<1>(t), color = std::get<2>(t);
    if (y >= 0 && y < new_height && x >= 0 && x < new_width) {
      cell[static_cast<size_t>(y) * W + x] = color;
      has[static_cast<size_t>(y) * W + x] = 1;
    }
  }

  // Rebuild the scene in place: clear all nodes/edges/counter and re-dim.
  // The rebuild resets the graph, the layer map, height/width and the node
  // counter, but keeps the background colour and scene.grid (the stale grid
  // keeps its pre-reset value; actions that read it guard on shape).
  std::optional<int> bg;
  if (scene.has_background()) bg = scene.background();
  bool had_grid = scene.has_grid();
  Grid saved_grid = had_grid ? scene.grid() : Grid();
  scene = Scene(new_width, new_height);
  scene.set_background(bg);
  if (had_grid) scene.set_grid(saved_grid);
  // (Scene ctor takes (width, height); node_counter starts 0.)

  // The 8-connected same-color components. A row-major scan reaches each
  // component first at its (y,x)-least pixel; the pixel list is sorted.
  struct Comp {
    int color;
    Coord min;
    std::vector<Coord> px;
  };
  std::vector<Comp> comps;
  std::vector<char> seen(has.size(), 0);
  std::vector<int> stack;
  for (int y0 = 0; y0 < H; ++y0)
    for (int x0 = 0; x0 < W; ++x0) {
      const size_t s0 = static_cast<size_t>(y0) * W + x0;
      if (!has[s0] || seen[s0]) continue;
      const int color = cell[s0];
      Comp c{color, {y0, x0}, {}};
      seen[s0] = 1;
      stack.assign(1, static_cast<int>(s0));
      while (!stack.empty()) {
        const int p = stack.back();
        stack.pop_back();
        const int y = p / W, x = p % W;
        c.px.emplace_back(y, x);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int ny = y + dy, nx = x + dx;
            if ((dy == 0 && dx == 0) || ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            const size_t q = static_cast<size_t>(ny) * W + nx;
            if (has[q] && !seen[q] && cell[q] == color) {
              seen[q] = 1;
              stack.push_back(static_cast<int>(q));
            }
          }
      }
      std::sort(c.px.begin(), c.px.end());
      comps.push_back(std::move(c));
    }
  // Ascending color; within a color by (len, min-coord), which is a total order
  // (the min coords differ).
  std::sort(comps.begin(), comps.end(), [](const Comp& a, const Comp& b) {
    if (a.color != b.color) return a.color < b.color;
    if (a.px.size() != b.px.size()) return a.px.size() < b.px.size();
    return a.min < b.min;
  });

  scene.reserve_nodes(comps.size());
  for (Comp& comp : comps) {
    scene.node_counter += 1;
    Node node;
    node.node_id = "sub_" + std::to_string(scene.node_counter);
    node.node_type = "SubObject";
    node.color = comp.color;
    node.pixel_list = std::move(comp.px);
    node.post_init();
    node.compute_shape_hash();
    node.classify_shape();
    node.check_boundary_touches(scene.width(), scene.height());
    scene.add_node(std::move(node));
  }
}

}  // namespace arc

#endif  // ARC_DSL_SCENE_OPS_HPP
