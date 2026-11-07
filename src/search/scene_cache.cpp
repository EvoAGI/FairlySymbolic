// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SceneCache implementation. See scene_cache.hpp.

#include "arc/scene_cache.hpp"

#include "arc/codec.hpp"   // encoder_signature_repr
#include "arc/encode.hpp"  // encode_minimal

namespace arc {

std::size_t estimate_scene_bytes(const Scene& scene) {
  std::size_t b = sizeof(Scene);

  // Original input grid + step-2 grid-line pixel set.
  b += scene.grid().cells().capacity() * sizeof(int);
  b += scene.grid_line_pixels().size() * (sizeof(Coord) + 32);  // + rb-tree node

  const std::vector<Node>& nodes = scene.nodes();
  b += nodes.capacity() * sizeof(Node);
  for (const Node& n : nodes) {
    b += n.node_id.capacity() + n.node_type.capacity();
    b += n.pixel_list.capacity() * sizeof(Coord);
    b += n.perimeter_pixels.get().capacity() * sizeof(Coord);  // shared with the copies
    b += n.interior_pixels.get().capacity() * sizeof(Coord);
    if (n.attrs) b += n.attrs->capacity() * sizeof(int);  // shared with the copies
    b += n.child_nodes.capacity() * 32;
    b += n.zf_inside_colors.capacity() * sizeof(int);
    b += n.zf_outside_colors.capacity() * sizeof(int);
    b += n.color_histogram.capacity() * sizeof(std::pair<int, int>);
    b += n.colors.capacity() * sizeof(int);
    b += n.subobject_ids.capacity() * 32;
    b += n.constituent_ids.capacity() * 32;
    b += n.meta_h_lines.capacity() * sizeof(int);
    b += n.meta_v_lines.capacity() * sizeof(int);
    if (n.shape_hash) b += n.shape_hash->capacity();
    if (n.shape_mask) b += n.shape_mask->capacity();
    if (n.enclosing_object_id) b += n.enclosing_object_id->capacity();
    if (n.pattern_type) b += n.pattern_type->capacity();
    if (n.implicit_type) b += n.implicit_type->capacity();
    if (n.param_type) b += n.param_type->capacity();
  }

  b += scene.edge_bytes();

  // The node index (id -> index, layer map) is private; approximate
  // from the node count.
  b += nodes.size() * 80;  // node index entries
  return b;
}

std::string SceneCache::make_key(const Grid& grid, const GraphConfig& cfg) const {
  // Collision-free: raw grid bytes (dims + cells) then the encoder signature.
  const std::vector<int>& cells = grid.cells();
  int w = grid.width();
  int h = grid.height();
  std::string k;
  k.reserve(cells.size() * sizeof(int) + 64);
  k.append(reinterpret_cast<const char*>(&w), sizeof(w));
  k.append(reinterpret_cast<const char*>(&h), sizeof(h));
  if (!cells.empty())
    k.append(reinterpret_cast<const char*>(cells.data()), cells.size() * sizeof(int));
  k.push_back('\x1f');
  k += encoder_signature_repr(cfg);
  return k;
}

Scene SceneCache::get(const Grid& grid, const GraphConfig& cfg) {
  std::string key = make_key(grid, cfg);
  auto it = map_.find(key);
  if (it != map_.end()) {
    ++hits_;
    // Move to front (most-recently used). splice keeps the iterator valid.
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->scene;  // private copy for the caller to mutate
  }

  ++misses_;
  Scene scene = encode_minimal(grid, cfg);
  std::size_t sz = estimate_scene_bytes(scene);
  lru_.push_front(Entry{key, scene, sz});
  map_[std::move(key)] = lru_.begin();
  bytes_ += sz;
  evict_if_needed();
  if (bytes_ > peak_bytes_) peak_bytes_ = bytes_;
  return scene;
}

void SceneCache::evict_if_needed() {
  // Keep at least one entry even if a single scene exceeds the cap.
  while (bytes_ > max_bytes_ && lru_.size() > 1) {
    Entry& back = lru_.back();
    bytes_ -= back.bytes;
    map_.erase(back.key);
    lru_.pop_back();
    ++evictions_;
  }
}

}  // namespace arc
