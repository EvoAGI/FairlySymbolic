// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_SCENE_CACHE_HPP
#define ARC_SCENE_CACHE_HPP

// Per-task encoder scene cache.
//
// The DSL executor re-encodes the input grid into a Scene on every candidate
// (execute_program_gout -> encode_minimal), yet within one task the input grids
// are CONSTANT and only the encoder_config varies (and repeats heavily across
// the population). On busy AGI-2 grids the 17-step encoder costs 100-200 ms per
// call and starves the GA. This cache memoises the encoded scene keyed by
// (grid-bytes, encoder-signature).
//
// The Scene is a plain deep-copyable value type whose copy is ~100-1000x
// cheaper than encode_minimal, so the cache stores LIVE Scenes and hands back a
// private COPY on a hit. A copied scene is byte-identical to a freshly built
// one, so program output is the same with or without the cache.
//
// NOT thread-safe by design: one task runs on exactly one worker thread and the
// per-task GA is single-threaded, so no mutex is needed. One instance per task
// (owned by run_task_full); memory is bounded per task by an LRU byte budget.

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>

#include "arc/config.hpp"
#include "arc/graph.hpp"
#include "arc/grid.hpp"

namespace arc {

// Cheap heap-footprint estimate for LRU byte accounting: sums the dominant
// containers (pixel lists, edge attrs, grid cells, index maps). It is an
// approximation of the real RAM footprint, NOT a serialization size -- that is
// what we want for bounding memory.
std::size_t estimate_scene_bytes(const Scene& scene);

class SceneCache {
 public:
  explicit SceneCache(std::size_t max_bytes = 1500ULL * 1024 * 1024)
      : max_bytes_(max_bytes) {}

  // Return an independent Scene for (grid, cfg). Hit -> a copy of the stored
  // scene (safe for the caller to mutate). Miss -> encode_minimal, store a
  // copy, evict to the byte budget, return the built scene.
  Scene get(const Grid& grid, const GraphConfig& cfg);

  std::size_t hits() const { return hits_; }
  std::size_t misses() const { return misses_; }
  std::size_t evictions() const { return evictions_; }
  std::size_t entries() const { return map_.size(); }
  std::size_t bytes() const { return bytes_; }
  std::size_t peak_bytes() const { return peak_bytes_; }
  std::size_t max_bytes() const { return max_bytes_; }

 private:
  struct Entry {
    std::string key;
    Scene scene;
    std::size_t bytes;
  };
  std::string make_key(const Grid& grid, const GraphConfig& cfg) const;
  void evict_if_needed();

  std::list<Entry> lru_;  // front = most-recently used
  std::unordered_map<std::string, std::list<Entry>::iterator> map_;
  std::size_t bytes_ = 0;
  std::size_t max_bytes_;
  std::size_t peak_bytes_ = 0;
  std::size_t hits_ = 0, misses_ = 0, evictions_ = 0;
};

}  // namespace arc

#endif  // ARC_SCENE_CACHE_HPP
