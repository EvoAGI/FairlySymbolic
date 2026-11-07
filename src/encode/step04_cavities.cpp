// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 4: cavity detection (detect_cavities + find_enclosing_object).
//
// Detects enclosed empty spaces (holes) within objects via a border flood fill,
// then BFS-components the unreachable background pixels into CavityObjects.
//
// Determinism notes:
//   * Skipped entirely when there is no background color (no "empty space").
//   * foreground = union of subobject pixels (the post-3d subobject list, NOT
//     the ClusterObjects created at 3c).
//   * The border-reachability BFS seeds the outer ring in a fixed order
//     (top/bottom rows over x, then left/right cols over y=1..h-2), is FIFO,
//     and visits neighbours in get_orthogonal_neighbors order. Order does not
//     affect the reachable SET.
//   * cavity_pixels are collected in row-major (y-outer, x-inner) scan order;
//     the per-cavity BFS starts from the first unvisited such pixel, so cavity
//     ids are assigned in that scan order. No hash-set order drives iteration:
//     cavity_visited is only a membership test.
//   * find_enclosing_object: smallest bbox-area (obj.width*obj.height) object
//     whose bbox fully encloses the cavity bbox; strict '<' -> first-min-wins
//     over subobject-list order.
//   * CavityObject creation calls check_boundary_touches ONLY (no shape hash /
//     classify_shape), and cavity_color = background_color.

#include "arc/encode.hpp"

#include <deque>
#include <string>
#include <vector>

namespace arc {

namespace {

// find_enclosing_object: cavity bbox = (min y, min x, max y, max x) over the
// cavity component; pick the smallest-area enclosing subobject (strict '<').
// Returns the node_id, or an empty optional when none encloses.
std::optional<std::string> find_enclosing_object(const std::vector<Coord>& cavity_pixels,
                                                 const std::vector<Node>& subobjects) {
  if (cavity_pixels.empty()) return std::nullopt;
  int cymin = cavity_pixels[0].first, cymax = cavity_pixels[0].first;
  int cxmin = cavity_pixels[0].second, cxmax = cavity_pixels[0].second;
  for (const auto& p : cavity_pixels) {
    if (p.first < cymin) cymin = p.first;
    if (p.first > cymax) cymax = p.first;
    if (p.second < cxmin) cxmin = p.second;
    if (p.second > cxmax) cxmax = p.second;
  }

  const Node* best = nullptr;
  // best_area starts at +inf; strict '<' comparison keeps the first minimum.
  long long best_area = 0;
  bool have_best = false;
  for (const auto& obj : subobjects) {
    if (!obj.has_bbox) continue;  // no bounding box
    if (obj.bbox_ymin <= cymin && obj.bbox_ymax >= cymax && obj.bbox_xmin <= cxmin &&
        obj.bbox_xmax >= cxmax) {
      long long area = static_cast<long long>(obj.width) * obj.height;
      if (!have_best || area < best_area) {
        best_area = area;
        best = &obj;
        have_best = true;
      }
    }
  }
  if (best == nullptr) return std::nullopt;
  return best->node_id;
}

}  // namespace

std::vector<Node> detect_cavities(Scene& scene, const Grid& grid,
                                  const std::vector<Node>& subobjects) {
  std::vector<Node> cavities;
  if (!scene.has_background()) return cavities;  // no background color

  const int h = grid.height(), w = grid.width();

  std::vector<uint8_t> foreground(static_cast<size_t>(h) * w, 0);
  auto fg = [&](int y, int x) -> uint8_t& { return foreground[static_cast<size_t>(y) * w + x]; };
  for (const auto& obj : subobjects)
    for (const auto& p : obj.pixel_list) fg(p.first, p.second) = 1;

  std::vector<uint8_t> reachable(static_cast<size_t>(h) * w, 0);
  auto rc = [&](int y, int x) -> uint8_t& { return reachable[static_cast<size_t>(y) * w + x]; };
  std::deque<Coord> queue;

  for (int x = 0; x < w; ++x) {
    if (!fg(0, x)) {
      queue.emplace_back(0, x);
      rc(0, x) = 1;
    }
    if (h > 1 && !fg(h - 1, x)) {
      queue.emplace_back(h - 1, x);
      rc(h - 1, x) = 1;
    }
  }
  for (int y = 1; y < h - 1; ++y) {
    if (!fg(y, 0)) {
      queue.emplace_back(y, 0);
      rc(y, 0) = 1;
    }
    if (w > 1 && !fg(y, w - 1)) {
      queue.emplace_back(y, w - 1);
      rc(y, w - 1) = 1;
    }
  }

  // FIFO over the queue (neighbours in get_orthogonal_neighbors order).
  static const int d4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  while (!queue.empty()) {
    Coord cur = queue.front();
    queue.pop_front();
    for (const auto& d : d4) {
      const int ny = cur.first + d[0], nx = cur.second + d[1];
      if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
      if (!rc(ny, nx) && !fg(ny, nx)) {
        rc(ny, nx) = 1;
        queue.emplace_back(ny, nx);
      }
    }
  }

  // cavity_pixels: row-major scan of unreachable background cells.
  std::vector<Coord> cavity_pixels;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (!fg(y, x) && !rc(y, x)) cavity_pixels.emplace_back(y, x);

  if (cavity_pixels.empty()) return cavities;

  // The cavity cells are exactly the cells neither foreground nor reachable.
  std::vector<uint8_t> cavity_visited(static_cast<size_t>(h) * w, 0);
  auto cv = [&](int y, int x) -> uint8_t& { return cavity_visited[static_cast<size_t>(y) * w + x]; };

  const int bg = scene.background();

  for (const auto& seed : cavity_pixels) {
    if (cv(seed.first, seed.second)) continue;

    // BFS (FIFO): the component list is the queue, in discovery order.
    std::vector<Coord> component;
    component.push_back(seed);
    cv(seed.first, seed.second) = 1;
    for (size_t head = 0; head < component.size(); ++head) {
      const Coord cur = component[head];
      for (const auto& d : d4) {
        const int ny = cur.first + d[0], nx = cur.second + d[1];
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
        if (!cv(ny, nx) && !fg(ny, nx) && !rc(ny, nx)) {
          cv(ny, nx) = 1;
          component.emplace_back(ny, nx);
        }
      }
    }

    std::optional<std::string> enclosing = find_enclosing_object(component, subobjects);

    scene.node_counter += 1;
    Node cavity;
    cavity.node_id = "cavity_" + std::to_string(scene.node_counter);
    cavity.node_type = "CavityObject";
    cavity.cavity_color = bg;
    cavity.enclosing_object_id = enclosing;
    cavity.pixel_list = std::move(component);
    cavity.post_init();  // bbox, centroid, dims, mass, density
    cavity.check_boundary_touches(w, h);
    cavities.push_back(std::move(cavity));
  }

  return cavities;
}

}  // namespace arc
