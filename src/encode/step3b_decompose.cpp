// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 3b: decompose irregular objects into rectangles or lines
// (decompose_objects).
//
// Determinism notes:
//   * mode="rectangles": find_largest_rectangle scans a fixed
//     (min_y..max_y) x (min_x..max_x) coordinate range with membership tests
//     only; the leftover pixels' order does not matter because pixel_list is
//     sorted at serialization. So the decomposition is fully deterministic and
//     order-independent.
//   * mode="lines" / "both": rows/cols are scanned in ascending order (std::map)
//     and leftover pixels in (y,x) order (std::set). This order fixes the
//     equal-length tie-break (which of two same-length runs wins) and the
//     leftover-pixel group order. "both" returns whichever of lines/rects has
//     fewer groups (lines wins ties).

#include "arc/encode.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "arc/cpyset.hpp"

namespace arc {

namespace {

// find_largest_rectangle: largest axis-aligned filled rectangle. Returns the
// pixel list (row-major over the rect) or empty if best_area < 2.
std::vector<Coord> find_largest_rectangle(const std::set<Coord>& pixels) {
  if (pixels.empty()) return {};
  int min_y = pixels.begin()->first, max_y = min_y;
  int min_x = pixels.begin()->second, max_x = min_x;
  for (const auto& p : pixels) {
    min_y = std::min(min_y, p.first);
    max_y = std::max(max_y, p.first);
    min_x = std::min(min_x, p.second);
    max_x = std::max(max_x, p.second);
  }
  auto has = [&](int y, int x) { return pixels.find(Coord(y, x)) != pixels.end(); };

  std::vector<Coord> best_rect;
  int best_area = 0;
  for (int y1 = min_y; y1 <= max_y; ++y1) {
    for (int x1 = min_x; x1 <= max_x; ++x1) {
      if (!has(y1, x1)) continue;
      int max_x2 = max_x;
      for (int y2 = y1; y2 <= max_y; ++y2) {
        if (!has(y2, x1)) break;
        int x2 = x1;
        while (x2 + 1 <= max_x2 && has(y2, x2 + 1)) x2 += 1;
        max_x2 = std::min(max_x2, x2);
        if (max_x2 < x1) break;
        int area = (y2 - y1 + 1) * (max_x2 - x1 + 1);
        if (area > best_area) {
          best_area = area;
          best_rect.clear();
          for (int y = y1; y <= y2; ++y)
            for (int x = x1; x <= max_x2; ++x) best_rect.emplace_back(y, x);
        }
      }
    }
  }
  return best_area >= 2 ? best_rect : std::vector<Coord>{};
}

// decompose_rectangles: iteratively extract the largest rectangle. Returns a
// list of pixel groups, or empty if <= 1 group.
std::vector<std::vector<Coord>> decompose_rectangles(const std::vector<Coord>& pixel_list) {
  std::set<Coord> remaining(pixel_list.begin(), pixel_list.end());
  std::vector<std::vector<Coord>> rects;
  const int max_iterations = 50;
  for (int it = 0; it < max_iterations; ++it) {
    if (remaining.empty()) break;
    std::vector<Coord> rect = find_largest_rectangle(remaining);
    if (rect.size() < 2) {
      if (!remaining.empty()) {
        rects.emplace_back(remaining.begin(), remaining.end());
        remaining.clear();
      }
      break;
    }
    for (const auto& p : rect) remaining.erase(p);
    rects.push_back(std::move(rect));
  }
  if (!remaining.empty()) rects.emplace_back(remaining.begin(), remaining.end());
  if (rects.empty()) return {};
  return rects.size() > 1 ? rects : std::vector<std::vector<Coord>>{};
}

// find_best_line: longest horizontal OR vertical maximal consecutive run in
// the pixel set. Rows are scanned in ascending order (then columns in ascending
// order) via std::map, for a deterministic tie-break. Horizontal runs are
// tested before vertical, and only a strictly longer run replaces the best (so
// equal-length H beats V, and the first row/col encountered wins among equal
// H/V).
std::vector<Coord> find_best_line(const std::set<Coord>& pixels) {
  std::vector<Coord> best;

  std::map<int, std::vector<int>> rows;  // y -> xs (ascending y)
  for (const auto& p : pixels) rows[p.first].push_back(p.second);
  for (auto& kv : rows) {
    int y = kv.first;
    std::vector<int>& xs = kv.second;
    std::sort(xs.begin(), xs.end());
    size_t run_start = 0;
    for (size_t i = 1; i <= xs.size(); ++i) {
      if (i == xs.size() || xs[i] != xs[i - 1] + 1) {
        if (i - run_start > best.size()) {
          std::vector<Coord> run;
          for (size_t j = run_start; j < i; ++j) run.emplace_back(y, xs[j]);
          best = std::move(run);
        }
        run_start = i;
      }
    }
  }

  std::map<int, std::vector<int>> cols;  // x -> ys (ascending x)
  for (const auto& p : pixels) cols[p.second].push_back(p.first);
  for (auto& kv : cols) {
    int x = kv.first;
    std::vector<int>& ys = kv.second;
    std::sort(ys.begin(), ys.end());
    size_t run_start = 0;
    for (size_t i = 1; i <= ys.size(); ++i) {
      if (i == ys.size() || ys[i] != ys[i - 1] + 1) {
        if (i - run_start > best.size()) {
          std::vector<Coord> run;
          for (size_t j = run_start; j < i; ++j) run.emplace_back(ys[j], x);
          best = std::move(run);
        }
        run_start = i;
      }
    }
  }

  return best.size() >= 2 ? best : std::vector<Coord>{};
}

// decompose_lines: greedily pick the longest line covering remaining pixels
// until none remain; a <2-length remainder becomes one leftover group. Returns
// >1 groups or empty.
std::vector<std::vector<Coord>> decompose_lines(const std::set<Coord>& pixels) {
  std::set<Coord> remaining = pixels;
  std::vector<std::vector<Coord>> lines;
  while (!remaining.empty()) {
    std::vector<Coord> best_line = find_best_line(remaining);
    if (best_line.size() < 2) {
      if (!remaining.empty()) lines.emplace_back(remaining.begin(), remaining.end());
      break;
    }
    for (const auto& p : best_line) remaining.erase(p);
    lines.push_back(std::move(best_line));
  }
  if (lines.empty()) return {};
  return lines.size() > 1 ? lines : std::vector<std::vector<Coord>>{};
}

// decompose_object: dispatch on mode. Returns groups or empty.
std::vector<std::vector<Coord>> decompose_object(const std::vector<Coord>& pixel_list,
                                                 const std::string& mode) {
  // Objects with at most 3 distinct pixels are not decomposed.
  std::set<Coord> uniq(pixel_list.begin(), pixel_list.end());
  if (uniq.size() <= 3) return {};

  std::vector<std::vector<Coord>> result;
  if (mode == "rectangles") {
    result = decompose_rectangles(pixel_list);
  } else if (mode == "lines") {
    result = decompose_lines(uniq);
  } else if (mode == "both") {
    // Try both; return the one with fewer sub-shapes (lines wins ties).
    std::vector<std::vector<Coord>> lines = decompose_lines(uniq);
    std::vector<std::vector<Coord>> rects = decompose_rectangles(pixel_list);
    if (lines.empty() && rects.empty()) return {};
    if (lines.empty())
      result = std::move(rects);
    else if (rects.empty())
      result = std::move(lines);
    else
      result = (lines.size() <= rects.size()) ? std::move(lines) : std::move(rects);
  } else {
    return {};
  }
  if (result.size() <= 1) return {};
  return result;
}

}  // namespace

std::vector<Node> decompose_objects(Scene& scene, std::vector<Node> subobjects,
                                    const Grid& grid, const GraphConfig& cfg) {
  std::vector<Node> result;
  result.reserve(subobjects.size());

  for (auto& obj : subobjects) {
    // Only objects with no shape mask or an "Irregular" one are decomposed.
    bool mask_none = !obj.shape_mask.has_value();
    bool mask_irr = obj.shape_mask.has_value() && *obj.shape_mask == "Irregular";
    if (!(mask_none || mask_irr)) {
      result.push_back(std::move(obj));
      continue;
    }
    if (obj.mass <= 3) {
      result.push_back(std::move(obj));
      continue;
    }

    std::vector<std::vector<Coord>> parts =
        decompose_object(obj.pixel_list, cfg.decomposition_mode);
    if (parts.size() <= 1) {
      result.push_back(std::move(obj));
      continue;
    }

    for (const auto& group : parts) {
      scene.node_counter += 1;
      // dominant: the most frequent color in the group. Colors are scanned in
      // hash-set iteration order (cpython_set_order_int), so on a count tie the
      // first color in that order wins.
      std::map<int, int> counts;               // color -> count (any order)
      std::vector<int> seen;                    // first-seen colors (for the set)
      for (const auto& p : group) {
        int c = grid.at(p.first, p.second);
        if (counts.find(c) == counts.end()) seen.push_back(c);
        counts[c] += 1;
      }
      std::vector<int> color_order = cpython_set_order_int(seen);
      int dominant = color_order.front();
      int best = counts[dominant];
      for (int c : color_order) {
        if (counts[c] > best) {
          best = counts[c];
          dominant = c;
        }
      }
      Node sub;
      sub.node_id = "sub_" + std::to_string(scene.node_counter);
      sub.node_type = "SubObject";
      sub.color = dominant;
      sub.pixel_list = group;
      sub.post_init();
      sub.check_boundary_touches(scene.width(), scene.height());
      sub.compute_shape_hash();
      sub.classify_shape();
      result.push_back(std::move(sub));
    }
  }

  return result;
}

}  // namespace arc
