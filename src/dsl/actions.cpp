// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// DSL actions: the grid <-> scene bridge, the grid-level shape operations and
// apply_action_to_scene. Shared helpers live in arg_coerce.hpp / scene_ops.hpp,
// which also state the ordering rules.

#include "arc/dsl.hpp"
#include "arc/concepts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arc/encode.hpp"
#include "arc/scene_cache.hpp"

#include "arg_coerce.hpp"
#include "scene_ops.hpp"

namespace arc {

// ===========================================================================
// Actions.
// ===========================================================================

namespace {

// iter_subobjects(scene): get_nodes_by_type("SubObject") in lexicographic id
// order (NOT the mass sort). Returns pointers into the scene node vector.

// collect_colored_pixels(scene): (y,x,color) for every SubObject pixel whose
// object color != background, in iter_subobjects order.

// reset_scene_subobjects: clear the scene and rebuild SubObjects from the
// colored-pixel list. Groups pixels by color (last-write-wins per coord), then
// per color sorts 8-connected components by (len, min-coord) and emits sub_{n}
// in that order. Each SubObject is computed as post_init -> compute_shape_hash
// -> classify_shape -> check_boundary_touches; perimeter/interior are left
// empty.

// update_geometry(data): recompute mass/bbox/width/height/centroid/density
// from pixel_list, IN PLACE, without touching shape_hash / shape_mask /
// perimeter / interior. An empty pixel list clears bbox/centroid.
void update_geometry(Node& n) {
  if (n.pixel_list.empty()) {
    n.mass = 0;
    n.has_bbox = false;
    n.width = 0;
    n.height = 0;
    n.has_centroid = false;
    n.density = 0.0;
    return;
  }
  int ymin = n.pixel_list[0].first, ymax = ymin;
  int xmin = n.pixel_list[0].second, xmax = xmin;
  long long sy = 0, sx = 0;
  for (const auto& p : n.pixel_list) {
    ymin = std::min(ymin, p.first);
    ymax = std::max(ymax, p.first);
    xmin = std::min(xmin, p.second);
    xmax = std::max(xmax, p.second);
    sy += p.first;
    sx += p.second;
  }
  n.mass = static_cast<int>(n.pixel_list.size());
  n.bbox_ymin = ymin;
  n.bbox_xmin = xmin;
  n.bbox_ymax = ymax;
  n.bbox_xmax = xmax;
  n.has_bbox = true;
  n.width = xmax - xmin + 1;
  n.height = ymax - ymin + 1;
  double nn = static_cast<double>(n.pixel_list.size());
  n.centroid_y = static_cast<double>(sy) / nn;
  n.centroid_x = static_cast<double>(sx) / nn;
  n.has_centroid = true;
  long long area = static_cast<long long>(n.width) * n.height;
  n.density = area > 0 ? static_cast<double>(n.mass) / static_cast<double>(area) : 0.0;
}

// Mutable-node lookup by id. Returns nullptr when absent.
Node* find_node_mut(Scene& scene, const std::string& nid) {
  if (!scene.has_node(nid)) return nullptr;
  return &scene.node_ref(scene.index_of(nid));
}

// delete_nodes(scene, ids): remove each present node.
void delete_nodes(Scene& scene, const std::vector<std::string>& ids) { scene.remove_nodes(ids); }

// Banker's rounding (round half to even) for a double. Used by
// resolve_anchor (bbox_center / centroid) and the COMPLETE_D4_SYMMETRY pivot.
long long py_round(double x) {
  double fl = std::floor(x);
  double diff = x - fl;
  if (diff < 0.5) return static_cast<long long>(fl);
  if (diff > 0.5) return static_cast<long long>(fl) + 1;
  // Exactly .5 -> round to even.
  long long lo = static_cast<long long>(fl);
  return (lo % 2 == 0) ? lo : lo + 1;
}

// resolve_anchor(data, anchor) -> (y,x) or absent.
bool resolve_anchor(const Node& n, const std::string& anchor, int* oy, int* ox) {
  if (anchor == "bbox_center") {
    if (!n.has_bbox) return false;
    *oy = static_cast<int>(py_round((n.bbox_ymin + n.bbox_ymax) / 2.0));
    *ox = static_cast<int>(py_round((n.bbox_xmin + n.bbox_xmax) / 2.0));
    return true;
  }
  if (anchor == "min_corner") {
    if (!n.has_bbox) return false;
    *oy = n.bbox_ymin;
    *ox = n.bbox_xmin;
    return true;
  }
  if (!n.has_centroid) return false;
  *oy = static_cast<int>(py_round(n.centroid_y));
  *ox = static_cast<int>(py_round(n.centroid_x));
  return true;
}

// resolve_action_color(data, action_args, bg): color_source in {self(default),
// constant, background/bg}. constant -> as_int(args["color"], bg); bg -> bg;
// else -> node color (as_int(data.color, bg)).
int resolve_action_color(const Node& n, const DslArgs& args, int bg) {
  std::string mode = to_lower(strip(arg_str(args, "color_source", "self")));
  if (mode == "constant") return as_int(args_get(args, "color"), bg);
  if (mode == "background" || mode == "bg") return bg;
  return n.color.has_value() ? *n.color : bg;
}

// neighbor_offsets already defined above (file-scope helper).

// prune_endpoints_pixels: iteratively drop degree<=1 (or ==1 if
// protect_junctions) pixels. connectivity 4/8; iterations>=1. Order-independent
// over a set.
std::set<Coord> prune_endpoints_pixels(const std::set<Coord>& pixels, int iterations,
                                       int connectivity, bool protect_junctions) {
  if (pixels.empty()) return pixels;
  std::vector<Coord> nbrs = neighbor_offsets(connectivity);
  std::set<Coord> current = pixels;
  int iters = std::max(1, iterations);
  for (int it = 0; it < iters; ++it) {
    std::vector<Coord> to_remove;
    for (const auto& p : current) {
      int degree = 0;
      for (const auto& off : nbrs) {
        Coord q = {p.first + off.first, p.second + off.second};
        if (current.find(q) != current.end()) ++degree;
      }
      bool should_remove = protect_junctions ? (degree == 1) : (degree <= 1);
      if (should_remove) to_remove.push_back(p);
    }
    if (to_remove.empty()) break;
    for (const auto& p : to_remove) current.erase(p);
    if (current.empty()) break;
  }
  return current;
}

// dilation_offsets(radius, mask, connectivity). mask auto -> cross(conn4)/
// square(else). Emits (dy,dx) in row-major order.
std::vector<Coord> dilation_offsets(int radius, std::string mask, int connectivity) {
  int r = std::max(0, radius);
  if (mask == "auto") mask = (connectivity == 4) ? "cross" : "square";
  std::vector<Coord> out;
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx) {
      if (mask == "cross") {
        if (!(dy == 0 || dx == 0)) continue;
      } else if (mask == "diamond") {
        if (std::abs(dy) + std::abs(dx) > r) continue;
      } else {  // square
        if (std::max(std::abs(dy), std::abs(dx)) > r) continue;
      }
      out.push_back({dy, dx});
    }
  return out;
}

// stamp_pattern_offsets(pattern) for a STRING pattern name. (Seed args always
// pass strings via JSON; a list pattern can't arrive through the arg tree.)
std::vector<Coord> stamp_pattern_offsets(const std::string& raw) {
  std::string name = to_lower(strip(raw));
  if (name == "dot" || name == "center") return {{0, 0}};
  if (name == "cross" || name == "plus")
    return {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  if (name == "x" || name == "diag")
    return {{0, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  if (name == "ring" || name == "hollow_square")
    return {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  // default: full 3x3
  std::vector<Coord> out;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) out.push_back({dy, dx});
  return out;
}

// ==== Grid <-> scene bridge, grid-level shape and delta operations ==========
// The delta / shape-op actions operate on a dense grid. Grid2D is a row-major
// vector of rows.
using Grid2D = std::vector<std::vector<int>>;

// render_scene_to_grid2d(scene): full-grid render INCLUDING structural
// SubObjects, bg-filled. (Delta/shape-op handlers read the whole grid.)
Grid2D render_scene_to_grid2d(const Scene& scene) {
  int h = scene.height(), w = scene.width();
  int bg = scene.has_background() ? scene.background() : 0;
  Grid2D g(h, std::vector<int>(w, bg));
  std::vector<size_t> idxs = scene.get_nodes_by_type("SubObject", true);  // include_structural
  for (size_t i : idxs) {
    const Node& n = scene.nodes()[i];
    int color = n.color.has_value() ? *n.color : bg;
    if (color == bg) continue;
    for (const auto& px : n.pixel_list)
      if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
        g[px.first][px.second] = color;
  }
  return g;
}

// install_grid_into_scene(scene, grid): rebuild subobjects from grid (bg =
// scene bg). Resizes the scene to grid dims.
void install_grid_into_scene(Scene& scene, const Grid2D& g) {
  int h = static_cast<int>(g.size());
  int w = h ? static_cast<int>(g[0].size()) : 0;
  int bg = scene.has_background() ? scene.background() : 0;
  std::vector<std::tuple<int, int, int>> updated;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (g[y][x] != bg) updated.emplace_back(y, x, g[y][x]);
  reset_scene_subobjects(scene, updated, h, w);
}

// detect_bg(grid): most-common color, ties -> smallest value. Counts over ALL
// cells.
int detect_bg2d(const Grid2D& g) {
  std::map<int, int> counts;  // value -> count (ascending value)
  for (const auto& row : g) for (int v : row) counts[v]++;
  if (counts.empty()) return 0;
  // pick max count; on tie smallest value (map already ascending, strict >).
  int best_val = counts.begin()->first, best_cnt = counts.begin()->second;
  for (const auto& kv : counts)
    if (kv.second > best_cnt) { best_cnt = kv.second; best_val = kv.first; }
  return best_val;
}

// components_4(mask): 4-connected components of a boolean mask, row-major scan,
// DFS (stack). Returns the cell lists in discovery order.
std::vector<std::vector<Coord>> components_4(const std::vector<std::vector<char>>& mask) {
  int H = static_cast<int>(mask.size());
  int W = H ? static_cast<int>(mask[0].size()) : 0;
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  std::vector<std::vector<Coord>> comps;
  static const int d4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (!mask[y][x] || seen[y][x]) continue;
      std::vector<Coord> stack{{y, x}};
      seen[y][x] = 1;
      std::vector<Coord> cells;
      while (!stack.empty()) {
        Coord c = stack.back(); stack.pop_back();
        cells.push_back(c);
        for (const auto& d : d4) {
          int ny = c.first + d[0], nx = c.second + d[1];
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && mask[ny][nx] && !seen[ny][nx]) {
            seen[ny][nx] = 1;
            stack.push_back({ny, nx});
          }
        }
      }
      comps.push_back(std::move(cells));
    }
  return comps;
}

// Object record (make_obj): holds the fields select_object reads.
struct ShapeObj {
  std::vector<Coord> cells;
  int ncells;
  int ymin, xmin, ymax, xmax;
  long long bbox_area;
  std::vector<std::pair<int, int>> colors;  // (color, count) in first-seen order
  double density;
};

ShapeObj make_obj(const Grid2D& g, const std::vector<Coord>& cells) {
  ShapeObj o;
  o.cells = cells;
  o.ncells = static_cast<int>(cells.size());
  o.ymin = cells[0].first; o.ymax = cells[0].first;
  o.xmin = cells[0].second; o.xmax = cells[0].second;
  for (const auto& c : cells) {
    o.ymin = std::min(o.ymin, c.first); o.ymax = std::max(o.ymax, c.first);
    o.xmin = std::min(o.xmin, c.second); o.xmax = std::max(o.xmax, c.second);
  }
  // colset: colours in first-seen order over cells.
  for (const auto& c : cells) {
    int v = g[c.first][c.second];
    bool found = false;
    for (auto& kv : o.colors) if (kv.first == v) { kv.second++; found = true; break; }
    if (!found) o.colors.emplace_back(v, 1);
  }
  o.bbox_area = static_cast<long long>(o.ymax - o.ymin + 1) * (o.xmax - o.xmin + 1);
  o.density = o.bbox_area ? static_cast<double>(cells.size()) / o.bbox_area : 0.0;
  return o;
}

// objects(grid, bg, mode). mono: per non-bg color (ascending), 4-conn comps.
// multi: 4-conn comps of (grid != bg).
std::vector<ShapeObj> shape_objects(const Grid2D& g, int bg, const std::string& mode) {
  int H = static_cast<int>(g.size());
  int W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<ShapeObj> objs;
  if (mode == "mono") {
    std::set<int> colors;
    for (const auto& row : g) for (int v : row) colors.insert(v);
    for (int c : colors) {  // ascending
      if (c == bg) continue;
      std::vector<std::vector<char>> mask(H, std::vector<char>(W, 0));
      for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) mask[y][x] = (g[y][x] == c) ? 1 : 0;
      for (auto& cells : components_4(mask)) objs.push_back(make_obj(g, cells));
    }
  } else {
    std::vector<std::vector<char>> mask(H, std::vector<char>(W, 0));
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) mask[y][x] = (g[y][x] != bg) ? 1 : 0;
    for (auto& cells : components_4(mask)) objs.push_back(make_obj(g, cells));
  }
  return objs;
}

// select_object(objs, which). Returns index or -1 (no object). Ties: largest /
// smallest need a unique extremum; densest unique within 1e-9; unique_color /
// odd_one_out need exactly one singleton.
int select_object(const std::vector<ShapeObj>& objs, const std::string& which) {
  if (objs.empty()) return -1;
  if (which == "largest" || which == "smallest") {
    // best = max/min by ncells (the first one on a tie).
    int best = 0;
    for (int i = 1; i < static_cast<int>(objs.size()); ++i) {
      if (which == "largest") { if (objs[i].ncells > objs[best].ncells) best = i; }
      else { if (objs[i].ncells < objs[best].ncells) best = i; }
    }
    int cnt = 0;
    for (const auto& o : objs) if (o.ncells == objs[best].ncells) ++cnt;
    return cnt == 1 ? best : -1;
  }
  if (which == "densest") {
    int best = 0;
    for (int i = 1; i < static_cast<int>(objs.size()); ++i)
      if (objs[i].density > objs[best].density) best = i;
    int cnt = 0;
    for (const auto& o : objs) if (std::abs(o.density - objs[best].density) < 1e-9) ++cnt;
    return cnt == 1 ? best : -1;
  }
  if (which == "unique_color") {
    // dom(o) = max color entry by (count, -color).
    auto dom = [](const ShapeObj& o) {
      int c = o.colors[0].first, cnt = o.colors[0].second;
      for (const auto& kv : o.colors)
        if (kv.second > cnt || (kv.second == cnt && (-kv.first) > (-c))) { c = kv.first; cnt = kv.second; }
      return c;
    };
    std::vector<int> doms;
    for (const auto& o : objs) doms.push_back(dom(o));
    std::map<int, int> cnt;
    for (int d : doms) cnt[d]++;
    std::vector<int> singletons;
    for (int i = 0; i < static_cast<int>(doms.size()); ++i) if (cnt[doms[i]] == 1) singletons.push_back(i);
    return singletons.size() == 1 ? singletons[0] : -1;
  }
  if (which == "odd_one_out") {
    if (objs.size() < 3) return -1;
    // shape_key = the set of normalized (y-ymin, x-xmin).
    std::vector<std::set<Coord>> keys;
    for (const auto& o : objs) {
      std::set<Coord> k;
      for (const auto& c : o.cells) k.insert({c.first - o.ymin, c.second - o.xmin});
      keys.push_back(k);
    }
    // Count of each shape key.
    std::vector<std::pair<std::set<Coord>, int>> cnt;
    for (const auto& k : keys) {
      bool found = false;
      for (auto& kv : cnt) if (kv.first == k) { kv.second++; found = true; break; }
      if (!found) cnt.emplace_back(k, 1);
    }
    auto count_of = [&](const std::set<Coord>& k) {
      for (const auto& kv : cnt) if (kv.first == k) return kv.second;
      return 0;
    };
    std::vector<int> singletons;
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) if (count_of(keys[i]) == 1) singletons.push_back(i);
    return singletons.size() == 1 ? singletons[0] : -1;
  }
  return -1;
}

// recolor_crop(crop, recolor, bg). Returns (ok, out); ok=false: no result.
bool recolor_crop(const Grid2D& crop, const std::string& recolor, int bg, Grid2D* out) {
  if (recolor == "none") { *out = crop; return true; }
  // Colour counts of crop; nz = non-bg (value, count) in ascending value.
  std::map<int, int> counts;
  for (const auto& row : crop) for (int v : row) counts[v]++;
  std::vector<std::pair<int, int>> nz;  // (value, count) ascending value
  for (const auto& kv : counts) if (kv.first != bg) nz.emplace_back(kv.first, kv.second);
  if (nz.size() < 2) return false;
  if (recolor == "swap_two") {
    // sort by (-count, value).
    std::stable_sort(nz.begin(), nz.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });
    int a = nz[0].first, b = nz[1].first;
    *out = crop;
    for (auto& row : *out) for (int& v : row) { if (v == a) v = b; else if (v == b) v = a; }
    return true;
  }
  if (recolor == "to_marker_color") {
    // sort by (count, value); paint all non-bg to nz[0].first.
    std::stable_sort(nz.begin(), nz.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    });
    int marker = nz[0].first;
    *out = crop;
    for (auto& row : *out) for (int& v : row) if (v != bg) v = marker;
    return true;
  }
  return false;
}

// crop_to_object(grid, which, recolor, obj_mode). Returns (ok, out).
bool crop_to_object(const Grid2D& g, const std::string& which, const std::string& recolor,
                    const std::string& obj_mode, Grid2D* out) {
  if (g.empty() || g[0].empty()) return false;
  int bg = detect_bg2d(g);
  std::vector<ShapeObj> objs = shape_objects(g, bg, obj_mode);
  int sel = select_object(objs, which);
  if (sel < 0) return false;
  const ShapeObj& o = objs[sel];
  Grid2D crop;
  for (int y = o.ymin; y <= o.ymax; ++y) {
    std::vector<int> row;
    for (int x = o.xmin; x <= o.xmax; ++x) row.push_back(g[y][x]);
    crop.push_back(row);
  }
  return recolor_crop(crop, recolor, bg, out);
}

// count_objects(grid, bg, color_strict, diag). BFS (FIFO queue) over
// non-bg cells; color_strict skips neighbors of a different colour than the
// component seed.
int count_objects(const Grid2D& g, int bg, bool color_strict, bool diag) {
  int H = static_cast<int>(g.size());
  int W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  std::vector<Coord> nbrs = diag
      ? std::vector<Coord>{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
      : std::vector<Coord>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  int n = 0;
  for (int i = 0; i < H; ++i)
    for (int j = 0; j < W; ++j) {
      if (seen[i][j] || g[i][j] == bg) continue;
      ++n;
      int c0 = g[i][j];
      std::deque<Coord> dq{{i, j}};
      seen[i][j] = 1;
      while (!dq.empty()) {
        Coord cur = dq.front(); dq.pop_front();
        for (const auto& off : nbrs) {
          int ny = cur.first + off.first, nx = cur.second + off.second;
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && !seen[ny][nx] && g[ny][nx] != bg) {
            if (color_strict && g[ny][nx] != c0) continue;
            seen[ny][nx] = 1;
            dq.push_back({ny, nx});
          }
        }
      }
    }
  return n;
}

// scale_by_count(grid, mode, count_type, color_strict, diag). Returns (ok, out);
// ok=false when the count is outside [1, 30] or an output side would exceed 30.
bool scale_by_count(const Grid2D& g, const std::string& mode, const std::string& count_type,
                    bool color_strict, bool diag, Grid2D* out) {
  if (g.empty() || g[0].empty()) return false;
  int bg = detect_bg2d(g);
  int k;
  if (count_type == "objects") {
    k = count_objects(g, bg, color_strict, diag);
  } else if (count_type == "colors") {
    std::set<int> colors;
    for (const auto& row : g) for (int v : row) colors.insert(v);
    colors.erase(bg);
    k = static_cast<int>(colors.size());
  } else {
    return false;
  }
  if (k < 1 || k > 30) return false;
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  if (mode == "scale") {
    if (H * k > 30 || W * k > 30) return false;
    // Kronecker upscale: each cell expands to a k x k block.
    Grid2D r(H * k, std::vector<int>(W * k, 0));
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int a = 0; a < k; ++a)
          for (int b = 0; b < k; ++b) r[y * k + a][x * k + b] = g[y][x];
    *out = r; return true;
  }
  if (mode == "tile_h") {
    if (W * k > 30) return false;
    Grid2D r(H, std::vector<int>(W * k, 0));
    for (int y = 0; y < H; ++y)
      for (int t = 0; t < k; ++t)
        for (int x = 0; x < W; ++x) r[y][t * W + x] = g[y][x];
    *out = r; return true;
  }
  if (mode == "tile_v") {
    if (H * k > 30) return false;
    Grid2D r(H * k, std::vector<int>(W, 0));
    for (int t = 0; t < k; ++t)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) r[t * H + y][x] = g[y][x];
    *out = r; return true;
  }
  if (mode == "tile_both") {
    if (H * k > 30 || W * k > 30) return false;
    Grid2D r(H * k, std::vector<int>(W * k, 0));
    for (int ty = 0; ty < k; ++ty)
      for (int tx = 0; tx < k; ++tx)
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) r[ty * H + y][tx * W + x] = g[y][x];
    *out = r; return true;
  }
  return false;
}

// find_uniform_divider(grid, axis): indices of fully-uniform columns (axis v)
// or rows (axis h).
std::vector<int> find_uniform_divider(const Grid2D& g, const std::string& axis) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<int> res;
  if (axis == "v") {
    for (int c = 0; c < W; ++c) {
      bool uniform = true;
      for (int r = 1; r < H; ++r) if (g[r][c] != g[0][c]) { uniform = false; break; }
      if (uniform) res.push_back(c);
    }
  } else {
    for (int r = 0; r < H; ++r) {
      bool uniform = true;
      for (int c = 1; c < W; ++c) if (g[r][c] != g[r][0]) { uniform = false; break; }
      if (uniform) res.push_back(r);
    }
  }
  return res;
}

// Extract a sub-grid [ (v) columns start..end ) full rows | (h) rows start..end ).
Grid2D slice_panel(const Grid2D& g, const std::string& axis, int start, int end) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  Grid2D p;
  if (axis == "v") {
    for (int y = 0; y < H; ++y) {
      std::vector<int> row;
      for (int x = start; x < end; ++x) row.push_back(g[y][x]);
      p.push_back(row);
    }
  } else {
    for (int y = start; y < end; ++y) {
      std::vector<int> row(g[y].begin(), g[y].begin() + W);
      p.push_back(row);
    }
  }
  return p;
}

// boolean_combine(masks, op) over vector-of boolean masks (row-major). Returns
// (ok, result). Masks are same-shape (guaranteed by caller).
bool boolean_combine(const std::vector<std::vector<std::vector<char>>>& masks,
                     const std::string& op, std::vector<std::vector<char>>* out) {
  if (masks.empty()) return false;
  std::vector<std::vector<char>> r = masks[0];
  auto invert = [&]() { for (auto& row : r) for (char& v : row) v = v ? 0 : 1; };
  if (op == "and") { for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] &= masks[i][y][x]; *out = r; return true; }
  if (op == "or")  { for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] |= masks[i][y][x]; *out = r; return true; }
  if (op == "xor") { for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] ^= masks[i][y][x]; *out = r; return true; }
  if (op == "nand"){ for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] &= masks[i][y][x]; invert(); *out = r; return true; }
  if (op == "nor") { for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] |= masks[i][y][x]; invert(); *out = r; return true; }
  if (op == "xnor"){ for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] ^= masks[i][y][x]; invert(); *out = r; return true; }
  if (op == "diff"){ for (size_t i = 1; i < masks.size(); ++i) for (size_t y = 0; y < r.size(); ++y) for (size_t x = 0; x < r[y].size(); ++x) r[y][x] &= (masks[i][y][x] ? 0 : 1); *out = r; return true; }
  return false;
}

std::vector<std::vector<char>> nonbg_mask(const Grid2D& p, int bg) {
  std::vector<std::vector<char>> m(p.size());
  for (size_t y = 0; y < p.size(); ++y) {
    m[y].resize(p[y].size());
    for (size_t x = 0; x < p[y].size(); ++x) m[y][x] = (p[y][x] != bg) ? 1 : 0;
  }
  return m;
}

// split_panels(grid, axis, n_panels) -> (nodiv, has_nodiv, withdiv, has_withdiv).
void split_panels(const Grid2D& g, const std::string& axis, int n_panels,
                  std::vector<Grid2D>* nodiv, bool* has_nodiv,
                  std::vector<Grid2D>* withdiv, bool* has_withdiv) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  int L = (axis == "v") ? W : H;
  *has_nodiv = false; *has_withdiv = false;
  if (L % n_panels == 0) {
    int seg = L / n_panels;
    std::vector<Grid2D> ps;
    for (int i = 0; i < n_panels; ++i) ps.push_back(slice_panel(g, axis, i * seg, (i + 1) * seg));
    *nodiv = ps; *has_nodiv = true;
  }
  int rem = L - (n_panels - 1);
  if (rem > 0 && rem % n_panels == 0) {
    int seg = rem / n_panels;
    if (seg >= 1) {
      std::vector<int> div_idx;
      for (int i = 0; i < n_panels - 1; ++i) div_idx.push_back(seg + i * (seg + 1));
      std::vector<int> divs = find_uniform_divider(g, axis);
      std::set<int> div_positions(divs.begin(), divs.end());
      bool all_div = true;
      for (int di : div_idx) if (div_positions.find(di) == div_positions.end()) { all_div = false; break; }
      if (all_div) {
        std::vector<Grid2D> ps;
        int start = 0;
        for (int i = 0; i < n_panels; ++i) {
          int end = start + seg;
          ps.push_back(slice_panel(g, axis, start, end));
          start = end + 1;
        }
        *withdiv = ps; *has_withdiv = true;
      }
    }
  }
}

// split_2x2(grid) -> list of 4-panel groups (row-major TL,TR,BL,BR).
std::vector<std::vector<Grid2D>> split_2x2(const Grid2D& g) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<std::vector<Grid2D>> results;
  auto block = [&](int y0, int y1, int x0, int x1) {
    Grid2D b;
    for (int y = y0; y < y1; ++y) {
      std::vector<int> row;
      for (int x = x0; x < x1; ++x) row.push_back(g[y][x]);
      b.push_back(row);
    }
    return b;
  };
  if (H % 2 == 0 && W % 2 == 0) {
    int h2 = H / 2, w2 = W / 2;
    results.push_back({block(0, h2, 0, w2), block(0, h2, w2, W),
                       block(h2, H, 0, w2), block(h2, H, w2, W)});
  }
  if (H % 2 == 1 && W % 2 == 1) {
    int h2 = H / 2, w2 = W / 2;
    // middle row / col must be uniform.
    bool row_uniform = true;
    for (int x = 1; x < W; ++x) if (g[h2][x] != g[h2][0]) { row_uniform = false; break; }
    bool col_uniform = true;
    for (int y = 1; y < H; ++y) if (g[y][w2] != g[0][w2]) { col_uniform = false; break; }
    if (row_uniform && col_uniform) {
      results.push_back({block(0, h2, 0, w2), block(0, h2, w2 + 1, W),
                         block(h2 + 1, H, 0, w2), block(h2 + 1, H, w2 + 1, W)});
    }
  }
  return results;
}

bool same_shape(const std::vector<Grid2D>& panels) {
  if (panels.empty()) return false;
  size_t h = panels[0].size(), w = h ? panels[0][0].size() : 0;
  for (const auto& p : panels) {
    if (p.size() != h) return false;
    for (const auto& row : p) if (row.size() != w) return false;
  }
  return true;
}

// panel_split_boolean(grid, axis, op, out_color, bg, layout). Returns (ok, out).
bool panel_split_boolean(const Grid2D& g, const std::string& axis, const std::string& op,
                         int out_color, int bg, const std::string& layout, Grid2D* out) {
  auto emit = [&](const std::vector<std::vector<char>>& combined) {
    Grid2D r(combined.size());
    for (size_t y = 0; y < combined.size(); ++y) {
      r[y].resize(combined[y].size());
      for (size_t x = 0; x < combined[y].size(); ++x) r[y][x] = combined[y][x] ? out_color : bg;
    }
    return r;
  };
  if (layout == "2x2") {
    for (const auto& panels : split_2x2(g)) {
      if (!same_shape(panels)) continue;
      std::vector<std::vector<std::vector<char>>> masks;
      for (const auto& p : panels) masks.push_back(nonbg_mask(p, bg));
      std::vector<std::vector<char>> combined;
      if (!boolean_combine(masks, op, &combined)) return false;
      *out = emit(combined);
      return true;
    }
    return false;
  }
  int n_panels = (layout == "2") ? 2 : 3;
  std::vector<std::string> axes = (axis == "auto") ? std::vector<std::string>{"v", "h"}
                                                   : std::vector<std::string>{axis};
  for (const auto& ax : axes) {
    std::vector<Grid2D> nodiv, withdiv;
    bool has_nodiv = false, has_withdiv = false;
    split_panels(g, ax, n_panels, &nodiv, &has_nodiv, &withdiv, &has_withdiv);
    // Try withdiv first, then nodiv.
    for (int which = 0; which < 2; ++which) {
      const std::vector<Grid2D>& panels = (which == 0) ? withdiv : nodiv;
      bool present = (which == 0) ? has_withdiv : has_nodiv;
      if (!present || !same_shape(panels)) continue;
      std::vector<std::vector<std::vector<char>>> masks;
      for (const auto& p : panels) masks.push_back(nonbg_mask(p, bg));
      std::vector<std::vector<char>> combined;
      if (!boolean_combine(masks, op, &combined)) return false;
      *out = emit(combined);
      return true;
    }
  }
  return false;
}

// apply_fill_enclosed_bg(grid, fill_color, bg): flood from border over bg cells;
// paint every unreached bg cell fill_color.
Grid2D fill_enclosed_bg(const Grid2D& g, int fill_color, int bg) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  std::vector<std::vector<char>> visited(H, std::vector<char>(W, 0));
  std::vector<Coord> stack;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if ((y == 0 || y == H - 1 || x == 0 || x == W - 1) && g[y][x] == bg) stack.push_back({y, x});
  while (!stack.empty()) {
    Coord c = stack.back(); stack.pop_back();
    if (visited[c.first][c.second] || g[c.first][c.second] != bg) continue;
    visited[c.first][c.second] = 1;
    static const int d4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto& d : d4) {
      int ny = c.first + d[0], nx = c.second + d[1];
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && !visited[ny][nx] && g[ny][nx] == bg)
        stack.push_back({ny, nx});
    }
  }
  Grid2D out = g;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g[y][x] == bg && !visited[y][x]) out[y][x] = fill_color;
  return out;
}

// apply_gravity_to_edge(grid, direction, bg).
Grid2D gravity_to_edge(const Grid2D& g, const std::string& direction, int bg) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  Grid2D out(H, std::vector<int>(W, bg));
  if (direction == "down") {
    for (int x = 0; x < W; ++x) {
      std::vector<int> col;
      for (int y = 0; y < H; ++y) if (g[y][x] != bg) col.push_back(g[y][x]);
      for (size_t k = 0; k < col.size(); ++k) out[H - static_cast<int>(col.size()) + static_cast<int>(k)][x] = col[k];
    }
  } else if (direction == "up") {
    for (int x = 0; x < W; ++x) {
      std::vector<int> col;
      for (int y = 0; y < H; ++y) if (g[y][x] != bg) col.push_back(g[y][x]);
      for (size_t k = 0; k < col.size(); ++k) out[static_cast<int>(k)][x] = col[k];
    }
  } else if (direction == "right") {
    for (int y = 0; y < H; ++y) {
      std::vector<int> row;
      for (int x = 0; x < W; ++x) if (g[y][x] != bg) row.push_back(g[y][x]);
      for (size_t k = 0; k < row.size(); ++k) out[y][W - static_cast<int>(row.size()) + static_cast<int>(k)] = row[k];
    }
  } else if (direction == "left") {
    for (int y = 0; y < H; ++y) {
      std::vector<int> row;
      for (int x = 0; x < W; ++x) if (g[y][x] != bg) row.push_back(g[y][x]);
      for (size_t k = 0; k < row.size(); ++k) out[y][static_cast<int>(k)] = row[k];
    }
  } else {
    return g;  // unchanged copy
  }
  return out;
}

// apply_fill_diagonal_periodic(grid, diagonal, period, bg). Returns an unchanged
// copy of the grid on inconsistency.
Grid2D fill_diagonal_periodic(const Grid2D& g, const std::string& diagonal, int period, int bg) {
  int H = static_cast<int>(g.size()), W = H ? static_cast<int>(g[0].size()) : 0;
  std::map<int, int> cycle;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int c = g[y][x];
      if (c == bg) continue;
      int k = ((diagonal == "anti" ? (y + x) : (y - x)) % period + period) % period;
      auto it = cycle.find(k);
      if (it == cycle.end()) cycle[k] = c;
      else if (it->second != c) return g;  // inconsistent -> copy
    }
  Grid2D out(H, std::vector<int>(W, bg));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int k = ((diagonal == "anti" ? (y + x) : (y - x)) % period + period) % period;
      auto it = cycle.find(k);
      if (it != cycle.end()) out[y][x] = it->second;
    }
  return out;
}

// apply_tile_fold_or_nonbg(grid, out_h, out_w, bg). Returns an unchanged copy
// of the grid when dims don't divide.
Grid2D tile_fold_or_nonbg(const Grid2D& g, int out_h, int out_w, int bg) {
  int ih = static_cast<int>(g.size()), iw = ih ? static_cast<int>(g[0].size()) : 0;
  if (out_h <= 0 || out_w <= 0 || ih % out_h != 0 || iw % out_w != 0) return g;
  int ny = ih / out_h, nx = iw / out_w;
  Grid2D out(out_h, std::vector<int>(out_w, bg));
  for (int y = 0; y < out_h; ++y)
    for (int x = 0; x < out_w; ++x) {
      bool done = false;
      for (int ty = 0; ty < ny && !done; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
          int v = g[ty * out_h + y][tx * out_w + x];
          if (v != bg) { out[y][x] = v; done = true; break; }
        }
    }
  return out;
}

}  // namespace


// Per-action handlers. apply_action_to_scene tries them in a fixed order;
// several handlers cover a group of kinds and re-branch on action.kind
// internally, so the dispatch is deliberately NOT a switch. Every handler ends
// in an unconditional `return;`, so none falls through.

namespace {

void act_recolor([[maybe_unused]] Scene& scene,
                 [[maybe_unused]] const std::vector<std::string>& selected_ids,
                 [[maybe_unused]] const DslAction& action) {

    int new_color = arg_int(action.args, "color", 0);
    for (const auto& nid : selected_ids) {
      if (scene.has_node(nid)) scene.node_ref(scene.index_of(nid)).color = new_color;
    }
    return;
}

void act_delete([[maybe_unused]] Scene& scene,
                [[maybe_unused]] const std::vector<std::string>& selected_ids,
                [[maybe_unused]] const DslAction& action) {

    // delete_nodes: remove each selected node (Scene::remove_node also drops
    // it from the layer map).
    scene.remove_nodes(selected_ids);
    return;
}

void act_keep_selected([[maybe_unused]] Scene& scene,
                       [[maybe_unused]] const std::vector<std::string>& selected_ids,
                       [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    std::vector<std::tuple<int, int, int>> current = collect_colored_pixels(scene);
    // selected: iterate iter_subobjects, keep those in selected_ids, capturing
    // (color, pixel_list). Order does not affect the kept-pixel set.
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    std::vector<std::pair<int, std::vector<Coord>>> selected;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) != selected_set.end())
        selected.emplace_back(node_color(*n, 0), n->pixel_list);
    }
    // KEEP_SELECTED: kept = selected pixels only (in-bounds), then reset.
    std::vector<std::tuple<int, int, int>> kept;
    for (const auto& sc : selected) {
      int color = sc.first;
      for (const auto& px : sc.second) {
        if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
          kept.emplace_back(px.first, px.second, color);
      }
    }
    reset_scene_subobjects(scene, kept, h, w);
    return;
}

void act_swap_colors([[maybe_unused]] Scene& scene,
                     [[maybe_unused]] const std::vector<std::string>& selected_ids,
                     [[maybe_unused]] const DslAction& action) {

    int a = arg_int(action.args, "color_a", 0);
    int b = arg_int(action.args, "color_b", 0);
    if (a == b) return;
    int bg = scene.has_background() ? scene.background() : -1;
    if (bg == a) scene.set_background(b);
    else if (bg == b) scene.set_background(a);
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      int cur = n->color.has_value() ? *n->color : -1;
      if (cur == a) n->color = b;
      else if (cur == b) n->color = a;
    }
    return;
}

// RECOLOR_TO to=background | most | least (formerly recolor_to_background,
// recolor_to_most_frequent, recolor_to_least_frequent).
void act_recolor_to([[maybe_unused]] Scene& scene,
                    [[maybe_unused]] const std::vector<std::string>& selected_ids,
                    [[maybe_unused]] const DslAction& action) {

    // Resolve the bulk recolour target.
    int new_color;
    const std::string to = action_variant(action);
    if (to == "background") {
      new_color = scene.has_background() ? scene.background() : 0;
    } else {
      // color_mass: mass per colour over iter_subobjects, in first-seen (lex-id) order.
      std::vector<std::pair<int, int>> color_mass;  // (color, mass) in first-seen order
      auto bump = [&](int c, int m) {
        for (auto& kv : color_mass) if (kv.first == c) { kv.second += m; return; }
        color_mass.emplace_back(c, m);
      };
      for (const Node* n : iter_subobjects(scene)) bump(node_color(*n, 0), n->mass);
      if (color_mass.empty()) return;
      if (to == "most") {
        // max by (mass, -color): higher mass; on tie LOWER color.
        new_color = color_mass[0].first; int bm = color_mass[0].second;
        for (const auto& kv : color_mass)
          if (kv.second > bm || (kv.second == bm && (-kv.first) > (-new_color))) {
            new_color = kv.first; bm = kv.second;
          }
      } else {
        // min by (mass, color): lower mass; on tie LOWER color.
        new_color = color_mass[0].first; int bm = color_mass[0].second;
        for (const auto& kv : color_mass)
          if (kv.second < bm || (kv.second == bm && kv.first < new_color)) {
            new_color = kv.first; bm = kv.second;
          }
      }
    }
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n) n->color = new_color;
    }
    return;
}

void act_recolor_from_edge_majority([[maybe_unused]] Scene& scene,
                                    [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                    [[maybe_unused]] const DslAction& action) {

    std::string edge_type_name = to_upper(strip(arg_str(action.args, "edge_type", "TOUCHES")));
    std::string direction = to_lower(arg_str(action.args, "direction", "both"));
    const EdgeNeighbors nbr(scene, edge_type_name, direction);
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      // votes: count per neighbour colour, max by (count, -color).
      std::vector<std::pair<int, int>> votes;  // (color, count) in first-seen order
      auto bump = [&](int c) {
        for (auto& kv : votes) if (kv.first == c) { kv.second += 1; return; }
        votes.emplace_back(c, 1);
      };
      nbr.for_each(nid, [&](const std::string& nb) {
        const Node* nn = find_node(scene, nb);
        if (nn != nullptr) bump(node_color(*nn, 0));
        return false;
      });
      if (votes.empty()) continue;
      int best = votes[0].first; int bc = votes[0].second;
      for (const auto& kv : votes)
        if (kv.second > bc || (kv.second == bc && (-kv.first) > (-best))) {
          best = kv.first; bc = kv.second;
        }
      n->color = best;
    }
    return;
}

void act_recolor_sequence([[maybe_unused]] Scene& scene,
                          [[maybe_unused]] const std::vector<std::string>& selected_ids,
                          [[maybe_unused]] const DslAction& action) {

    std::string order = to_lower(strip(arg_str(action.args, "order", "y_then_x")));
    int start_color = as_int(args_get(action.args, "start_color"), 1);
    int step = as_int(args_get(action.args, "step"), 1);
    // selected_nodes: in selected_ids order (present only), sorted by the order key below.
    struct SelRow { std::string nid; Node* n; };
    std::vector<SelRow> rows;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n) rows.push_back({nid, n});
    }
    // Order key (data, order, node_id): a tuple chosen by `order`. bbox missing
    // sentinel (1e9,1e9,-1,-1). Uses stable sort.
    auto key = [&](const SelRow& r, long long out[4]) {
      long long ymin, xmin;
      long long mass = r.n->mass;
      if (r.n->has_bbox) { ymin = r.n->bbox_ymin; xmin = r.n->bbox_xmin; }
      else { ymin = 1000000000LL; xmin = 1000000000LL; }
      if (order == "x_then_y") { out[0] = xmin; out[1] = ymin; out[2] = -mass; }
      else if (order == "mass_desc") { out[0] = -mass; out[1] = ymin; out[2] = xmin; }
      else if (order == "mass_asc") { out[0] = mass; out[1] = ymin; out[2] = xmin; }
      else { out[0] = ymin; out[1] = xmin; out[2] = -mass; }
    };
    std::stable_sort(rows.begin(), rows.end(), [&](const SelRow& a, const SelRow& b) {
      long long ka[4], kb[4];
      key(a, ka); key(b, kb);
      for (int i = 0; i < 3; ++i) if (ka[i] != kb[i]) return ka[i] < kb[i];
      return a.nid < b.nid;
    });
    for (size_t i = 0; i < rows.size(); ++i) {
      long long c = (static_cast<long long>(start_color) +
                     static_cast<long long>(i) * step) % 10;
      if (c < 0) c += 10;  // wrap to a non-negative colour (floor modulo)
      rows[i].n->color = static_cast<int>(c);
    }
    return;
}

void act_recolor_from_reference([[maybe_unused]] Scene& scene,
                                [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                [[maybe_unused]] const DslAction& action) {

    // act_recolor_from_reference. Nested selector picks a reference set;
    // policies nearest / majority / only, match same_shape / same_shape_strict /
    // any.
    std::string sel_kind_raw =
        strip(arg_str(action.args, "reference_selector_kind",
                      to_string(SelectorKind::NON_BACKGROUND)));
    const DslValue* sa = args_get(action.args, "reference_selector_args");
    DslArgs sel_args = (sa && sa->tag == DslValue::Tag::Dict && sa->dict) ? *sa->dict : DslArgs{};
    std::string color_policy = to_lower(strip(arg_str(action.args, "color_policy", "nearest")));
    if (color_policy != "nearest" && color_policy != "majority" && color_policy != "only")
      color_policy = "nearest";
    std::string match_policy = to_lower(strip(arg_str(action.args, "match_policy", "same_shape")));
    if (match_policy != "same_shape" && match_policy != "same_shape_strict" &&
        match_policy != "any")
      match_policy = "same_shape";
    bool exclude_selected = as_bool(args_get(action.args, "exclude_selected"), true);

    SelectorKind ref_kind;
    if (!selector_from_name(sel_kind_raw, &ref_kind, &sel_args)) ref_kind = SelectorKind::NON_BACKGROUND;
    DslSelector ref_sel; ref_sel.kind = ref_kind; ref_sel.args = sel_args;
    std::vector<std::string> references = select_node_ids(scene, ref_sel);
    if (exclude_selected) {
      std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
      std::vector<std::string> filt;
      for (const auto& rid : references)
        if (selected_set.find(rid) == selected_set.end()) filt.push_back(rid);
      references = filt;
    }
    if (references.empty()) return;

    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::vector<std::string> candidates = references;
      if (match_policy == "same_shape" || match_policy == "same_shape_strict") {
        std::string target_shape = n->shape_hash.has_value() ? *n->shape_hash : std::string();
        bool target_has = n->shape_hash.has_value();
        std::vector<std::string> same;
        for (const auto& rid : references) {
          const Node* rn = find_node(scene, rid);
          if (rn == nullptr) continue;
          bool rhas = rn->shape_hash.has_value();
          std::string rshape = rhas ? *rn->shape_hash : std::string();
          // shape_hash comparison: equal iff both present and equal, or both
          // absent.
          if (rhas == target_has && rshape == target_shape) same.push_back(rid);
        }
        if (!same.empty()) candidates = same;
        else if (match_policy == "same_shape_strict") continue;
      }
      if (candidates.empty()) continue;

      int new_color;
      if (color_policy == "majority") {
        std::vector<std::pair<int, int>> votes;  // (color, weight) in first-seen order
        auto bump = [&](int c, int w) {
          for (auto& kv : votes) if (kv.first == c) { kv.second += w; return; }
          votes.emplace_back(c, w);
        };
        for (const auto& rid : candidates) {
          const Node* rn = find_node(scene, rid);
          if (rn == nullptr) continue;
          bump(node_color(*rn, 0), std::max(1, rn->mass));
        }
        if (votes.empty()) continue;
        new_color = votes[0].first; int bc = votes[0].second;
        for (const auto& kv : votes)
          if (kv.second > bc || (kv.second == bc && (-kv.first) > (-new_color))) {
            new_color = kv.first; bc = kv.second;
          }
      } else if (color_policy == "only") {
        // set of colors; require exactly one distinct.
        std::set<int> colors;
        for (const auto& rid : candidates) {
          const Node* rn = find_node(scene, rid);
          if (rn) colors.insert(node_color(*rn, 0));
        }
        if (colors.size() != 1) continue;
        new_color = *colors.begin();
      } else {  // nearest
        double tcy = n->has_centroid ? n->centroid_y : 0.0;
        double tcx = n->has_centroid ? n->centroid_x : 0.0;
        bool have_best = false;
        double best_dist = 0.0;
        std::string best_id;
        int best_color = 0;
        for (const auto& rid : candidates) {
          const Node* rn = find_node(scene, rid);
          if (rn == nullptr) continue;
          double rcy = rn->has_centroid ? rn->centroid_y : 0.0;
          double rcx = rn->has_centroid ? rn->centroid_x : 0.0;
          double d = (tcy - rcy) * (tcy - rcy) + (tcx - rcx) * (tcx - rcx);
          // key = (dist, reference id); pick min.
          if (!have_best || d < best_dist ||
              (d == best_dist && rid < best_id)) {
            have_best = true; best_dist = d; best_id = rid;
            best_color = node_color(*rn, 0);
          }
        }
        if (!have_best) continue;
        new_color = best_color;
      }
      n->color = new_color;
    }
    return;
}

void act_prune_endpoints([[maybe_unused]] Scene& scene,
                         [[maybe_unused]] const std::vector<std::string>& selected_ids,
                         [[maybe_unused]] const DslAction& action) {

    int iterations = std::max(1, as_int(args_get(action.args, "iterations"), 1));
    int connectivity = as_int(args_get(action.args, "connectivity"), 4);
    if (connectivity != 4 && connectivity != 8) connectivity = 4;
    bool protect = as_bool(args_get(action.args, "protect_junctions"), false);
    std::vector<std::string> empty_nodes;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::set<Coord> pixels(n->pixel_list.begin(), n->pixel_list.end());
      std::set<Coord> pruned = prune_endpoints_pixels(pixels, iterations, connectivity, protect);
      n->pixel_list.assign(pruned.begin(), pruned.end());  // sorted (set is ordered)
      update_geometry(*n);
      if (pruned.empty()) empty_nodes.push_back(nid);
    }
    if (!empty_nodes.empty()) delete_nodes(scene, empty_nodes);
    return;
}

void act_shrink_to_core([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    int connectivity = as_int(args_get(action.args, "connectivity"), 4);
    if (connectivity != 4 && connectivity != 8) connectivity = 4;
    std::vector<std::string> empty_nodes;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::set<Coord> current(n->pixel_list.begin(), n->pixel_list.end());
      if (current.empty()) { empty_nodes.push_back(nid); continue; }
      while (true) {
        std::set<Coord> nxt = prune_endpoints_pixels(current, 1, connectivity, false);
        if (nxt.size() == current.size()) break;
        current = nxt;
        if (current.empty()) break;
      }
      n->pixel_list.assign(current.begin(), current.end());
      update_geometry(*n);
      if (current.empty()) empty_nodes.push_back(nid);
    }
    if (!empty_nodes.empty()) delete_nodes(scene, empty_nodes);
    return;
}

void act_dilate_selected([[maybe_unused]] Scene& scene,
                         [[maybe_unused]] const std::vector<std::string>& selected_ids,
                         [[maybe_unused]] const DslAction& action) {

    int radius = std::max(1, as_int(args_get(action.args, "radius"), 1));
    int connectivity = as_int(args_get(action.args, "connectivity"), 4);
    if (connectivity != 4 && connectivity != 8) connectivity = 4;
    std::string mask = to_lower(strip(arg_str(action.args, "mask", "auto")));
    if (mask != "auto" && mask != "cross" && mask != "square" && mask != "diamond")
      mask = "auto";
    std::vector<Coord> offsets = dilation_offsets(radius, mask, connectivity);
    int h = scene.height(), w = scene.width();
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    int bg = scene.has_background() ? scene.background() : 0;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) == selected_set.end()) continue;
      int color = node_color(*n, bg);
      for (const auto& px : n->pixel_list)
        for (const auto& off : offsets) {
          int ny = px.first + off.first, nx = px.second + off.second;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) updated.emplace_back(ny, nx, color);
        }
    }
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_hollow([[maybe_unused]] Scene& scene,
                [[maybe_unused]] const std::vector<std::string>& selected_ids,
                [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    int bg = scene.has_background() ? scene.background() : 0;
    int fill = as_int(args_get(action.args, "color"), bg);
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    std::set<Coord> occupied;
    for (const Node* n : iter_subobjects(scene)) {
      if (node_color(*n, bg) == bg) continue;
      for (const auto& px : n->pixel_list) occupied.insert(px);
    }
    std::vector<std::tuple<int, int, int>> new_pixels;
    for (const Node* n : iter_subobjects(scene)) {
      int color = node_color(*n, bg);
      if (color == bg) continue;
      const std::vector<Coord>& pixels = n->pixel_list;
      if (pixels.empty()) continue;  // degenerate node with no pixels: nothing to
                                     // paint in any branch, and it would make the
                                     // FILL_BBOX pixels[0] below an out-of-bounds
                                     // read (container-overflow / SIGSEGV at scale).
      if (selected_set.find(n->node_id) == selected_set.end()) {
        for (const auto& px : pixels) new_pixels.emplace_back(px.first, px.second, color);
        continue;
      }
      std::set<Coord> pixset(pixels.begin(), pixels.end());
      // FILL_BBOX part=interior (formerly hollow): the object's interior
      // cells take the fill colour.
      if (action_variant(action) == "interior") {
        static const int d4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& px : pixels) {
          bool interior = true;
          for (const auto& d : d4) {
            Coord q = {px.first + d[0], px.second + d[1]};
            if (pixset.find(q) == pixset.end()) { interior = false; break; }
          }
          int c = interior ? fill : color;
          if (c != bg) new_pixels.emplace_back(px.first, px.second, c);
        }
      } else {  // FILL_BBOX part=gaps: the free cells of the bbox take it
        int ymn = pixels[0].first, ymx = pixels[0].first;
        int xmn = pixels[0].second, xmx = pixels[0].second;
        for (const auto& px : pixels) {
          new_pixels.emplace_back(px.first, px.second, color);
          ymn = std::min(ymn, px.first); ymx = std::max(ymx, px.first);
          xmn = std::min(xmn, px.second); xmx = std::max(xmx, px.second);
        }
        if (fill != bg) {
          for (int y = ymn; y <= ymx; ++y)
            for (int x = xmn; x <= xmx; ++x)
              if (occupied.find({y, x}) == occupied.end())
                new_pixels.emplace_back(y, x, fill);
        }
      }
    }
    reset_scene_subobjects(scene, new_pixels, h, w);
    return;
}

void act_outline_object([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    int bg = scene.has_background() ? scene.background() : 0;
    int outline = as_int(args_get(action.args, "color"), bg);
    if (outline == bg) return;
    int connectivity = as_int(args_get(action.args, "connectivity"), 8);
    if (connectivity != 4 && connectivity != 8) connectivity = 8;
    std::vector<Coord> offsets = neighbor_offsets(connectivity);
    std::set<Coord> occupied;
    for (const Node* n : iter_subobjects(scene)) {
      if (node_color(*n, bg) == bg) continue;
      for (const auto& px : n->pixel_list) occupied.insert(px);
    }
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    std::set<Coord> ring;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) == selected_set.end()) continue;
      for (const auto& px : n->pixel_list)
        for (const auto& off : offsets) {
          int ny = px.first + off.first, nx = px.second + off.second;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w && occupied.find({ny, nx}) == occupied.end())
            ring.insert({ny, nx});
        }
    }
    for (const auto& p : ring) updated.emplace_back(p.first, p.second, outline);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_translate([[maybe_unused]] Scene& scene,
                   [[maybe_unused]] const std::vector<std::string>& selected_ids,
                   [[maybe_unused]] const DslAction& action) {

    int dy = arg_int(action.args, "dy", 0);
    int dx = arg_int(action.args, "dx", 0);
    int h = scene.height(), w = scene.width();
    // TRANSLATE mode=move | wrap (formerly translate, translate_wrap).
    bool wrap = action_variant(action) == "wrap";
    if (wrap && (h <= 0 || w <= 0)) return;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::set<Coord> moved;
      for (const auto& px : n->pixel_list) {
        if (wrap) {
          int ny = ((px.first + dy) % h + h) % h;
          int nx = ((px.second + dx) % w + w) % w;
          moved.insert({ny, nx});
        } else {
          int ny = px.first + dy, nx = px.second + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) moved.insert({ny, nx});
        }
      }
      n->pixel_list.assign(moved.begin(), moved.end());
      update_geometry(*n);
    }
    return;
}

void act_translate_by_own_extent([[maybe_unused]] Scene& scene,
                                 [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                 [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    std::string src = to_lower(strip(arg_str(action.args, "extent_source", "width")));
    if (src != "width" && src != "height" && src != "mass") src = "width";
    int mult_y = as_int(args_get(action.args, "mult_y"), 0);
    int mult_x = as_int(args_get(action.args, "mult_x"), 0);
    if (mult_y == 0 && mult_x == 0) return;
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    for (const Node* cn : iter_subobjects(scene)) {
      if (selected_set.find(cn->node_id) == selected_set.end()) continue;
      Node* n = find_node_mut(scene, cn->node_id);
      int base = (src == "width") ? n->width : (src == "height") ? n->height : n->mass;
      if (base <= 0) continue;
      int dy = base * mult_y, dx = base * mult_x;
      std::set<Coord> moved;
      for (const auto& px : n->pixel_list) {
        int ny = px.first + dy, nx = px.second + dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) moved.insert({ny, nx});
      }
      n->pixel_list.assign(moved.begin(), moved.end());
      update_geometry(*n);
    }
    return;
}

void act_normalize_to_origin([[maybe_unused]] Scene& scene,
                             [[maybe_unused]] const std::vector<std::string>& selected_ids,
                             [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    std::vector<Coord> selected_pixels;
    for (const auto& nid : selected_ids) {
      const Node* n = find_node(scene, nid);
      if (n) selected_pixels.insert(selected_pixels.end(), n->pixel_list.begin(), n->pixel_list.end());
    }
    if (selected_pixels.empty()) return;
    int min_y = selected_pixels[0].first, min_x = selected_pixels[0].second;
    for (const auto& p : selected_pixels) {
      min_y = std::min(min_y, p.first);
      min_x = std::min(min_x, p.second);
    }
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::set<Coord> moved;
      for (const auto& px : n->pixel_list) {
        int ny = px.first - min_y, nx = px.second - min_x;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) moved.insert({ny, nx});
      }
      n->pixel_list.assign(moved.begin(), moved.end());
      update_geometry(*n);
    }
    return;
}

void act_mirror_h([[maybe_unused]] Scene& scene,
                  [[maybe_unused]] const std::vector<std::string>& selected_ids,
                  [[maybe_unused]] const DslAction& action) {

    // MIRROR axis=h | v (formerly mirror_h, mirror_v).
    bool horizontal = action_variant(action) == "h";
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      if (!n->has_bbox) continue;
      int ymin = n->bbox_ymin, xmin = n->bbox_xmin, ymax = n->bbox_ymax, xmax = n->bbox_xmax;
      std::set<Coord> mirrored;
      for (const auto& px : n->pixel_list) {
        if (horizontal) mirrored.insert({px.first, xmin + xmax - px.second});
        else mirrored.insert({ymin + ymax - px.first, px.second});
      }
      n->pixel_list.assign(mirrored.begin(), mirrored.end());
      update_geometry(*n);
    }
    return;
}

void act_copy_translate([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    std::vector<std::tuple<int, int, int>> current = collect_colored_pixels(scene);
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    // selected: (color, pixels) in iter_subobjects order.
    std::vector<std::pair<int, std::vector<Coord>>> selected;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) != selected_set.end())
        selected.emplace_back(node_color(*n, 0), n->pixel_list);
    }
    // TRANSLATE mode=copy (formerly copy_translate).
    if (action.kind == ActionKind::TRANSLATE) {
      int dy = arg_int(action.args, "dy", 0);
      int dx = arg_int(action.args, "dx", 0);
      std::vector<std::tuple<int, int, int>> updated = current;
      for (const auto& sc : selected) {
        int color = sc.first;
        for (const auto& px : sc.second) {
          int ny = px.first + dy, nx = px.second + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) updated.emplace_back(ny, nx, color);
        }
      }
      reset_scene_subobjects(scene, updated, h, w);
      return;
    }
    // KEEP_SELECTED out=crop (formerly keep_selected_normalized).
    std::vector<std::tuple<int, int, int>> kept;
    for (const auto& sc : selected) {
      int color = sc.first;
      for (const auto& px : sc.second)
        if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
          kept.emplace_back(px.first, px.second, color);
    }
    if (kept.empty()) { reset_scene_subobjects(scene, kept, h, w); return; }
    int min_y = std::get<0>(kept[0]), min_x = std::get<1>(kept[0]);
    for (const auto& t : kept) {
      min_y = std::min(min_y, std::get<0>(t));
      min_x = std::min(min_x, std::get<1>(t));
    }
    std::vector<std::tuple<int, int, int>> normalized;
    int new_h = 0, new_w = 0;
    for (const auto& t : kept) {
      int y = std::get<0>(t) - min_y, x = std::get<1>(t) - min_x;
      normalized.emplace_back(y, x, std::get<2>(t));
      new_h = std::max(new_h, y);
      new_w = std::max(new_w, x);
    }
    reset_scene_subobjects(scene, normalized, new_h + 1, new_w + 1);
    return;
}

void act_stamp_local_pattern([[maybe_unused]] Scene& scene,
                             [[maybe_unused]] const std::vector<std::string>& selected_ids,
                             [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    std::vector<Coord> offsets = stamp_pattern_offsets(arg_str(action.args, "pattern", "cross"));
    std::string anchor = to_lower(strip(arg_str(action.args, "anchor", "centroid")));
    if (anchor != "centroid" && anchor != "bbox_center" && anchor != "min_corner")
      anchor = "centroid";
    int bg = scene.has_background() ? scene.background() : 0;
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      int ay, ax;
      if (!resolve_anchor(*n, anchor, &ay, &ax)) continue;
      int color = resolve_action_color(*n, action.args, bg);
      for (const auto& off : offsets) {
        int ny = ay + off.first, nx = ax + off.second;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) updated.emplace_back(ny, nx, color);
      }
    }
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_complete_d4_symmetry([[maybe_unused]] Scene& scene,
                              [[maybe_unused]] const std::vector<std::string>& selected_ids,
                              [[maybe_unused]] const DslAction& action) {

    bool include_pivot_color = as_bool(args_get(action.args, "include_pivot_color"), false);
    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    // color_pixels (a set per colour) and color_mass, with colours in first-seen
    // order over iter_subobjects.
    std::vector<int> color_order;  // first-seen colors
    std::map<int, std::set<Coord>> color_pixels;
    std::vector<std::pair<int, int>> color_mass;  // (color, mass) in first-seen order
    auto bump_mass = [&](int c, int m) {
      for (auto& kv : color_mass) if (kv.first == c) { kv.second += m; return; }
      color_mass.emplace_back(c, m);
    };
    for (const Node* n : iter_subobjects(scene)) {
      int color = n->color.has_value() ? *n->color : -1;
      if (color < 0 || n->pixel_list.empty()) continue;
      bump_mass(color, static_cast<int>(n->pixel_list.size()));
      if (color_pixels.find(color) == color_pixels.end()) color_order.push_back(color);
      for (const auto& px : n->pixel_list) color_pixels[color].insert(px);
    }
    if (color_mass.empty()) return;
    // pivot = max by (mass, -color).
    int pivot_color = color_mass[0].first; int bm = color_mass[0].second;
    for (const auto& kv : color_mass)
      if (kv.second > bm || (kv.second == bm && (-kv.first) > (-pivot_color))) {
        pivot_color = kv.first; bm = kv.second;
      }
    auto pit = color_pixels.find(pivot_color);
    if (pit == color_pixels.end() || pit->second.empty()) return;
    long long sy = 0, sx = 0;
    for (const auto& p : pit->second) { sy += p.first; sx += p.second; }
    int npv = static_cast<int>(pit->second.size());
    int py = static_cast<int>(py_round(static_cast<double>(sy) / npv));
    int px = static_cast<int>(py_round(static_cast<double>(sx) / npv));
    std::vector<std::tuple<int, int, int>> additions;
    // Iterate color_pixels in first-seen colour order.
    for (int color : color_order) {
      if (!include_pivot_color && color == pivot_color) continue;
      for (const auto& p : color_pixels[color]) {
        int dy = p.first - py, dx = p.second - px;
        int orbit[8][2] = {{dy, dx}, {-dy, dx}, {dy, -dx}, {-dy, -dx},
                           {dx, dy}, {-dx, dy}, {dx, -dy}, {-dx, -dy}};
        for (auto& o : orbit) {
          int ny = py + o[0], nx = px + o[1];
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) additions.emplace_back(ny, nx, color);
        }
      }
    }
    if (additions.empty()) return;
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& a : additions) updated.push_back(a);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_project_inner_quad_to_outer_corners([[maybe_unused]] Scene& scene,
                                             [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                             [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    std::vector<const Node*> subs = iter_subobjects(scene);
    if (subs.empty()) return;
    // largest = max by (mass, node id).
    const Node* largest = subs[0];
    for (const Node* n : subs)
      if (n->mass > largest->mass || (n->mass == largest->mass && n->node_id > largest->node_id))
        largest = n;
    if (!largest->has_bbox) return;
    int ymin = largest->bbox_ymin, xmin = largest->bbox_xmin;
    int ymax = largest->bbox_ymax, xmax = largest->bbox_xmax;
    // inner singleton nodes strictly inside the largest bbox.
    std::vector<std::pair<Coord, int>> inner;
    for (const Node* n : subs) {
      if (n->node_id == largest->node_id) continue;
      if (n->pixel_list.size() != 1) continue;
      int y = n->pixel_list[0].first, x = n->pixel_list[0].second;
      if (!(ymin < y && y < ymax && xmin < x && x < xmax)) continue;
      inner.emplace_back(Coord{y, x}, n->color.has_value() ? *n->color : 0);
    }
    if (inner.size() != 4) return;
    // pos_to_color map; ys/xs sorted unique.
    std::map<Coord, int> pos_to_color;
    for (const auto& kv : inner) pos_to_color[kv.first] = kv.second;
    std::set<int> ys, xs;
    for (const auto& kv : pos_to_color) { ys.insert(kv.first.first); xs.insert(kv.first.second); }
    if (ys.size() != 2 || xs.size() != 2) return;
    std::vector<int> ysv(ys.begin(), ys.end()), xsv(xs.begin(), xs.end());
    Coord ul = {ysv[0], xsv[0]}, ur = {ysv[0], xsv[1]};
    Coord ll = {ysv[1], xsv[0]}, lr = {ysv[1], xsv[1]};
    for (const Coord& c : {ul, ur, ll, lr})
      if (pos_to_color.find(c) == pos_to_color.end()) return;
    Coord t_tl = {ymin - 1, xmin - 1}, t_tr = {ymin - 1, xmax + 1};
    Coord t_bl = {ymax + 1, xmin - 1}, t_br = {ymax + 1, xmax + 1};
    for (const Coord& t : {t_tl, t_tr, t_bl, t_br})
      if (!(t.first >= 0 && t.first < h && t.second >= 0 && t.second < w)) return;
    std::vector<std::tuple<int, int, int>> updated;
    for (const auto& t : collect_colored_pixels(scene)) {
      Coord c = {std::get<0>(t), std::get<1>(t)};
      if (pos_to_color.find(c) == pos_to_color.end()) updated.push_back(t);
    }
    updated.emplace_back(t_tl.first, t_tl.second, pos_to_color[lr]);
    updated.emplace_back(t_tr.first, t_tr.second, pos_to_color[ll]);
    updated.emplace_back(t_bl.first, t_bl.second, pos_to_color[ur]);
    updated.emplace_back(t_br.first, t_br.second, pos_to_color[ul]);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_project_singletons_to_full_lines([[maybe_unused]] Scene& scene,
                                          [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                          [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    int bg = scene.has_background() ? scene.background() : 0;
    // base_grid: scene.grid if it matches (h,w); else zeros filled from
    // collect_colored_pixels (bg cells -> 0, NOT scene bg).
    std::vector<std::vector<int>> base_grid(h, std::vector<int>(w, 0));
    bool use_scene_grid = scene.has_grid() && scene.grid().height() == h && scene.grid().width() == w;
    if (use_scene_grid) {
      const Grid& g = scene.grid();
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) base_grid[y][x] = g.at(y, x);
    } else {
      for (const auto& t : collect_colored_pixels(scene)) {
        int y = std::get<0>(t), x = std::get<1>(t), c = std::get<2>(t);
        if (y >= 0 && y < h && x >= 0 && x < w) base_grid[y][x] = c;
      }
    }
    // line_candidates[color] = list of (orientation, line_pos) in scan order:
    // rows first (top->bottom), then cols (left->right).
    std::map<int, std::vector<std::pair<int, int>>> line_candidates;  // color -> [(orient01,pos)] 0=h,1=v
    for (int y = 0; y < h; ++y) {
      std::set<int> row_colors;
      for (int x = 0; x < w; ++x) row_colors.insert(base_grid[y][x]);
      if (row_colors.size() == 1) {
        int c = *row_colors.begin();
        if (c != bg) line_candidates[c].push_back({0, y});
      }
    }
    for (int x = 0; x < w; ++x) {
      std::set<int> col_colors;
      for (int y = 0; y < h; ++y) col_colors.insert(base_grid[y][x]);
      if (col_colors.size() == 1) {
        int c = *col_colors.begin();
        if (c != bg) line_candidates[c].push_back({1, x});
      }
    }
    // singleton_nodes: mass-1 SubObjects with color >= 0, iter_subobjects order.
    struct SglNode { int color, y, x; };
    std::vector<SglNode> singletons;
    for (const Node* n : iter_subobjects(scene)) {
      if (n->mass != 1) continue;
      int color = n->color.has_value() ? *n->color : -1;
      if (color < 0) continue;
      if (n->pixel_list.empty()) continue;
      singletons.push_back({color, n->pixel_list[0].first, n->pixel_list[0].second});
    }
    if (line_candidates.empty()) return;
    // additions map + delete positions.
    std::map<Coord, int> additions;
    std::set<Coord> delete_positions;
    for (const auto& s : singletons) {
      delete_positions.insert({s.y, s.x});
      auto it = line_candidates.find(s.color);
      if (it == line_candidates.end() || it->second.empty()) continue;
      bool have_best = false;
      int best_orient = 0, best_pos = 0, best_dist = 0;
      for (const auto& cand : it->second) {
        int orient = cand.first, line_pos = cand.second;
        int dist = std::abs((orient == 0 ? s.y : s.x) - line_pos);
        if (!have_best || dist < best_dist) {
          have_best = true; best_orient = orient; best_pos = line_pos; best_dist = dist;
        }
      }
      if (!have_best) continue;
      int ty, tx;
      if (best_orient == 0) {  // horizontal
        if (s.y == best_pos) continue;
        ty = s.y < best_pos ? best_pos - 1 : best_pos + 1;
        tx = s.x;
      } else {  // vertical
        if (s.x == best_pos) continue;
        ty = s.y;
        tx = s.x < best_pos ? best_pos - 1 : best_pos + 1;
      }
      if (ty >= 0 && ty < h && tx >= 0 && tx < w) additions[{ty, tx}] = s.color;
    }
    if (additions.empty() && delete_positions.empty()) return;
    // out_grid = base_grid copy; clear singleton positions to bg; apply additions.
    std::vector<std::vector<int>> out_grid = base_grid;
    for (const auto& p : delete_positions)
      if (p.first >= 0 && p.first < h && p.second >= 0 && p.second < w) out_grid[p.first][p.second] = bg;
    for (const auto& kv : additions)
      if (kv.first.first >= 0 && kv.first.first < h && kv.first.second >= 0 && kv.first.second < w)
        out_grid[kv.first.first][kv.first.second] = kv.second;
    std::vector<std::tuple<int, int, int>> updated;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (out_grid[y][x] != bg) updated.emplace_back(y, x, out_grid[y][x]);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_symmetrize_object([[maybe_unused]] Scene& scene,
                           [[maybe_unused]] const std::vector<std::string>& selected_ids,
                           [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    int bg = scene.has_background() ? scene.background() : 0;
    std::string orientation = to_lower(strip(arg_str(action.args, "orientation", "v")));
    if (orientation != "h" && orientation != "v") orientation = "v";
    std::string mode = to_lower(strip(arg_str(action.args, "mode", "intersect")));
    if (mode != "intersect" && mode != "union") mode = "intersect";
    int radius = as_int(args_get(action.args, "cluster_radius"), 1);
    if (radius < 0) radius = 0;

    // Full colour grid (all non-bg pixels), iter_subobjects order (last wins).
    std::vector<std::vector<int>> grid(h, std::vector<int>(w, bg));
    for (const Node* n : iter_subobjects(scene)) {
      int c = n->color.has_value() ? *n->color : bg;
      for (const auto& px : n->pixel_list)
        if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
          grid[px.first][px.second] = c;
    }
    // Selected pixels per color (gathered in iter_subobjects order).
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    std::map<int, std::vector<Coord>> sel_by_color;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) == selected_set.end()) continue;
      int c = n->color.has_value() ? *n->color : bg;
      if (c == bg) continue;
      for (const auto& px : n->pixel_list)
        if (px.first >= 0 && px.first < h && px.second >= 0 && px.second < w)
          sel_by_color[c].push_back(px);
    }
    if (sel_by_color.empty()) return;

    std::set<Coord> remove;
    std::vector<std::tuple<int, int, int>> add;
    for (const auto& kv : sel_by_color) {  // std::map -> ascending color
      int color = kv.first;
      std::set<Coord> pix(kv.second.begin(), kv.second.end());
      // mask + dilate by radius (8-connected), label components.
      std::vector<std::vector<char>> mask(h, std::vector<char>(w, 0));
      for (const auto& p : pix) mask[p.first][p.second] = 1;
      // 8-connected dilation `radius` times.
      for (int it = 0; it < radius; ++it) {
        std::vector<std::vector<char>> nxt = mask;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            if (mask[y][x])
              for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                  int ny = y + dy, nx = x + dx;
                  if (ny >= 0 && ny < h && nx >= 0 && nx < w) nxt[ny][nx] = 1;
                }
        mask = nxt;
      }
      // Label 8-connected components of dilated mask (row-major scan, DFS).
      std::vector<std::vector<int>> labels(h, std::vector<int>(w, 0));
      int cur = 0;
      for (int sy = 0; sy < h; ++sy)
        for (int sx = 0; sx < w; ++sx) {
          if (!mask[sy][sx] || labels[sy][sx]) continue;
          ++cur;
          std::vector<Coord> stack{{sy, sx}};
          labels[sy][sx] = cur;
          while (!stack.empty()) {
            Coord c = stack.back(); stack.pop_back();
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                int ny = c.first + dy, nx = c.second + dx;
                if (ny >= 0 && ny < h && nx >= 0 && nx < w && mask[ny][nx] && !labels[ny][nx]) {
                  labels[ny][nx] = cur;
                  stack.push_back({ny, nx});
                }
              }
          }
        }
      for (int lab = 1; lab <= cur; ++lab) {
        // fig = sorted pixels of `pix` whose label == lab.
        std::vector<Coord> fig;
        for (const auto& p : pix)
          if (labels[p.first][p.second] == lab) fig.push_back(p);
        std::sort(fig.begin(), fig.end());
        if (fig.size() < 2) continue;
        int ymn = fig[0].first, ymx = fig[0].first, xmn = fig[0].second, xmx = fig[0].second;
        for (const auto& p : fig) {
          ymn = std::min(ymn, p.first); ymx = std::max(ymx, p.first);
          xmn = std::min(xmn, p.second); xmx = std::max(xmx, p.second);
        }
        int lo = (orientation == "h") ? 2 * ymn : 2 * xmn;
        int hi = (orientation == "h") ? 2 * ymx : 2 * xmx;
        bool have_best = false;
        int best_agree = 0; double best_ratio = 0.0; int best_a2 = 0;
        for (int a2 = lo; a2 <= hi; ++a2) {
          int agree = 0, total = 0;
          for (const auto& p : fig) {
            int my = (orientation == "h") ? a2 - p.first : p.first;
            int mx = (orientation == "h") ? p.second : a2 - p.second;
            if (my >= 0 && my < h && mx >= 0 && mx < w) {
              ++total;
              if (grid[my][mx] == color) ++agree;
            }
          }
          if (total == 0) continue;
          double ratio = static_cast<double>(agree) / total;
          // score = (agree, agree/total); pick max (strict >).
          if (!have_best || agree > best_agree ||
              (agree == best_agree && ratio > best_ratio)) {
            have_best = true; best_agree = agree; best_ratio = ratio; best_a2 = a2;
          }
        }
        if (!have_best) continue;
        int a2 = best_a2;
        for (const auto& p : fig) {
          int my = (orientation == "h") ? a2 - p.first : p.first;
          int mx = (orientation == "h") ? p.second : a2 - p.second;
          bool in_bounds = my >= 0 && my < h && mx >= 0 && mx < w;
          if (mode == "intersect") {
            if (!(in_bounds && grid[my][mx] == color)) remove.insert(p);
          } else {  // union
            if (in_bounds && grid[my][mx] == bg) add.emplace_back(my, mx, color);
          }
        }
      }
    }
    if (remove.empty() && add.empty()) return;
    std::vector<std::tuple<int, int, int>> colored;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (grid[y][x] != bg && remove.find({y, x}) == remove.end())
          colored.emplace_back(y, x, grid[y][x]);
    for (const auto& a : add) colored.push_back(a);
    reset_scene_subobjects(scene, colored, h, w);
    return;
}

void act_crop_to_object([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    Grid2D out;
    if (!crop_to_object(g, arg_str(action.args, "which", "largest"),
                        arg_str(action.args, "recolor", "none"),
                        arg_str(action.args, "obj_mode", "multi"), &out))
      return;
    install_grid_into_scene(scene, out);
    return;
}

void act_scale_by_count([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    Grid2D out;
    if (!scale_by_count(g, arg_str(action.args, "mode", "scale"),
                        arg_str(action.args, "count_type", "colors"),
                        as_bool(args_get(action.args, "color_strict"), false),
                        as_bool(args_get(action.args, "diag"), false), &out))
      return;
    install_grid_into_scene(scene, out);
    return;
}

void act_extend_line_until([[maybe_unused]] Scene& scene,
                           [[maybe_unused]] const std::vector<std::string>& selected_ids,
                           [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    std::string direction = to_lower(strip(arg_str(action.args, "direction", "ray")));
    std::string stop_condition = to_lower(strip(arg_str(action.args, "stop_condition", "boundary")));
    if (stop_condition != "boundary" && stop_condition != "color!=bg" &&
        stop_condition != "color_not_bg" && stop_condition != "touch_object")
      stop_condition = "boundary";
    std::string anchor = to_lower(strip(arg_str(action.args, "anchor", "centroid")));
    if (anchor != "centroid" && anchor != "bbox_center" && anchor != "min_corner")
      anchor = "centroid";
    std::vector<Coord> directions;
    if (direction == "ray") directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    else if (direction == "ray8")
      directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    else if (direction == "diag") directions = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    else if (direction == "up") directions = {{-1, 0}};
    else if (direction == "down") directions = {{1, 0}};
    else if (direction == "left") directions = {{0, -1}};
    else if (direction == "right") directions = {{0, 1}};
    else if (direction == "up_left") directions = {{-1, -1}};
    else if (direction == "up_right") directions = {{-1, 1}};
    else if (direction == "down_left") directions = {{1, -1}};
    else if (direction == "down_right") directions = {{1, 1}};
    else directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    bool stop_on_same = as_bool(args_get(action.args, "stop_on_same_color"), false);
    const DslValue* inter_v = args_get(action.args, "intersection_color");
    int inter_c = (inter_v != nullptr && inter_v->tag != DslValue::Tag::Null)
                      ? as_int(inter_v, -1) : -1;

    int bg_color = scene.has_background() ? scene.background() : 0;
    Grid2D base_grid(h, std::vector<int>(w, bg_color));
    for (const auto& t : collect_colored_pixels(scene)) {
      int y = std::get<0>(t), x = std::get<1>(t), c = std::get<2>(t);
      if (y >= 0 && y < h && x >= 0 && x < w) base_grid[y][x] = c;
    }
    std::map<Coord, int> additions;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      std::set<Coord> own(n->pixel_list.begin(), n->pixel_list.end());
      int ay, ax;
      if (!resolve_anchor(*n, anchor, &ay, &ax)) continue;
      int color = resolve_action_color(*n, action.args, bg_color);
      for (const auto& d : directions) {
        int ny = ay + d.first, nx = ax + d.second;
        while (ny >= 0 && ny < h && nx >= 0 && nx < w) {
          int cell = base_grid[ny][nx];
          bool occupied_foreign = cell != bg_color && own.find({ny, nx}) == own.end();
          if (stop_on_same && occupied_foreign && cell == color) break;
          if ((stop_condition == "color!=bg" || stop_condition == "color_not_bg" ||
               stop_condition == "touch_object") && occupied_foreign)
            break;
          auto it = additions.find({ny, nx});
          if (it != additions.end() && it->second != color && inter_c >= 0) {
            additions[{ny, nx}] = inter_c;
          } else if (it == additions.end()) {
            additions[{ny, nx}] = color;
          }
          if (stop_condition == "boundary" && (ny == 0 || ny == h - 1 || nx == 0 || nx == w - 1))
            break;
          ny += d.first; nx += d.second;
        }
      }
    }
    if (additions.empty()) return;
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& kv : additions) updated.emplace_back(kv.first.first, kv.first.second, kv.second);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_project_marker([[maybe_unused]] Scene& scene,
                        [[maybe_unused]] const std::vector<std::string>& selected_ids,
                        [[maybe_unused]] const DslAction& action) {

    std::string side = to_lower(strip(arg_str(action.args, "side", "bottom")));
    if (side != "top" && side != "bottom" && side != "left" && side != "right") side = "bottom";
    std::string align = to_lower(strip(arg_str(action.args, "align", "centroid")));
    if (align != "centroid" && align != "bbox_center") align = "centroid";
    std::string color_mode = to_lower(strip(arg_str(action.args, "color_mode", "self")));
    if (color_mode != "self" && color_mode != "constant" && color_mode != "background")
      color_mode = "self";
    int const_color = as_int(args_get(action.args, "color"), 4);
    int h = scene.height(), w = scene.width();
    int bg_color = scene.has_background() ? scene.background() : 0;
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      int base_y, base_x;
      if (!resolve_anchor(*n, align, &base_y, &base_x)) continue;
      int py, px;
      if (side == "top") { py = 0; px = std::max(0, std::min(w - 1, base_x)); }
      else if (side == "bottom") { py = std::max(0, h - 1); px = std::max(0, std::min(w - 1, base_x)); }
      else if (side == "left") { py = std::max(0, std::min(h - 1, base_y)); px = 0; }
      else { py = std::max(0, std::min(h - 1, base_y)); px = std::max(0, w - 1); }
      int color;
      if (color_mode == "constant") color = const_color;
      else if (color_mode == "background") color = bg_color;
      else color = n->color.has_value() ? *n->color : bg_color;
      if (py >= 0 && py < h && px >= 0 && px < w) updated.emplace_back(py, px, color);
    }
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_attach_same_color_singletons([[maybe_unused]] Scene& scene,
                                      [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                      [[maybe_unused]] const DslAction& action) {

    int min_line_length = std::max(2, as_int(args_get(action.args, "min_line_length"), 4));
    bool delete_source = as_bool(args_get(action.args, "delete_source"), true);
    int h = scene.height(), w = scene.width();
    std::set<std::string> selected_set;
    for (const auto& nid : selected_ids) if (scene.has_node(nid)) selected_set.insert(nid);
    if (selected_set.empty()) return;
    // singleton_refs: non-selected mass-1 SubObjects, iter_subobjects order.
    struct RefRow { std::string nid; int color; int y, x; };
    std::vector<RefRow> singleton_refs;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) != selected_set.end()) continue;
      if (n->mass != 1) continue;
      int color = n->color.has_value() ? *n->color : 0;
      int y = n->pixel_list.empty() ? 0 : n->pixel_list[0].first;
      int x = n->pixel_list.empty() ? 0 : n->pixel_list[0].second;
      singleton_refs.push_back({n->node_id, color, y, x});
    }
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    std::set<Coord> delete_coords;
    for (const auto& nid : selected_ids) {
      Node* n = find_node_mut(scene, nid);
      if (n == nullptr) continue;
      if (!n->has_bbox) continue;
      int ymin = n->bbox_ymin, xmin = n->bbox_xmin, ymax = n->bbox_ymax, xmax = n->bbox_xmax;
      int obj_h = ymax - ymin + 1, obj_w = xmax - xmin + 1;
      int color = n->color.has_value() ? *n->color : (scene.has_background() ? scene.background() : 0);
      bool is_vertical = obj_h >= min_line_length && obj_h > obj_w && obj_w <= 2;
      bool is_horizontal = obj_w >= min_line_length && obj_w > obj_h && obj_h <= 2;
      if (!(is_vertical || is_horizontal)) continue;
      for (const auto& r : singleton_refs) {
        if (!scene.has_node(r.nid)) continue;  // reference already removed
        if (r.color != color) continue;
        int ry = r.y, rx = r.x, ty, tx;
        if (is_vertical) {
          if (rx < xmin) tx = xmin - 1;
          else if (rx > xmax) tx = xmax + 1;
          else continue;
          ty = std::max(ymin, std::min(ymax, ry));
        } else {
          if (ry < ymin) ty = ymin - 1;
          else if (ry > ymax) ty = ymax + 1;
          else continue;
          tx = std::max(xmin, std::min(xmax, rx));
        }
        if (ty >= 0 && ty < h && tx >= 0 && tx < w) {
          updated.emplace_back(ty, tx, color);
          if (delete_source) delete_coords.insert({ry, rx});
        }
      }
    }
    if (!delete_coords.empty()) {
      std::vector<std::tuple<int, int, int>> filt;
      for (const auto& t : updated)
        if (delete_coords.find({std::get<0>(t), std::get<1>(t)}) == delete_coords.end())
          filt.push_back(t);
      updated = filt;
    }
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_fill_between_collinear_same_color_pairs([[maybe_unused]] Scene& scene,
                                                 [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                                 [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0 || selected_ids.empty()) return;
    const DslValue* fc = args_get(action.args, "fill_color");
    bool fill_endpoint = true;
    int fill_const = 0;
    if (fc != nullptr) {
      if (fc->tag == DslValue::Tag::Str && fc->s == "endpoint") fill_endpoint = true;
      else { fill_endpoint = false; fill_const = as_int(fc, 0); }
    }
    std::string axes = to_lower(strip(arg_str(action.args, "axes", "both")));
    if (axes != "both" && axes != "horizontal" && axes != "vertical" &&
        axes != "diagonal" && axes != "all")
      axes = "both";
    bool do_h = axes == "both" || axes == "horizontal" || axes == "all";
    bool do_v = axes == "both" || axes == "vertical" || axes == "all";
    bool do_d = axes == "diagonal" || axes == "all";
    int bg = scene.has_background() ? scene.background() : 0;
    Grid2D base_grid(h, std::vector<int>(w, bg));
    for (const auto& t : collect_colored_pixels(scene)) {
      int y = std::get<0>(t), x = std::get<1>(t), c = std::get<2>(t);
      if (y >= 0 && y < h && x >= 0 && x < w) base_grid[y][x] = c;
    }
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    // singletons: selected mass-1, color >=0 and != bg, in iter_subobjects order.
    struct Sgl { int y, x, c; };
    std::vector<Sgl> singletons;
    for (const Node* n : iter_subobjects(scene)) {
      if (selected_set.find(n->node_id) == selected_set.end()) continue;
      if (n->mass != 1) continue;
      if (n->pixel_list.empty()) continue;
      int color = n->color.has_value() ? *n->color : -1;
      if (color < 0 || color == bg) continue;
      singletons.push_back({n->pixel_list[0].first, n->pixel_list[0].second, color});
    }
    if (singletons.size() < 2) return;
    // by_color: colours in first-seen order, preserving per-color point order.
    std::vector<int> color_order;
    std::map<int, std::vector<Coord>> by_color;
    for (const auto& s : singletons) {
      if (by_color.find(s.c) == by_color.end()) color_order.push_back(s.c);
      by_color[s.c].push_back({s.y, s.x});
    }
    std::map<Coord, int> additions;
    for (int c : color_order) {
      int fill = fill_endpoint ? c : fill_const;
      const std::vector<Coord>& pts = by_color[c];
      for (size_t i = 0; i < pts.size(); ++i)
        for (size_t j = i + 1; j < pts.size(); ++j) {
          int y1 = pts[i].first, x1 = pts[i].second;
          int y2 = pts[j].first, x2 = pts[j].second;
          int dy = y2 - y1, dx = x2 - x1;
          if (do_h && y1 == y2) {
            int lo = std::min(x1, x2), hi = std::max(x1, x2);
            bool blocked = false;
            for (int xx = lo + 1; xx < hi; ++xx) if (base_grid[y1][xx] != bg) { blocked = true; break; }
            if (blocked) continue;
            for (int xx = lo + 1; xx < hi; ++xx) additions[{y1, xx}] = fill;
          } else if (do_v && x1 == x2) {
            int lo = std::min(y1, y2), hi = std::max(y1, y2);
            bool blocked = false;
            for (int yy = lo + 1; yy < hi; ++yy) if (base_grid[yy][x1] != bg) { blocked = true; break; }
            if (blocked) continue;
            for (int yy = lo + 1; yy < hi; ++yy) additions[{yy, x1}] = fill;
          } else if (do_d && std::abs(dy) == std::abs(dx) && dy != 0) {
            int step_y = dy > 0 ? 1 : -1, step_x = dx > 0 ? 1 : -1;
            int yy = y1 + step_y, xx = x1 + step_x;
            std::vector<Coord> path;
            bool clear = true;
            while (!(yy == y2 && xx == x2)) {
              if (base_grid[yy][xx] != bg) { clear = false; break; }
              path.push_back({yy, xx});
              yy += step_y; xx += step_x;
            }
            if (clear) for (const auto& p : path) additions[{p.first, p.second}] = fill;
          }
        }
    }
    if (additions.empty()) return;
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& kv : additions) updated.emplace_back(kv.first.first, kv.first.second, kv.second);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_delete_reference_after_transfer([[maybe_unused]] Scene& scene,
                                         [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                         [[maybe_unused]] const DslAction& action) {

    std::string sel_kind_raw =
        strip(arg_str(action.args, "reference_selector_kind",
                      to_string(SelectorKind::REFERENCE_SINGLETONS)));
    const DslValue* sa = args_get(action.args, "reference_selector_args");
    DslArgs sel_args = (sa && sa->tag == DslValue::Tag::Dict && sa->dict) ? *sa->dict : DslArgs{};
    SelectorKind sk;
    if (!selector_from_name(sel_kind_raw, &sk, &sel_args)) sk = SelectorKind::REFERENCE_SINGLETONS;
    DslSelector sel; sel.kind = sk; sel.args = sel_args;
    std::vector<std::string> refs = select_node_ids(scene, sel);
    if (!refs.empty()) delete_nodes(scene, refs);
    return;
}

void act_fill_cavity([[maybe_unused]] Scene& scene,
                     [[maybe_unused]] const std::vector<std::string>& selected_ids,
                     [[maybe_unused]] const DslAction& action) {

    const DslValue* tc = args_get(action.args, "color");
    if (tc == nullptr || tc->tag == DslValue::Tag::Null) return;
    int target_color = as_int(tc, 0);
    std::set<std::string> selected_set(selected_ids.begin(), selected_ids.end());
    std::vector<Coord> cavity_pixels;
    // Iterate ALL scene nodes in insertion order for CavityObjects.
    for (const Node& n : scene.nodes()) {
      if (n.node_type != "CavityObject") continue;
      bool directly = selected_set.find(n.node_id) != selected_set.end();
      bool enclosed = n.enclosing_object_id.has_value() &&
                      selected_set.find(*n.enclosing_object_id) != selected_set.end();
      if (directly || enclosed)
        for (const auto& px : n.pixel_list) cavity_pixels.push_back(px);
    }
    if (cavity_pixels.empty()) return;
    int h = scene.height(), w = scene.width();
    std::vector<std::tuple<int, int, int>> updated = collect_colored_pixels(scene);
    for (const auto& p : cavity_pixels)
      if (p.first >= 0 && p.first < h && p.second >= 0 && p.second < w)
        updated.emplace_back(p.first, p.second, target_color);
    reset_scene_subobjects(scene, updated, h, w);
    return;
}

void act_panel_split_boolean([[maybe_unused]] Scene& scene,
                             [[maybe_unused]] const std::vector<std::string>& selected_ids,
                             [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    std::string bg_policy = arg_str(action.args, "bg_policy", "zero");
    int bg = (bg_policy == "detected") ? detect_bg2d(g) : 0;
    Grid2D out;
    if (!panel_split_boolean(g, arg_str(action.args, "axis", "auto"),
                             arg_str(action.args, "op", "and"),
                             as_int(args_get(action.args, "out_color"), 1), bg,
                             arg_str(action.args, "layout", "2"), &out))
      return;
    install_grid_into_scene(scene, out);
    return;
}

void act_fill_enclosed_bg([[maybe_unused]] Scene& scene,
                          [[maybe_unused]] const std::vector<std::string>& selected_ids,
                          [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    int bg = scene.has_background() ? scene.background() : 0;
    Grid2D out = fill_enclosed_bg(g, as_int(args_get(action.args, "fill_color"), 4), bg);
    install_grid_into_scene(scene, out);
    return;
}

void act_gravity_to_edge([[maybe_unused]] Scene& scene,
                         [[maybe_unused]] const std::vector<std::string>& selected_ids,
                         [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    int bg = scene.has_background() ? scene.background() : 0;
    Grid2D out = gravity_to_edge(g, arg_str(action.args, "direction", "down"), bg);
    install_grid_into_scene(scene, out);
    return;
}

void act_fill_diagonal_periodic([[maybe_unused]] Scene& scene,
                                [[maybe_unused]] const std::vector<std::string>& selected_ids,
                                [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    int bg = scene.has_background() ? scene.background() : 0;
    int period = arg_int(action.args, "period", 3);
    if (period <= 0) return;
    Grid2D out = fill_diagonal_periodic(g, arg_str(action.args, "diagonal", "anti"), period, bg);
    install_grid_into_scene(scene, out);
    return;
}

void act_tile_fold_or_nonbg([[maybe_unused]] Scene& scene,
                            [[maybe_unused]] const std::vector<std::string>& selected_ids,
                            [[maybe_unused]] const DslAction& action) {

    int h = scene.height(), w = scene.width();
    if (h <= 0 || w <= 0) return;
    int out_h = arg_int(action.args, "out_h", 0);
    int out_w = arg_int(action.args, "out_w", 0);
    if (out_h <= 0 || out_w <= 0) {
      int div_y = arg_int(action.args, "div_y", 0);
      int div_x = arg_int(action.args, "div_x", 0);
      if (div_y <= 0 || div_x <= 0) return;
      if (h % div_y != 0 || w % div_x != 0) return;
      out_h = h / div_y; out_w = w / div_x;
    }
    if (out_h <= 0 || out_w <= 0 || h % out_h != 0 || w % out_w != 0) return;
    Grid2D g = render_scene_to_grid2d(scene);
    int bg = scene.has_background() ? scene.background() : 0;
    Grid2D out = tile_fold_or_nonbg(g, out_h, out_w, bg);
    install_grid_into_scene(scene, out);
    return;
}

}  // namespace

void apply_action_to_scene(Scene& scene, const std::vector<std::string>& selected_ids,
                           const DslAction& action) {

  if (const ConceptOp* op = find_concept_op(action.kind)) { apply_concept_op(*op, scene, selected_ids, action); return; }
  if (action.kind == ActionKind::RECOLOR) { act_recolor(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::DELETE) { act_delete(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::KEEP_SELECTED) {
    if (action_variant(action) == "crop") act_copy_translate(scene, selected_ids, action);
    else act_keep_selected(scene, selected_ids, action);
    return;
  }
  if (action.kind == ActionKind::SWAP_COLORS) { act_swap_colors(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::RECOLOR_TO) { act_recolor_to(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::RECOLOR_FROM_EDGE_MAJORITY) { act_recolor_from_edge_majority(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::RECOLOR_SEQUENCE) { act_recolor_sequence(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::RECOLOR_FROM_REFERENCE) { act_recolor_from_reference(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::PRUNE_ENDPOINTS) { act_prune_endpoints(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::SHRINK_TO_CORE) { act_shrink_to_core(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::DILATE_SELECTED) { act_dilate_selected(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::FILL_BBOX) { act_hollow(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::OUTLINE_OBJECT) { act_outline_object(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::TRANSLATE) {
    const std::string mode = action_variant(action);
    // mode=origin: formerly normalize_to_origin (dy, dx unused).
    if (mode == "copy") act_copy_translate(scene, selected_ids, action);
    else if (mode == "origin") act_normalize_to_origin(scene, selected_ids, action);
    else act_translate(scene, selected_ids, action);
    return;
  }
  if (action.kind == ActionKind::TRANSLATE_BY_OWN_EXTENT) { act_translate_by_own_extent(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::MIRROR) { act_mirror_h(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::STAMP_LOCAL_PATTERN) { act_stamp_local_pattern(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::COMPLETE_D4_SYMMETRY) { act_complete_d4_symmetry(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::PROJECT_INNER_QUAD_TO_OUTER_CORNERS) { act_project_inner_quad_to_outer_corners(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::PROJECT_SINGLETONS_TO_FULL_LINES) { act_project_singletons_to_full_lines(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::SYMMETRIZE_OBJECT) { act_symmetrize_object(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::CROP_TO_OBJECT) { act_crop_to_object(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::SCALE_BY_COUNT) { act_scale_by_count(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::EXTEND_LINE_UNTIL) { act_extend_line_until(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::PROJECT_MARKER) { act_project_marker(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::ATTACH_SAME_COLOR_SINGLETONS) { act_attach_same_color_singletons(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::FILL_BETWEEN_COLLINEAR_SAME_COLOR_PAIRS) { act_fill_between_collinear_same_color_pairs(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::DELETE_REFERENCE_AFTER_TRANSFER) { act_delete_reference_after_transfer(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::FILL_CAVITY) { act_fill_cavity(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::PANEL_SPLIT_BOOLEAN) { act_panel_split_boolean(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::FILL_ENCLOSED_BG) { act_fill_enclosed_bg(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::GRAVITY_TO_EDGE) { act_gravity_to_edge(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::FILL_DIAGONAL_PERIODIC) { act_fill_diagonal_periodic(scene, selected_ids, action); return; }
  if (action.kind == ActionKind::TILE_FOLD_OR_NONBG) { act_tile_fold_or_nonbg(scene, selected_ids, action); return; }
}


}  // namespace arc
