// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 18: object attributes (see arc/object_attrs.hpp).
//
// Shape:     height, width, area, holes (enclosed background regions), hole_cells,
//            perimeter (cells with a 4-neighbour outside the object), is_rect,
//            is_frame, is_hline, is_vline, is_square, is_single, sym_lr, sym_ud
// Relations: n_same_color, n_same_shape, n_same_size, n_touching (other objects
//            8-adjacent), n_inside (objects whose bbox lies inside this bbox),
//            is_inside (inside another object's bbox), dist_border
// Ranks:     rank_size (1 = largest, dense), rank_size_asc, rank_height,
//            rank_width, rank_x (1 = leftmost), rank_y (1 = topmost)
// Position:  side_y, side_x (bbox centre vs grid centre: 0 before, 1 on, 2 after),
//            on_center (bbox covers the grid centre), rank_center (1 = closest
//            to the grid centre)
// Reference: rel_y_big, rel_x_big (0 before, 1 overlapping rows / columns,
//            2 after, 3 = the largest object itself), touch_big (8-adjacent to
//            the largest object), in_big (bbox inside the largest object's bbox),
//            rel_y_line, rel_x_line (0 before, 2 after the divider row / column
//            with the most cells, 1 none / on it). A divider spans the grid edge to edge (a
//            dashed one may end one cell short) in one colour, filling at least half of it (dashed lines count), and
//            the rows / columns beside it are not that colour where it is.
// Shape:     orient (0 taller than wide, 1 square bbox, 2 wider than tall)
// Position:  rank_center_small (rank_center with ties broken by mass, smaller first)
// Reference: rel_y_uniq, rel_x_uniq (as rel_y_big, against the first object whose
//            colour no other object has, the largest of them; 1 when none)
// Depth:     is_front (a straight line through the object leaves it into the same
//            other colour on both sides, or it covers part of the bbox of a
//            touching other-colour object that misses cells there), is_behind
//            (the object it lies across), is_top (in front, behind nothing: the
//            top of a stack)
// Colour:    color_rank (1 = the colour with the most object cells, dense),
//            is_rare_color (the colour with the fewest), rank_in_color (by mass among its colour, 1 = largest)
// Layout:    n_row_aligned / n_col_aligned (other objects sharing a row / a column),
//            is_between (objects on both sides along a shared row or column),
//            gap_near (empty cells to the nearest object, Chebyshev, capped at 9)
// Shape:     is_diag (a diagonal line), shape_kind (0 single, 1 horizontal line,
//            2 vertical line, 3 diagonal line, 4 solid rectangle, 5 frame, 6 other),
//            n_same_kind (other objects of the same shape kind)

#include "arc/object_attrs.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc {

namespace {

enum A {
  HEIGHT, WIDTH, AREA, HOLES, HOLE_CELLS, PERIMETER, IS_RECT, IS_FRAME, IS_HLINE, IS_VLINE,
  IS_SQUARE, IS_SINGLE, SYM_LR, SYM_UD, N_SAME_COLOR, N_SAME_SHAPE, N_SAME_SIZE, N_TOUCHING,
  N_INSIDE, IS_INSIDE, DIST_BORDER, RANK_SIZE, RANK_SIZE_ASC, RANK_HEIGHT, RANK_WIDTH, RANK_X,
  RANK_Y, SIDE_Y, SIDE_X, ON_CENTER, RANK_CENTER, REL_Y_BIG, REL_X_BIG, TOUCH_BIG, IN_BIG,
  REL_Y_LINE, REL_X_LINE, ORIENT, RANK_CENTER_SMALL, REL_Y_UNIQ, REL_X_UNIQ, IS_FRONT, IS_BEHIND,
  COLOR_RANK, IS_RARE_COLOR, RANK_IN_COLOR, N_ROW_ALIGNED, N_COL_ALIGNED, IS_BETWEEN,
  GAP_NEAR, IS_DIAG, SHAPE_KIND, N_SAME_KIND, IS_TOP, N_ATTRS
};

// 0 when [a0, a1] lies before [b0, b1], 2 after, 1 overlapping.
int side_of(int a0, int a1, int b0, int b1) { return a1 < b0 ? 0 : a0 > b1 ? 2 : 1; }

struct Obj {
  size_t idx;
  int y0, x0, y1, x1, mass, color;
  // bbox-local occupancy, row-major: mask[y * mw + x]
  int mw = 0;
  std::vector<char> mask;
  char m(int y, int x) const { return mask[static_cast<size_t>(y) * mw + x]; }
  std::string shape;                     // mask as a string, for equality
  int v[N_ATTRS] = {};
};

void shape_attrs(Obj& o) {
  int h = o.y1 - o.y0 + 1, w = o.x1 - o.x0 + 1;
  o.v[HEIGHT] = h;
  o.v[WIDTH] = w;
  o.v[AREA] = h * w;
  o.v[IS_RECT] = o.mass == h * w;
  o.v[IS_HLINE] = h == 1 && w > 1 && o.mass == w;
  o.v[IS_VLINE] = w == 1 && h > 1 && o.mass == h;
  o.v[IS_SQUARE] = h == w;
  o.v[IS_SINGLE] = o.mass == 1;
  o.v[ORIENT] = h > w ? 0 : h < w ? 2 : 1;
  bool frame = h >= 3 && w >= 3;
  for (int y = 0; y < h && frame; ++y)
    for (int x = 0; x < w && frame; ++x) {
      bool border = y == 0 || y == h - 1 || x == 0 || x == w - 1;
      frame = border ? o.m(y, x) : !o.m(y, x);
    }
  o.v[IS_FRAME] = frame;
  bool lr = true, ud = true;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      lr = lr && o.m(y, x) == o.m(y, w - 1 - x);
      ud = ud && o.m(y, x) == o.m(h - 1 - y, x);
    }
  o.v[SYM_LR] = lr;
  o.v[SYM_UD] = ud;
  int per = 0;
  const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (!o.m(y, x)) continue;
      for (int k = 0; k < 4; ++k) {
        int p = y + dy[k], q = x + dx[k];
        if (p < 0 || p >= h || q < 0 || q >= w || !o.m(p, q)) { ++per; break; }
      }
    }
  o.v[PERIMETER] = per;
  // Enclosed background regions: 4-connected empty regions of the bbox that do
  // not reach its border.
  std::vector<int> lab_(static_cast<size_t>(h) * w, 0);
  auto lab = [&](int y, int x) -> int& { return lab_[static_cast<size_t>(y) * w + x]; };
  int holes = 0, cells = 0, next = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (o.m(y, x) || lab(y, x)) continue;
      ++next;
      std::vector<std::pair<int, int>> st{{y, x}};
      lab(y, x) = next;
      bool open = false;
      int n = 0;
      while (!st.empty()) {
        auto c = st.back();
        st.pop_back();
        ++n;
        if (c.first == 0 || c.first == h - 1 || c.second == 0 || c.second == w - 1) open = true;
        for (int k = 0; k < 4; ++k) {
          int p = c.first + dy[k], q = c.second + dx[k];
          if (p < 0 || p >= h || q < 0 || q >= w || o.m(p, q) || lab(p, q)) continue;
          lab(p, q) = next;
          st.push_back({p, q});
        }
      }
      if (!open) { ++holes; cells += n; }
    }
  o.v[HOLES] = holes;
  o.v[HOLE_CELLS] = cells;
}

// Dense rank of key(o) over objs; ascending unless desc.
template <typename F>
void dense_rank(std::vector<Obj>& objs, int slot, bool desc, F key) {
  int lo = key(objs[0]), hi = lo;
  for (const Obj& o : objs) { lo = std::min(lo, key(o)); hi = std::max(hi, key(o)); }
  if (static_cast<long long>(hi) - lo < 4096) {  // small range: rank by a presence table
    std::vector<int> rank(static_cast<size_t>(hi - lo) + 1, 0);
    for (const Obj& o : objs) rank[key(o) - lo] = 1;
    int n = 0;
    for (int& r : rank) r = r ? ++n : 0;  // 1-based ascending rank of each present value
    for (Obj& o : objs) {
      const int pos = rank[key(o) - lo] - 1;
      o.v[slot] = desc ? n - pos : pos + 1;
    }
    return;
  }
  std::vector<int> vals;
  vals.reserve(objs.size());
  for (const Obj& o : objs) vals.push_back(key(o));
  std::sort(vals.begin(), vals.end());
  vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
  const int n = static_cast<int>(vals.size());
  for (Obj& o : objs) {
    const int pos = static_cast<int>(std::lower_bound(vals.begin(), vals.end(), key(o)) - vals.begin());
    o.v[slot] = desc ? n - pos : pos + 1;
  }
}

}  // namespace

const std::vector<std::string>& object_attr_names() {
  static const std::vector<std::string> names = {
      "height", "width", "area", "holes", "hole_cells", "perimeter", "is_rect", "is_frame",
      "is_hline", "is_vline", "is_square", "is_single", "sym_lr", "sym_ud", "n_same_color",
      "n_same_shape", "n_same_size", "n_touching", "n_inside", "is_inside", "dist_border",
      "rank_size", "rank_size_asc", "rank_height", "rank_width", "rank_x", "rank_y",
      "side_y", "side_x", "on_center", "rank_center", "rel_y_big", "rel_x_big", "touch_big",
      "in_big", "rel_y_line", "rel_x_line", "orient", "rank_center_small", "rel_y_uniq", "rel_x_uniq", "is_front", "is_behind",
      "color_rank", "is_rare_color", "rank_in_color", "n_row_aligned", "n_col_aligned",
      "is_between", "gap_near", "is_diag", "shape_kind", "n_same_kind", "is_top"};
  return names;
}

bool node_attr(const Node& n, const std::string& name, int* value) {
  if (!n.attrs || n.attrs->empty()) return false;
  static const std::unordered_map<std::string, int> slot = [] {
    std::unordered_map<std::string, int> m;
    const auto& names = object_attr_names();
    for (size_t i = 0; i < names.size(); ++i) m.emplace(names[i], static_cast<int>(i));
    return m;
  }();
  auto it = slot.find(name);
  if (it == slot.end() || it->second >= static_cast<int>(n.attrs->size())) return false;
  *value = (*n.attrs)[it->second];
  return true;
}

void compute_object_attributes(Scene& scene) {
  std::vector<size_t> idxs = scene.get_nodes_by_type("SubObject", false);
  std::vector<Obj> objs;
  for (size_t i : idxs) {
    const Node& n = scene.nodes()[i];
    if (n.pixel_list.empty()) continue;
    Obj o;
    o.idx = i;
    o.y0 = o.y1 = n.pixel_list[0].first;
    o.x0 = o.x1 = n.pixel_list[0].second;
    for (const auto& p : n.pixel_list) {
      o.y0 = std::min(o.y0, p.first); o.y1 = std::max(o.y1, p.first);
      o.x0 = std::min(o.x0, p.second); o.x1 = std::max(o.x1, p.second);
    }
    o.mass = static_cast<int>(n.pixel_list.size());
    o.color = n.color.has_value() ? *n.color : -1;
    o.mw = o.x1 - o.x0 + 1;
    o.mask.assign(static_cast<size_t>(o.y1 - o.y0 + 1) * o.mw, 0);
    for (const auto& p : n.pixel_list) o.mask[static_cast<size_t>(p.first - o.y0) * o.mw + (p.second - o.x0)] = 1;
    o.shape = std::to_string(o.mw) + ":";
    o.shape.reserve(o.shape.size() + o.mask.size());
    for (char c : o.mask) o.shape += c ? '1' : '0';
    shape_attrs(o);
    objs.push_back(std::move(o));
  }
  if (objs.empty()) return;

  std::map<int, int> by_color;
  std::map<std::string, int> by_shape;
  std::map<std::pair<int, int>, int> by_size;
  for (const Obj& o : objs) {
    by_color[o.color]++;
    by_shape[o.shape]++;
    by_size[{o.v[HEIGHT], o.v[WIDTH]}]++;
  }
  int H = scene.height(), W = scene.width();
  // Cell -> object index, for adjacency.
  std::vector<std::vector<int>> owner(H, std::vector<int>(W, -1));
  for (size_t k = 0; k < objs.size(); ++k)
    for (const auto& p : scene.nodes()[objs[k].idx].pixel_list)
      if (p.first >= 0 && p.first < H && p.second >= 0 && p.second < W) owner[p.first][p.second] = static_cast<int>(k);
  std::vector<int> touch_stamp(objs.size(), -1);
  for (size_t k = 0; k < objs.size(); ++k) {
    Obj& o = objs[k];
    o.v[N_SAME_COLOR] = by_color[o.color] - 1;
    o.v[N_SAME_SHAPE] = by_shape[o.shape] - 1;
    o.v[N_SAME_SIZE] = by_size[{o.v[HEIGHT], o.v[WIDTH]}] - 1;
    o.v[DIST_BORDER] = std::min(std::min(o.y0, o.x0), std::min(H - 1 - o.y1, W - 1 - o.x1));
    int n_touch = 0;  // distinct other owners of the 8 neighbours (stamped with k)
    for (const auto& p : scene.nodes()[o.idx].pixel_list)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          int y = p.first + dy, x = p.second + dx;
          if (y < 0 || y >= H || x < 0 || x >= W) continue;
          int w = owner[y][x];
          if (w >= 0 && w != static_cast<int>(k) && touch_stamp[w] != static_cast<int>(k)) {
            touch_stamp[w] = static_cast<int>(k);
            ++n_touch;
          }
        }
    o.v[N_TOUCHING] = n_touch;
  }
  if (objs.size() <= 300) {  // bbox containment is quadratic
    for (Obj& a : objs)
      for (const Obj& b : objs) {
        if (&a == &b) continue;
        bool b_in_a = b.y0 >= a.y0 && b.y1 <= a.y1 && b.x0 >= a.x0 && b.x1 <= a.x1 &&
                      (b.y0 > a.y0 || b.y1 < a.y1 || b.x0 > a.x0 || b.x1 < a.x1);
        if (b_in_a) a.v[N_INSIDE]++;
        bool a_in_b = a.y0 >= b.y0 && a.y1 <= b.y1 && a.x0 >= b.x0 && a.x1 <= b.x1 &&
                      (a.y0 > b.y0 || a.y1 < b.y1 || a.x0 > b.x0 || a.x1 < b.x1);
        if (a_in_b) a.v[IS_INSIDE] = 1;
      }
  }
  dense_rank(objs, RANK_SIZE, true, [](const Obj& o) { return o.mass; });
  dense_rank(objs, RANK_SIZE_ASC, false, [](const Obj& o) { return o.mass; });
  dense_rank(objs, RANK_HEIGHT, true, [](const Obj& o) { return o.v[HEIGHT]; });
  dense_rank(objs, RANK_WIDTH, true, [](const Obj& o) { return o.v[WIDTH]; });
  dense_rank(objs, RANK_X, false, [](const Obj& o) { return o.x0; });
  dense_rank(objs, RANK_Y, false, [](const Obj& o) { return o.y0; });

  // Position against the grid centre (doubled coordinates, so no rounding).
  for (Obj& o : objs) {
    int cy = o.y0 + o.y1 - (H - 1), cx = o.x0 + o.x1 - (W - 1);
    o.v[SIDE_Y] = cy < 0 ? 0 : cy > 0 ? 2 : 1;
    o.v[SIDE_X] = cx < 0 ? 0 : cx > 0 ? 2 : 1;
    o.v[ON_CENTER] = 2 * o.y0 <= H - 1 && 2 * o.y1 >= H - 1 && 2 * o.x0 <= W - 1 && 2 * o.x1 >= W - 1;
  }
  dense_rank(objs, RANK_CENTER, false, [H, W](const Obj& o) {
    return std::max(std::abs(o.y0 + o.y1 - (H - 1)), std::abs(o.x0 + o.x1 - (W - 1)));
  });
  dense_rank(objs, RANK_CENTER_SMALL, false, [H, W](const Obj& o) {
    return std::max(std::abs(o.y0 + o.y1 - (H - 1)), std::abs(o.x0 + o.x1 - (W - 1))) * 1000 + o.mass;
  });

  // Against the largest object (first one on ties).
  size_t big = 0;
  for (size_t k = 1; k < objs.size(); ++k)
    if (objs[k].mass > objs[big].mass) big = k;
  const Obj& B = objs[big];
  for (size_t k = 0; k < objs.size(); ++k) {
    Obj& o = objs[k];
    if (k == big) { o.v[REL_Y_BIG] = o.v[REL_X_BIG] = 3; continue; }
    o.v[REL_Y_BIG] = side_of(o.y0, o.y1, B.y0, B.y1);
    o.v[REL_X_BIG] = side_of(o.x0, o.x1, B.x0, B.x1);
    o.v[IN_BIG] = o.y0 >= B.y0 && o.y1 <= B.y1 && o.x0 >= B.x0 && o.x1 <= B.x1;
    for (const auto& p : scene.nodes()[o.idx].pixel_list) {
      for (int dy = -1; dy <= 1 && !o.v[TOUCH_BIG]; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          int y = p.first + dy, x = p.second + dx;
          if (y >= 0 && y < H && x >= 0 && x < W && owner[y][x] == static_cast<int>(big)) {
            o.v[TOUCH_BIG] = 1;
            break;
          }
        }
      if (o.v[TOUCH_BIG]) break;
    }
  }

  // Against the fullest divider row / column. Structural SubObjects (the
  // separator lines the encoder sets apart) count here: they are the dividers.
  std::vector<std::vector<int>> cell(H, std::vector<int>(W, -1));
  int scene_bg = scene.has_background() ? scene.background() : 0;
  for (size_t i : scene.get_nodes_by_type("SubObject", true)) {
    const Node& n = scene.nodes()[i];
    int color = n.color.has_value() ? *n.color : -1;
    if (color == scene_bg) continue;
    for (const auto& p : n.pixel_list)
      if (p.first >= 0 && p.first < H && p.second >= 0 && p.second < W) cell[p.first][p.second] = color;
  }
  auto at = [&](bool row, int i, int j) { return row ? cell[i][j] : cell[j][i]; };
  auto divider = [&](bool row) {
    int n = row ? H : W, len = row ? W : H, best = -1, best_on = 0;
    for (int i = 0; i < n; ++i) {
      // It reaches both edges, give or take the gap of a dashed line.
      int c = at(row, i, 0) >= 0 ? at(row, i, 0) : len > 1 ? at(row, i, 1) : -1;
      int e = at(row, i, len - 1) >= 0 ? at(row, i, len - 1) : len > 1 ? at(row, i, len - 2) : -1;
      if (c < 0 || e != c) continue;
      int on = 0, beside = 0;
      bool mono = true;
      for (int j = 0; j < len && mono; ++j) {
        int v = at(row, i, j);
        if (v < 0) continue;
        mono = v == c;
        ++on;
        beside += (i > 0 && at(row, i - 1, j) == c) || (i < n - 1 && at(row, i + 1, j) == c);
      }
      if (mono && 2 * on >= len && 2 * beside < on && on > best_on) { best = i; best_on = on; }
    }
    return best;
  };
  int hrow = divider(true), vcol = divider(false);
  for (Obj& o : objs) {
    o.v[REL_Y_LINE] = hrow < 0 ? 1 : side_of(o.y0, o.y1, hrow, hrow);
    o.v[REL_X_LINE] = vcol < 0 ? 1 : side_of(o.x0, o.x1, vcol, vcol);
  }

  // Against the largest object of a colour no other object has.
  int uniq = -1;
  for (size_t k = 0; k < objs.size(); ++k)
    if (by_color[objs[k].color] == 1 && (uniq < 0 || objs[k].mass > objs[uniq].mass)) uniq = static_cast<int>(k);
  for (size_t k = 0; k < objs.size(); ++k) {
    Obj& o = objs[k];
    if (uniq < 0) { o.v[REL_Y_UNIQ] = o.v[REL_X_UNIQ] = 1; continue; }
    if (static_cast<int>(k) == uniq) { o.v[REL_Y_UNIQ] = o.v[REL_X_UNIQ] = 3; continue; }
    const Obj& U = objs[uniq];
    o.v[REL_Y_UNIQ] = side_of(o.y0, o.y1, U.y0, U.y1);
    o.v[REL_X_UNIQ] = side_of(o.x0, o.x1, U.x0, U.x1);
  }

  // Depth: walk each row / column run of an object; when the cells just past
  // both ends belong to objects of one other colour, the object lies in front.
  for (size_t k = 0; k < objs.size(); ++k) {
    const Obj& o = objs[k];
    for (const auto& p : scene.nodes()[o.idx].pixel_list) {
      for (int d = 0; d < 2; ++d) {
        int dy = d == 0 ? 0 : 1, dx = d == 0 ? 1 : 0;
        int y0 = p.first, x0 = p.second;
        if (y0 - dy >= 0 && x0 - dx >= 0 && owner[y0 - dy][x0 - dx] == static_cast<int>(k)) continue;  // not a run start
        int y1 = y0, x1 = x0;
        while (y1 + dy < H && x1 + dx < W && owner[y1 + dy][x1 + dx] == static_cast<int>(k)) { y1 += dy; x1 += dx; }
        int by = y0 - dy, bx = x0 - dx, ay = y1 + dy, ax = x1 + dx;
        if (by < 0 || bx < 0 || ay >= H || ax >= W) continue;
        int a = owner[by][bx], b = owner[ay][ax];
        if (a < 0 || b < 0 || objs[a].color != objs[b].color || objs[a].color == o.color) continue;
        objs[k].v[IS_FRONT] = 1;
        objs[a].v[IS_BEHIND] = objs[b].v[IS_BEHIND] = 1;
      }
    }
  }
  // Or it covers part of the bounding box of a touching object of another
  // colour that misses cells there (a corner or an edge hidden behind it).
  // Only the objects owning a 4-neighbour of one of its cells can touch it.
  std::vector<int> touched_by(objs.size(), -1), near;
  for (size_t k = 0; k < objs.size(); ++k) {
    const Obj& o = objs[k];
    const std::vector<Coord>& px = scene.nodes()[o.idx].pixel_list;
    near.clear();
    for (const auto& p : px) {
      const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
      for (int d = 0; d < 4; ++d) {
        int y = p.first + dy[d], x = p.second + dx[d];
        if (y < 0 || y >= H || x < 0 || x >= W) continue;
        int q = owner[y][x];
        if (q < 0 || q == static_cast<int>(k) || touched_by[q] == static_cast<int>(k)) continue;
        touched_by[q] = static_cast<int>(k);
        near.push_back(q);
      }
    }
    for (int q : near) {
      const Obj& b = objs[q];
      if (b.color == o.color || b.mass == b.v[AREA]) continue;
      bool inside = false;
      for (const auto& p : px)
        if (p.first >= b.y0 && p.first <= b.y1 && p.second >= b.x0 && p.second <= b.x1) { inside = true; break; }
      if (inside) {
        objs[k].v[IS_FRONT] = 1;
        objs[q].v[IS_BEHIND] = 1;
      }
    }
  }

  // Colour statistics over the objects.
  {
    std::map<int, int> cells;
    for (const Obj& o : objs) cells[o.color] += o.mass;
    std::set<int> counts;
    for (const auto& kv : cells) counts.insert(kv.second);
    std::vector<int> desc(counts.rbegin(), counts.rend());
    int rare = cells.begin()->first;
    for (const auto& kv : cells)
      if (kv.second < cells[rare]) rare = kv.first;
    // Per colour, its objects' distinct masses (ascending).
    std::map<int, std::vector<int>> masses;
    for (const Obj& o : objs) masses[o.color].push_back(o.mass);
    for (auto& kv : masses) {
      std::sort(kv.second.begin(), kv.second.end());
      kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }
    for (Obj& o : objs) {
      o.v[COLOR_RANK] = static_cast<int>(std::find(desc.begin(), desc.end(), cells[o.color]) - desc.begin()) + 1;
      o.v[IS_RARE_COLOR] = cells.size() > 1 && o.color == rare;
      // 1 + the distinct masses above it among the objects of its colour.
      const std::vector<int>& m = masses[o.color];
      o.v[RANK_IN_COLOR] = 1 + static_cast<int>(m.end() - std::upper_bound(m.begin(), m.end(), o.mass));
    }
  }
  // Layout against the other objects (q != k), without visiting every pair:
  //  * n_row_aligned: the boxes whose rows meet [y0, y1] are all but those
  //    ending above it and those starting below it (sorted ends, binary search);
  //  * left / right / up / down: per grid row the least x1 and greatest x0 of
  //    the boxes covering it (per column the least y1 / greatest y0); a box
  //    meets the object's rows iff it covers one of them. The object's own
  //    box never passes these tests, so it may stay in;
  //  * gap_near (the least max(gy, gx), capped at 9): max(gy, gx) <= g iff the
  //    box meets the object's box grown by g + 1, i.e. iff the box-coverage
  //    count over that rectangle exceeds the object's own area (2D prefix sums).
  const size_t nobj = objs.size();
  int ylo = objs[0].y0, yhi = objs[0].y1, xlo = objs[0].x0, xhi = objs[0].x1;
  for (const Obj& o : objs) {
    ylo = std::min(ylo, o.y0); yhi = std::max(yhi, o.y1);
    xlo = std::min(xlo, o.x0); xhi = std::max(xhi, o.x1);
  }
  const int RH = yhi - ylo + 1, RW = xhi - xlo + 1;
  std::vector<int> sy0(nobj), sy1(nobj), sx0(nobj), sx1(nobj);
  for (size_t q = 0; q < nobj; ++q) {
    sy0[q] = objs[q].y0; sy1[q] = objs[q].y1; sx0[q] = objs[q].x0; sx1[q] = objs[q].x1;
  }
  std::sort(sy0.begin(), sy0.end()); std::sort(sy1.begin(), sy1.end());
  std::sort(sx0.begin(), sx0.end()); std::sort(sx1.begin(), sx1.end());
  const int INF = std::numeric_limits<int>::max(), NINF = std::numeric_limits<int>::min();
  std::vector<int> row_min_x1(RH, INF), row_max_x0(RH, NINF), col_min_y1(RW, INF), col_max_y0(RW, NINF);
  std::vector<long long> cov(static_cast<size_t>(RH + 1) * (RW + 1), 0);  // difference array, then prefix sums
  auto C = [&](int y, int x) -> long long& { return cov[static_cast<size_t>(y) * (RW + 1) + x]; };
  for (const Obj& b : objs) {
    for (int r = b.y0; r <= b.y1; ++r) {
      row_min_x1[r - ylo] = std::min(row_min_x1[r - ylo], b.x1);
      row_max_x0[r - ylo] = std::max(row_max_x0[r - ylo], b.x0);
    }
    for (int c = b.x0; c <= b.x1; ++c) {
      col_min_y1[c - xlo] = std::min(col_min_y1[c - xlo], b.y1);
      col_max_y0[c - xlo] = std::max(col_max_y0[c - xlo], b.y0);
    }
    C(b.y0 - ylo, b.x0 - xlo) += 1;
    C(b.y0 - ylo, b.x1 - xlo + 1) -= 1;
    C(b.y1 - ylo + 1, b.x0 - xlo) -= 1;
    C(b.y1 - ylo + 1, b.x1 - xlo + 1) += 1;
  }
  for (int y = 0; y <= RH; ++y)  // coverage per cell
    for (int x = 0; x <= RW; ++x)
      C(y, x) += (y ? C(y - 1, x) : 0) + (x ? C(y, x - 1) : 0) - (y && x ? C(y - 1, x - 1) : 0);
  std::vector<long long> pre(static_cast<size_t>(RH + 1) * (RW + 1), 0);  // pre(y, x): cells [0, y) x [0, x)
  auto P = [&](int y, int x) -> long long& { return pre[static_cast<size_t>(y) * (RW + 1) + x]; };
  for (int y = 1; y <= RH; ++y)
    for (int x = 1; x <= RW; ++x) P(y, x) = C(y - 1, x - 1) + P(y - 1, x) + P(y, x - 1) - P(y - 1, x - 1);
  auto covered = [&](int y0, int y1, int x0, int x1) {  // coverage over [y0, y1] x [x0, x1], clipped
    y0 = std::max(y0, ylo) - ylo; y1 = std::min(y1, yhi) - ylo;
    x0 = std::max(x0, xlo) - xlo; x1 = std::min(x1, xhi) - xlo;
    if (y0 > y1 || x0 > x1) return 0LL;
    return P(y1 + 1, x1 + 1) - P(y0, x1 + 1) - P(y1 + 1, x0) + P(y0, x0);
  };
  for (size_t k = 0; k < objs.size(); ++k) {
    Obj& o = objs[k];
    const int oy0 = o.y0, oy1 = o.y1, ox0 = o.x0, ox1 = o.x1;
    const int above = static_cast<int>(std::lower_bound(sy1.begin(), sy1.end(), oy0) - sy1.begin());
    const int below = static_cast<int>(sy0.end() - std::upper_bound(sy0.begin(), sy0.end(), oy1));
    const int lefts = static_cast<int>(std::lower_bound(sx1.begin(), sx1.end(), ox0) - sx1.begin());
    const int rights = static_cast<int>(sx0.end() - std::upper_bound(sx0.begin(), sx0.end(), ox1));
    o.v[N_ROW_ALIGNED] += static_cast<int>(nobj) - above - below - 1;
    o.v[N_COL_ALIGNED] += static_cast<int>(nobj) - lefts - rights - 1;
    bool left = false, right = false, up = false, down = false;
    for (int r = oy0; r <= oy1; ++r) {
      left = left || row_min_x1[r - ylo] < ox0;
      right = right || row_max_x0[r - ylo] > ox1;
    }
    for (int c = ox0; c <= ox1; ++c) {
      up = up || col_min_y1[c - xlo] < oy0;
      down = down || col_max_y0[c - xlo] > oy1;
    }
    const long long own = static_cast<long long>(oy1 - oy0 + 1) * (ox1 - ox0 + 1);
    int gap = 9;
    for (int g = 0; g < 9; ++g)
      if (covered(oy0 - g - 1, oy1 + g + 1, ox0 - g - 1, ox1 + g + 1) > own) {
        gap = g;
        break;
      }
    o.v[IS_BETWEEN] = (left && right) || (up && down);
    o.v[GAP_NEAR] = gap;
    o.v[IS_DIAG] = o.mass > 1 && o.mass == o.v[HEIGHT] && o.v[HEIGHT] == o.v[WIDTH] &&
                   (o.m(0, 0) ? o.m(o.v[HEIGHT] - 1, o.v[WIDTH] - 1) : o.m(0, o.v[WIDTH] - 1));
    if (o.v[IS_DIAG]) {
      bool main = o.m(0, 0), ok = true;
      for (int i = 0; i < o.v[HEIGHT]; ++i) ok = ok && o.m(i, main ? i : o.v[WIDTH] - 1 - i);
      o.v[IS_DIAG] = ok;
    }
    o.v[SHAPE_KIND] = o.v[IS_SINGLE] ? 0 : o.v[IS_HLINE] ? 1 : o.v[IS_VLINE] ? 2 : o.v[IS_DIAG] ? 3
                    : o.v[IS_RECT] ? 4 : o.v[IS_FRAME] ? 5 : 6;
  }

  std::map<int, int> kinds;
  for (const Obj& o : objs) kinds[o.v[SHAPE_KIND]]++;
  for (Obj& o : objs) {
    o.v[N_SAME_KIND] = kinds[o.v[SHAPE_KIND]] - 1;
    o.v[IS_TOP] = o.v[IS_FRONT] && !o.v[IS_BEHIND];
  }

  for (const Obj& o : objs) scene.node_ref(o.idx).attrs = LocalShared<std::vector<int>>(std::vector<int>(o.v, o.v + N_ATTRS));
}

}  // namespace arc
