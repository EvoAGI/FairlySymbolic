// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// PATCH: background cells are filled to complete a shape. ConceptARC
// "CompleteShape" (under / over: "AboveBelow").
//
// args: mode  bridge  a background cell whose two neighbours left and right, or
//                     above and below, share a colour takes that colour (one-cell
//                     gaps in lines, frames and solid shapes)
//             extent  the bounding box of all non-background cells is filled with
//                     their most frequent colour
//             objbox  each single-colour object (4-connected) fills the background
//                     cells of its own bbox, with its colour (fill=own) or with the
//                     most frequent other colour (fill=other), with one colour
//                     (fill=0-9), or with the colour a map "c:f,..." gives its
//                     colour (learned by the seeds)
//             swap_sides  two parallel lines of one colour with the same span are
//                     replaced by the two other sides of the rectangle they bound
//             span    a background cell with cells of one colour somewhere to its
//                     left and to its right on its row takes that colour's fill
//                     colour (fill as for objbox; colours with more cells go
//                     first): a cup filled to its brim, dots joined
//             span_v  the same along columns
//             lattice inside each hollow object (8-connected, one colour; the
//                     cells that cannot reach its bbox border), when what is drawn
//                     there is one colour, the background cells of the period-2
//                     classes that colour occupies take it (a checkerboard,
//                     stripes or dots completed). The classes are those of
//                     (y + x) % 2, y % 2, x % 2 or (y % 2, x % 2), whichever
//                     fills the fewest cells
//             under   a background region (4-connected) lying under the
//                     drawing (some of its cells have a non-background cell right
//                     above them, none right below) takes the colour of those
//                     cells above (the side below a dividing line filled)
//             over    the same upwards: regions lying over the drawing
//       fill  (objbox, span, span_v) own | other | 0-9 | map "c:f,..."

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

thread_local std::vector<int> g_task_colors;

namespace {

DslArgs make(const std::string& mode, const std::string& fill = "own") {
  DslArgs a;
  a.emplace_back("mode", cdv_str(mode));
  if (mode == "objbox" || mode == "span" || mode == "span_v") a.emplace_back("fill", cdv_str(fill));
  return a;
}

std::map<int, int> parse_map(const std::string& s) {
  std::map<int, int> m;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t c = item.find(':');
    if (c != std::string::npos) m[std::atoi(item.substr(0, c).c_str())] = std::atoi(item.substr(c + 1).c_str());
  }
  return m;
}

// Non-background cell counts per colour.
std::map<int, int> colour_counts(const CGrid& g, int bg) {
  std::map<int, int> n;
  for (const auto& row : g)
    for (int v : row)
      if (v != bg) n[v]++;
  return n;
}

// The colour `fill` gives to cells drawn for colour c (n: colour counts, m: the
// parsed map); -1 for none.
int fill_colour(const std::string& fill, int c, int bg, const std::map<int, int>& n, const std::map<int, int>& m) {
  if (fill == "own") return c;
  if (fill == "other") {
    int cnt = 0, f = -1;
    for (const auto& kv : n)
      if (kv.first != c && kv.second > cnt) { f = kv.first; cnt = kv.second; }
    for (size_t k = 0; f < 0 && k < g_task_colors.size(); ++k)
      if (g_task_colors[k] != c && g_task_colors[k] != bg) f = g_task_colors[k];
    return f;
  }
  if (fill.size() == 1 && fill[0] >= '0' && fill[0] <= '9') return fill[0] - '0';
  auto it = m.find(c);
  return it == m.end() ? -1 : it->second;
}

CGrid objbox(const CGrid& g, int bg, const std::string& fill) {
  CGrid out = g;
  std::map<int, int> n = colour_counts(g, bg), m = parse_map(fill);
  for (const auto& comp : concept_components(g, bg, false, true)) {
    int c = g[comp[0].first][comp[0].second], f = fill_colour(fill, c, bg, n, m);
    if (f < 0) continue;
    Box b = bbox_of(comp);
    for (int y = b.y0; y <= b.y1; ++y)
      for (int x = b.x0; x <= b.x1; ++x)
        if (g[y][x] == bg) out[y][x] = f;
  }
  return out;
}

// Background cells between the first and last cell of a colour on their row
// (cols: column) take that colour's fill colour; colours with more cells first.
CGrid span(const CGrid& g, int bg, const std::string& fill, bool cols) {
  CGrid out = g;
  std::map<int, int> n = colour_counts(g, bg), m = parse_map(fill);
  std::vector<std::pair<int, int>> order;  // (-count, colour)
  for (const auto& kv : n) order.push_back({-kv.second, kv.first});
  std::sort(order.begin(), order.end());
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  int lines = cols ? W : H, len = cols ? H : W;
  for (const auto& oc : order) {
    int c = oc.second, f = fill_colour(fill, c, bg, n, m);
    if (f < 0) continue;
    for (int i = 0; i < lines; ++i) {
      int first = -1, last = -1;
      for (int j = 0; j < len; ++j)
        if ((cols ? g[j][i] : g[i][j]) == c) {
          if (first < 0) first = j;
          last = j;
        }
      for (int j = first + 1; j < last; ++j) {
        int& o = cols ? out[j][i] : out[i][j];
        if ((cols ? g[j][i] : g[i][j]) == bg && o == bg) o = f;
      }
    }
  }
  return out;
}

// Cells of comp's bbox that cannot reach the bbox border (4-connected).
std::vector<Coord> enclosed(const std::vector<Coord>& comp) {
  Box b = bbox_of(comp);
  int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
  std::vector<std::vector<char>> obj(h, std::vector<char>(w, 0)), reach(h, std::vector<char>(w, 0));
  for (const Coord& c : comp) obj[c.first - b.y0][c.second - b.x0] = 1;
  std::vector<Coord> st;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !obj[y][x]) {
        reach[y][x] = 1;
        st.push_back({y, x});
      }
  const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
  while (!st.empty()) {
    Coord c = st.back();
    st.pop_back();
    for (int k = 0; k < 4; ++k) {
      int p = c.first + dy[k], q = c.second + dx[k];
      if (p < 0 || p >= h || q < 0 || q >= w || obj[p][q] || reach[p][q]) continue;
      reach[p][q] = 1;
      st.push_back({p, q});
    }
  }
  std::vector<Coord> res;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (!obj[y][x] && !reach[y][x]) res.push_back({b.y0 + y, b.x0 + x});
  return res;
}

CGrid lattice(const CGrid& g, int bg) {
  CGrid out = g;
  auto key = [](int l, const Coord& c) {
    int y = c.first & 1, x = c.second & 1;
    return l == 0 ? (y + x) & 1 : l == 1 ? y : l == 2 ? x : 2 * y + x;
  };
  for (const auto& comp : concept_components(g, bg, true, true)) {
    std::vector<Coord> in = enclosed(comp);
    int k = -1;
    bool mixed = false;
    for (const Coord& c : in) {
      int v = g[c.first][c.second];
      if (v == bg) continue;
      if (k >= 0 && v != k) mixed = true;
      k = v;
    }
    if (k < 0 || mixed) continue;
    int best_l = -1, best_n = 0;
    std::vector<char> best_hit;
    for (int l = 0; l < 4; ++l) {
      std::vector<char> hit(4, 0);
      for (const Coord& c : in)
        if (g[c.first][c.second] == k) hit[key(l, c)] = 1;
      int cnt = 0;
      for (const Coord& c : in) cnt += hit[key(l, c)];
      if (best_l < 0 || cnt < best_n) {
        best_l = l;
        best_n = cnt;
        best_hit = hit;
      }
    }
    for (const Coord& c : in)
      if (g[c.first][c.second] == bg && best_hit[key(best_l, c)]) out[c.first][c.second] = k;
  }
  return out;
}

// mode=under (up=false) / over (up=true).
CGrid side(const CGrid& g, int bg, bool up) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid out = g;
  std::vector<std::vector<char>> seen(H, std::vector<char>(W, 0));
  const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
  int toward = up ? 1 : -1;  // the row the drawing is in, seen from the region
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (g[y][x] != bg || seen[y][x]) continue;
      std::vector<Coord> st{{y, x}}, region;
      seen[y][x] = 1;
      std::map<int, int> n;
      bool away = false;
      while (!st.empty()) {
        Coord c = st.back();
        st.pop_back();
        region.push_back(c);
        int a = c.first + toward, b = c.first - toward;
        if (a >= 0 && a < H && g[a][c.second] != bg) n[g[a][c.second]]++;
        if (b >= 0 && b < H && g[b][c.second] != bg) away = true;
        for (int k = 0; k < 4; ++k) {
          int p = c.first + dy[k], q = c.second + dx[k];
          if (p < 0 || p >= H || q < 0 || q >= W || seen[p][q] || g[p][q] != bg) continue;
          seen[p][q] = 1;
          st.push_back({p, q});
        }
      }
      if (away || n.empty()) continue;
      int f = n.begin()->first;
      for (const auto& kv : n)
        if (kv.second > n[f]) f = kv.first;
      for (const Coord& c : region) out[c.first][c.second] = f;
    }
  return out;
}

CGrid to_cgrid(const Grid& gr) {
  CGrid g(gr.height(), std::vector<int>(gr.width()));
  const auto& c = gr.cells();
  for (int y = 0; y < gr.height(); ++y)
    for (int x = 0; x < gr.width(); ++x) g[y][x] = c[y * gr.width() + x];
  return g;
}

CGrid swap_sides(const CGrid& g, int bg) {
  CGrid out = g;
  std::map<int, std::vector<Box>> lines;
  for (const auto& comp : concept_components(g, bg, false, true)) {
    Box b = bbox_of(comp);
    int area = (b.y1 - b.y0 + 1) * (b.x1 - b.x0 + 1);
    if (area != static_cast<int>(comp.size()) || area < 2 || (b.y0 != b.y1 && b.x0 != b.x1)) continue;
    lines[g[comp[0].first][comp[0].second]].push_back(b);
  }
  for (const auto& kv : lines) {
    if (kv.second.size() != 2) continue;
    Box a = kv.second[0], b = kv.second[1];
    bool hor = a.y0 == a.y1 && b.y0 == b.y1 && a.x0 == b.x0 && a.x1 == b.x1 && a.x1 > a.x0;
    bool ver = a.x0 == a.x1 && b.x0 == b.x1 && a.y0 == b.y0 && a.y1 == b.y1 && a.y1 > a.y0;
    if (!hor && !ver) continue;
    if (hor && std::abs(a.y0 - b.y0) < 2) continue;
    if (ver && std::abs(a.x0 - b.x0) < 2) continue;
    for (const Box& l : {a, b})
      for (int y = l.y0; y <= l.y1; ++y)
        for (int x = l.x0; x <= l.x1; ++x) out[y][x] = bg;
    if (hor) {
      int r0 = std::min(a.y0, b.y0), r1 = std::max(a.y0, b.y0);
      for (int y = r0 + 1; y < r1; ++y) out[y][a.x0] = out[y][a.x1] = kv.first;
    } else {
      int c0 = std::min(a.x0, b.x0), c1 = std::max(a.x0, b.x0);
      for (int x = c0 + 1; x < c1; ++x) out[a.y0][x] = out[a.y1][x] = kv.first;
    }
  }
  return out;
}

void apply(Scene& scene, const std::vector<std::string>&, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene), H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  CGrid out = g;
  std::string mode = arg_str(action.args, "mode", "bridge");
  if (mode == "swap_sides") {
    out = swap_sides(g, bg);
  } else if (mode == "span" || mode == "span_v") {
    out = span(g, bg, arg_str(action.args, "fill", "own"), mode == "span_v");
  } else if (mode == "lattice") {
    out = lattice(g, bg);
  } else if (mode == "under" || mode == "over") {
    out = side(g, bg, mode == "over");
  } else if (mode == "objbox") {
    out = objbox(g, bg, arg_str(action.args, "fill", "own"));
  } else if (mode == "extent") {
    int y0 = H, y1 = -1, x0 = W, x1 = -1;
    std::map<int, int> n;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (g[y][x] != bg) {
          y0 = std::min(y0, y); y1 = std::max(y1, y);
          x0 = std::min(x0, x); x1 = std::max(x1, x);
          n[g[y][x]]++;
        }
    if (y1 < 0) return;
    int top = bg, cnt = 0;
    for (const auto& kv : n)
      if (kv.second > cnt) { top = kv.first; cnt = kv.second; }
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) out[y][x] = top;
  } else {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (g[y][x] != bg) continue;
        if (x > 0 && x < W - 1 && g[y][x - 1] != bg && g[y][x - 1] == g[y][x + 1]) out[y][x] = g[y][x - 1];
        else if (y > 0 && y < H - 1 && g[y - 1][x] != bg && g[y - 1][x] == g[y + 1][x]) out[y][x] = g[y - 1][x];
      }
  }
  concept_install(scene, out);
}

DslArgs sample(ArcRandom& rng) {
  return make(pick_str(rng, {"bridge", "extent", "objbox", "swap_sides", "span", "span_v", "lattice", "under", "over"}),
              pick_str(rng, {"own", "other", "own", "other", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}));
}

// Seeds only when every train output has the input's size and keeps every
// non-background input cell.
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  bool swap_ok = true;
  for (size_t i = 0; i < ins.size() && swap_ok; ++i) swap_ok = swap_sides(to_cgrid(ins[i]), 0) == to_cgrid(outs[i]);
  if (swap_ok) res.push_back(make("swap_sides"));
  for (size_t i = 0; i < ins.size(); ++i) {
    const auto& a = ins[i].cells();
    const auto& b = outs[i].cells();
    for (size_t k = 0; k < a.size(); ++k)
      if (a[k] != 0 && a[k] != b[k]) return res;
  }
  res.push_back(make("bridge"));
  res.push_back(make("extent"));
  res.push_back(make("objbox", "own"));
  res.push_back(make("objbox", "other"));
  res.push_back(make("span", "other"));
  res.push_back(make("lattice"));
  // Learn a colour map: each object's bbox gaps take one output colour per object colour.
  std::map<int, int> m;
  std::vector<CGrid> gi, go;
  for (size_t i = 0; i < ins.size(); ++i) {
    gi.push_back(to_cgrid(ins[i]));
    go.push_back(to_cgrid(outs[i]));
    for (const auto& comp : concept_components(gi[i], 0, false, true)) {
      int c = gi[i][comp[0].first][comp[0].second];
      Box b = bbox_of(comp);
      for (int y = b.y0; y <= b.y1; ++y)
        for (int x = b.x0; x <= b.x1; ++x) {
          if (gi[i][y][x] != 0) continue;
          int f = go[i][y][x];
          if (f == 0 || (m.count(c) && m[c] != f)) return res;
          m[c] = f;
        }
    }
  }
  if (m.empty()) return res;
  std::string fs;
  for (const auto& kv : m) fs += (fs.empty() ? "" : ",") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
  for (size_t i = 0; i < gi.size(); ++i)
    if (objbox(gi[i], 0, fs) != go[i]) return res;
  res.push_back(make("objbox", fs));
  return res;
}

}  // namespace

ConceptOp patch_op() { return {ActionKind::PATCH, 2, apply, sample, seeds}; }

}  // namespace arc
