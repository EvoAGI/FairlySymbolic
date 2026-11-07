// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// ATTR_PAINT: each object is painted, or has its holes filled, with a colour
// looked up from one of its attributes (fill by orientation, recolour by size).
// ConceptARC "HorizontalVertical", "FilledNotFilled", "Count"; common in ARC.
//
// args: attr    orient | mass | height | width | holes | rank_size (1 = the
//               largest mass) | rank_size_asc (1 = the smallest) | is_rect |
//               border (touches the grid edge) | run_orient (per cell, not per object: the
//               longer of its same-colour row and column runs, 0 = column,
//               2 = row, 1 = tie; lines that touch at a corner are told apart)
//               | twin (1 when another object has the same shape, any colour,
//               else 0)
//               | rel_touching | rel_nearest | rel_largest | rel_bottom | rel_top
//               (formerly the op recolor_rel source=...: each selected object
//               takes the colour of a related object; no map; target object |
//               box_bg; concept_recolor_rel.cpp)
//       target  object (its cells) | cavity (background enclosed by it)
//       map     "value:colour,value:colour,..."; objects with an unlisted value
//               are left alone
// Objects are 8-connected single-colour groups. The seed learns the map from the
// train pairs and fires only when it explains every changed cell.

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

void recolor_rel(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action);
DslArgs recolor_rel_sample(ArcRandom& rng);
std::vector<DslArgs> recolor_rel_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs);

namespace {

// rank_size, is_rect and border stay available: saved ARC-AGI-1/2 programs use
// them.
const std::vector<std::string> kAttrs = {"orient", "mass", "height", "width", "holes",
                                         "rank_size", "rank_size_asc", "is_rect", "border", "run_orient", "twin"};

struct Obj {
  std::vector<Coord> cells, cavity;
  int color;
  std::map<std::string, int> a;
};

std::vector<Obj> objects(const CGrid& g, int bg) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<Obj> res;
  const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
  for (auto& comp : concept_components(g, bg, true, true)) {
    Obj o;
    o.color = g[comp[0].first][comp[0].second];
    Box b = bbox_of(comp);
    int h = b.y1 - b.y0 + 1, w = b.x1 - b.x0 + 1;
    std::vector<std::vector<char>> in(h, std::vector<char>(w, 0)), reach(h, std::vector<char>(w, 0));
    for (const Coord& c : comp) in[c.first - b.y0][c.second - b.x0] = 1;
    std::vector<Coord> st;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !in[y][x]) {
          reach[y][x] = 1;
          st.push_back({y, x});
        }
    while (!st.empty()) {
      Coord c = st.back();
      st.pop_back();
      for (int k = 0; k < 4; ++k) {
        int p = c.first + dy[k], q = c.second + dx[k];
        if (p < 0 || p >= h || q < 0 || q >= w || in[p][q] || reach[p][q]) continue;
        reach[p][q] = 1;
        st.push_back({p, q});
      }
    }
    // Holes: 4-connected regions of the unreached cells.
    std::vector<std::vector<char>> seen(h, std::vector<char>(w, 0));
    int holes = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (in[y][x] || reach[y][x] || seen[y][x]) continue;
        ++holes;
        st = {{y, x}};
        seen[y][x] = 1;
        while (!st.empty()) {
          Coord c = st.back();
          st.pop_back();
          if (g[b.y0 + c.first][b.x0 + c.second] == bg) o.cavity.push_back({b.y0 + c.first, b.x0 + c.second});
          for (int k = 0; k < 4; ++k) {
            int p = c.first + dy[k], q = c.second + dx[k];
            if (p < 0 || p >= h || q < 0 || q >= w || in[p][q] || reach[p][q] || seen[p][q]) continue;
            seen[p][q] = 1;
            st.push_back({p, q});
          }
        }
      }
    o.a["orient"] = h > w ? 0 : h < w ? 2 : 1;
    o.a["mass"] = static_cast<int>(comp.size());
    o.a["height"] = h;
    o.a["width"] = w;
    o.a["holes"] = holes;
    o.a["is_rect"] = static_cast<int>(comp.size()) == h * w;
    o.a["border"] = b.y0 == 0 || b.x0 == 0 || b.y1 == H - 1 || b.x1 == W - 1;
    o.cells = std::move(comp);
    res.push_back(std::move(o));
  }
  std::map<std::vector<Coord>, int> shapes;  // cells relative to the bbox corner
  std::vector<std::vector<Coord>> shape_of;
  for (const Obj& o : res) {
    Box b = bbox_of(o.cells);
    std::vector<Coord> rel;
    for (const Coord& c : o.cells) rel.push_back({c.first - b.y0, c.second - b.x0});
    std::sort(rel.begin(), rel.end());
    shapes[rel]++;
    shape_of.push_back(rel);
  }
  for (size_t i = 0; i < res.size(); ++i) res[i].a["twin"] = shapes[shape_of[i]] > 1;
  std::set<int> masses;
  for (const Obj& o : res) masses.insert(o.a.at("mass"));
  std::vector<int> asc(masses.begin(), masses.end());
  for (Obj& o : res) {
    int r = static_cast<int>(std::find(asc.begin(), asc.end(), o.a["mass"]) - asc.begin());
    o.a["rank_size_asc"] = r + 1;
    o.a["rank_size"] = static_cast<int>(asc.size()) - r;
  }
  return res;
}

std::map<int, int> parse_map(const std::string& s) {
  std::map<int, int> m;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t c = item.find(':');
    if (c == std::string::npos) continue;
    m[std::atoi(item.substr(0, c).c_str())] = std::atoi(item.substr(c + 1).c_str());
  }
  return m;
}

std::string map_str(const std::map<int, int>& m) {
  std::string s;
  for (const auto& kv : m) s += (s.empty() ? "" : ",") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
  return s;
}

DslArgs make(const std::string& attr, const std::string& target, const std::string& map) {
  DslArgs a;
  a.emplace_back("attr", cdv_str(attr));
  a.emplace_back("target", cdv_str(target));
  a.emplace_back("map", cdv_str(map));
  return a;
}

CGrid paint(const CGrid& g, int bg, const std::string& attr, bool cavity, const std::map<int, int>& m) {
  CGrid out = g;
  if (attr == "run_orient") {
    if (cavity) return out;
    int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int v = g[y][x];
        if (v == bg) continue;
        int row = 1, col = 1;
        for (int q = x - 1; q >= 0 && g[y][q] == v; --q) ++row;
        for (int q = x + 1; q < W && g[y][q] == v; ++q) ++row;
        for (int p = y - 1; p >= 0 && g[p][x] == v; --p) ++col;
        for (int p = y + 1; p < H && g[p][x] == v; ++p) ++col;
        auto it = m.find(col > row ? 0 : col < row ? 2 : 1);
        if (it != m.end()) out[y][x] = it->second;
      }
    return out;
  }
  for (const Obj& o : objects(g, bg)) {
    auto it = m.find(o.a.at(attr));
    if (it == m.end()) continue;
    for (const Coord& c : cavity ? o.cavity : o.cells) out[c.first][c.second] = it->second;
  }
  return out;
}

void apply(Scene& scene, const std::vector<std::string>& selected_ids, const DslAction& action) {
  if (arg_str(action.args, "attr", "orient").rfind("rel_", 0) == 0) return recolor_rel(scene, selected_ids, action);
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  std::string attr = arg_str(action.args, "attr", "orient");
  if (std::find(kAttrs.begin(), kAttrs.end(), attr) == kAttrs.end()) return;
  auto m = parse_map(arg_str(action.args, "map", ""));
  if (m.empty()) return;
  concept_install(scene, paint(g, concept_bg(scene), attr, arg_str(action.args, "target", "object") == "cavity", m));
}

// Half the draws are rel_* (the weight of the former RECOLOR_REL op).
DslArgs sample(ArcRandom& rng) {
  if (pick_int(rng, {0, 1})) return recolor_rel_sample(rng);
  std::map<int, int> m;
  for (int v : {0, 1, 2}) m[v] = pick_int(rng, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  return make(pick_str(rng, kAttrs), pick_str(rng, {"object", "cavity"}), map_str(m));
}

CGrid to_cgrid(const Grid& gr) {
  CGrid g(gr.height(), std::vector<int>(gr.width()));
  const auto& c = gr.cells();
  for (int y = 0; y < gr.height(); ++y)
    for (int x = 0; x < gr.width(); ++x) g[y][x] = c[y * gr.width() + x];
  return g;
}

// For each attribute and target, read off the colour each object's cells (or
// cavity) take in the output; keep the maps that are consistent, reproduce
// every train output exactly, and see each listed value on two objects or more
// (a value seen once is memorised, not learned).
std::vector<DslArgs> map_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  if (ins.empty() || ins.size() != outs.size()) return res;
  std::vector<CGrid> gi, go;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
    gi.push_back(to_cgrid(ins[i]));
    go.push_back(to_cgrid(outs[i]));
    if (gi.back() == go.back()) return res;
  }
  std::vector<std::vector<Obj>> objs;
  for (const CGrid& g : gi) objs.push_back(objects(g, 0));
  for (const std::string& attr : kAttrs)
    for (bool cavity : {false, true}) {
      if (attr == "run_orient") continue;  // a cell attribute, not learned here
      std::map<int, int> m, support;
      bool ok = true;
      for (size_t i = 0; i < gi.size() && ok; ++i)
        for (const Obj& o : objs[i]) {
          const auto& cells = cavity ? o.cavity : o.cells;
          if (cells.empty()) continue;
          int col = go[i][cells[0].first][cells[0].second];
          for (const Coord& c : cells) ok = ok && go[i][c.first][c.second] == col;
          if (!ok) break;
          if (col == gi[i][cells[0].first][cells[0].second]) continue;  // unchanged
          int v = o.a.at(attr);
          if (m.count(v) && m[v] != col) { ok = false; break; }
          m[v] = col;
          support[v]++;
        }
      if (!ok || m.empty()) continue;
      for (const auto& kv : support) ok = ok && kv.second >= 2;
      for (size_t i = 0; i < gi.size() && ok; ++i) ok = paint(gi[i], 0, attr, cavity, m) == go[i];
      if (ok) res.push_back(make(attr, cavity ? "cavity" : "object", map_str(m)));
    }
  return res;
}

// The learned maps, then the rel_* seeds (the former recolor_rel's).
std::vector<DslArgs> seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res = map_seeds(ins, outs);
  for (DslArgs& a : recolor_rel_seeds(ins, outs)) res.push_back(std::move(a));
  return res;
}

}  // namespace

ConceptOp attr_paint_op() { return {ActionKind::ATTR_PAINT, 4, apply, sample, seeds}; }

}  // namespace arc
