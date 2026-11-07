// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// SORT_OBJECTS key=panel_<key> (formerly the op SORT_PANELS key=<key>): the grid
// is cut into equal panels by evenly spaced lines of one colour; the panels are
// sorted by a count and put back in the same slots. ConceptARC "Order".
// Reached from concept_sort_objects.cpp.
//
// args: key   panel_ + minority (cells not of the panel's main colour) | nonbg |
//             minority_row (first row holding a stray cell: a cell of neither the
//             divider colour nor the panel's main colour, or a divider-coloured
//             hole in a row of the bar, the rows holding the main colour; none
//             counts as the height)
//       desc  descending order
//       axis  v (panels side by side) | h (panels stacked)
//       clean the bar rows (rows where some panel holds its main colour) of every
//             panel are filled with its main colour, its other rows with the
//             divider colour

#include <algorithm>
#include <map>
#include <tuple>

#include "arc/concepts.hpp"
#include "arg_coerce.hpp"

namespace arc {

namespace {

CGrid transpose(const CGrid& g) {
  CGrid t(g[0].size(), std::vector<int>(g.size()));
  for (size_t y = 0; y < g.size(); ++y)
    for (size_t x = 0; x < g[0].size(); ++x) t[x][y] = g[y][x];
  return t;
}

// Column spans between divider columns: for each colour, the uniform columns of
// that colour; kept when they cut the grid into >= 2 spans of equal width.
std::vector<std::pair<int, int>> panel_spans(const CGrid& g, int* div_color = nullptr) {
  int H = static_cast<int>(g.size()), W = static_cast<int>(g[0].size());
  std::vector<std::pair<int, int>> best;
  for (int c = 0; c <= 9; ++c) {
    std::vector<int> div;
    for (int x = 0; x < W; ++x) {
      bool u = true;
      for (int y = 0; y < H && u; ++y) u = g[y][x] == c;
      if (u) div.push_back(x);
    }
    div.push_back(W);
    std::vector<std::pair<int, int>> spans;
    int s = 0;
    for (int x : div) {
      if (x > s) spans.emplace_back(s, x);
      s = x + 1;
    }
    bool equal = spans.size() >= 2;
    for (const auto& sp : spans) equal = equal && sp.second - sp.first == spans[0].second - spans[0].first;
    if (equal && spans.size() > best.size()) {
      best = spans;
      if (div_color) *div_color = c;
    }
  }
  return best;
}

bool sort_columns(CGrid& g, int bg, const std::string& key, bool desc, bool clean) {
  int dc = bg;
  auto spans = panel_spans(g, &dc);
  if (spans.size() < 2) return false;
  int H = static_cast<int>(g.size());
  std::vector<std::tuple<int, size_t>> order;
  std::vector<int> main_color;
  for (size_t i = 0; i < spans.size(); ++i) {
    std::map<int, int> hist;
    for (int y = 0; y < H; ++y)
      for (int x = spans[i].first; x < spans[i].second; ++x) hist[g[y][x]]++;
    int dom = hist.begin()->first;
    for (const auto& kv : hist)
      if (kv.second > hist[dom]) dom = kv.first;
    int fg = -1;
    for (const auto& kv : hist)
      if (kv.first != dc && (fg < 0 || kv.second > hist[fg])) fg = kv.first;
    main_color.push_back(fg);
    int n = 0;
    if (key == "minority_row") {
      n = H;
      for (int y = 0; y < H && n == H; ++y) {
        bool bar = false, stray = false, hole = false;
        for (int x = spans[i].first; x < spans[i].second; ++x) {
          bar = bar || g[y][x] == fg;
          stray = stray || (g[y][x] != dc && g[y][x] != fg);
          hole = hole || g[y][x] == dc;
        }
        if (stray || (bar && hole)) n = y;
      }
    } else {
      for (const auto& kv : hist)
        if (key == "nonbg" ? kv.first != bg : kv.first != dom) n += kv.second;
    }
    order.emplace_back(n, i);
  }
  std::stable_sort(order.begin(), order.end());
  if (desc) std::reverse(order.begin(), order.end());
  CGrid out = g;
  std::vector<char> bar_row(H, 0);
  for (int y = 0; y < H; ++y)
    for (size_t i = 0; i < spans.size(); ++i)
      for (int x = spans[i].first; x < spans[i].second; ++x) bar_row[y] = bar_row[y] || g[y][x] == main_color[i];
  for (size_t slot = 0; slot < spans.size(); ++slot) {
    size_t src = std::get<1>(order[slot]);
    for (int y = 0; y < H; ++y) {
      bool bar = bar_row[y];
      for (int d = 0; d < spans[slot].second - spans[slot].first; ++d) {
        int v = g[y][spans[src].first + d];
        if (clean && main_color[src] >= 0) v = bar ? main_color[src] : dc;
        out[y][spans[slot].first + d] = v;
      }
    }
  }
  g = out;
  return true;
}

DslArgs make(const std::string& key, bool desc, const std::string& axis, bool clean = false) {
  DslArgs a = {{"key", cdv_str("panel_" + key)}, {"desc", cdv_bool(desc)}, {"axis", cdv_str(axis)}};
  if (clean) a.emplace_back("clean", cdv_bool(true));
  return a;
}

}  // namespace

// SORT_OBJECTS key=panel_* (concept_sort_objects.cpp).
void sort_panels(Scene& scene, const DslAction& action) {
  CGrid g = concept_render(scene);
  if (g.empty() || g[0].empty()) return;
  int bg = concept_bg(scene);
  std::string key = arg_str(action.args, "key", "panel_minority").substr(6);
  bool desc = as_bool(args_get(action.args, "desc"), false);
  bool rows = arg_str(action.args, "axis", "v") == "h";
  if (rows) g = transpose(g);
  if (!sort_columns(g, bg, key, desc, as_bool(args_get(action.args, "clean"), false))) return;
  if (rows) g = transpose(g);
  concept_install(scene, g);
}

DslArgs sort_panels_sample(ArcRandom& rng) {
  return make(pick_str(rng, {"minority", "nonbg", "minority_row"}), pick_int(rng, {0, 1}) == 1,
              pick_str(rng, {"v", "h"}), pick_int(rng, {0, 1}) == 1);
}

std::vector<DslArgs> sort_panels_seeds(const std::vector<Grid>& ins, const std::vector<Grid>& outs) {
  std::vector<DslArgs> res;
  for (size_t i = 0; i < ins.size(); ++i)
    if (ins[i].height() != outs[i].height() || ins[i].width() != outs[i].width()) return res;
  for (const char* k : {"minority", "nonbg", "minority_row"})
    for (bool desc : {false, true})
      for (const char* axis : {"v", "h"})
        for (bool clean : {false, true}) res.push_back(make(k, desc, axis, clean));
  return res;
}

}  // namespace arc
