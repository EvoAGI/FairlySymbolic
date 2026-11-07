// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Scene / Node / Edge container. See graph.hpp for the determinism contract.
#include "arc/graph.hpp"

#include <algorithm>
#include <set>

#include "arc/md5.hpp"

namespace arc {

// --------------------------------------------------------------------------
// Node: derived fields.
// --------------------------------------------------------------------------

void Node::compute_bounding_box() {
  if (pixel_list.empty()) return;
  int ymin = pixel_list[0].first, ymax = pixel_list[0].first;
  int xmin = pixel_list[0].second, xmax = pixel_list[0].second;
  for (const auto& p : pixel_list) {
    ymin = std::min(ymin, p.first);
    ymax = std::max(ymax, p.first);
    xmin = std::min(xmin, p.second);
    xmax = std::max(xmax, p.second);
  }
  bbox_ymin = ymin;
  bbox_xmin = xmin;
  bbox_ymax = ymax;
  bbox_xmax = xmax;
  has_bbox = true;
}

void Node::compute_centroid() {
  if (pixel_list.empty()) return;
  // Mean y and x: integer sums, then floating-point division.
  long long sy = 0, sx = 0;
  for (const auto& p : pixel_list) {
    sy += p.first;
    sx += p.second;
  }
  double n = static_cast<double>(pixel_list.size());
  centroid_y = static_cast<double>(sy) / n;
  centroid_x = static_cast<double>(sx) / n;
  has_centroid = true;
}

void Node::compute_dimensions() {
  if (!has_bbox) return;
  height = bbox_ymax - bbox_ymin + 1;
  width = bbox_xmax - bbox_xmin + 1;
}

void Node::compute_density() {
  if (has_bbox && mass > 0) {
    long long area = static_cast<long long>(width) * height;
    density = area > 0 ? static_cast<double>(mass) / static_cast<double>(area) : 0.0;
  }
}

void Node::post_init() {
  // Fixed order: bbox, centroid, dimensions, mass, density.
  if (!pixel_list.empty() && !has_bbox) compute_bounding_box();
  if (!pixel_list.empty() && !has_centroid) compute_centroid();
  if (has_bbox) compute_dimensions();
  if (mass == 0 && !pixel_list.empty()) mass = static_cast<int>(pixel_list.size());
  if (has_bbox && mass > 0) compute_density();
}

// --------------------------------------------------------------------------
// Node: node / SubObject methods.
// --------------------------------------------------------------------------

void Node::check_boundary_touches(int grid_width, int grid_height) {
  if (!has_bbox) return;
  touches_top = (bbox_ymin == 0);
  touches_bottom = (bbox_ymax == grid_height - 1);
  touches_left = (bbox_xmin == 0);
  touches_right = (bbox_xmax == grid_width - 1);
}

namespace {

// Occupancy of a pixel list over its bounding box (a pixel set without a tree).
struct PixelMask {
  int y0 = 0, x0 = 0, h = 0, w = 0;
  std::vector<uint8_t> on;
  explicit PixelMask(const std::vector<Coord>& px) {
    int y1 = px[0].first, x1 = px[0].second;
    y0 = y1;
    x0 = x1;
    for (const auto& p : px) {
      y0 = std::min(y0, p.first);
      y1 = std::max(y1, p.first);
      x0 = std::min(x0, p.second);
      x1 = std::max(x1, p.second);
    }
    h = y1 - y0 + 1;
    w = x1 - x0 + 1;
    on.assign(static_cast<size_t>(h) * w, 0);
  }
  void set(const Coord& p) {
    if (p.first >= y0 && p.first < y0 + h && p.second >= x0 && p.second < x0 + w)
      on[static_cast<size_t>(p.first - y0) * w + (p.second - x0)] = 1;
  }
  bool has(int y, int x) const {
    return y >= y0 && y < y0 + h && x >= x0 && x < x0 + w && on[static_cast<size_t>(y - y0) * w + (x - x0)];
  }
};

}  // namespace

void Node::compute_perimeter_pixels() {
  std::vector<Coord> out;
  if (!pixel_list.empty()) {
    PixelMask m(pixel_list);
    for (const auto& p : pixel_list) m.set(p);
    for (const auto& p : pixel_list) {
      const int y = p.first, x = p.second;
      if (!m.has(y - 1, x) || !m.has(y + 1, x) || !m.has(y, x - 1) || !m.has(y, x + 1)) out.push_back(p);
    }
  }
  perimeter_pixels.set(std::move(out));
}

void Node::compute_interior_pixels() {
  if (perimeter_pixels.empty()) compute_perimeter_pixels();
  std::vector<Coord> out;
  if (!pixel_list.empty()) {
    PixelMask perim(pixel_list);  // the perimeter is a subset of the pixels
    for (const auto& p : perimeter_pixels.get()) perim.set(p);
    for (const auto& p : pixel_list)
      if (!perim.has(p.first, p.second)) out.push_back(p);
  }
  interior_pixels.set(std::move(out));
}

std::vector<uint8_t> Node::normalized_matrix(int& out_h, int& out_w) const {
  if (pixel_list.empty()) {
    out_h = 1;
    out_w = 0;
    return {};
  }
  int ymin = bbox_ymin, xmin = bbox_xmin, ymax = bbox_ymax, xmax = bbox_xmax;
  int h = ymax - ymin + 1;
  int w = xmax - xmin + 1;
  out_h = h;
  out_w = w;
  std::vector<uint8_t> m(static_cast<size_t>(h) * w, 0);
  for (const auto& p : pixel_list) m[static_cast<size_t>(p.first - ymin) * w + (p.second - xmin)] = 1;
  return m;
}

void Node::compute_shape_hash() {
  int h = 0, w = 0;
  std::vector<uint8_t> m = normalized_matrix(h, w);
  // The hashed bytes: one 0/1 byte per cell, row-major. Empty -> "".
  shape_hash = md5_hex(m.data(), m.size());
}

std::string Node::classify_shape() {
  if (pixel_list.empty()) {
    shape_mask = "Empty";
    return *shape_mask;
  }
  if (mass == 1) {
    shape_mask = "Point";
    return *shape_mask;
  }

  int h = 0, w = 0;
  std::vector<uint8_t> m = normalized_matrix(h, w);
  auto M = [&](int y, int x) -> uint8_t { return m[static_cast<size_t>(y) * w + x]; };

  // Filled rectangle: every cell set.
  bool all_filled = true;
  for (uint8_t v : m)
    if (!v) {
      all_filled = false;
      break;
    }
  if (all_filled) {
    if (h == 1 || w == 1)
      shape_mask = "Line";
    else if (h == w)
      shape_mask = "Square";
    else
      shape_mask = "Rectangle";
    return *shape_mask;
  }

  // Frame: interior all empty, all four borders filled (h>=3 && w>=3).
  if (h >= 3 && w >= 3) {
    bool interior_empty = true;
    for (int y = 1; y < h - 1 && interior_empty; ++y)
      for (int x = 1; x < w - 1; ++x)
        if (M(y, x)) {
          interior_empty = false;
          break;
        }
    bool top_row = true, bot_row = true, left_col = true, right_col = true;
    for (int x = 0; x < w; ++x) {
      if (!M(0, x)) top_row = false;
      if (!M(h - 1, x)) bot_row = false;
    }
    for (int y = 0; y < h; ++y) {
      if (!M(y, 0)) left_col = false;
      if (!M(y, w - 1)) right_col = false;
    }
    if (interior_empty && top_row && bot_row && left_col && right_col) {
      shape_mask = "Frame";
      return *shape_mask;
    }
  }

  // Plus/cross: odd h,w >=3; middle row & col full; corners empty.
  if (h >= 3 && w >= 3 && (h % 2 == 1) && (w % 2 == 1)) {
    int mid_y = h / 2, mid_x = w / 2;
    bool mid_row = true, mid_col = true;
    for (int x = 0; x < w; ++x)
      if (!M(mid_y, x)) mid_row = false;
    for (int y = 0; y < h; ++y)
      if (!M(y, mid_x)) mid_col = false;
    if (mid_row && mid_col) {
      bool corners_empty = !M(0, 0) && !M(0, w - 1) && !M(h - 1, 0) && !M(h - 1, w - 1);
      if (corners_empty) {
        shape_mask = "Plus";
        return *shape_mask;
      }
    }
  }

  // L-shape / T-shape: density < 0.6 and mass >= 3, count corners.
  if (density < 0.6 && mass >= 3) {
    int corners = 0;
    static const int deltas[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (!M(y, x)) continue;
        int n = 0;
        int d0[2] = {0, 0}, d1[2] = {0, 0};
        for (const auto& d : deltas) {
          int ny = y + d[0], nx = x + d[1];
          if (0 <= ny && ny < h && 0 <= nx && nx < w && M(ny, nx)) {
            if (n == 0) {
              d0[0] = d[0];
              d0[1] = d[1];
            } else if (n == 1) {
              d1[0] = d[0];
              d1[1] = d[1];
            }
            ++n;
          }
        }
        if (n == 2) {
          // corner iff the two neighbors are perpendicular:
          // (d1[0]==0) != (d2[0]==0)
          if ((d0[0] == 0) != (d1[0] == 0)) ++corners;
        }
      }
    }
    if (corners == 1) {
      shape_mask = "L-Shape";
      return *shape_mask;
    } else if (corners == 2) {
      shape_mask = "T-Shape";
      return *shape_mask;
    }
  }

  // Diagonal: square, mass == h, main or anti diagonal fully set.
  if (h == w && mass == h) {
    bool main_diag = true;
    for (int i = 0; i < h; ++i)
      if (!M(i, i)) main_diag = false;
    if (main_diag) {
      shape_mask = "Diagonal";
      return *shape_mask;
    }
    bool anti_diag = true;
    for (int i = 0; i < h; ++i)
      if (!M(i, w - 1 - i)) anti_diag = false;
    if (anti_diag) {
      shape_mask = "Diagonal";
      return *shape_mask;
    }
  }

  shape_mask = "Irregular";
  return *shape_mask;
}

// classify_shape_cluster: Empty / Point / filled-rectangle (Line /
// Square / Rectangle) / Irregular. No Frame/Plus/L/T/Diagonal branches.
std::string Node::classify_shape_cluster() {
  if (pixel_list.empty()) {
    shape_mask = "Empty";
    return *shape_mask;
  }
  if (mass == 1) {
    shape_mask = "Point";
    return *shape_mask;
  }

  int h = 0, w = 0;
  std::vector<uint8_t> m = normalized_matrix(h, w);
  bool all_filled = true;
  for (uint8_t v : m)
    if (!v) {
      all_filled = false;
      break;
    }
  if (all_filled) {
    if (h == 1 || w == 1)
      shape_mask = "Line";
    else if (h == w)
      shape_mask = "Square";
    else
      shape_mask = "Rectangle";
    return *shape_mask;
  }

  shape_mask = "Irregular";
  return *shape_mask;
}

size_t Scene::add_node(const Node& node) { return add_node(Node(node)); }

size_t Scene::add_node(Node&& node) {
  NodeIndex& ix = ix_.mut();
  ix.by_type.clear();
  size_t idx = nodes_.size();
  ix.id_to_index[node.node_id] = idx;
  ix.layer_map[node.node_type].push_back(node.node_id);
  nodes_.push_back(std::move(node));
  return idx;
}

std::string Scene::add_implicit_node_coord(const std::string& implicit_type,
                                           const std::string& param_key, double coord) {
  node_counter += 1;
  Node n;
  n.node_id = "implicit_" + std::to_string(node_counter);
  n.node_type = "ImplicitObject";
  n.implicit_type = implicit_type;
  if (param_key == "y") {
    n.has_param_y = true;
    n.param_y = coord;
  } else {  // "x"
    n.has_param_x = true;
    n.param_x = coord;
  }
  add_node(n);
  return n.node_id;
}

std::string Scene::add_implicit_node_type(const std::string& implicit_type,
                                          const std::string& param_type) {
  node_counter += 1;
  Node n;
  n.node_id = "implicit_" + std::to_string(node_counter);
  n.node_type = "ImplicitObject";
  n.implicit_type = implicit_type;
  n.param_type = param_type;
  add_node(n);
  return n.node_id;
}

const std::string& edge_type_name(EdgeType t) {
  static const std::string names[] = {
      "IS_IMMEDIATELY_ABOVE", "IS_IMMEDIATELY_BELOW", "IS_IMMEDIATELY_LEFT_OF",
      "IS_IMMEDIATELY_RIGHT_OF", "OVERLAPS", "TOUCHES", "CONTAINS", "IS_PART_OF",
      "IS_INSIDE_ZONE", "SHARES_ROW", "SHARES_COLUMN", "COLLINEAR_TO", "DIAGONAL_TO",
      "SAME_CENTROID_ROW", "SAME_CENTROID_COLUMN", "PROJECTS_RAY_TO", "FALLS_TOWARDS",
      "BRIDGED_TO", "SAME_COLOR", "SAME_SHAPE", "IS_SYMMETRIC_TO", "FORMS_REPEATING_PATTERN",
      ""};
  static_assert(sizeof(names) / sizeof(names[0]) == static_cast<size_t>(EdgeType::NONE) + 1,
                "one name per EdgeType");
  return names[static_cast<size_t>(t)];
}

EdgeType edge_type_of(const std::string& name) {
  for (size_t i = 0; i < static_cast<size_t>(EdgeType::NONE); ++i)
    if (edge_type_name(static_cast<EdgeType>(i)) == name) return static_cast<EdgeType>(i);
  return EdgeType::NONE;
}

namespace {

size_t edge_slot(uint32_t src, uint32_t dst, EdgeType type, size_t mask) {
  uint64_t h = src * 0x9E3779B97F4A7C15ULL ^ dst * 0xC2B2AE3D27D4EB4FULL ^
               static_cast<uint64_t>(type) * 0x165667B19E3779F9ULL;
  h ^= h >> 29;
  return static_cast<size_t>(h) & mask;
}

}  // namespace

namespace {
uint32_t type_bit(EdgeType t) { return 1u << static_cast<unsigned>(t); }
}  // namespace

void Scene::rebuild_edge_slots(EdgeStore& es) {
  size_t indexed = 0;
  for (const EdgeRec& e : es.edges) indexed += !(es.unindexed & type_bit(e.type));
  size_t n = 16;
  while (n < 2 * indexed + 2) n <<= 1;
  es.slots.assign(n, 0);
  const size_t mask = n - 1;
  for (size_t i = 0; i < es.edges.size(); ++i) {
    const EdgeRec& e = es.edges[i];
    if (es.unindexed & type_bit(e.type)) continue;
    size_t p = edge_slot(e.src, e.dst, e.type, mask);
    while (es.slots[p]) p = (p + 1) & mask;
    es.slots[p] = static_cast<uint32_t>(i + 1);
  }
  es.n_indexed = indexed;
  es.index_stale = false;
}

void Scene::add_new_edge_slow(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr) {
  EdgeStore& es = edges_mut();
  es.edges.push_back(EdgeRec{src, dst, type, attr});
  es.unindexed |= type_bit(type);
}

uint32_t Scene::edge_attr_record(EdgeAttrs&& attrs) {
  EdgeStore& es = edges_mut();
  if (es.attrs.use_count() > 1) es.attrs = std::make_shared<std::vector<EdgeAttrs>>(*es.attrs);
  es.attrs->push_back(std::move(attrs));
  return static_cast<uint32_t>(es.attrs->size());
}

void Scene::add_clique(EdgeType type, std::vector<uint32_t> members) {
  if (members.size() < 2) return;
  EdgeStore& es = edges_mut();
  es.cliques.push_back(Clique{type, es.edges.size(), std::move(members)});
  es.unindexed |= type_bit(type);
}

void Scene::expand_cliques(EdgeStore& es) {
  size_t n = es.edges.size();
  for (const Clique& c : es.cliques) n += c.members.size() * (c.members.size() - 1);
  std::vector<EdgeRec> all;
  all.reserve(n);
  size_t c = 0;
  for (size_t i = 0; i <= es.edges.size(); ++i) {
    for (; c < es.cliques.size() && es.cliques[c].pos == i; ++c) {
      const std::vector<uint32_t>& m = es.cliques[c].members;
      for (size_t a = 0; a < m.size(); ++a)
        for (size_t b = a + 1; b < m.size(); ++b) {
          all.push_back(EdgeRec{m[a], m[b], es.cliques[c].type, 0});
          all.push_back(EdgeRec{m[b], m[a], es.cliques[c].type, 0});
        }
    }
    if (i < es.edges.size()) all.push_back(es.edges[i]);
  }
  es.edges.swap(all);
  es.cliques.clear();
  es.slots.clear();
  es.index_stale = true;
}

uint32_t Scene::edge_node(const std::string& node_id) {
  EdgeStore& es = edges_mut();
  auto it = es.idt->handle.find(node_id);
  if (it != es.idt->handle.end()) return it->second;
  if (es.idt.use_count() > 1) es.idt = std::make_shared<EdgeIds>(*es.idt);
  const uint32_t h = static_cast<uint32_t>(es.idt->ids.size());
  es.idt->ids.push_back(node_id);
  es.idt->handle.emplace(node_id, h);
  return h;
}

uint32_t Scene::find_edge_node(const std::string& node_id) const {
  if (!es_) return kNoEdgeNode;
  auto it = es_->idt->handle.find(node_id);
  return it == es_->idt->handle.end() ? kNoEdgeNode : it->second;
}

void Scene::put_edge(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr_rec) {
  EdgeStore& es = edges_mut();
  if (es.unindexed & type_bit(type)) {  // index the type's appended edges first
    if (!es.cliques.empty()) expand_cliques(es);
    es.unindexed &= ~type_bit(type);
    rebuild_edge_slots(es);
  }
  const size_t mask = es.slots.size() - 1;
  size_t p = edge_slot(src, dst, type, mask);
  for (; es.slots[p]; p = (p + 1) & mask) {
    EdgeRec& x = es.edges[es.slots[p] - 1];
    if (x.src == src && x.dst == dst && x.type == type) {
      x.attr = attr_rec;  // the new edge's attributes replace the old's
      return;
    }
  }
  es.edges.push_back(EdgeRec{src, dst, type, attr_rec});
  if (2 * ++es.n_indexed + 2 > es.slots.size())
    rebuild_edge_slots(es);
  else
    es.slots[p] = static_cast<uint32_t>(es.edges.size());
}

const std::vector<EdgeRec>& Scene::no_edge_recs() {
  static const std::vector<EdgeRec> none;
  return none;
}

const std::vector<Edge>& Scene::edges() const {
  static const std::vector<Edge> none;
  if (!edges_gone_.empty()) drop_gone_edges();
  if (!es_) return none;
  if (!es_->spelled) {
    auto v = std::make_shared<std::vector<Edge>>();
    v->reserve(es_->edges.size());
    const std::vector<std::string>& ids = es_->idt->ids;
    for_each_edge([&](const EdgeRec& r) {
      Edge e{ids[r.src], ids[r.dst], edge_type_name(r.type), {}, {}, {}};
      if (r.attr) {
        const EdgeAttrs& a = (*es_->attrs)[r.attr - 1];
        e.int_attrs = a.int_attrs;
        e.str_attrs = a.str_attrs;
        e.dbl_attrs = a.dbl_attrs;
      }
      v->push_back(std::move(e));
    });
    es_->spelled = std::move(v);
  }
  return *es_->spelled;
}

void Scene::drop_gone_edges() const {
  std::set<std::string> gone;
  gone.swap(edges_gone_);
  if (!es_) return;
  const EdgeIds& t = *es_->idt;
  std::vector<char> hit(t.ids.size(), 0);
  bool any = false;
  for (const auto& id : gone) {
    auto it = t.handle.find(id);
    if (it != t.handle.end()) hit[it->second] = 1, any = true;
  }
  if (!any) return;
  auto is_hit = [&hit](const EdgeRec& e) { return hit[e.src] || hit[e.dst]; };
  const std::vector<EdgeRec>& cur = es_->edges;
  bool in_clique = false;
  for (const Clique& c : es_->cliques)
    for (uint32_t m : c.members) in_clique = in_clique || hit[m];
  auto first = std::find_if(cur.begin(), cur.end(), is_hit);
  if (first == cur.end() && !in_clique) return;
  // The kept edges in order; a clique loses its removed members (the clique of
  // the rest is what removing their edges leaves) and moves to the kept index.
  std::vector<size_t> kept_before(cur.size() + 1, 0);
  auto kept = std::make_shared<EdgeStore>();
  kept->idt = es_->idt;
  kept->attrs = es_->attrs;
  kept->unindexed = es_->unindexed;
  kept->edges.reserve(cur.size());
  for (size_t i = 0; i < cur.size(); ++i) {
    kept_before[i] = kept->edges.size();
    if (!is_hit(cur[i])) kept->edges.push_back(cur[i]);
  }
  kept_before[cur.size()] = kept->edges.size();
  for (const Clique& c : es_->cliques) {
    Clique k{c.type, kept_before[c.pos], {}};
    for (uint32_t m : c.members)
      if (!hit[m]) k.members.push_back(m);
    if (k.members.size() >= 2) kept->cliques.push_back(std::move(k));
  }
  kept->index_stale = true;
  es_ = std::move(kept);
}

Scene::EdgeStore& Scene::edges_mut() {
  if (!edges_gone_.empty()) drop_gone_edges();
  if (!es_) {
    es_ = std::make_shared<EdgeStore>();
    es_->idt = std::make_shared<EdgeIds>();
    es_->attrs = std::make_shared<std::vector<EdgeAttrs>>();
  } else if (es_.use_count() > 1) {
    es_ = std::make_shared<EdgeStore>(*es_);  // the node table and attributes stay shared
  }
  EdgeStore& es = *es_;
  es.spelled.reset();
  if (es.index_stale || es.slots.empty()) rebuild_edge_slots(es);
  return es;
}

std::unordered_map<std::string, int> Scene::edge_degrees(EdgeType type) const {
  std::unordered_map<std::string, int> deg;
  for_each_edge([&](const EdgeRec& e) {
    if (e.type != type) return;
    ++deg[edge_node_id(e.src)];
    ++deg[edge_node_id(e.dst)];
  });
  return deg;
}

size_t Scene::edge_bytes() const {
  if (!es_) return 0;
  size_t b = es_->edges.capacity() * sizeof(EdgeRec) + es_->slots.capacity() * sizeof(uint32_t);
  for (const Clique& c : es_->cliques) b += sizeof(Clique) + c.members.capacity() * sizeof(uint32_t);
  b += es_->attrs->capacity() * sizeof(EdgeAttrs);
  for (const EdgeAttrs& a : *es_->attrs) {
    b += a.int_attrs.capacity() * sizeof(std::pair<std::string, int>);
    b += a.str_attrs.capacity() * sizeof(std::pair<std::string, std::string>);
    b += a.dbl_attrs.capacity() * sizeof(std::pair<std::string, double>);
    for (const auto& kv : a.str_attrs) b += kv.first.capacity() + kv.second.capacity();
  }
  for (const std::string& id : es_->idt->ids) b += sizeof(std::string) + id.capacity() + 48;  // + map node
  if (es_->spelled) b += es_->spelled->capacity() * (sizeof(Edge) + 32);
  return b;
}

void Scene::remove_node(const std::string& node_id) {
  if (!ix_.get().id_to_index.count(node_id)) return;
  node_handle_.clear();
  NodeIndex& ix = ix_.mut();
  ix.by_type.clear();
  size_t idx = ix.id_to_index.find(node_id)->second;
  std::string type = nodes_[idx].node_type;

  // Erase from the node vector and rebuild the id->index map (indices shift).
  nodes_.erase(nodes_.begin() + static_cast<std::ptrdiff_t>(idx));
  ix.id_to_index.clear();
  for (size_t i = 0; i < nodes_.size(); ++i) ix.id_to_index[nodes_[i].node_id] = i;

  // Discard the id from its layer bucket.
  auto lit = ix.layer_map.find(type);
  if (lit != ix.layer_map.end()) {
    std::vector<std::string>& ids = lit->second;
    ids.erase(std::remove(ids.begin(), ids.end(), node_id), ids.end());
  }

  // Drop every incident edge (both directions): removing a node deletes the
  // node AND all edges touching it (deferred: edges_gone_).
  // The step 5d / noise_first callers run before any edges exist, so this is a
  // no-op for them; the DSL DELETE action relies on it.
  if (es_ && (!es_->edges.empty() || !es_->cliques.empty())) edges_gone_.insert(node_id);
}

void Scene::remove_nodes(const std::vector<std::string>& node_ids) {
  std::set<std::string> gone;
  for (const auto& id : node_ids)
    if (ix_.get().id_to_index.count(id)) gone.insert(id);
  if (gone.empty()) return;
  if (gone.size() == 1) return remove_node(*gone.begin());
  node_handle_.clear();
  NodeIndex& ix = ix_.mut();
  ix.by_type.clear();
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                              [&](const Node& n) { return gone.count(n.node_id) > 0; }),
               nodes_.end());
  ix.id_to_index.clear();
  for (size_t i = 0; i < nodes_.size(); ++i) ix.id_to_index[nodes_[i].node_id] = i;
  for (auto& kv : ix.layer_map) {
    std::vector<std::string>& ids = kv.second;
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](const std::string& id) { return gone.count(id) > 0; }),
              ids.end());
  }
  if (es_ && (!es_->edges.empty() || !es_->cliques.empty())) edges_gone_.insert(gone.begin(), gone.end());
}

void Scene::reclassify_as_noise(const std::string& node_id) {
  const auto& id_to_index = ix_.get().id_to_index;
  auto it = id_to_index.find(node_id);
  if (it == id_to_index.end()) return;
  size_t idx = it->second;
  if (nodes_[idx].node_type != "SubObject") return;

  NodeIndex& ix = ix_.mut();
  ix.by_type.clear();
  nodes_[idx].node_type = "NoiseObject";

  auto sit = ix.layer_map.find("SubObject");
  if (sit != ix.layer_map.end()) {
    std::vector<std::string>& ids = sit->second;
    ids.erase(std::remove(ids.begin(), ids.end(), node_id), ids.end());
  }
  ix.layer_map["NoiseObject"].push_back(node_id);
}

std::vector<size_t> Scene::get_nodes_by_type(const std::string& node_type,
                                             bool include_structural) const {
  std::vector<size_t> out;
  const NodeIndex& ix = ix_.get();
  auto it = ix.layer_map.find(node_type);
  if (it == ix.layer_map.end()) return out;

  const std::vector<size_t>* sorted = nullptr;
  for (const auto& kv : ix.by_type)
    if (kv.first == node_type) sorted = &kv.second;
  if (!sorted) {
    // Byte-wise lexicographic sort on the node-id string (canonical order).
    std::vector<std::string> ids = it->second;
    std::stable_sort(ids.begin(), ids.end());
    std::vector<size_t> idx;
    idx.reserve(ids.size());
    for (const auto& id : ids) idx.push_back(ix.id_to_index.at(id));
    ix.by_type.emplace_back(node_type, std::move(idx));
    sorted = &ix.by_type.back().second;
  }
  out.reserve(sorted->size());
  for (size_t idx : *sorted) {
    if (!include_structural && nodes_[idx].is_structural) continue;
    out.push_back(idx);
  }
  return out;
}

}  // namespace arc
