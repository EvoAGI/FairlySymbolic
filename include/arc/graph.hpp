// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_GRAPH_HPP
#define ARC_GRAPH_HPP

// Scene / Node / Edge container: the scene graph (a directed multigraph) the
// encoder builds and the DSL actions edit.
//
// Determinism contract:
//   * A single global node_counter per Scene (starts at 0). Ids are strings
//     built from it ("sub_1", "sub_2", ...).
//   * get_nodes_by_type returns nodes sorted BYTE-WISE LEXICOGRAPHICALLY on
//     the node-id string (so "sub_10" < "sub_2"), excluding is_structural by
//     default.
//   * Edges are a multigraph keyed on (src, dst, edge_type); re-adding the same
//     key updates rather than duplicates. They are stored compactly (EdgeRec:
//     interned endpoint handles and a type code); edges() spells them out.

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arc/grid.hpp"

namespace arc {

// Values a node keeps from its encode and no action changes: the copies of a
// node share them (a scene-cache hit copies every node), so copying one is a
// reference count, not an allocation. LocalShared<T>: an immutable T with a
// plain (non-atomic) count -- a scene and its copies stay on the thread of
// their task, and atomic counts cost a lock on some toolchains (MinGW).
template <typename T>
class LocalShared {
 public:
  LocalShared() = default;
  explicit LocalShared(T v) : p_(new Box{std::move(v), 1}) {}
  LocalShared(const LocalShared& o) : p_(o.p_) {
    if (p_) ++p_->count;
  }
  LocalShared(LocalShared&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  LocalShared& operator=(LocalShared o) noexcept {
    std::swap(p_, o.p_);
    return *this;
  }
  ~LocalShared() {
    if (p_ && --p_->count == 0) delete p_;
  }
  explicit operator bool() const { return p_ != nullptr; }
  const T& operator*() const { return p_->value; }
  const T* operator->() const { return &p_->value; }

 private:
  struct Box {
    T value;
    long count;
  };
  Box* p_ = nullptr;
};
// LocalCow<T>: a T shared by the copies of a scene until one of them changes
// it (mut() copies it then); plain count, same thread rule as LocalShared.
template <typename T>
class LocalCow {
 public:
  LocalCow() : p_(new Box{T{}, 1}) {}
  LocalCow(const LocalCow& o) : p_(o.p_) { ++p_->count; }
  LocalCow& operator=(const LocalCow& o) {
    LocalCow t(o);
    std::swap(p_, t.p_);
    return *this;
  }
  ~LocalCow() {
    if (--p_->count == 0) delete p_;
  }
  const T& get() const { return p_->value; }
  T& mut() {
    if (p_->count > 1) {
      --p_->count;
      p_ = new Box{p_->value, 1};
    }
    return p_->value;
  }

 private:
  struct Box {
    T value;
    long count;
  };
  Box* p_;
};
// SharedStr: an optional string (the std::optional<std::string> interface the
// readers use).
class SharedStr {
 public:
  SharedStr() = default;
  SharedStr& operator=(std::string s) {
    p_ = LocalShared<std::string>(std::move(s));
    return *this;
  }
  bool has_value() const { return static_cast<bool>(p_); }
  explicit operator bool() const { return static_cast<bool>(p_); }
  const std::string& operator*() const { return *p_; }
  const std::string* operator->() const { return p_.operator->(); }

 private:
  LocalShared<std::string> p_;
};
// SharedCoords: a pixel list set once (perimeter / interior, which only the
// canonical JSON reads).
class SharedCoords {
 public:
  const std::vector<Coord>& get() const {
    static const std::vector<Coord> none;
    return p_ ? *p_ : none;
  }
  bool empty() const { return !p_ || p_->empty(); }
  void set(std::vector<Coord>&& v) {
    if (v.empty()) p_ = LocalShared<std::vector<Coord>>();
    else p_ = LocalShared<std::vector<Coord>>(std::move(v));
  }

 private:
  LocalShared<std::vector<Coord>> p_;
};

// A node in the scene graph. One struct holds the shared base fields and the
// type-specific fields of every node type (SubObject, ZoneNode, CavityObject,
// ClusterObject, PatternObject, ImplicitObject, ...).
struct Node {
  // Core identity.
  std::string node_id;
  std::string node_type = "ArcNode";

  // Absolute spatial attributes. bounding_box = (ymin, xmin, ymax, xmax).
  bool has_bbox = false;
  int bbox_ymin = 0, bbox_xmin = 0, bbox_ymax = 0, bbox_xmax = 0;
  int width = 0;
  int height = 0;
  bool has_centroid = false;
  double centroid_y = 0.0;
  double centroid_x = 0.0;

  // Pixel tracking (insertion order == BFS discovery order, preserved).
  std::vector<Coord> pixel_list;
  SharedCoords perimeter_pixels;
  SharedCoords interior_pixels;

  // Physical & ordering attributes.
  int mass = 0;
  double density = 0.0;
  int z_index = 0;
  // spatial_order keys assigned by compute_spatial_ordering (0 = not set,
  // which the canonical JSON writes as 0 too).
  struct SpatialOrder {
    int from_left = 0, from_right = 0, from_top = 0, from_bottom = 0, by_size = 0;
  };
  SpatialOrder spatial_order;

  // Step 9 (assign_local_coordinates): bbox / centroid relative to the
  // smallest containing ZoneNode. local_bounding_box = (ymin,xmin,ymax,xmax).
  bool has_local_bbox = false;
  int local_bbox_ymin = 0, local_bbox_xmin = 0, local_bbox_ymax = 0, local_bbox_xmax = 0;
  bool has_local_centroid = false;
  double local_centroid_y = 0.0;
  double local_centroid_x = 0.0;

  // Step 9b (compute_zone_features / zone_row-col propagation). zone_row/col
  // land on ZoneNodes AND (via CONTAINS/IS_INSIDE_ZONE) their child objects.
  std::optional<int> zone_row;
  std::optional<int> zone_col;

  // Step 9 (assign_local_coordinates): ZoneNode child list, in the order
  // objects were assigned to the zone (node-insertion iteration order).
  std::vector<std::string> child_nodes;

  // Step 9b (compute_zone_features): per-ZoneNode aggregate features.
  // has_zone_features gates emission (absent -> null). dominant_color may be
  // the background (nullopt when there is no background).
  bool has_zone_features = false;
  int zf_n_colors = 0;
  std::optional<int> zf_dominant_color;
  int zf_n_objects = 0;
  double zf_fill_ratio = 0.0;
  int zf_n_inside_objects = 0;
  int zf_n_outside_objects = 0;
  std::vector<int> zf_inside_colors;   // sorted
  std::vector<int> zf_outside_colors;  // sorted
  bool zf_has_color[10] = {false, false, false, false, false,
                           false, false, false, false, false};

  // Topology & hashing.
  SharedStr shape_hash;

  // State & boundary flags.
  bool is_structural = false;
  bool touches_top = false;
  bool touches_bottom = false;
  bool touches_left = false;
  bool touches_right = false;

  double noise_score = 0.0;

  // SubObject-specific.
  std::optional<int> color;
  std::optional<std::string> shape_mask;

  // ZoneNode-specific.
  std::optional<int> zone_color;

  // CavityObject-specific. enclosing_object_id is the node_id of the smallest
  // enclosing SubObject (nullopt when there is none); cavity_color is the scene
  // background color the cavity was filled from.
  std::optional<std::string> enclosing_object_id;
  std::optional<int> cavity_color;

  // ClusterObject-specific. color_histogram preserves insertion order (the
  // order colors are first seen while scanning pixel_list). `colors` (step 5c)
  // is the sorted key list, only populated for ClusterObjects.
  std::vector<std::pair<int, int>> color_histogram;  // (color, count) insertion-ordered
  std::vector<int> colors;                            // sorted color_histogram keys
  bool has_colors = false;                            // colors field was populated
  // ClusterObject member SubObject ids, in component BFS order (step 5). Feeds
  // the step-8 cluster->subobject CONTAINS / IS_PART_OF edges.
  std::vector<std::string> subobject_ids;

  // PatternObject-specific (step 16). pattern_type is "HLine"/"VLine"/"DiagLine";
  // constituent_ids are the member SubObject ids in emission order. A PatternObject
  // carries no color/shape_hash/shape_mask (never runs classify/compute_shape_hash).
  std::optional<std::string> pattern_type;
  std::vector<std::string> constituent_ids;

  // ImplicitObject-specific (step 17). implicit_type is "HorizontalAxis" /
  // "VerticalAxis" / "DiagonalAxis"; the parameters dict holds exactly ONE key:
  // "y"/"x" (a float axis coordinate) for H/V axes, or "type" ("main"/"anti") for
  // a diagonal axis. These flags select which serialize.
  std::optional<std::string> implicit_type;
  bool has_param_y = false;
  double param_y = 0.0;
  bool has_param_x = false;
  double param_x = 0.0;
  std::optional<std::string> param_type;  // "main" / "anti" (DiagonalAxis)

  // Marks a SubObject exempt from the min_object_size filter.
  bool merged = false;

  // Metadata flags used by gridline SubObjects.
  bool is_gridline_meta = false;
  std::vector<int> meta_h_lines;
  std::vector<int> meta_v_lines;

  // Step 18 object attributes (arc/object_attrs.hpp): one value per name of
  // object_attr_names(), in that order; empty when not computed. Set once by
  // compute_object_attributes and shared by the node's copies.
  LocalShared<std::vector<int>> attrs;

  // ---- derived-field computations ----------------------------------------
  void compute_bounding_box();
  void compute_centroid();
  void compute_dimensions();
  void compute_density();
  void post_init();  // runs the four computations above, in order

  // ---- node / SubObject methods -----------------------------------------
  void check_boundary_touches(int grid_width, int grid_height);
  void compute_perimeter_pixels();
  void compute_interior_pixels();
  void compute_shape_hash();          // MD5 over the normalized matrix
  std::string classify_shape();       // a SubObject's shape class
  std::string classify_shape_cluster();  // a ClusterObject's shape class (subset)

  // Normalized boolean matrix (tight bbox, row-major) as 0/1 bytes. Public so
  // step 15's symmetry test can compare flips/rotations.
  std::vector<uint8_t> normalized_matrix(int& out_h, int& out_w) const;

 private:
};

// Edge types, one code per edge type name. NONE is the code of a name no edge
// can have (a DSL argument that names no type).
enum class EdgeType : uint8_t {
  IS_IMMEDIATELY_ABOVE, IS_IMMEDIATELY_BELOW, IS_IMMEDIATELY_LEFT_OF, IS_IMMEDIATELY_RIGHT_OF,
  OVERLAPS, TOUCHES, CONTAINS, IS_PART_OF, IS_INSIDE_ZONE,
  SHARES_ROW, SHARES_COLUMN, COLLINEAR_TO, DIAGONAL_TO, SAME_CENTROID_ROW, SAME_CENTROID_COLUMN,
  PROJECTS_RAY_TO, FALLS_TOWARDS, BRIDGED_TO, SAME_COLOR, SAME_SHAPE, IS_SYMMETRIC_TO,
  FORMS_REPEATING_PATTERN,
  NONE
};
const std::string& edge_type_name(EdgeType t);
EdgeType edge_type_of(const std::string& name);  // NONE for an unknown name

// Edge attributes, in insertion order. Integer:
// OVERLAPS overlap_size, FALLS_TOWARDS fall_distance. String: COLLINEAR_TO axis,
// PROJECTS_RAY_TO direction, BRIDGED_TO bridge_node. Double: PROJECTS_RAY_TO
// distance.
struct EdgeAttrs {
  std::vector<std::pair<std::string, int>> int_attrs;
  std::vector<std::pair<std::string, std::string>> str_attrs;
  std::vector<std::pair<std::string, double>> dbl_attrs;
};

// A stored edge: endpoints are handles into the scene's edge-node table (the
// node ids, interned when an edge first names them); attr is 1 + an index into
// the store's EdgeAttrs, 0 when the edge has none. Edges may share a record.
struct EdgeRec {
  uint32_t src;
  uint32_t dst;
  EdgeType type;
  uint32_t attr;
};

// An edge with its ids, type name and attributes spelled out (Scene::edges(),
// built on demand from the EdgeRecs for the canonical JSON and the checks).
struct Edge {
  std::string source_id;
  std::string target_id;
  std::string edge_type;  // edge type name
  std::vector<std::pair<std::string, int>> int_attrs;
  std::vector<std::pair<std::string, std::string>> str_attrs;
  std::vector<std::pair<std::string, double>> dbl_attrs;

  bool has_int_attr(const std::string& k) const {
    for (const auto& kv : int_attrs)
      if (kv.first == k) return true;
    return false;
  }
  int get_int_attr(const std::string& k) const {
    for (const auto& kv : int_attrs)
      if (kv.first == k) return kv.second;
    return 0;
  }
  bool has_str_attr(const std::string& k) const {
    for (const auto& kv : str_attrs)
      if (kv.first == k) return true;
    return false;
  }
  const std::string& get_str_attr(const std::string& k) const {
    static const std::string empty;
    for (const auto& kv : str_attrs)
      if (kv.first == k) return kv.second;
    return empty;
  }
  bool has_dbl_attr(const std::string& k) const {
    for (const auto& kv : dbl_attrs)
      if (kv.first == k) return true;
    return false;
  }
  double get_dbl_attr(const std::string& k) const {
    for (const auto& kv : dbl_attrs)
      if (kv.first == k) return kv.second;
    return 0.0;
  }
};

class Scene {
 public:
  Scene(int width, int height) : width_(width), height_(height) {}

  int width() const { return width_; }
  int height() const { return height_; }
  void set_dims(int width, int height) {
    width_ = width;
    height_ = height;
  }

  bool has_background() const { return background_.has_value(); }
  int background() const { return background_.value(); }
  void set_background(std::optional<int> bg) { background_ = bg; }

  // Grid-line pixel set from step 2. Consumed by step 9b
  // (compute_zone_features) to exclude grid-line cells from zone pixel colors.
  const std::set<Coord>& grid_line_pixels() const {
    static const std::set<Coord> none;
    return grid_line_pixels_ ? *grid_line_pixels_ : none;
  }
  void set_grid_line_pixels(const std::set<Coord>& p) { grid_line_pixels_ = LocalShared<std::set<Coord>>(p); }

  // Original input grid, set by the encoder. A few DSL actions
  // (PROJECT_SINGLETONS_TO_FULL_LINES) read it directly; has_grid() tells
  // whether it is set.
  bool has_grid() const { return has_grid_; }
  const Grid& grid() const {
    static const Grid none;
    return grid_ ? *grid_ : none;
  }
  void set_grid(const Grid& g) {
    grid_ = LocalShared<Grid>(g);
    has_grid_ = true;
  }

  // Single global node counter.
  int64_t node_counter = 0;

  // add_node: stores the node and records its id under its type. Returns index.
  size_t add_node(const Node& node);
  size_t add_node(Node&& node);
  void reserve_nodes(size_t extra) { nodes_.reserve(nodes_.size() + extra); }

  // add_implicit_node: advances the shared
  // node_counter, builds an ImplicitObject with id "implicit_{n}" of the given
  // implicit_type + single-key parameters, adds it, and returns the node id.
  // param_key is "y"/"x" (float coord) or "type" (string "main"/"anti").
  std::string add_implicit_node_coord(const std::string& implicit_type,
                                       const std::string& param_key, double coord);
  std::string add_implicit_node_type(const std::string& implicit_type,
                                      const std::string& param_type);

  // Edges form a multigraph keyed on (src, dst, type): re-adding a key keeps
  // its place and replaces its attributes.
  // Endpoints are edge-node handles; edge_node interns a node id.
  static constexpr uint32_t kNoEdgeNode = 0xffffffffu;
  uint32_t edge_node(const std::string& node_id);
  uint32_t find_edge_node(const std::string& node_id) const;  // kNoEdgeNode if never interned
  // edge_node of nodes()[idx]'s id, remembered per node index while the
  // encoder runs (dropped when a node is removed; drop_node_handles() ends it).
  uint32_t edge_node_at(size_t idx) {
    if (idx < node_handle_.size() && node_handle_[idx] != kNoEdgeNode) return node_handle_[idx];
    const uint32_t h = edge_node(nodes_[idx].node_id);
    if (node_handle_.size() <= idx) node_handle_.resize(nodes_.size(), kNoEdgeNode);
    node_handle_[idx] = h;
    return h;
  }
  void drop_node_handles() { std::vector<uint32_t>().swap(node_handle_); }
  const std::string& edge_node_id(uint32_t h) const { return es_->idt->ids[h]; }
  void add_edge(uint32_t src, uint32_t dst, EdgeType type) { put_edge(src, dst, type, 0); }
  // With attributes: attr_rec from edge_attr_record (shareable, see below).
  void add_edge(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr_rec) {
    put_edge(src, dst, type, attr_rec);
  }
  // add_edge for a key the caller knows is not in the scene (a loop that emits
  // each of its keys once, of a type no earlier step adds): appended without a
  // lookup. The type's edges join the (src, dst, type) index only if a later
  // add_edge of that type needs it.
  // attr: 0 or a record from edge_attr_record. Any number of edges may share
  // a record (a record never changes once made; an update points the edge at
  // another one).
  void add_new_edge(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr = 0) {
    if (es_ && es_.use_count() == 1 && edges_gone_.empty() && !es_->spelled) {  // the encoder's case
      es_->edges.push_back(EdgeRec{src, dst, type, attr});
      es_->unindexed |= 1u << static_cast<unsigned>(type);
      return;
    }
    add_new_edge_slow(src, dst, type, attr);
  }
  uint32_t edge_attr_record(EdgeAttrs&& attrs);
  // The edges of a clique, for i < j: (m_i, m_j) then (m_j, m_i), kept as the
  // member list in their place among the edges (on a 30x30 grid SAME_COLOR /
  // SAME_SHAPE / IS_SYMMETRIC_TO cliques are most of the 200-400k edges). The
  // keys must be new, as for add_new_edge. Readers see the edges.
  void add_clique(EdgeType type, std::vector<uint32_t> members);
  void add_edge(const std::string& src, const std::string& dst, EdgeType type) {
    add_edge(edge_node(src), edge_node(dst), type);
  }

  // remove_node: erase a node by id from nodes_ and the node index (id -> index, layer map).
  // Used by step 5d.
  // Edges are not touched (step 5d runs before any edges are created).
  void remove_node(const std::string& node_id);
  // remove_nodes: the same end state as remove_node on each id in turn (absent
  // ids skipped), in one pass over nodes and edges instead of one per id.
  void remove_nodes(const std::vector<std::string>& node_ids);

  // reclassify_as_noise: retype a SubObject to NoiseObject and move its layer-map
  // entry. Only fires when noise_threshold
  // < 1.0 (default off). confidence is not part of the canonical scene JSON so it
  // is not tracked.
  void reclassify_as_noise(const std::string& node_id);

  const std::vector<Node>& nodes() const { return nodes_; }
  std::vector<Node>& nodes() { return nodes_; }
  Node& node_ref(size_t idx) { return nodes_[idx]; }

  // get_nodes_by_type: lexicographically-sorted node ids of the given type,
  // excluding is_structural unless include_structural. Returns indices into
  // nodes_.
  std::vector<size_t> get_nodes_by_type(const std::string& node_type,
                                        bool include_structural = false) const;

  // The stored edges in creation order, and an edge's attributes (null if
  // none). Valid until the next edge write or node removal. Expands the
  // cliques into the list; for_each_edge reads them without that.
  const std::vector<EdgeRec>& edge_recs() const {
    if (!edges_gone_.empty()) drop_gone_edges();
    if (es_ && !es_->cliques.empty()) expand_cliques(*es_);
    return es_ ? es_->edges : no_edge_recs();
  }
  // The edges in creation order as stored: on_edge(const EdgeRec&) for each
  // edge, on_clique(type, members) for each clique where its edges stand.
  template <typename FE, typename FC>
  void for_each_edge_block(FE&& on_edge, FC&& on_clique) const {
    if (!edges_gone_.empty()) drop_gone_edges();
    if (!es_) return;
    const std::vector<EdgeRec>& es = es_->edges;
    const std::vector<Clique>& cs = es_->cliques;
    size_t c = 0;
    for (size_t i = 0; i <= es.size(); ++i) {
      for (; c < cs.size() && cs[c].pos == i; ++c) on_clique(cs[c].type, cs[c].members);
      if (i < es.size()) on_edge(es[i]);
    }
  }
  // f(const EdgeRec&) for every edge in creation order, clique edges made on
  // the fly (attr 0).
  template <typename F>
  void for_each_edge(F&& f) const {
    if (!edges_gone_.empty()) drop_gone_edges();
    if (!es_) return;
    const std::vector<EdgeRec>& es = es_->edges;
    const std::vector<Clique>& cs = es_->cliques;
    size_t c = 0;
    for (size_t i = 0; i <= es.size(); ++i) {
      for (; c < cs.size() && cs[c].pos == i; ++c) {
        const std::vector<uint32_t>& m = cs[c].members;
        for (size_t a = 0; a < m.size(); ++a)
          for (size_t b = a + 1; b < m.size(); ++b) {
            f(EdgeRec{m[a], m[b], cs[c].type, 0});
            f(EdgeRec{m[b], m[a], cs[c].type, 0});
          }
      }
      if (i < es.size()) f(es[i]);
    }
  }
  const EdgeAttrs* edge_attrs(const EdgeRec& e) const {
    return e.attr ? &(*es_->attrs)[e.attr - 1] : nullptr;
  }

  // The edges spelled out (ids, type name, attributes) in creation order.
  // Built on demand and kept until the edges change: for the canonical JSON
  // and the checks, not for the search.
  const std::vector<Edge>& edges() const;

  // Node id -> number of edges of `type` incident to it in either direction
  // (its neighbour count over `type` edges in both directions), for every node
  // with at least one, in one pass over the edges.
  std::unordered_map<std::string, int> edge_degrees(EdgeType type) const;

  // Bytes held by the edge store (scene-cache accounting).
  size_t edge_bytes() const;
  size_t edge_node_count() const { return es_ ? es_->idt->ids.size() : 0; }

  // Look up a node index by id (npos if absent).
  bool has_node(const std::string& node_id) const {
    return ix_.get().id_to_index.count(node_id) > 0;
  }
  size_t index_of(const std::string& node_id) const { return ix_.get().id_to_index.at(node_id); }

 private:
  int width_;
  int height_;
  std::optional<int> background_;
  // Set by the encoder and shared by the scene's copies.
  LocalShared<std::set<Coord>> grid_line_pixels_;  // from step 2
  LocalShared<Grid> grid_;                         // original input grid
  bool has_grid_ = false;
  std::vector<Node> nodes_;
  // The node index, shared by the scene's copies until one of them adds,
  // removes or retypes a node (a scene-cache hit then copies no map).
  struct NodeIndex {
    std::unordered_map<std::string, size_t> id_to_index;  // node_id -> index in nodes_ (never iterated)
    std::map<std::string, std::vector<std::string>> layer_map;  // type -> ids (insertion order)
    // get_nodes_by_type's sorted ids as node indices, per type, structural
    // included; dropped whenever layer_map or id_to_index change.
    mutable std::vector<std::pair<std::string, std::vector<size_t>>> by_type;
  };
  LocalCow<NodeIndex> ix_;
  std::vector<uint32_t> node_handle_;  // edge_node_at's cache, per node index
  // The edges, their edge-node table and their (src, dst, type) index, shared
  // by the copies of a scene until one of them changes an edge (copy on
  // write): a scene-cache hit copies the nodes but not the edges, which on a
  // 30x30 grid number 20-380k. The node table is shared on, by the stores
  // copied from one another, until one of them interns a new id. Scenes stay
  // on the thread of their task, so the plain use_count is exact. The index
  // serves add_edge only; after a removal it is rebuilt there.
  struct EdgeIds {
    std::vector<std::string> ids;
    std::unordered_map<std::string, uint32_t> handle;
  };
  // A clique's edges stand before edges[pos] (pos ascending in the list).
  struct Clique {
    EdgeType type;
    size_t pos;
    std::vector<uint32_t> members;
  };
  struct EdgeStore {
    std::shared_ptr<EdgeIds> idt;
    std::vector<EdgeRec> edges;
    std::vector<Clique> cliques;
    std::shared_ptr<std::vector<EdgeAttrs>> attrs;  // shared like idt, copied on write
    std::vector<uint32_t> slots;  // edge index + 1; 0 = empty; size a power of 2
    bool index_stale = false;
    uint32_t unindexed = 0;  // bit per EdgeType: has edges from add_new_edge, not in slots
    size_t n_indexed = 0;    // edges in slots
    std::shared_ptr<const std::vector<Edge>> spelled;  // edges(), null until asked
  };
  // Removed nodes whose edges are still in es_: they go at the next edge read
  // or write, in one pass that keeps the order (the result of dropping them at
  // each removal). A program that deletes objects and then only renders never
  // pays for it.
  mutable std::shared_ptr<EdgeStore> es_;  // null: no edges
  mutable std::set<std::string> edges_gone_;
  static const std::vector<EdgeRec>& no_edge_recs();
  static void expand_cliques(EdgeStore& es);  // the cliques' edges into es.edges, in place
  void drop_gone_edges() const;
  EdgeStore& edges_mut();
  void put_edge(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr_rec);
  void add_new_edge_slow(uint32_t src, uint32_t dst, EdgeType type, uint32_t attr);
  static void rebuild_edge_slots(EdgeStore& es);
};

}  // namespace arc

#endif  // ARC_GRAPH_HPP
