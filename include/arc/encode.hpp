// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

#ifndef ARC_ENCODE_HPP
#define ARC_ENCODE_HPP

// Encoder entry points: the numbered steps that turn a grid into a scene graph
// (step 1 background ... step 17 symmetries), the orchestrator, and the
// canonical scene serializers.

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arc/config.hpp"
#include "arc/graph.hpp"
#include "arc/grid.hpp"

namespace arc {

// Step 1: the background-color heuristic. Returns nullopt when the grid has no
// discernible background.
std::optional<int> calculate_background_heuristic(const Grid& grid);

// Step 1 orchestration: resolve the background color from config + heuristic.
std::optional<int> resolve_background(const Grid& grid, const GraphConfig& cfg);

// Step 2: detect grid-line zones. Emits structural gridline SubObject(s) and
// ZoneNode(s) into the scene, advancing the node counter, and returns the
// grid-line pixel set (fed to step 3's exclude_pixels).
std::set<Coord> detect_zones(Scene& scene, const Grid& grid, const GraphConfig& cfg);

// Step 3: extract SubObjects (connected_component / pixel / color_global). Does
// NOT add to the scene (the orchestrator adds them after 3b/3c/3d). Returns
// them in creation order with spatial ordering already stamped. exclude_pixels
// are skipped.
std::vector<Node> extract_step3(Scene& scene, const Grid& grid, const GraphConfig& cfg,
                                const std::set<Coord>& exclude_pixels);

// Step 3b: decompose irregular SubObjects into simpler sub-shapes.
std::vector<Node> decompose_objects(Scene& scene, std::vector<Node> subobjects,
                                    const Grid& grid, const GraphConfig& cfg);

// Step 3d: merge same-color objects by containment relationship.
std::vector<Node> merge_objects_by_containment(Scene& scene, std::vector<Node> subobjects,
                                               const Grid& grid);

// Step 4: detect enclosed cavities within the subobject foreground. Returns the
// CavityObjects (not added; the orchestrator adds them). subobjects is the
// post-3d subobject list (ClusterObjects from 3c are NOT part of the foreground).
std::vector<Node> detect_cavities(Scene& scene, const Grid& grid,
                                  const std::vector<Node>& subobjects);

// Step 5: group pixel-adjacent multi-color SubObjects into ClusterObjects.
// Step 5c (colors=sorted histogram keys) is applied inline on each cluster.
std::vector<Node> detect_clusters(Scene& scene, const std::vector<Node>& subobjects,
                                  const GraphConfig& cfg);

// Step 5d: split SubObjects straddling a larger object's bbox. Mutates the
// scene (removes the split node, adds two new sub_{n}). Default-off.
void split_by_containment(Scene& scene);

// Step 6: spatial edges (TOUCHES / OVERLAPS / directional). Iterates the scene
// node list in insertion order, physical types with non-empty pixels, O(n^2)
// i<j pairs. Uses spatial_touch_connectivity.
void compute_spatial_edges(Scene& scene, const GraphConfig& cfg);

// Step 7: noise scoring. Writes noise_score (0..1) onto every SubObject. The
// centroid-mean accumulation runs in lexicographic node-id order, so the float
// bits never depend on iteration order.
void score_noise_candidates(Scene& scene, const GraphConfig& cfg);

// Step 8: containment edges (CONTAINS / IS_PART_OF + cluster/cavity edges).
void compute_containment_edges(Scene& scene);

// Step 9: local coordinates + IS_INSIDE_ZONE. For every non-zone/non-implicit
// object with a centroid+bbox, find the smallest-area containing ZoneNode
// (candidates in lexicographic node-id order, so ties resolve the same way
// every run), stamp local_bounding_box / local_centroid, add an IS_INSIDE_ZONE
// edge, and append the object to the zone's child_nodes.
void assign_local_coordinates(Scene& scene);

// Step 9b: per-ZoneNode aggregate features (n_colors, dominant_color, fill_ratio,
// inside/outside object buckets, has_color_0..9) + zone_row/col grid position,
// propagated to child objects over CONTAINS out-edges / IS_INSIDE_ZONE in-edges.
void compute_zone_features(Scene& scene, const Grid& grid);

// Step 10: z_index from the CONTAINS containment DAG (custom BFS: roots =
// nodes with no CONTAINS parent, FIFO queue, max-wins on revisit).
void compute_z_indices(Scene& scene);

// Step 11: alignment edges (SHARES_ROW/COLUMN, COLLINEAR_TO, DIAGONAL_TO,
// SAME_CENTROID_ROW/COLUMN). Iterates the scene node list in insertion order,
// physical-typed nodes with a bbox, O(n^2) i<j pairs. Uses alignment_tolerance.
void compute_alignment_edges(Scene& scene, const GraphConfig& cfg);

// Step 12: ray edges (PROJECTS_RAY_TO). Casts a ray from each SubObject/Cluster
// centroid in 4 cardinal directions, connecting to the nearest object whose
// bbox spans the ray. Iterates nodes in insertion order.
void compute_ray_edges(Scene& scene);

// Step 13: gravity edges (FALLS_TOWARDS). Occupancy map over physical pixels;
// each object drops straight down to the nearest object below (grid floor = no
// edge). Iterates nodes in insertion order.
void compute_gravity_edges(Scene& scene);

// Step 14: bridge edges (BRIDGED_TO). Line/Rectangle 1-wide density-1 SubObjects
// act as bridges; any two objects both TOUCHing a bridge become BRIDGED_TO.
// Reads the bridge's TOUCHES neighbours: out-neighbours, then in-neighbours,
// each in the order they first appear among the edges.
void compute_bridge_edges(Scene& scene);

// Step 15: logic edges (SAME_SHAPE, SAME_COLOR, IS_SYMMETRIC_TO). Groups
// SubObjects by color / shape_hash / matrix-shape and emits undirected pairs
// inside each group; symmetry is a flip/rot90 matrix predicate.
void compute_logic_edges(Scene& scene);

// Step 16: repeating-pattern detection. Groups SubObjects by
// (color, shape_hash), finds regular horizontal/vertical/diagonal lines, and
// emits PatternObject nodes (id "pattern_{n}") + FORMS_REPEATING_PATTERN /
// IS_PART_OF edges. Uses cfg.pattern_min_size. Iterates SubObjects in
// lexicographic node-id order.
void detect_patterns(Scene& scene, const GraphConfig& cfg);

// Step 17: symmetry detection (detect_symmetries). Adds ImplicitObject axis
// nodes (id "implicit_{n}") for horizontal / vertical / diagonal grid symmetry
// and connects mirrored SubObjects to the H/V axes via IS_SYMMETRIC_TO edges.
// Uses cfg.symmetry_tolerance.
void detect_symmetries(Scene& scene, const Grid& grid, const GraphConfig& cfg);

// Orchestrator: run steps 1..3d (gated by config) and add the resulting nodes.
Scene encode_minimal(const Grid& grid, const GraphConfig& cfg);

// Canonical scene serializer: deterministic JSON string. Nodes sorted by
// lexicographic id; floats emitted as raw <d little-endian bits hex.
std::string scene_to_canonical_json(const Scene& scene);

// Canonical scene serializer WITH edges (steps 6/7/8). Identical node block to
// scene_to_canonical_json, plus an "edges" array sorted by (src, dst, type) with
// each edge's integer attributes (only overlap_size in step1..8) in a fixed key
// slot.
std::string scene_to_canonical_json_with_edges(const Scene& scene);

// Canonical scene serializer for steps 1..10: the with-edges node block
// extended with the step 9 / 9b / 10 node fields (local_bounding_box,
// local_centroid, zone_row/col, and the per-zone feature fields) in fixed key
// order. IS_INSIDE_ZONE edges appear in the edges[] array.
std::string scene_to_canonical_json_step10(const Scene& scene);

// Canonical scene serializer for steps 1..15: the step10 node block
// (unchanged -- steps 11..15 add no node fields) plus an edges[] array that now
// carries the alignment / ray / gravity / bridge / logic edge attributes
// (axis, direction, distance, fall_distance, bridge_node) in fixed key order.
std::string scene_to_canonical_json_step15(const Scene& scene);

// Canonical scene serializer for the FULL step1..17 encoder: the
// step15 edge block plus the step 16/17 node types. PatternObject appends
// pattern_type + constituent_ids; ImplicitObject appends implicit_type +
// parameters. New edge types (FORMS_REPEATING_PATTERN, IS_PART_OF from patterns,
// IS_SYMMETRIC_TO from axes) carry no new attributes, so the edge slot block is
// unchanged.
std::string scene_to_canonical_json_full(const Scene& scene);

}  // namespace arc

#endif  // ARC_ENCODE_HPP
