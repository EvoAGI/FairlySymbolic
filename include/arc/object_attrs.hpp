// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Step 18: object attributes. Each SubObject gets integer attributes that
// describe its shape, how it relates to the other objects, and its rank among
// them. The BY_ATTR selector filters or picks extremes on any of them, so the
// search can use the whole set without one selector per attribute.
#pragma once

#include <string>
#include <vector>

#include "arc/graph.hpp"

namespace arc {

// Attribute names, in the order they are stored on each node.
const std::vector<std::string>& object_attr_names();

// Fills Node::attrs of every SubObject of the scene.
void compute_object_attributes(Scene& scene);

// Value of attribute `name` on `n`; false when absent.
bool node_attr(const Node& n, const std::string& name, int* value);

}  // namespace arc
