# ƸӜƷ Fairly Symbolic
# Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
# Copyright 2026 INRAE / Politecnico di Torino
# SPDX-License-Identifier: MIT-0

# Makefile for the C++ ARC solver.
#
# Run `make` from the repo root. Default target builds libarc.a and the
# standalone arc_solver binary.

# ---- toolchain -------------------------------------------------------------
CXX      ?= clang++
CC       ?= clang

# Shared warning/optimization flags for all C and C++ translation units.
# -ffp-contract=off keeps floating-point results reproducible: without it the
# compiler may contract `a + b*c` into a single fused multiply-add, which rounds
# differently from two separate IEEE-754 operations (e.g. in uniform() and
# gauss()). -fno-builtin-pow keeps std::pow off the builtin fast path for the
# same reason.
COMMON_FLAGS := -O2 -fno-builtin-pow -ffp-contract=off -Wall -Wextra -fno-exceptions

# Portability: Kaggle's g++ / libstdc++ does NOT transitively expose several
# standard symbols (size_t, uint*_t, memcpy, ...) that Apple clang / libc++ leaks,
# so the tree builds locally but fails to compile there. Force-include the
# standard headers into every C++ translation unit. C++ ONLY -- these are C++
# headers and there are no C sources; leave CFLAGS untouched.
FORCE_INCLUDES := -include cstddef -include cstdint -include cstring -include cstdlib \
                  -include cstdio -include climits -include limits -include algorithm \
                  -include utility -include functional -include stdexcept

CXXFLAGS     := -std=c++17 $(COMMON_FLAGS) $(FORCE_INCLUDES)
CFLAGS       := -std=c11   $(COMMON_FLAGS)

INCLUDES := -Iinclude

# ---- sources ---------------------------------------------------------------
# All C++ under src/ (recursive). The real JSON parser (src/json.cpp) replaces
# the former third_party/cJSON placeholder (Phase R), so there is no C source.
# `-not -name '._*'` skips macOS AppleDouble junk files: a tarball built on macOS
# can embed `._foo.cpp` (xattr/resource-fork sidecars) that GNU tar extracts as
# real files; they are binary and must never be compiled.
CXX_SRCS := $(shell find src -name '*.cpp' -not -name 'main.cpp' -not -name '._*' 2>/dev/null)

CXX_OBJS := $(CXX_SRCS:.cpp=.o)
LIB_OBJS := $(CXX_OBJS)

# ---- artifacts -------------------------------------------------------------
LIBARC       := libarc.a
ARC_SOLVER   := arc_solver

# ---- top-level -------------------------------------------------------------
# Default target: static lib + the standalone solver.
.PHONY: all clean
all: $(LIBARC) $(ARC_SOLVER)

# ---- compile rules ---------------------------------------------------------
# -fPIC is kept verbatim from the source tree this was extracted from: it
# changes codegen, and the extraction is verified by byte-diffing this binary's
# submission.json against that tree's. Do not drop it without re-verifying.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -fPIC -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -fPIC -c $< -o $@

# ---- link ------------------------------------------------------------------
$(LIBARC): $(LIB_OBJS)
	$(AR) rcs $@ $(LIB_OBJS)

# arc_solver: the standalone deployment binary. Links main.cpp against the
# static lib. -pthread for the task-level std::thread worker pool.
$(ARC_SOLVER): src/runtime/main.cpp $(LIBARC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -pthread src/runtime/main.cpp $(LIBARC) -o $@

# ---- clean -----------------------------------------------------------------
clean:
	rm -f $(LIB_OBJS) $(LIB_OBJS:.o=.d) $(LIBARC) $(ARC_SOLVER)

# Header-dependency tracking: -MMD -MP emit .d files so editing a header
# rebuilds every object that includes it (prevents stale-object failures).
-include $(LIB_OBJS:.o=.d)
