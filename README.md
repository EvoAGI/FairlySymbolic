# Fairly Symbolic

An evolutionary symbolic approach for ARC-AGI

For each task, the solver evolves three things together: a *parser* configuration that turns a grid into a typed multigraph of objects and relations, a short *program* of rewrite rules over that graph, and a *projector* that draws the result back as a grid. Candidates are scored on the task's train pairs only. The solver is written in C++17, has no third-party dependencies and no neural component.

```
input grid --parser--> graph --program--> graph --projector--> predicted grid
```

In the code the parser is called the *encoder* and the projector the *decoder*.

## Authors

- Luca Favole [&lt;lucafavole0@gmail.com&gt;](mailto:lucafavole0@gmail.com)
- Giorgio Mongardi [&lt;giorgio.mongardi@polito.it&gt;](mailto:giorgio.mongardi@polito.it)
- Giovanni Squillero [&lt;giovanni.squillero@polito.it&gt;](mailto:giovanni.squillero@polito.it)
- Alberto Tonda [&lt;alberto.tonda@inrae.fr&gt;](mailto:alberto.tonda@inrae.fr)

## Build

```sh
make -j8            # builds libarc.a and the arc_solver binary
```

Any C++17 compiler on Linux or macOS works. Windows is not supported (the runtime uses POSIX calls); use WSL there.

Keep the flags in the `Makefile`. `-ffp-contract=off` stops fused multiply-add from changing floating-point results, and `-fno-builtin-pow` stops the compiler from replacing `pow(x, 0.5)` with a square root, so results stay the same across machines.

## Run

```sh
./arc_solver --data-dir path/to/tasks --out submission.json
```

`--data-dir` takes a flat folder of ARC task files (`<task_id>.json`). `--kaggle DIR --split training|evaluation|test` reads the layout of the ARC Prize dataset on Kaggle instead. The output follows the ARC Prize format: for every task, one `{attempt_1, attempt_2}` pair per test input.

```sh
./arc_solver --help                                      # every option
./arc_solver --data-dir DIR --progress-dir runs/myrun    # per-task results and programs, written as the run goes
```

`arc_config.conf` is read from the working directory (or from `--config FILE`). Settings are applied in this order, later ones winning: built-in defaults, the config file, `ARC_*` environment variables, command-line flags.

## Determinism

By default every task runs a fixed number of generations (60 per restart, 5 restarts, population 40). Each task seeds its random generator from its own identifier, so the output depends only on the seed and the configuration: `--n-workers 1` and `--n-workers 8` give the same `submission.json`.

A time limit is applied only when `--max-task-seconds` is given on the command line. The search then stops on the clock, and the result also depends on the speed and load of the machine.

## Reproducing the paper

The paper's runs used the configuration in this repository, the flags below, and a time limit per task (840 or 3,840 seconds; seeds 47, 41 and 42):

```sh
export ARC_CONCEPT_SEEDS=0 ARC_GENERALISE=1 ARC_INFORMED_ATTR=1 ARC_IMMIGRANTS=4 \
       ARC_NO_GUARDS=1 ARC_VOTE_TYPES=1 ARC_TASK_OTHER=1 ARC_MDL_PRIOR=1 \
       ARC_REENC_CACHE=1 ARC_VOTE_FIT=2 ARC_FITLOG=1
./arc_solver --config arc_config.conf --seed 47 \
    --data-dir path/to/ARC-AGI-1/evaluation \
    --max-task-seconds 840 --generations 100000000 --n-workers 17 \
    --out submission.json --progress-dir runs/g1eval-840-s47 --full-results
```

For ARC-AGI-2, replace `--data-dir` with `--kaggle path/to/arc-prize-2026-arc-agi-2 --split training` (or `evaluation`). They ran 17 tasks at a time on a 36-thread Intel Xeon Gold 6140. Because the runs are bounded by time, a rerun on another machine gives close but not identical numbers. The benchmark data is not included.

## Kaggle submission

`kaggle/v72/` holds the notebook of our ARC Prize 2026 entry and a description of how it was built and how its members are combined.

## Layout

```
include/arc/   public headers; every source includes "arc/<name>.hpp"
src/
  util/        random numbers, hashing, JSON, number formatting
  graph/       grids and graph containers
  encode/      the parser, one file per step
  dsl/         the program language: selectors, actions, executor, projector
  fitness/     object and pixel distances, lexicase cases
  search/      genetic search, genome encoding, restarts, seed programs
  runtime/     configuration, task input/output, main
```

## License

MIT No Attribution (MIT-0); see `LICENSE`.
