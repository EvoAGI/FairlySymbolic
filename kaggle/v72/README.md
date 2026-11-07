# Kaggle submission v72 (ARC Prize 2026)

`arc-2026-graph-solver-submission.ipynb` is version 72 of our Kaggle notebook,
submitted on 2 November 2026. It scored 2.50 on the semi-private evaluation set,
which ranks the entries during the competition. It was our last submission and
ties our best score. It is one of the two submissions selected as final, the
other being v50.

## Code

The notebook builds two solver binaries from the Kaggle dataset
`lucafavole/arc-graph-solver-code-combo`:

| Folder in the dataset | Source | Members |
|---|---|---|
| `csolver/` | this repository | 47, 41 |
| `csolver_v52/` | an earlier version of the solver (6 September 2026), included in the dataset | 777, 2718, 333 |

Each folder also holds a `library/` folder of mined macros, which v72 does not
use because no member sets `ARC_MACRO_SEED_LIBRARY`. Before building, the
notebook appends standard-header `-include` flags to `COMMON_FLAGS` in each
Makefile, because Kaggle's g++ does not pull in those headers indirectly.

## Members

The members run one after the other. The 10.5-hour budget is split evenly among
them, with a per-task cap of budget / 5 members / rounds of tasks x 0.95.
Every member runs with `--generations 10000000 --max-rules 5 --occam-continue 1 --view-continue 1`.

| Seed | Binary | Population | Restarts |
|---|---|---|---|
| 777 | `csolver_v52/` | 64 | 10 |
| 47 | `csolver/` | 40 | 5 |
| 41 | `csolver/` | 40 | 5 |
| 2718 | `csolver_v52/` | 640 | 5 |
| 333 | `csolver_v52/` | 160 | 5 |

The `csolver_v52/` members run with their built-in defaults. The `csolver/`
members read `arc_config.conf` and set:

```
ARC_CONCEPT_SEEDS=0 ARC_GENERALISE=1 ARC_INFORMED_ATTR=1 ARC_IMMIGRANTS=4
ARC_NO_GUARDS=1 ARC_ENSEMBLE=8 ARC_VOTE_TYPES=1 ARC_TASK_OTHER=1
ARC_MDL_PRIOR=1 ARC_REENC_CACHE=1
```

## Merge

For each test input:

- attempt 1 is attempt 1 of member 777;
- attempt 2 is the attempt 1 given by the most of members 47, 41, 2718 and
  333, among those that differ from attempt 1, with ties going to the member
  listed first;
- if no member differs, attempt 2 is member 777's own attempt 2, and failing
  that, attempt 1 again.

## Rerunning

Attach the dataset `lucafavole/arc-graph-solver-code-combo` and the competition data
to the notebook. On the public test file the notebook only runs a two-task
smoke test. The full search runs only in Kaggle's scoring rerun.
