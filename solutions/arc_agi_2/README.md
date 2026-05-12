# ARC-AGI-2 DSL Solver

Part of the **Explore Before You Solve** open-source release.
Competition: [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
License: CC0

## What it does

Solves ARC-AGI-2 grid-transformation tasks via program synthesis over a DSL of
grid primitives. For each task it:

1. Enumerates candidate programs (depth-1 and depth-2 compositions)
2. Verifies each candidate against all training input/output pairs
3. Returns the top-2 verified programs applied to the test input

## Quick start

```bash
pip install -r requirements.txt

# Show all options
python -m solutions.arc_agi_2.evaluate --help

# DSL-only baseline on evaluation set (no LLM needed)
python -m solutions.arc_agi_2.evaluate \
    --challenges data/arc-agi-2/arc-agi_evaluation_challenges.json \
    --solutions  data/arc-agi-2/arc-agi_evaluation_solutions.json \
    --out        submission.json \
    --time-budget 30.0

# Smoke test on 10 training tasks
python -m solutions.arc_agi_2.evaluate \
    --challenges data/arc-agi-2/arc-agi_training_challenges.json \
    --solutions  data/arc-agi-2/arc-agi_training_solutions.json \
    --out        smoke.json \
    --time-budget 10.0 \
    --max-tasks  10
```

## DSL primitives (29 total)

| Category | Primitives |
|----------|-----------|
| Geometric | rotate_90/180/270, flip_h/v, transpose, anti_transpose |
| Spatial | tile_2x1/1x2/2x2, crop_bbox, pad_one, shift_*, mirror_* |
| Structural | fill_holes, gravity_up/down/left/right, outline, keep_largest_object, keep_smallest_object |
| Color | invert_binary, invert_all, binarize, swap_A_B, recolor_A_to_B |

## Files

| File | Purpose |
|------|---------|
| `dsl.py` | All grid primitives |
| `solver.py` | Search loop with time budget |
| `llm_proposer.py` | `OllamaProposer`, `ClaudeProposer` (paper experiments only) |
| `evaluate.py` | CLI runner — writes `submission.json` |
| `data_loader.py` | Load challenge/solution JSON, score submission |

## Metric

`score = mean(attempt_1 == truth OR attempt_2 == truth)` over all test inputs.
Exact grid match required. 2 attempts per test input.
