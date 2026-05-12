# Explore Before You Solve

Code for: **"Explore Before You Solve: The [Speed, Depth] Commutator and RHAE-Optimal Epistemic Agents for ARC-AGI-3"**

ARC Prize 2026 Paper Track submission.

## Key Results

| Condition | RHAE | Games Solved |
|-----------|------|-------------|
| B1: No-explore (0.5B) | 0.0000 | 0/5 |
| AERA adaptive (0.5B) | 0.2645 | 1/5 |
| AERA budget=1 (0.5B) | **0.5290** | 2/5 |
| B1: No-explore (1.5B) | 0.2645 | 1/5 |
| AERA adaptive (1.5B) | 0.2645 | 1/5 |
| AERA budget=1 (1.5B) | 0.0000 | 0/5 |

## Quick Start

```bash
pip install -r requirements.txt
# or: docker build -t aera-arc3 .

# EXP-001: no-explore baseline
python run_eval.py --model ~/models/qwen2.5-7b.gguf --games sb26 ft09 cd82 tu93 r11l --no-explore

# EXP-002: AERA (primary claim)
python run_eval.py --model ~/models/qwen2.5-7b.gguf --games sb26 ft09 cd82 tu93 r11l
```

See REPRODUCIBILITY.md for full instructions including Kaggle notebook.

## Theoretical Contribution

RHAE = min(human_actions/AI_actions, 1.15)^2 is shown to have the structure of a [Speed, Depth] commutator penalty — measuring the cost of deviating from the Pareto-optimal exploration-execution frontier.

## License

CC0 — public domain. No restrictions on use, reproduction, or modification.
