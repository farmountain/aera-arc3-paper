# Explore Before You Solve

Code for: **"Explore Before You Solve: The Speed--Depth Trade-off in Epistemic Agents for ARC-AGI-3"**

ARC Prize 2026 Paper Track submission. arXiv: [pending].

## Central Finding

Systematic investigation reveals that **all 25 public ARC-AGI-3 games are reachable through non-intelligent strategies**: 10 in a single blind step, 5 after probing, 8 via repeated actions with sufficient budget, and 18 via a null-coordinate vulnerability. The public evaluation set cannot discriminate intelligent exploration from trivial heuristics — the private 55-game set is the only genuine intelligence test.

## Key Results

| Scope | System | RHAE | Solved |
|-------|--------|------|--------|
| 25-game (public) | AERA b=1 (Qwen2.5-0.5B) | **0.2116** | 4/25 (VC33, FT09, LP85, S5I5) |
| 25-game (public) | No-explore baseline | 0.0000 | 0/25 |
| 25-game (public) | Random baseline | 0.0000 | 0/25 |
| 25-game (public) | Systematic exhaustive | **all solved** | 25/25 (non-intelligent) |
| 55-game (private) | Linked code track entry | **0.30** | competition scale |

## Contributions

1. **Benchmark validity analysis** — demonstrates that current interactive reasoning benchmarks fail to measure the exploration capability they claim to require
2. **EXPLORE-before-PLAN framework** — AERA (Adaptive Epistemic Reasoning Agent) with Speed--Depth trade-off formalization
3. **Complete 25-game taxonomy** — per-game breakdown of solving strategies, failure modes, and action-selection biases

## Quick Start

```bash
pip install -r requirements.txt

# EXP-001: no-explore baseline (25 public games)
python run_eval.py --model ~/models/qwen2.5-7b.gguf --all --no-explore

# EXP-002: AERA (25 public games)
python run_eval.py --model ~/models/qwen2.5-7b.gguf --all
```

See `REPRODUCIBILITY.md` for full instructions.

## Paper

Canonical LaTeX source: [`paper/main.tex`](paper/main.tex)
Compiled PDF: [`paper/paper_final.pdf`](paper/paper_final.pdf)

## License

CC0 — public domain. No restrictions on use, reproduction, or modification.
