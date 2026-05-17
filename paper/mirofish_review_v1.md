# Mirofish Persona-Review #1 — 2026-05-18

Simulation reviewing `paper/main.tex` at v0.2 for arxiv cs.AI endorsement
quality. 5 simulated personas. Synthetic critique only — not real
peer review. Recorded for transparency.

## Value Function

```
V = 0.25 * rigor
  + 0.20 * novelty
  + 0.20 * honesty
  + 0.15 * human_fingerprint
  + 0.20 * (1 - red_flags)
```

Target for "endorser-defensible": V >= 0.85.

## Pre-Review Scores (v0.2)

| Component | Score | Notes |
|---|---|---|
| rigor              | 0.55 | Theorem 1 unproven; Lemma 1 evidence-claim mismatch; Corollary 1 over-stamped |
| novelty            | 0.65 | Commutator framing novel as analogy; AERA architecture conventional |
| honesty            | 0.80 | §8.2 Limitations thorough; FT09 mechanism acknowledged |
| human_fingerprint  | 0.20 | Uniform tone; no lived-experience signal |
| (1 - red_flags)    | 0.45 | Hallucinated cites (Bonnet, SOAR, CompressARC authors); em-dash density high; "(running)" placeholders |
| **V_pre**          | **0.547** | |

## Critiques by Persona (top issues)

| ID | Persona | Critique | Severity |
|---|---|---|---|
| C1.1 | Chollet-class | Theorem 1 convexity unproven | showstop |
| C2.2 | ARC team | "Human = Pareto-optimal" not in ARC source | showstop |
| C3.1 | Lake-class | Lemma 1 evidence does not establish Pareto-optimality | showstop |
| C4.1 | AI-skeptic | High LLM-prior alone may kill endorsement | showstop |
| C1.2 | Chollet-class | Corollary 1 over-stamped from n=1 environment | major |
| C1.3 | Chollet-class | FT09 win = bug-triggered, not exploration success | major |
| C1.5 | Chollet-class | 5/25 public games = unmitigated sampling bias | major |
| C2.1 | ARC team | RHAE cap not modeled in theorem | major |
| C3.2 | Lake-class | Theory-implementation gap obscured in abstract | major |
| C3.3 | Lake-class | "Commutator" word usage without commutator math | major |
| C5.1 | Industry vet | No random baseline | major |
| C5.2 | Industry vet | No projected RHAE for full 55-game eval | major |
| C5.3 | Industry vet | No compute cost / per-task budget analysis | major |

## Actions Applied (v0.2 -> v0.3)

| ID | Action | Status |
|---|---|---|
| A1 | Theorem 1 -> Proposition under explicit (A1)/(A2) assumptions | done |
| A2 | Lemma 1 -> Conjecture; evidence reframed | done |
| A3 | Corollary 1 -> "empirical heuristic from n=1 environment" | done |
| A4 | Notes-on-this-work paragraph reserved for human author (no synthetic anecdote) | done |
| A5 | Drop Kahneman pop-citation; replace System-1/2 language with technical wording | done |
|    | Fix hallucinated cites (Bonnet, SOAR-Akyurek, CompressARC-Liu, prize-placement details) | done (pre-mirofish, in commit b4a159e/c1dca74 -> c1dca74+1) |

## Actions Deferred (need other sessions or future iterations)

| ID | Action | Why deferred |
|---|---|---|
| A6 | Run all 25 public ARC-AGI-3 games | GPU compute (other session) |
| A7 | Random-action baseline | GPU compute |
| A8 | Compute-cost analysis per game | needs A6/A7 numbers |
| A9 | Em-dash density scrub pass | optional polish; small ΔV |
| A10 | Human author writes "Notes on this work" with real anecdote | requires human |

## Post-Review Scores (v0.3, estimated)

| Component | Score | Δ | Reason |
|---|---|---|---|
| rigor              | 0.78 | +0.23 | A1/A2/A3 demotions resolve C1.1, C2.2, C3.1 showstops |
| novelty            | 0.65 | 0     | unchanged |
| honesty            | 0.85 | +0.05 | A4 placeholder + A5 tone fix raise honesty signal |
| human_fingerprint  | 0.30 | +0.10 | FT09 provenance note + placeholder framing; not full credit until A10 |
| (1 - red_flags)    | 0.65 | +0.20 | Hallucinated cites fixed; theorem/lemma stamps removed; Kahneman dropped |
| **V_post**         | **0.687** | +0.140 | |

## Honest Assessment

V_post ~= 0.69. Target was 0.85. **Gap of 0.16 remains.** Closing it
requires:

1. Human author writes real "Notes on this work" (A10) -> hf 0.30 -> 0.55
2. Full 25-game run + random baseline (A6/A7) -> rigor 0.78 -> 0.88, red_flags 0.65 -> 0.80
3. Compute cost section (A8) -> rigor +0.02

Projected V after A6+A7+A8+A10: ~= 0.83. Close to target but not over.

**Conclusion:** Paper is now defensible against the worst landmines
(unproven theorem stamps, hallucinated citations, "human = optimal"
overclaim). It is NOT yet at top-tier endorser-defensible quality. The
remaining gap is empirical scope, not paper quality - it requires
running the full evaluation set.

## Recommendation

Submit to arxiv as v0.3 with explicit "small-n empirical study,
theoretical framework with stated assumptions" framing. Do not pitch as
"proves" or "demonstrates"; pitch as "proposes a framework and provides
preliminary 5-game evidence". Be honest about the empirical scope in
the abstract.
