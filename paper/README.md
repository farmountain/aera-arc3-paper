# Paper: Explore Before You Solve

LaTeX source + figures for the arxiv preprint.

## Files

| File | Purpose |
|---|---|
| `main.tex` | arxiv-ready LaTeX source (article.cls, 11pt) |
| `figures/gen_figures.py` | Regenerate Figure 1 + Figure 3 from EXP data |
| `figures/figure1_pareto.pdf` | [Speed, Depth] Pareto frontier with budget points (EXP-003) |
| `figures/figure3_entropy.pdf` | Per-step belief entropy per game (EXP-002) |

The full prose draft (markdown) is in `../paper_draft.md`.

## Build

### Option 1 — Overleaf

1. New project → Upload `main.tex` and `figures/` directory.
2. Compile with pdfLaTeX.

### Option 2 — Local

```bash
# Requires TeX Live or MiKTeX.
cd paper
pdflatex main.tex
pdflatex main.tex   # 2nd pass for refs
```

### Option 3 — arxiv submission

Arxiv compiles `.tex` server-side. Upload:
- `main.tex`
- `figures/figure1_pareto.pdf`
- `figures/figure3_entropy.pdf`

Set primary category: `cs.AI`. Cross-list: `cs.LG`.

## Regenerate Figures

```bash
python paper/figures/gen_figures.py
```

Requires `matplotlib` and `numpy`.

## License

CC0. No restrictions on reproduction or use.
