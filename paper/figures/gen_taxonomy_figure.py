"""Generate taxonomy figure: 25-game grid colored by solving strategy."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# All 25 public ARC-AGI-3 games with their taxonomy categories
games = [
    # (short_name, category, color)
    ("FT09", "Blind ACTION6", "#d62728"),
    ("CN04", "Blind ACTION6", "#d62728"),
    ("M0R0", "Blind ACTION6", "#d62728"),
    ("LF52", "Blind ACTION6", "#d62728"),
    ("BP35", "Blind ACTION6", "#d62728"),
    ("R11L", "Blind other", "#ff7f0e"),
    ("VC33", "Blind other", "#ff7f0e"),
    ("LP85", "Blind other", "#ff7f0e"),
    ("TN36", "Blind other", "#ff7f0e"),
    ("S5I5", "Blind other", "#ff7f0e"),
    ("SB26", "ACTION6 after probe", "#2ca02c"),
    ("CD82", "ACTION6 after probe", "#2ca02c"),
    ("AR25", "ACTION6 after probe", "#2ca02c"),
    ("SK48", "ACTION6 after probe", "#2ca02c"),
    ("DC22", "ACTION6 after probe", "#2ca02c"),
    ("SP80", "Repeated ACTION1", "#9467bd"),
    ("SU15", "Diverse explore", "#8c564b"),
    ("TU93", "Budget-constrained", "#1f77b4"),
    ("RE86", "Budget-constrained", "#1f77b4"),
    ("TR87", "Budget-constrained", "#1f77b4"),
    ("KA59", "Budget-constrained", "#1f77b4"),
    ("LS20", "Budget-constrained", "#1f77b4"),
    ("SC25", "Budget-constrained", "#1f77b4"),
    ("G50T", "Budget-constrained", "#1f77b4"),
    ("WA30", "Budget-constrained", "#1f77b4"),
]

# Mark which games AERA solves (b=1, Qwen2.5-0.5B)
aera_solved = {"FT09", "VC33", "LP85", "S5I5"}

fig, ax = plt.subplots(figsize=(12, 3.5))

cols = 13
rows = 2
for i, (name, cat, color) in enumerate(games):
    r = i // cols
    c = i % cols
    rect = mpatches.FancyBboxPatch(
        (c + 0.05, rows - 1 - r + 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=1.5,
        alpha=0.85
    )
    ax.add_patch(rect)
    # Bold if solved by AERA
    weight = "bold" if name in aera_solved else "normal"
    ax.text(c + 0.5, rows - 1 - r + 0.5, name, ha="center", va="center",
            fontsize=8 if name in aera_solved else 7.5,
            fontweight=weight, color="white" if color != "#ff7f0e" else "black")

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect("equal")
ax.axis("off")

# Legend
legend_patches = [
    mpatches.Patch(color="#d62728", label="Blind ACTION6 (1 step, 5 games)"),
    mpatches.Patch(color="#ff7f0e", label="Blind other (1 step, 5 games)"),
    mpatches.Patch(color="#2ca02c", label="ACTION6 after probe (5 games)"),
    mpatches.Patch(color="#9467bd", label="Repeated ACTION1 (1 game)"),
    mpatches.Patch(color="#8c564b", label="Diverse explore (1 game)"),
    mpatches.Patch(color="#1f77b4", label="Budget-constrained (8 games)"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=7.5)

ax.set_title(
    "Figure 1: Complete taxonomy of all 25 public ARC-AGI-3 games by solving strategy.\n"
    "Bold labels = solved by AERA (b=1, Qwen2.5-0.5B). All 25 games reachable without intelligent exploration.",
    fontsize=10, fontweight="bold", pad=12
)

plt.tight_layout()
plt.savefig("figure1_taxonomy.pdf", dpi=150, bbox_inches="tight")
plt.savefig("figure1_taxonomy.png", dpi=150, bbox_inches="tight")
print("Saved figure1_taxonomy.pdf and .png")
