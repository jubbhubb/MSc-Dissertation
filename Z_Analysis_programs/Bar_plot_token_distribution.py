"""
Generate bar charts from API token-usage table.

Chart 1: Total tokens per pathway (all 7 methods).
Chart 2: Horizontal stacked bar chart, one bar per pathway, each
         segmented into Coding / Subtheme / Report.
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Data transcribed from the LaTeX table
# Each stage cell is "input / output"
# ---------------------------------------------------------------
data = {
    "C \u2192 C": {"Coding": (138067, 4581),  "Subtheme": (4688, 4414),   "Report": (5492, 2601)},
    "C \u2192 S": {"Coding": (138067, 4581),  "Subtheme": (6032, 5025),   "Report": (5481, 2488)},
    "D \u2192 C": {"Coding": (148346, 42214), "Subtheme": (38871, 15913), "Report": (23874, 2802)},
    "D \u2192 D": {"Coding": (148346, 42214), "Subtheme": (43239, 42034), "Report": (49170, 2678)},
    "D \u2192 S": {"Coding": (148346, 42214), "Subtheme": (54999, 44586), "Report": (49276, 2624)},
    "S \u2192 D": {"Coding": (587704, 254872),"Subtheme": (167233, 142892),"Report": (174030, 2478)},
    "S \u2192 S": {"Coding": (587704, 254872),"Subtheme": (346321, 205547),"Report": (201312, 2367)},
}

# Total tokens per pathway (input + output summed across all stages)
totals = {
    pathway: sum(i + o for i, o in stages.values())
    for pathway, stages in data.items()
}

# ---------------------------------------------------------------
# Chart 1: Total tokens by pathway
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

pathways = list(totals.keys())
values = list(totals.values())

colors = ["#4C72B0" if p != "S \u2192 S" else "#C44E52" for p in pathways]

bars = ax.bar(pathways, values, color=colors)
ax.set_ylabel("Total tokens")
ax.set_title("Total token usage by pathway")
ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")

for bar, val in zip(bars, values):
    ax.annotate(f"{val:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("total_tokens_by_pathway.png", dpi=200)
plt.close()

# ---------------------------------------------------------------
# Chart 2: Horizontal stacked bar chart, one bar per pathway,
# each segmented into Coding / Subtheme / Report (input+output combined)
# ---------------------------------------------------------------
stage_names = ["Coding", "Subtheme", "Report"]
stage_colors = {"Coding": "#4C72B0", "Subtheme": "#DD8452", "Report": "#55A868"}

ordered_pathways = sorted(data.keys(), key=lambda p: totals[p])

stage_totals = {
    p: [data[p][s][0] + data[p][s][1] for s in stage_names]
    for p in ordered_pathways
}

fig, ax = plt.subplots(figsize=(10, 6))

left = [0] * len(ordered_pathways)
for i, stage in enumerate(stage_names):
    values = [stage_totals[p][i] for p in ordered_pathways]
    bars = ax.barh(ordered_pathways, values, left=left, label=stage, color=stage_colors[stage])
    for bar, val in zip(bars, values):
        if val > 30000:
            ax.annotate(f"{val:,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2),
                        ha="center", va="center", fontsize=8, color="white")
    left = [l + v for l, v in zip(left, values)]

for p, total_left in zip(ordered_pathways, left):
    ax.annotate(f"{total_left:,}", (total_left, p), textcoords="offset points",
                xytext=(6, 0), va="center", fontsize=9)

ax.set_xlabel("Total tokens")
ax.set_title("Token usage by pathway, split by processing stage")
ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
ax.legend(title="Stage", loc="lower right")

plt.tight_layout()
plt.savefig("pathway_breakdown_horizontal.png", dpi=200)
plt.close()

print("Totals:")
for p, v in totals.items():
    print(f"  {p}: {v:,}")