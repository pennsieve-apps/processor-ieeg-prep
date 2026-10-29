"""
QC figure per recording: a 10 s snapshot of the output traces (bad channels in red, named)
and a strip with the status of every source channel (good / bad / dropped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
STATUS = {"good": "#0ca30c", "bad": "#d03b3b", "dropped": "#a3a29c"}
MAX_TRACES = 48


def qc_figure(p: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, status, snip = p["names"], p["status"], p["snippet"]
    n = len(names)
    bad = [i for i in range(n) if status[i] == "bad"]
    good = [i for i in range(n) if status[i] != "bad"]
    room = max(MAX_TRACES - min(len(bad), MAX_TRACES // 2), 0)
    take = list(np.unique(np.linspace(0, len(good) - 1, min(room, len(good))).astype(int))) if good else []
    show = sorted([good[k] for k in take] + bad[: MAX_TRACES // 2])
    fig = plt.figure(figsize=(11, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1.3], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    t = p["snippet_start_s"] + np.arange(snip.shape[1]) / p["sfreq"]
    if show:
        x = snip[show].astype(np.float64)
        x = x - np.median(x, axis=1, keepdims=True)
        scale = np.median(np.percentile(np.abs(x), 99, axis=1)) or 1.0
        for row, i in enumerate(show):
            y = -row + 0.45 * x[row] / scale
            ax.plot(t, np.clip(y, -row - 1.2, -row + 1.2), lw=0.5,
                    color=STATUS["bad"] if status[i] == "bad" else INK, rasterized=True)
        ax.set_yticks([-r for r in range(len(show))])
        ax.set_yticklabels([names[i] for i in show], fontsize=6)
        for lab, i in zip(ax.get_yticklabels(), show):
            lab.set_color(STATUS["bad"] if status[i] == "bad" else INK2)
        ax.set_ylim(-len(show) + 0.2, 0.9)
        ax.text(1.0, 1.01, f"scale: ±{0.45 * scale:.0f} µV per row half-height", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK2)
    if p.get("onset_s") is not None and t[0] <= p["onset_s"] <= t[-1]:
        ax.axvline(p["onset_s"], color=INK2, lw=1, ls="--")
        ax.text(p["onset_s"], 0.6, " seizure onset", fontsize=7, color=INK2, va="center",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time from recording start (s)", fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=INK2, labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    nb = len(bad)
    nbc = sum(1 for _, st in p["contact_status"] if st == "bad")
    ax.set_title(f"{p['recording_id']}\n{p['montage']} montage, {n} channels ({n - nb} good, {nb} bad; "
                 f"{len(show)} shown, bad in red); {nbc} bad source contact(s); {p['kind']}", fontsize=9, color=INK,
                 loc="left")

    cs = p["contact_status"]
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(SURFACE)
    for k, (_, st) in enumerate(cs):
        ax2.add_patch(matplotlib.patches.Rectangle((k + 0.08, 0), 0.84, 1, color=STATUS.get(st, MUTED), lw=0))
    ax2.set_xlim(0, max(len(cs), 1))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    marked = [k for k, (_, st) in enumerate(cs) if st != "good"]
    if len(cs) <= 64:
        ax2.set_xticks(np.arange(len(cs)) + 0.5)
        ax2.set_xticklabels([c for c, _ in cs], rotation=90, fontsize=5, color=INK2)
    else:
        ax2.set_xticks([k + 0.5 for k in marked[:60]])
        ax2.set_xticklabels([cs[k][0] for k in marked[:60]], rotation=90, fontsize=5, color=INK2)
    ax2.tick_params(length=0)
    counts = {s: sum(1 for _, st in cs if st == s) for s in ("good", "bad", "dropped")}
    handles = [matplotlib.patches.Patch(color=STATUS[s], label=f"{s} {counts[s]}") for s in counts]
    ax2.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=7,
               labelcolor=INK, handlelength=1, title="source channels (names shown for those not good)",
               title_fontsize=7, alignment="left")
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
