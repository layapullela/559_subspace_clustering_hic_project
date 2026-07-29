"""
Hyperparameter sweep for tiled TV-SSC TAD detection
=====================================================

Sweeps over (window, step, boundary_tol, vote_frac) to find the configuration
that produces predicted TAD boundaries with the best insulation scores relative
to the Juicer/Arrowhead benchmark.

Metric
------
  delta_median = median(log2_I @ pred boundaries) − median(log2_I @ GT boundaries)
  → more negative means pred boundaries sit in *deeper* insulation valleys
    than the Arrowhead benchmark (i.e. your method beats Juicer).

Secondary metrics
-----------------
  pct_neg   – fraction of pred boundaries whose log2_I < 0 (stronger than
              chromosome median); higher is better
  n_pred    – number of predicted boundaries; should be comparable to GT count
  delta_p10 – 10th-percentile of pred log2_I minus GT 10th-percentile
              (checks tail quality, not just median)

Strategy
--------
  Tile ADMM (expensive) is run ONCE per (window, step) pair and the raw
  per-tile boundary candidates are cached.  The consensus step
  (boundary_tol, vote_frac) is then swept cheaply over the cached candidates.

Usage
-----
  python sweep_tad_hyperparams.py \\
      --hic   /path/to/sample.hic \\
      --gt    arrowhead_output/10000_blocks.bedpe \\
      --chrom 1 \\
      --start-bp 3000000 \\
      --out-csv  sweep_results.csv \\
      --out-png  sweep_results.png
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

# Re-use solver + clustering from existing module
from ssc_admm_nuc_total_var import ssc_admm_nuc_tv, cluster_from_C

# Re-use insulation utilities from compare_tad_boundaries
from compare_tad_boundaries import (
    load_chromosome_sparse,
    load_domains,
    domains_to_boundary_bins,
    insulation_track,
    log2_normalize,
)

# Re-use tiling helpers from full_chr_tv_experiment
from full_chr_tv_experiment import (
    normalize_window,
    labels_to_segments,
    window_boundaries,
    consensus_boundaries,
    boundaries_to_domains,
    write_bed,
    plot_results,
    tad_statistics,
    BINSIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Grid definition
# ─────────────────────────────────────────────────────────────────────────────

# Tile-level (expensive): (window_bins, step_bins)
WINDOW_STEP_GRID: list[tuple[int, int]] = [
    # (window, step) — step is roughly 25–50% of window for dense overlap
    (100,  25),
    (100,  50),
    (150,  38),
    (150,  75),
    (200,  50),
    (200, 100),
    (300,  75),
    (300, 150),
]

# Consensus-level (cheap): boundary_tol (bins) × vote_frac
BOUNDARY_TOL_GRID: list[int]   = [3, 5, 8, 10, 15]
VOTE_FRAC_GRID:    list[float] = [0.20, 0.30, 0.40, 0.50, 0.60]

# Fixed ADMM hyper-parameters (matching the user's best run so far)
GAMMA     = 1.0
LAMBDA_E  = 1.0
LAMBDA_Z  = 0.1
MU        = 1.0
SIGMA_ADM = 1.0
MAX_ITER  = 50
TOL_ADMM  = 1e-3
NORM      = "log1p"

# Fixed evaluation parameters
INSULATION_WINDOW_BP = 500_000   # 50 bins at 10 kbp
MIN_TAD_BINS         = 10        # same as default


# ─────────────────────────────────────────────────────────────────────────────
# Cache entry: raw tile results for a (window, step) pair
# ─────────────────────────────────────────────────────────────────────────────

class TileCache(NamedTuple):
    candidates: list[tuple[int, int]]   # (boundary_position, window_id)
    interiors:  list[tuple[int, int]]   # interior range of each window
    n_tiles:    int
    elapsed_s:  float


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: extract per-tile candidates (expensive)
# ─────────────────────────────────────────────────────────────────────────────

def extract_tile_candidates(
    M_sparse: csr_matrix,
    n_bins: int,
    start_bin: int,
    window: int,
    step: int,
    verbose: bool = True,
) -> TileCache:
    """Run ADMM on every tile and return raw boundary candidates."""
    edge_margin = max(1, window // 10)

    candidates: list[tuple[int, int]] = []
    interiors:  list[tuple[int, int]] = []

    starts = list(range(start_bin,
                        max(start_bin + 1, n_bins - window // 4),
                        step))
    t_wall = time.perf_counter()

    if verbose:
        print(
            f"\n  [window={window}, step={step}]  "
            f"{len(starts)} tiles to process ...",
            flush=True,
        )

    for idx, start in enumerate(starts):
        end    = min(start + window, n_bins)
        W_size = end - start
        if W_size < 10:
            continue

        Y_raw = M_sparse[start:end, start:end].toarray()
        Y     = normalize_window(Y_raw, NORM)

        _, svs1, _ = randomized_svd(Y, n_components=1, random_state=0)
        sigma1 = float(svs1[0])

        if sigma1 < 1e-10:
            continue

        X, _C, _E = ssc_admm_nuc_tv(
            Y,
            lambda_e=LAMBDA_E / sigma1,
            lambda_z=LAMBDA_Z / sigma1,
            gamma=GAMMA,
            mu=MU,
            sigma=SIGMA_ADM,
            max_iter=MAX_ITER,
            tol=TOL_ADMM,
        )

        pred   = cluster_from_C(X, k=None)
        bounds = window_boundaries(
            pred,
            win_start=start, win_end=end,
            region_start=start_bin, region_end=n_bins,
            edge_margin=edge_margin,
        )
        wid = len(interiors)
        candidates.extend((b, wid) for b in bounds)
        interiors.append((start + edge_margin, end - edge_margin))

    elapsed = time.perf_counter() - t_wall
    if verbose:
        print(
            f"  → {len(candidates)} raw candidates from {len(interiors)} tiles "
            f"in {elapsed:.1f} s",
            flush=True,
        )

    return TileCache(candidates, interiors, len(interiors), elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: evaluate one consensus configuration (cheap)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    cache: TileCache,
    boundary_tol: int,
    vote_frac: float,
    n_bins: int,
    start_bin: int,
    log2_track: np.ndarray,
    gt_bins: np.ndarray,
) -> dict:
    """Build TADs from cached tile data with the given consensus params and score."""
    boundaries = consensus_boundaries(
        cache.candidates,
        cache.interiors,
        boundary_tol,
        vote_frac,
    )
    tads = boundaries_to_domains(boundaries, start_bin, n_bins, MIN_TAD_BINS)
    stats = tad_statistics(tads)
    pred_bins = np.array(stats["boundaries"], dtype=int)

    # Insulation at predicted boundaries
    keep_pred = (pred_bins >= 0) & (pred_bins < len(log2_track))
    log2_pred = log2_track[pred_bins[keep_pred]]
    pred_fin  = log2_pred[np.isfinite(log2_pred)]

    # Insulation at GT boundaries
    keep_gt = (gt_bins >= 0) & (gt_bins < len(log2_track))
    log2_gt = log2_track[gt_bins[keep_gt]]
    gt_fin  = log2_gt[np.isfinite(log2_gt)]

    med_pred = float(np.median(pred_fin)) if pred_fin.size else np.nan
    med_gt   = float(np.median(gt_fin))   if gt_fin.size  else np.nan
    p10_pred = float(np.percentile(pred_fin, 10)) if pred_fin.size else np.nan
    p10_gt   = float(np.percentile(gt_fin,   10)) if gt_fin.size  else np.nan

    delta_median = (med_pred - med_gt) if (np.isfinite(med_pred) and np.isfinite(med_gt)) else np.nan
    delta_p10    = (p10_pred - p10_gt) if (np.isfinite(p10_pred) and np.isfinite(p10_gt)) else np.nan

    pct_neg = (float(np.sum(pred_fin < 0)) / pred_fin.size * 100.0) if pred_fin.size else 0.0

    return dict(
        boundary_tol=boundary_tol,
        vote_frac=vote_frac,
        n_tads=stats["n_tads"],
        n_pred_boundaries=len(pred_bins[keep_pred]),
        n_gt_boundaries=int(np.sum(keep_gt)),
        med_pred=med_pred,
        med_gt=med_gt,
        delta_median=delta_median,
        p10_pred=p10_pred,
        p10_gt=p10_gt,
        delta_p10=delta_p10,
        pct_neg_pred=pct_neg,
        median_tad_kbp=stats["median_kbp"],
        mean_tad_kbp=stats["mean_kbp"],
        tads=tads,           # stored for best-config re-use (not written to CSV)
        boundaries=boundaries,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _passes_constraints(
    r: dict,
    min_tads: int,
    min_median_kbp: float,
    max_median_kbp: float,
) -> bool:
    return (
        r["n_tads"] >= min_tads
        and min_median_kbp <= r["median_tad_kbp"] <= max_median_kbp
    )


def print_top_k(
    results: list[dict],
    k: int = 15,
    min_tads: int = 0,
    min_median_kbp: float = 0.0,
    max_median_kbp: float = float("inf"),
) -> None:
    finite = [r for r in results if np.isfinite(r["delta_median"])]

    # Unconstrained top-k
    top_all = sorted(finite, key=lambda r: r["delta_median"])[:k]

    # Constrained top-k
    constrained = [
        r for r in finite
        if _passes_constraints(r, min_tads, min_median_kbp, max_median_kbp)
    ]
    top_con = sorted(constrained, key=lambda r: r["delta_median"])[:k]

    header = (
        f"{'Rank':>4}  {'window':>6}  {'step':>5}  {'btol':>4}  {'vfrac':>5}  "
        f"{'Δmedian':>8}  {'Δp10':>7}  {'%neg':>5}  "
        f"{'n_tads':>6}  {'n_gt':>5}  {'med_TAD_kb':>10}"
    )
    sep = "─" * len(header)

    # ── print unconstrained ───────────────────────────────────────────────────
    print(f"\n{'═'*len(header)}")
    print("  TOP CONFIGURATIONS (unconstrained, lower Δmedian = pred beats GT)")
    print(header)
    print(sep)
    for rank, r in enumerate(top_all, 1):
        flag = ""
        if _passes_constraints(r, min_tads, min_median_kbp, max_median_kbp):
            flag = " ✓"
        print(
            f"{rank:>4}  {r['window']:>6}  {r['step']:>5}  {r['boundary_tol']:>4}  "
            f"{r['vote_frac']:>5.2f}  "
            f"{r['delta_median']:>+8.4f}  {r['delta_p10']:>+7.4f}  "
            f"{r['pct_neg_pred']:>4.1f}%  "
            f"{r['n_tads']:>6.0f}  {r['n_gt_boundaries']:>5}  "
            f"{r['median_tad_kbp']:>10.0f}{flag}"
        )
    print(f"{'═'*len(header)}\n")

    # ── print constrained ─────────────────────────────────────────────────────
    if min_tads > 0 or min_median_kbp > 0 or max_median_kbp < float("inf"):
        label = (
            f"n_tads ≥ {min_tads}  AND  "
            f"{min_median_kbp:.0f} ≤ median_TAD_kbp ≤ {max_median_kbp:.0f}"
        )
        print(f"{'═'*len(header)}")
        print(f"  TOP CONFIGURATIONS WITH CONSTRAINTS: {label}")
        print(f"  ({len(constrained)} configs qualify)")
        print(header)
        print(sep)
        if top_con:
            for rank, r in enumerate(top_con, 1):
                print(
                    f"{rank:>4}  {r['window']:>6}  {r['step']:>5}  {r['boundary_tol']:>4}  "
                    f"{r['vote_frac']:>5.2f}  "
                    f"{r['delta_median']:>+8.4f}  {r['delta_p10']:>+7.4f}  "
                    f"{r['pct_neg_pred']:>4.1f}%  "
                    f"{r['n_tads']:>6.0f}  {r['n_gt_boundaries']:>5}  "
                    f"{r['median_tad_kbp']:>10.0f}"
                )
        else:
            print("  (no configs meet the constraints)")
        print(f"{'═'*len(header)}\n")


def save_csv(results: list[dict], path: str) -> None:
    fields = [
        "window", "step", "boundary_tol", "vote_frac",
        "n_tads", "n_pred_boundaries", "n_gt_boundaries",
        "med_pred", "med_gt", "delta_median",
        "p10_pred", "p10_gt", "delta_p10",
        "pct_neg_pred", "median_tad_kbp", "mean_tad_kbp",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fields}
            w.writerow(row)
    print(f"Saved CSV → {path}  ({len(results)} rows)")


def save_plot(
    results: list[dict],
    path: str,
    min_tads: int = 0,
    min_median_kbp: float = 0.0,
    max_median_kbp: float = float("inf"),
) -> None:
    """
    Multi-panel summary:
      1. delta_median heat-map per (boundary_tol, vote_frac) averaged over all (window, step)
      2. delta_median vs window coloured by vote_frac (best step per window)
      3. Scatter: n_pred_boundaries vs delta_median
      4. Top-20 bar chart
    """
    finite = [r for r in results if np.isfinite(r["delta_median"])]
    if not finite:
        print("No finite results — skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    constraint_str = ""
    if min_tads > 0 or min_median_kbp > 0 or max_median_kbp < float("inf"):
        constraint_str = (
            f"\nConstraints: n_tads ≥ {min_tads},  "
            f"{min_median_kbp:.0f}–{max_median_kbp:.0f} kbp median TAD size  "
            f"(✓ markers)"
        )

    fig.suptitle(
        "TV-SSC Hyperparameter Sweep: Insulation Δmedian (pred − GT)\n"
        "More negative = predicted boundaries beat Arrowhead benchmark"
        + constraint_str,
        fontsize=12, fontweight="bold",
    )

    # ── 1. Heat-map: btol × vfrac, mean over (window, step) ─────────────────
    ax = axes[0, 0]
    btol_vals  = sorted(set(r["boundary_tol"] for r in finite))
    vfrac_vals = sorted(set(r["vote_frac"]    for r in finite))
    heat = np.full((len(btol_vals), len(vfrac_vals)), np.nan)
    for r in finite:
        i = btol_vals.index(r["boundary_tol"])
        j = vfrac_vals.index(r["vote_frac"])
        cur = heat[i, j]
        heat[i, j] = r["delta_median"] if np.isnan(cur) else min(cur, r["delta_median"])

    im = ax.imshow(heat, cmap="RdYlGn_r", aspect="auto",
                   vmin=np.nanmin(heat), vmax=np.nanmax(heat))
    ax.set_xticks(range(len(vfrac_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in vfrac_vals])
    ax.set_yticks(range(len(btol_vals)))
    ax.set_yticklabels([str(b) for b in btol_vals])
    ax.set_xlabel("vote_frac", fontsize=10)
    ax.set_ylabel("boundary_tol (bins)", fontsize=10)
    ax.set_title("Best Δmedian per (btol, vfrac)\nacross all (window, step)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate cells
    for i in range(len(btol_vals)):
        for j in range(len(vfrac_vals)):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i,j]:+.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    # ── 2. delta_median vs window, coloured by vote_frac ─────────────────────
    ax2 = axes[0, 1]
    window_vals = sorted(set(r["window"] for r in finite))
    cmap2 = plt.cm.plasma
    vfrac_unique = sorted(set(r["vote_frac"] for r in finite))
    norm2 = plt.Normalize(min(vfrac_unique), max(vfrac_unique))

    for vf in vfrac_unique:
        xs, ys = [], []
        for w in window_vals:
            subset = [r for r in finite if r["window"] == w and r["vote_frac"] == vf]
            if subset:
                best = min(subset, key=lambda r: r["delta_median"])
                xs.append(w)
                ys.append(best["delta_median"])
        ax2.plot(xs, ys, marker="o", color=cmap2(norm2(vf)),
                 label=f"vfrac={vf:.2f}", linewidth=1.5, markersize=5)

    ax2.axhline(0, color="grey", ls="--", lw=1, label="Δ = 0 (tie)")
    ax2.set_xlabel("window (bins)", fontsize=10)
    ax2.set_ylabel("Δ median log₂I (pred − GT)", fontsize=10)
    ax2.set_title("Best Δmedian vs window size\n(best step & btol per point)", fontsize=10)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── 3. Scatter: n_tads vs delta_median, mark constrained configs ──────────
    ax3 = axes[1, 0]
    # Split into constrained vs not
    con_pts  = [r for r in finite if _passes_constraints(r, min_tads, min_median_kbp, max_median_kbp)]
    uncon_pts = [r for r in finite if not _passes_constraints(r, min_tads, min_median_kbp, max_median_kbp)]

    if uncon_pts:
        sc3 = ax3.scatter(
            [r["n_tads"] for r in uncon_pts],
            [r["delta_median"] for r in uncon_pts],
            c=[r["window"] for r in uncon_pts], cmap="viridis",
            alpha=0.3, s=15, edgecolors="none",
        )
        plt.colorbar(sc3, ax=ax3, label="window (bins)", fraction=0.046, pad=0.04)
    if con_pts:
        ax3.scatter(
            [r["n_tads"] for r in con_pts],
            [r["delta_median"] for r in con_pts],
            c="tomato", marker="*", s=120, zorder=5,
            label=f"meets constraints ({len(con_pts)})",
        )

    ax3.axhline(0, color="grey", ls="--", lw=1)
    gt_n = finite[0]["n_gt_boundaries"] if finite else 0
    ax3.axvline(gt_n, color="steelblue", ls=":", lw=1.5,
                label=f"GT n_boundaries={gt_n}")
    if min_tads > 0:
        ax3.axvline(min_tads, color="red", ls="--", lw=1,
                    label=f"min_tads={min_tads}")
    ax3.set_xlabel("# TADs", fontsize=10)
    ax3.set_ylabel("Δ median log₂I", fontsize=10)
    ax3.set_title("TAD count vs insulation quality\n(★ = meets constraints)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. Top-20 bar chart (constrained if possible, else all) ───────────────
    ax4 = axes[1, 1]
    pool = con_pts if con_pts else finite
    top20 = sorted(pool, key=lambda r: r["delta_median"])[:20]
    title_suffix = " (constrained)" if con_pts else ""
    labels = [
        f"w={r['window']} s={r['step']}\nbt={r['boundary_tol']} vf={r['vote_frac']:.2f}"
        for r in top20
    ]
    vals = [r["delta_median"] for r in top20]
    colors = ["tomato" if v < 0 else "steelblue" for v in vals]
    ax4.barh(range(len(top20)), vals, color=colors, edgecolor="none")
    ax4.set_yticks(range(len(top20)))
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.axvline(0, color="black", lw=1)
    ax4.invert_yaxis()
    ax4.set_xlabel("Δ median log₂I (pred − GT)", fontsize=10)
    ax4.set_title(f"Top 20 configurations{title_suffix}\n(red = pred beats GT)", fontsize=10)
    ax4.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global GAMMA, LAMBDA_E, LAMBDA_Z, NORM, MAX_ITER  # may be overridden by CLI

    p = argparse.ArgumentParser(
        description="Hyperparameter sweep for TV-SSC TAD detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hic",       required=True,  help="Path to .hic file")
    p.add_argument("--gt",        required=True,  help="Arrowhead BEDPE or GT domain BED")
    p.add_argument("--chrom",     default="1",    help="Chromosome (e.g. 1 or chr1)")
    p.add_argument("--start-bp",  type=int, default=3_000_000,
                   help="Start bp to skip unmappable telomere region")
    p.add_argument("--insulation-window-bp", type=int, default=INSULATION_WINDOW_BP,
                   help="Insulation diamond half-width in bp")
    p.add_argument("--out-csv",   default="sweep_results.csv",
                   help="Output CSV with all configs and metrics")
    p.add_argument("--out-png",   default="sweep_results.png",
                   help="Output summary plot PNG")
    p.add_argument("--save-best-bed", default="best_tads.bed",
                   help="Write TAD BED for the single best configuration")
    p.add_argument("--save-best-png", default="best_tads.png",
                   help="Write TAD PNG for the single best configuration")
    p.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Override window sizes (bins). Default grid: "
            + str([ws[0] for ws in WINDOW_STEP_GRID])
        ),
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Override step sizes (bins) — must pair 1:1 with --windows if given.",
    )
    p.add_argument(
        "--btols",
        nargs="+",
        type=int,
        default=BOUNDARY_TOL_GRID,
        help="boundary_tol values to sweep",
    )
    p.add_argument(
        "--vfracs",
        nargs="+",
        type=float,
        default=VOTE_FRAC_GRID,
        help="vote_frac values to sweep",
    )
    p.add_argument("--gamma",     type=float, default=GAMMA)
    p.add_argument("--lambda-e",  type=float, default=LAMBDA_E)
    p.add_argument("--lambda-z",  type=float, default=LAMBDA_Z)
    p.add_argument("--norm",      default=NORM, choices=["none", "log1p", "zscore"])
    p.add_argument("--max-iter",  type=int, default=MAX_ITER)

    # Biological validity constraints for "best" config selection
    p.add_argument("--min-tads",        type=int,   default=0,
                   help="Minimum number of TADs required (default: no constraint)")
    p.add_argument("--min-median-kbp",  type=float, default=0.0,
                   help="Minimum median TAD size in kbp (default: no constraint)")
    p.add_argument("--max-median-kbp",  type=float, default=float("inf"),
                   help="Maximum median TAD size in kbp (default: no constraint)")
    args = p.parse_args()

    # Override module-level ADMM params from CLI
    GAMMA    = args.gamma
    LAMBDA_E = args.lambda_e
    LAMBDA_Z = args.lambda_z
    NORM     = args.norm
    MAX_ITER = args.max_iter

    # Build (window, step) grid
    if args.windows:
        if args.steps:
            if len(args.steps) != len(args.windows):
                raise SystemExit("--windows and --steps must have the same length")
            window_step_grid = list(zip(args.windows, args.steps))
        else:
            # default: 25% and 50% of each window
            window_step_grid = []
            for w in args.windows:
                window_step_grid.append((w, max(1, w // 4)))
                window_step_grid.append((w, max(1, w // 2)))
    else:
        window_step_grid = WINDOW_STEP_GRID

    chrom     = args.chrom.removeprefix("chr")
    start_bin = args.start_bp // BINSIZE

    # ── 1. Load Hi-C once ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading Hi-C: {args.hic}")
    M_sparse, n_bins = load_chromosome_sparse(
        args.hic, chrom, binsize=BINSIZE, normalization="KR"
    )

    # ── 2. Compute insulation track once ─────────────────────────────────────
    w_bins = max(1, args.insulation_window_bp // BINSIZE)
    print(
        f"\nComputing insulation track  "
        f"(window = {args.insulation_window_bp // 1000} kb = {w_bins} bins) ..."
    )
    raw_track  = insulation_track(M_sparse, w_bins)
    log2_track = log2_normalize(raw_track)
    chrom_fin  = log2_track[np.isfinite(log2_track)]
    print(
        f"  chrom: n={chrom_fin.size:,}  "
        f"median={np.median(chrom_fin):.4f}  mean={chrom_fin.mean():.4f}"
    )

    # ── 3. Load GT boundaries once ────────────────────────────────────────────
    print(f"\nLoading GT domains from {args.gt} ...")
    gt_domains = load_domains(args.gt, chrom=chrom)
    if not gt_domains:
        raise SystemExit(f"No domains found for chrom {chrom} in {args.gt}")
    gt_bins = domains_to_boundary_bins(gt_domains, binsize=BINSIZE)
    print(f"  {len(gt_domains)} domains → {len(gt_bins)} GT boundaries")

    # ── 4. Sweep ──────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    caches: dict[tuple[int, int], TileCache] = {}

    n_ws_combos   = len(window_step_grid)
    n_cons_combos = len(args.btols) * len(args.vfracs)
    total_configs = n_ws_combos * n_cons_combos

    print(f"\n{'='*60}")
    print(
        f"Sweep grid:  {n_ws_combos} (window, step) pairs  ×  "
        f"{n_cons_combos} (btol, vfrac) combos  =  {total_configs} total configs"
    )
    print(f"{'='*60}\n")

    t_sweep_start = time.perf_counter()

    for ws_idx, (window, step) in enumerate(window_step_grid):
        print(
            f"\n[{ws_idx+1}/{n_ws_combos}]  Extracting tile candidates "
            f"for window={window}, step={step} ...",
            flush=True,
        )
        cache = extract_tile_candidates(
            M_sparse, n_bins, start_bin, window, step, verbose=True
        )
        caches[(window, step)] = cache

        # Sweep cheap consensus params over cached candidates
        combo_results = []
        for boundary_tol, vote_frac in itertools.product(args.btols, args.vfracs):
            r = evaluate_config(
                cache, boundary_tol, vote_frac,
                n_bins, start_bin, log2_track, gt_bins,
            )
            r["window"] = window
            r["step"]   = step
            combo_results.append(r)

        # Print mini-table for this (window, step)
        finite_sub = [r for r in combo_results if np.isfinite(r["delta_median"])]
        if finite_sub:
            best_sub = min(finite_sub, key=lambda r: r["delta_median"])
            print(
                f"  Best for (w={window}, s={step}):  "
                f"btol={best_sub['boundary_tol']}  vfrac={best_sub['vote_frac']:.2f}  "
                f"Δmedian={best_sub['delta_median']:+.4f}  "
                f"n_pred={best_sub['n_pred_boundaries']}  "
                f"pct_neg={best_sub['pct_neg_pred']:.1f}%"
            )

        all_results.extend(combo_results)

    t_sweep = time.perf_counter() - t_sweep_start
    print(f"\nTotal sweep time: {t_sweep:.1f} s  ({t_sweep/60:.1f} min)")

    # ── 5. Report ──────────────────────────────────────────────────────────────
    print_top_k(
        all_results,
        k=20,
        min_tads=args.min_tads,
        min_median_kbp=args.min_median_kbp,
        max_median_kbp=args.max_median_kbp,
    )

    # Save stripped results (remove non-serialisable tads/boundaries keys)
    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k not in ("tads", "boundaries")}

    save_csv([strip(r) for r in all_results], args.out_csv)
    save_plot(
        [strip(r) for r in all_results],
        args.out_png,
        min_tads=args.min_tads,
        min_median_kbp=args.min_median_kbp,
        max_median_kbp=args.max_median_kbp,
    )

    # ── 6. Best config: prefer constrained, fall back to unconstrained ────────
    finite_all = [r for r in all_results if np.isfinite(r["delta_median"])]
    if not finite_all:
        print("WARNING: no finite delta_median values — cannot select best config.")
        return

    constrained_all = [
        r for r in finite_all
        if _passes_constraints(r, args.min_tads, args.min_median_kbp, args.max_median_kbp)
    ]
    if constrained_all:
        best = min(constrained_all, key=lambda r: r["delta_median"])
        best_label = "BEST (meets constraints)"
    else:
        print(
            "WARNING: no config meets the n_tads / median_kbp constraints.\n"
            "         Falling back to global best delta_median."
        )
        best = min(finite_all, key=lambda r: r["delta_median"])
        best_label = "BEST (unconstrained — no config met size/count constraints)"
    print(f"\n{'═'*60}")
    print(f"  ★  {best_label}  ★")
    print(f"  window         = {best['window']} bins  ({best['window']*BINSIZE//1000} kbp)")
    print(f"  step           = {best['step']} bins  ({best['step']*BINSIZE//1000} kbp)")
    print(f"  boundary_tol   = {best['boundary_tol']} bins")
    print(f"  vote_frac      = {best['vote_frac']:.2f}")
    print(f"  Δ median log2I = {best['delta_median']:+.4f}  "
          f"({'pred BEATS GT ▼' if best['delta_median'] < 0 else 'pred weaker than GT ▲'})")
    print(f"  # pred boundaries = {best['n_pred_boundaries']}")
    print(f"  # GT  boundaries  = {best['n_gt_boundaries']}")
    print(f"  median TAD size   = {best['median_tad_kbp']:.0f} kbp")
    print(f"{'═'*60}\n")

    write_bed(best["tads"], chrom, args.save_best_bed)
    plot_results(
        M_sparse, n_bins, best["tads"],
        tad_statistics(best["tads"]),
        chrom=chrom,
        out_png=args.save_best_png,
        max_vis_bins=3000,
    )

    # ── 7. Print the exact commands to reproduce the best run ─────────────────
    print("\n── Reproduce best config ─────────────────────────────────────────")
    print(f"python full_chr_tv_experiment.py \\")
    print(f"    --hic      {args.hic} \\")
    print(f"    --chrom    {args.chrom} \\")
    print(f"    --start-bp {args.start_bp} \\")
    print(f"    --norm     {NORM} \\")
    print(f"    --window   {best['window']} \\")
    print(f"    --step     {best['step']} \\")
    print(f"    --boundary-tol {best['boundary_tol']} \\")
    print(f"    --vote-frac    {best['vote_frac']} \\")
    print(f"    --gamma    {GAMMA} \\")
    print(f"    --out-png  best_tads_repro.png \\")
    print(f"    --out-bed  best_tads_repro.bed")
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
