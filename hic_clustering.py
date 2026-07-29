"""
Hi-C → SSC-ADMM clustering demo
==============================

Loads an intra-chromosomal Hi-C contact matrix from a `.hic` file using the
`straw` (hic-straw) bindings, extracts a 100x100 numpy array, runs
Sparse Subspace Clustering (SSC) from `ssc_admm.py`, and saves diagnostic plots.

Install deps (example):
  pip install numpy matplotlib scikit-learn hic-straw

Example:
  python hic_clustering.py \
    --hic "/path/to/sample_mouse.hic" \
    --chrom chr2 \
    --binsize 10000 \
    --start-bp 0 \
    --k 4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.cm import get_cmap

from ssc_admm import ssc_admm
from ssc_admm_nuc import ssc_admm as ssc_admm_nuc
from ssc_admm_nuc_lap import ssc_admm_nuc_lap, cluster_from_C
from ssc_admm_nuc_l1 import ssc_admm_nuc_l1
from ssc_admm_nuc_total_var import ssc_admm_nuc_tv


@dataclass(frozen=True)
class HicWindow:
    chrom: str
    binsize: int
    start_bp: int
    n_bins: int = 300

    @property
    def end_bp(self) -> int:
        return self.start_bp + self.n_bins * self.binsize


def _import_straw():
    """
    hic-straw has shown up as either:
      - `from straw import straw`
      - `import straw; straw.straw(...)`
      - `import hicstraw; hicstraw.straw(...)`
    Support both.
    """
    try:
        from straw import straw as straw_fn  # type: ignore

        return straw_fn
    except Exception:
        try:
            import straw  # type: ignore

            return straw.straw
        except Exception:
            import hicstraw  # type: ignore

            return hicstraw.straw


def load_hic_window_observed(
    hic_path: str,
    window: HicWindow,
    normalization: str = "NONE",
    unit: str = "BP",
) -> np.ndarray:
    """
    Returns an (n_bins, n_bins) dense contact matrix for a contiguous window.

    Notes:
    - Uses `observed` counts for (chrom, chrom).
    - `straw` returns sparse triplets (binX, binY, count) in base-pair coords.
    """
    straw_fn = _import_straw()

    chrom = window.chrom
    binsize = int(window.binsize)
    n = int(window.n_bins)
    start_bp = int(window.start_bp)
    end_bp = int(window.end_bp)

    # Query whole-chromosome sparse observed contacts at this resolution.
    # We then slice down to the requested [start_bp, end_bp) window.
    records = straw_fn("observed", normalization, hic_path, chrom, chrom, unit, binsize)

    M = np.zeros((n, n), dtype=np.float64)
    start_bin = start_bp // binsize
    end_bin = (end_bp - 1) // binsize  # inclusive bin index

    for rec in records:
        # hicstraw returns `contactRecord` objects with fields like (binX, binY, counts)
        # while some straw bindings return plain (x_bp, y_bp, value) tuples.
        if hasattr(rec, "binX") and hasattr(rec, "binY"):
            x_bp = int(rec.binX)
            y_bp = int(rec.binY)
            v = float(rec.counts)
        else:
            x_bp, y_bp, v = rec

        bx = int(x_bp) // binsize
        by = int(y_bp) // binsize

        if bx < start_bin or bx > end_bin or by < start_bin or by > end_bin:
            continue

        i = bx - start_bin
        j = by - start_bin
        if 0 <= i < n and 0 <= j < n:
            M[i, j] = float(v)

    # Symmetrize (records are usually upper-tri, but not guaranteed)
    M = np.maximum(M, M.T)
    return M


def _normalize_matrix(Y: np.ndarray, method: str) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)

    # clamp to percentile 99%
    Y = np.where(Y > np.percentile(Y, 95), np.percentile(Y, 95), Y)
    print(f"Clamped to percentile 99%: {np.percentile(Y, 99)}")

    if method == "none":
        return Y
    if method == "log1p":
        return np.log1p(Y)
    if method == "zscore":
        mu = Y.mean()
        sd = Y.std()
        return (Y - mu) / (sd + 1e-12)
    raise ValueError(f"Unknown normalization method: {method}")


def _reorder_by_labels(M: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(labels)
    return M[np.ix_(order, order)]


def _contiguous_boundaries(labels: np.ndarray) -> list[int]:
    """
    Return indices where the label changes when walking left→right.
    These are the natural "segment boundaries" in the original genomic order.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    return (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()


def visualize_hic_ssc(
    Y: np.ndarray,
    X_sp: np.ndarray,   pred_sp: np.ndarray,
    X_nuc: np.ndarray,  pred_nuc: np.ndarray,
    X_lap: np.ndarray,  pred_lap: np.ndarray,
    X_l1: np.ndarray,   pred_l1: np.ndarray,
    X_tv: np.ndarray,   pred_tv: np.ndarray,
    k: int,
    out_png: str,
) -> None:
    """
    2×5 figure: Sparse | Nuc | NucLap | NucL1 | NucTV — Hi-C boundaries (top)
    and predicted cluster label tracks in original genomic order (bottom).
    Cluster colors are consistent across all columns: cluster i always has the
    same color regardless of method.
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 8))

    Y_pos = Y[Y > 0]
    vmin = float(np.percentile(Y_pos, 5)) if Y_pos.size else 1.0
    vmax = float(np.percentile(Y_pos, 99.5)) if Y_pos.size else 1.0
    norm_hic = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6) * 1.01))

    # Shared discrete colormap — same k colors used for every column.
    _palette = [
        "#440154",
        "#3B528B",
        "#21918C",
        "#90D743",
        "#FDE725",
        "#31688E",
        "#35B779",
        "#443A83",
    ]
    cluster_colors = [_palette[i % len(_palette)] for i in range(k)]
    from matplotlib.colors import ListedColormap
    cmap_clusters = ListedColormap(cluster_colors)
    bounds_norm = BoundaryNorm(np.arange(-0.5, k, 1.0), k)

    methods = [
        ("Sparse",  pred_sp,  _contiguous_boundaries(pred_sp)),
        ("Nuc",     pred_nuc, _contiguous_boundaries(pred_nuc)),
        ("NucLap",  pred_lap, _contiguous_boundaries(pred_lap)),
        ("NucL1",   pred_l1,  _contiguous_boundaries(pred_l1)),
        ("NucTV",   pred_tv,  _contiguous_boundaries(pred_tv)),
    ]

    for col, (name, pred, bounds) in enumerate(methods):
        # Row 0 — Hi-C matrix with cluster boundaries overlaid
        ax = axes[0, col]
        im = ax.imshow(Y, cmap="Blues", interpolation="nearest", aspect="auto", norm=norm_hic)
        ax.set_title(f"Hi-C  |  {name} boundaries", fontsize=11)
        for b in bounds:
            ax.axhline(b - 0.5, color="black", lw=0.7, alpha=0.7)
            ax.axvline(b - 0.5, color="black", lw=0.7, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1 — 1-D cluster label track in original genomic order
        ax = axes[1, col]
        track = np.asarray(pred, dtype=int)[None, :]
        im = ax.imshow(track, aspect="auto", interpolation="nearest",
                       cmap=cmap_clusters, norm=bounds_norm)
        ax.set_yticks([])
        ax.set_xlabel("genomic bin index (original order)", fontsize=9)
        ax.set_title(f"{name}  — predicted clusters", fontsize=11)
        for b in bounds:
            ax.axvline(b - 0.5, color="black", lw=0.7, alpha=0.7)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(k))
        cbar.ax.set_yticklabels([str(i + 1) for i in range(k)])

    fig.suptitle(
        "Hi-C SSC clustering  —  Sparse | Nuc | NucLap | NucL1 | NucTV  (original genomic order)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Load a 100x100 Hi-C window and run SSC-ADMM.")
    p.add_argument("--hic", required=True, help="Path to .hic file")
    p.add_argument("--chrom", default="chr1", help="Chromosome name, e.g. chr1")
    p.add_argument("--binsize", type=int, default=10_000, help="Bin size (BP units)")
    p.add_argument("--start-bp", type=int, default=0, help="Window start in base pairs")
    p.add_argument("--k", type=int, default=None, help="Number of clusters for spectral clustering")
    p.add_argument("--norm", choices=["none", "log1p", "zscore"], default="log1p", help="Pre-normalization")

    # SSC hyperparams (forwarded to ssc_admm.ssc_admm)
    p.add_argument("--lambda-e", type=float, default=1.0)
    p.add_argument("--lambda-z", type=float, default=10.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    # NucLap-specific (default: first singular value of Y)
    p.add_argument("--gamma", type=float, default=None,
                   help="NucLap Laplacian penalty (default: σ₁(Y))")
    # NucL1-specific (default: σ₁(Y) / 2 each)
    p.add_argument("--lambda-sp", type=float, default=None,
                   help="NucL1 L1/sparsity weight (default: σ₁(Y)/2)")
    p.add_argument("--lambda-lr", type=float, default=None,
                   help="NucL1 nuclear-norm/low-rank weight (default: σ₁(Y)/2)")
    # NucTV-specific
    p.add_argument("--gamma-tv", type=float, default=None,
                   help="NucTV total-variation penalty (default: same as --gamma)")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="NucTV ADMM penalty for TV auxiliary constraints (default: 1.0)")

    p.add_argument("--out", default="hic_ssc.png", help="Output PNG filename")
    args = p.parse_args()

    window = HicWindow(chrom=args.chrom, binsize=args.binsize, start_bp=args.start_bp, n_bins=250)
    print(
        f"Loading Hi-C: {args.hic}\n"
        f"  chrom={window.chrom} binsize={window.binsize} start_bp={window.start_bp} end_bp={window.end_bp}"
    )

    Y_raw = load_hic_window_observed(args.hic, window, normalization="KR", unit="BP")
    Y = _normalize_matrix(Y_raw, args.norm)

    # Compute first singular value once; used as data-adaptive default scale.
    sigma1 = float(np.linalg.svd(Y, compute_uv=False)[0])
    if sigma1 == 0.0:
        print("WARNING: σ₁(Y) = 0 (matrix appears to be all-zeros). "
              "Falling back to σ₁ = 1.0 for scaling.")
        sigma1 = 1.0
    gamma = sigma1
    gamma_tv = args.gamma_tv if args.gamma_tv is not None else gamma
    lambda_sp = 1
    lambda_lr = 1
    print(f"σ₁(Y)={sigma1:.4g}  →  gamma={gamma:.4g}  gamma_tv={gamma_tv:.4g}  "
          f"lambda_sp={lambda_sp:.4g}  lambda_lr={lambda_lr:.4g}")

    # ── Sparse (L1 only) ──────────────────────────────────────────────────────
    print("\nRunning SSC-ADMM Sparse ...")
    t0 = time.perf_counter()
    X_sp, C_sp, E_sp = ssc_admm(
        Y,
        lambda_e=args.lambda_e / sigma1,
        lambda_z=args.lambda_z / sigma1,
        mu=args.mu,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed_sp = time.perf_counter() - t0
    pred_sp = cluster_from_C(X_sp, args.k)
    recon_sp = np.linalg.norm(Y - Y @ X_sp - E_sp, "fro") / (np.linalg.norm(Y, "fro") + 1e-12)
    metrics_sp = {
        "time_s":   round(elapsed_sp, 2),
        "recon":    round(float(recon_sp), 4),
        "sparsity": round(float(np.mean(np.abs(X_sp) < 1e-8)), 4),
    }
    print(f"Sparse done. {metrics_sp}")

    # ── Nuc (nuclear norm only) ───────────────────────────────────────────────
    print("\nRunning SSC-ADMM Nuc ...")
    t0 = time.perf_counter()
    X_nuc, C_nuc, J_nuc, E_nuc = ssc_admm_nuc(
        Y,
        lambda_e=args.lambda_e / sigma1,
        lambda_z=args.lambda_z / sigma1,
        mu=args.mu,
        rho=1.0,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed_nuc = time.perf_counter() - t0
    pred_nuc = cluster_from_C(X_nuc, args.k)
    print(f"Nuc done. time={elapsed_nuc:.2f}s  "
          f"recon={np.linalg.norm(Y - Y @ X_nuc - E_nuc, 'fro') / (np.linalg.norm(Y, 'fro') + 1e-12):.4f}")

    # ── NucLap ────────────────────────────────────────────────────────────────
    print(f"\nRunning SSC-ADMM NucLap (gamma={gamma:.4g}) ...")
    t0 = time.perf_counter()
    X_lap, J_lap, C_lap, E_lap = ssc_admm_nuc_lap(
        Y,
        lambda_e=args.lambda_e / sigma1,
        lambda_z=args.lambda_z / sigma1,
        mu=args.mu,
        rho=1,
        gamma=gamma,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed_lap = time.perf_counter() - t0
    pred_lap = cluster_from_C(X_lap, args.k)
    recon_lap = np.linalg.norm(Y - Y @ X_lap - E_lap, "fro") / (np.linalg.norm(Y, "fro") + 1e-12)
    metrics_lap = {
        "time_s": round(elapsed_lap, 2),
        "recon": round(float(recon_lap), 4),
        "sparsity": round(float(np.mean(np.abs(X_lap) < 1e-8)), 4),
    }
    print(f"NucLap done. {metrics_lap}")

    # ── NucL1 ─────────────────────────────────────────────────────────────────
    print(f"\nRunning SSC-ADMM NucL1 (λ_sp={lambda_sp:.4g}, λ_lr={lambda_lr:.4g}) ...")
    t0 = time.perf_counter()
    X_l1, C_l1, S_l1, J_l1, E_l1 = ssc_admm_nuc_l1(
        Y,
        lambda_sp=1,
        lambda_lr=1,
        lambda_e=args.lambda_e / sigma1,
        lambda_z=args.lambda_z / sigma1,
        mu=args.mu,
        rho_sp=1.0,
        rho_e=1.0,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed_l1 = time.perf_counter() - t0
    pred_l1 = cluster_from_C(X_l1, args.k)
    recon_l1 = np.linalg.norm(Y - Y @ X_l1 - E_l1, "fro") / (np.linalg.norm(Y, "fro") + 1e-12)
    metrics_l1 = {
        "time_s": round(elapsed_l1, 2),
        "recon": round(float(recon_l1), 4),
        "sparsity": round(float(np.mean(np.abs(X_l1) < 1e-8)), 4),
    }
    print(f"NucL1 done. {metrics_l1}")

    # ── NucTV ─────────────────────────────────────────────────────────────────
    print(f"\nRunning SSC-ADMM NucTV (gamma_tv={gamma_tv:.4g}, sigma={args.sigma:.4g}) ...")
    t0 = time.perf_counter()
    X_tv, _C_tv, E_tv = ssc_admm_nuc_tv(
        Y,
        lambda_e=args.lambda_e / sigma1,
        lambda_z=args.lambda_z / sigma1,
        mu=args.mu,
        sigma=args.sigma,
        gamma=gamma_tv,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed_tv = time.perf_counter() - t0
    pred_tv = cluster_from_C(X_tv, args.k)
    recon_tv = np.linalg.norm(Y - Y @ X_tv - E_tv, "fro") / (np.linalg.norm(Y, "fro") + 1e-12)
    metrics_tv = {
        "time_s":   round(elapsed_tv, 2),
        "recon":    round(float(recon_tv), 4),
        "sparsity": round(float(np.mean(np.abs(X_tv) < 1e-8)), 4),
    }
    print(f"NucTV done. {metrics_tv}")

    # When k was auto-detected per method, derive the visualization k as the
    # maximum number of unique clusters found across all methods.
    all_preds = [pred_sp, pred_nuc, pred_lap, pred_l1, pred_tv]
    k_viz = args.k if args.k is not None else max(len(np.unique(p)) for p in all_preds)

    visualize_hic_ssc(Y_raw,
                      X_sp,  pred_sp,
                      X_nuc, pred_nuc,
                      X_lap, pred_lap,
                      X_l1,  pred_l1,
                      X_tv,  pred_tv,
                      k_viz,
                      args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
