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
from matplotlib.colors import LogNorm

#from ssc_admm import ssc_admm, cluster_from_C
from ssc_admm_nuc_lap import ssc_admm_nuc_lap as ssc_admm, cluster_from_C


@dataclass(frozen=True)
class HicWindow:
    chrom: str
    binsize: int
    start_bp: int
    n_bins: int = 100

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
    X: np.ndarray,
    pred_labels: np.ndarray,
    metrics: dict[str, float],
    out_png: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Hi-C counts are heavy-tailed; a log color scale (or robust vmax) makes
    # off-diagonal structure visible instead of washing it out.
    Y_pos = Y[Y > 0]
    vmin = float(np.percentile(Y_pos, 5)) if Y_pos.size else 1.0
    vmax = float(np.percentile(Y_pos, 99.5)) if Y_pos.size else 1.0
    norm_hic = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6) * 1.01))

    boundaries = _contiguous_boundaries(pred_labels)

    ax = axes[0, 0]
    im = ax.imshow(Y, cmap="Blues", interpolation="nearest", aspect="auto", norm=norm_hic)
    ax.set_title("Hi-C window (original order)")
    # Overlay predicted contiguous segment boundaries (no reordering).
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=0.6, alpha=0.6)
        ax.axvline(b - 0.5, color="black", lw=0.6, alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    W = np.abs(X) + np.abs(X.T)
    im = ax.imshow(W, cmap="hot", interpolation="nearest", aspect="auto")
    ax.set_title("SSC affinity  W = |X| + |X|^T")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1D label track in original order (lets you see contiguity directly).
    ax = axes[1, 0]
    track = np.asarray(pred_labels, dtype=float)[None, :]
    im = ax.imshow(track, aspect="auto", interpolation="nearest", cmap="tab20")
    ax.set_yticks([])
    ax.set_xlabel("genomic bin index (original order)")
    ax.set_title("Predicted cluster id (original order)")
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", lw=0.6, alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    uniq, counts = np.unique(pred_labels, return_counts=True)
    ax.bar([str(u) for u in uniq], counts, color="#4C72B0", edgecolor="white")
    ax.set_title("Cluster sizes")
    ax.set_xlabel("cluster id")
    ax.set_ylabel("count")

    title_bits = [f"{k}={v:.4g}" for k, v in metrics.items()]
    fig.suptitle(" | ".join(title_bits), fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Load a 100x100 Hi-C window and run SSC-ADMM.")
    p.add_argument("--hic", required=True, help="Path to .hic file")
    p.add_argument("--chrom", default="chr1", help="Chromosome name, e.g. chr1")
    p.add_argument("--binsize", type=int, default=10_000, help="Bin size (BP units)")
    p.add_argument("--start-bp", type=int, default=0, help="Window start in base pairs")
    p.add_argument("--k", type=int, default=4, help="Number of clusters for spectral clustering")
    p.add_argument("--norm", choices=["none", "log1p", "zscore"], default="log1p", help="Pre-normalization")

    # SSC hyperparams (forwarded to ssc_admm.ssc_admm)
    p.add_argument("--lambda-e", type=float, default=1.0)
    p.add_argument("--lambda-z", type=float, default=10.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)

    p.add_argument("--out", default="hic_ssc.png", help="Output PNG filename")
    args = p.parse_args()

    window = HicWindow(chrom=args.chrom, binsize=args.binsize, start_bp=args.start_bp, n_bins=300)
    print(
        f"Loading Hi-C: {args.hic}\n"
        f"  chrom={window.chrom} binsize={window.binsize} start_bp={window.start_bp} end_bp={window.end_bp}"
    )

    Y_raw = load_hic_window_observed(args.hic, window, normalization="KR", unit="BP")
    Y = _normalize_matrix(Y_raw, args.norm)

    # SSC expects a data matrix Y of shape (n, N). In the included demos, Y is (N,N),
    # so we follow that convention here.
    print("Running SSC-ADMM ...")
    t0 = time.perf_counter()
    X, C, J, E = ssc_admm(
        Y,
        lambda_e=args.lambda_e,
        lambda_z=args.lambda_z,
        mu=args.mu,
        rho=1,
        gamma=2,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    # X, C, E = ssc_admm(
    #     Y,
    #     lambda_e=args.lambda_e,
    #     lambda_z=args.lambda_z,
    #     mu=args.mu,
    #     max_iter=args.max_iter,
    #     tol=args.tol,
    # )
    elapsed = time.perf_counter() - t0

    pred = cluster_from_C(X, args.k)

    # Simple diagnostics (no ground-truth labels for real Hi-C windows)
    recon = np.linalg.norm(Y - Y @ X - E, ord="fro") / (np.linalg.norm(Y, ord="fro") + 1e-12)
    sparsity = float(np.mean(np.abs(X) < 1e-8))
    metrics = {
        "time_s": float(elapsed),
        "recon_rel_fro": float(recon),
        "sparsity(|X|<1e-8)": float(sparsity),
    }
    print(f"Done. {metrics}")

    visualize_hic_ssc(Y_raw, X, pred, metrics, args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
