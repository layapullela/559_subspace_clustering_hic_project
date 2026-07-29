"""
Full-Chromosome TAD Detection via Tiled NucTV-SSC
==================================================

Loads a full intra-chromosomal Hi-C contact matrix at 10 kbp resolution,
tiles the chromosome into overlapping windows, runs the TV SSC-ADMM solver
on each tile, and reconstructs a genome-wide TAD *partition* by consensus
voting on domain boundaries.

Per-window logic
----------------
  After clustering a window of N bins into k communities:
    1. Convert predicted labels to contiguous segment runs.
    2. Emit each internal segment boundary (the cut between two adjacent
       segments) as a candidate boundary in global genomic coordinates.
       Boundaries at the window edge are ignored — they are cut off by the
       tiling and unreliable — unless the edge is the true region edge.

Boundary consensus across windows
---------------------------------
  Overlapping windows vote on boundary positions.  Candidate boundaries
  within ``tol`` bins of one another are grouped into a single consensus
  boundary.  A group is accepted when its number of votes is at least
  ``vote_frac`` of its coverage (the number of windows that span the
  position with the boundary interior to them).  Accepted boundaries plus
  the region endpoints define a set of consecutive, non-overlapping TAD
  domains; domains shorter than ``min_tad_bins`` are merged away.

Statistics reported
-------------------
  TADs detected  – number of domains in the partition
  Boundaries     – number of internal consensus boundaries
  Median/Mean    – domain length in kbp

Usage
-----
  python full_chr_tv_experiment.py \\
      --hic  /path/to/sample.hic \\
      --chrom chr1 \\
      --window 500 --step 150 \\
      --out-png full_chr_tads.png \\
      --out-bed full_chr_tads.bed
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings('ignore')

from ssc_admm_nuc_total_var import ssc_admm_nuc_tv, cluster_from_C

BINSIZE = 10_000   # 10 kbp


# ── Hi-C loading ──────────────────────────────────────────────────────────────

def _import_straw():
    try:
        from straw import straw as fn
        return fn
    except Exception:
        try:
            import straw
            return straw.straw
        except Exception:
            import hicstraw
            return hicstraw.straw


def load_chromosome_sparse(
    hic_path: str,
    chrom: str,
    binsize: int = BINSIZE,
    normalization: str = "KR",
) -> tuple[csr_matrix, int]:
    """
    Load a full intra-chromosomal Hi-C contact map as a scipy sparse matrix.

    Returns
    -------
    M      : csr_matrix  shape (n_bins, n_bins), symmetrized
    n_bins : int         number of genomic bins
    """
    straw_fn = _import_straw()
    print(
        f"  Querying straw:  chrom={chrom}  binsize={binsize} bp  "
        f"norm={normalization} ...",
        flush=True,
    )
    records = straw_fn("observed", normalization, hic_path, chrom, chrom, "BP", binsize)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    max_bin = 0

    print('begin loading matrix')
    for rec in records:
        if hasattr(rec, "binX"):
            bx = int(rec.binX) // binsize
            by = int(rec.binY) // binsize
            v  = float(rec.counts)
        else:
            bx = int(rec[0]) // binsize
            by = int(rec[1]) // binsize
            v  = float(rec[2])

        if v <= 0 or not np.isfinite(v):
            continue

        rows.append(bx); cols.append(by); vals.append(v)
        if bx != by:
            rows.append(by); cols.append(bx); vals.append(v)
        if bx > max_bin:
            max_bin = bx
        if by > max_bin:
            max_bin = by

    n_bins = max_bin + 1
    M = csr_matrix(
        (vals, (rows, cols)),
        shape=(n_bins, n_bins),
        dtype=np.float64,
    )
    print(
        f"  Loaded:  {n_bins} bins  "
        f"({n_bins * binsize / 1e6:.1f} Mbp)  "
        f"nnz={M.nnz:,}"
    )
    return M, n_bins


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_window(Y: np.ndarray, method: str = "log1p") -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    pos = Y[Y > 0]
    if pos.size:
        Y = np.minimum(Y, float(np.percentile(pos, 95)))
    if method == "log1p":
        return np.log1p(Y)
    if method == "zscore":
        mu, sd = Y.mean(), Y.std()
        return (Y - mu) / (sd + 1e-12)
    return Y


# ── TAD interval extraction ───────────────────────────────────────────────────

def labels_to_segments(pred: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a label array to a list of (start, end) half-open intervals
    in local coordinates, one per contiguous run of equal labels.
    """
    pred = np.asarray(pred)
    n    = len(pred)
    segs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and pred[j] == pred[i]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def window_boundaries(
    pred: np.ndarray,
    win_start: int,
    win_end: int,
    region_start: int,
    region_end: int,
    edge_margin: int,
) -> list[int]:
    """
    Global bin positions of the internal segment cuts detected in one window.

    A cut is the start of a segment other than the first.  Cuts within
    ``edge_margin`` bins of a window edge are discarded (the tiling can slice
    through a domain there, so those positions are unreliable) — unless the
    edge coincides with the true region edge, where the cut is genuine.
    """
    W = win_end - win_start
    at_region_start = win_start <= region_start
    at_region_end   = win_end   >= region_end

    bounds: list[int] = []
    for s, _e in labels_to_segments(pred)[1:]:   # skip the first segment (s == 0)
        near_left  = s < edge_margin            and not at_region_start
        near_right = s > (W - edge_margin)      and not at_region_end
        if near_left or near_right:
            continue
        bounds.append(win_start + s)
    return bounds


# ── Boundary consensus → TAD partition ────────────────────────────────────────

def consensus_boundaries(
    candidates: list[tuple[int, int]],
    windows: list[tuple[int, int]],
    tol: int,
    vote_frac: float,
) -> list[int]:
    """
    Reduce per-window candidate boundaries to a consensus boundary set.

    Candidate positions within ``tol`` bins of one another are grouped into a
    single boundary (placed at the vote-weighted mean).  A group is accepted
    when the number of windows voting for it is at least ``vote_frac`` of its
    coverage — the number of windows whose interior spans the boundary and
    could therefore have voted.

    Parameters
    ----------
    candidates : (position, window_id) pairs, one per detected cut
    windows    : (interior_lo, interior_hi) half-open interior range of each
                 processed window, indexed by window_id
    tol        : max bin distance for two cuts to be the same boundary
    vote_frac  : fraction of covering windows that must agree
    """
    if not candidates:
        return []

    candidates = sorted(candidates)
    groups: list[list[tuple[int, int]]] = [[candidates[0]]]
    for pos, wid in candidates[1:]:
        if pos - groups[-1][-1][0] <= tol:
            groups[-1].append((pos, wid))
        else:
            groups.append([(pos, wid)])

    accepted: list[int] = []
    for group in groups:
        positions = [p for p, _ in group]
        rep = int(round(float(np.mean(positions))))
        votes = len({wid for _, wid in group})
        coverage = sum(1 for lo, hi in windows if lo <= rep < hi)
        if coverage == 0:
            continue
        if votes >= max(1, int(np.ceil(vote_frac * coverage))):
            accepted.append(rep)
    return sorted(set(accepted))


def boundaries_to_domains(
    boundaries: list[int],
    region_start: int,
    region_end: int,
    min_bins: int,
) -> list[tuple[int, int]]:
    """
    Turn a boundary set into consecutive, non-overlapping TAD domains.

    Domains shorter than ``min_bins`` are removed by dropping their weaker
    bounding boundary until every domain meets the minimum length.
    """
    edges = [region_start] + [b for b in sorted(set(boundaries))
                              if region_start < b < region_end] + [region_end]

    while len(edges) > 2:
        sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        i = int(np.argmin(sizes))
        if sizes[i] >= min_bins:
            break
        # drop the internal edge that borders this too-short domain
        edges.pop(i + 1 if i + 1 < len(edges) - 1 else i)

    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


# ── TAD statistics ────────────────────────────────────────────────────────────

def tad_statistics(tads: list[tuple[int, int]]) -> dict:
    """Summary statistics for a TAD partition."""
    boundaries = [s for s, _e in tads[1:]]   # internal domain starts
    sizes_kbp  = [(e - s) * BINSIZE / 1000.0 for s, e in tads]
    return dict(
        n_tads=len(tads),
        n_boundaries=len(boundaries),
        median_kbp=float(np.median(sizes_kbp)) if sizes_kbp else 0.0,
        mean_kbp=float(np.mean(sizes_kbp)) if sizes_kbp else 0.0,
        boundaries=boundaries,
    )


# ── Tiled SSC-TV pipeline ────────────────────────────────────────────────────

def run_tiled_tv_tad(
    M_sparse: csr_matrix,
    n_bins: int,
    start_bin: int = 0,
    window: int = 500,
    step: int | None = None,
    edge_margin: int | None = None,
    boundary_tol: int = 3,
    vote_frac: float = 0.5,
    min_tad_bins: int = 10,
    norm: str = "log1p",
    lambda_e: float = 1.0,
    lambda_z: float = 0.1,
    gamma: float = 0.5,
    mu: float = 1.0,
    sigma: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-3,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """
    Run TV-SSC on overlapping tiles and reconstruct a genome-wide TAD partition
    by consensus voting on domain boundaries (start_bin .. n_bins).

    Each window contributes candidate boundaries (internal segment cuts, away
    from the unreliable window edges).  Overlapping windows vote; boundaries
    supported by at least ``vote_frac`` of the windows covering them are kept,
    then converted to consecutive non-overlapping domains.

    Parameters
    ----------
    start_bin    : first bin to process (= start_bp // BINSIZE)
    window       : tile width in bins (500 = 5 Mbp at 10 kbp)
    step         : stride in bins (default = window // 2, 50 % overlap)
    edge_margin  : bins near each window edge whose cuts are ignored
                   (default = window // 10)
    boundary_tol : max bin distance for cuts from different windows to be
                   treated as the same boundary
    vote_frac    : fraction of covering windows that must agree on a boundary
    min_tad_bins : minimum domain size in bins

    Returns
    -------
    tads     : list of (global_start, global_end) non-overlapping domains
    tile_log : per-tile diagnostic records
    """
    if step is None:
        step = max(1, window // 2)   # default: 50 % overlap
    if edge_margin is None:
        edge_margin = max(1, window // 10)

    candidates: list[tuple[int, int]] = []   # (boundary_position, window_id)
    interiors:  list[tuple[int, int]] = []   # interior range of each window
    tile_log:   list[dict] = []

    starts = list(range(start_bin, max(start_bin + 1, n_bins - window // 4), step))
    if verbose:
        start_mbp = start_bin * BINSIZE / 1_000_000
        print(
            f"\nTiling: {len(starts)} windows  "
            f"(window={window} bins = {window * BINSIZE // 1_000_000} Mbp,  "
            f"step={step} bins,  overlap={window - step} bins,  "
            f"region_start={start_mbp:.1f} Mbp)\n"
        )

    for idx, start in enumerate(starts):
        end    = min(start + window, n_bins)
        W_size = end - start
        if W_size < 10:
            continue

        Y_raw = M_sparse[start:end, start:end].toarray()
        Y     = normalize_window(Y_raw, norm)

        # Fast σ₁ estimate via single-component randomized SVD
        _, svs1, _ = randomized_svd(Y, n_components=1, random_state=0)
        sigma1 = float(svs1[0])
        tag = f"[{idx+1:3d}/{len(starts)}]  bins [{start:5d}–{end:5d})"
        
        if sigma1 < 1e-10:
            if verbose:
                print(f"  {tag}  SKIPPED (all-zero window)")
            tile_log.append(dict(start=start, end=end, k=0, skipped=True, time_s=0.0))
            continue

        t0 = time.perf_counter()
        X, _C, _E = ssc_admm_nuc_tv(
            Y,
            lambda_e=lambda_e / sigma1,
            lambda_z=lambda_z / sigma1,
            gamma=gamma, mu=mu, sigma=sigma,
            max_iter=max_iter, tol=tol,
        )
        elapsed = time.perf_counter() - t0

        pred   = cluster_from_C(X, k=None)
        bounds = window_boundaries(
            pred, win_start=start, win_end=end,
            region_start=start_bin, region_end=n_bins,
            edge_margin=edge_margin,
        )
        candidates.extend((b, idx) for b in bounds)
        interiors.append((start + edge_margin, end - edge_margin))

        k_det = len(np.unique(pred))
        if verbose:
            print(
                f"  {tag}  k={k_det}  cuts={len(bounds):3d}  "
                f"σ₁={sigma1:.3g}  t={elapsed:.1f}s"
            )
        tile_log.append(dict(start=start, end=end, k=k_det, skipped=False, time_s=elapsed))

    if verbose:
        print(
            f"\nBuilding consensus from {len(candidates)} candidate boundaries "
            f"(tol={boundary_tol} bins, vote_frac={vote_frac:.0%}) ..."
        )
    boundaries = consensus_boundaries(candidates, interiors, boundary_tol, vote_frac)
    tads = boundaries_to_domains(boundaries, start_bin, n_bins, min_tad_bins)
    if verbose:
        print(f"  → {len(boundaries)} consensus boundaries → {len(tads)} TADs\n")

    return tads, tile_log


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    M_sparse: csr_matrix,
    n_bins: int,
    tads: list[tuple[int, int]],
    stats: dict,
    chrom: str,
    out_png: str,
    max_vis_bins: int = 3000,
) -> None:
    """
    Two-panel figure:
      Left  – coverage track with TAD boundaries marked in red
      Right – Hi-C heatmap with TAD boundary lines and alternating diagonal
              TAD block shading
    """
    vis_end = min(n_bins, max_vis_bins)
    Y_vis   = M_sparse[:vis_end, :vis_end].toarray()
    Y_log   = np.log1p(Y_vis)

    # Visible TADs and boundaries
    vis_tads = [(s, e) for s, e in tads if s < vis_end]
    boundaries = stats["boundaries"]
    vis_bounds = [b for b in boundaries if b < vis_end]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ── Panel 1: coverage + TAD boundary marks ────────────────────────────────
    ax = axes[0]
    coverage = np.asarray(M_sparse[:vis_end, :].sum(axis=1)).ravel()
    ax.fill_between(range(vis_end), coverage, color='steelblue', alpha=0.55, lw=0)
    for b in vis_bounds:
        ax.axvline(b, color='red', lw=0.8, alpha=0.5)
    ax.set_xlim(0, vis_end)
    ax.set_title(
        f"Coverage + TAD boundaries  ({chrom}, first "
        f"{vis_end * BINSIZE // 1_000_000} Mbp)",
        fontsize=11,
    )
    ax.set_xlabel("Genomic bin (10 kbp)")
    ax.set_ylabel("Sum of contacts")

    # ── Panel 2: Hi-C heatmap + TAD blocks ───────────────────────────────────
    ax = axes[1]
    pos  = Y_log[Y_log > 0]
    vmin = float(np.percentile(pos, 5))  if pos.size else 0.0
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    im = ax.imshow(
        Y_log, cmap='Reds', aspect='auto', interpolation='nearest',
        vmin=vmin, vmax=vmax,
    )

    # Shade each TAD block on the diagonal with alternating transparency
    palette = ['#3498db', '#2ecc71']
    for i, (s, e) in enumerate(vis_tads):
        e_clipped = min(e, vis_end)
        color = palette[i % 2]
        rect = plt.Rectangle(
            (s - 0.5, s - 0.5), e_clipped - s, e_clipped - s,
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.12,
        )
        ax.add_patch(rect)
        # TAD boundary lines
        for b in [s, e_clipped]:
            if 0 < b < vis_end:
                ax.axhline(b - 0.5, color='navy', lw=0.5, alpha=0.6)
                ax.axvline(b - 0.5, color='navy', lw=0.5, alpha=0.6)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"Hi-C (log1p) + TAD partition  ({chrom})", fontsize=11
    )
    ax.set_xlabel("Genomic bin (10 kbp)")
    ax.set_ylabel("Genomic bin (10 kbp)")
    ax.set_xlim(-0.5, vis_end - 0.5)
    ax.set_ylim(vis_end - 0.5, -0.5)

    fig.suptitle(
        f"TV-SSC TAD detection  |  TADs={stats['n_tads']}  "
        f"boundaries={stats['n_boundaries']}  "
        f"median size={stats['median_kbp']:.0f} kbp",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved figure → {out_png}")
    plt.close(fig)


# ── BED export ────────────────────────────────────────────────────────────────

def write_bed(
    tads: list[tuple[int, int]],
    chrom: str,
    out_bed: str,
) -> None:
    """Write the TAD partition as a BED file (one domain per line)."""
    with open(out_bed, 'w') as f:
        f.write(
            f"track name='TVSSC_TADs' "
            f"description='TV-SSC TAD domains ({chrom})'\n"
        )
        for i, (s, e) in enumerate(tads):
            f.write(f"{chrom}\t{s * BINSIZE}\t{e * BINSIZE}\tTAD_{i+1}\n")
    print(f"Saved BED  → {out_bed}  ({len(tads)} TADs)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Full-chromosome TAD detection via tiled NucTV-SSC at 10 kbp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hic",      required=True,  help="Path to .hic file")
    p.add_argument("--chrom",   default="chr1", help="Chromosome (e.g. chr1 or 1)")
    p.add_argument("--norm",    default="log1p",
                   choices=["none", "log1p", "zscore"],
                   help="Per-window normalisation")
    p.add_argument("--start-bp", type=int, default=0,
                   help="Start position in bp to skip unmappable regions "
                        "(e.g. 3000000 to skip 0–3 Mbp telomere). "
                        "Rounded down to the nearest bin.")
    p.add_argument("--window",  type=int, default=1000,
                   help="Tile width in bins (500 = 5 Mbp at 10 kbp)")
    p.add_argument("--step",    type=int, default=None,
                   help="Stride in bins (default = window // 2, 50 %% overlap)")
    p.add_argument("--edge-margin", type=int, default=None,
                   help="Bins near each window edge whose cuts are ignored "
                        "(default = window // 10)")
    p.add_argument("--boundary-tol", type=int, default=10,
                   help="Max bin distance for cuts to count as the same boundary")
    p.add_argument("--vote-frac", type=float, default=0.5,
                   help="Fraction of covering windows that must agree on a boundary")
    p.add_argument("--min-tad-bins", type=int, default=10,
                   help="Minimum TAD size in bins")

    # SSC-TV hyperparameters
    p.add_argument("--lambda-e", type=float, default=1.0)
    p.add_argument("--lambda-z", type=float, default=0.1)
    p.add_argument("--gamma",    type=float, default=0.5,
                   help="TV regularisation weight")
    p.add_argument("--mu",       type=float, default=1.0)
    p.add_argument("--sigma",    type=float, default=1.0)
    p.add_argument("--max-iter", type=int,   default=50)
    p.add_argument("--tol",      type=float, default=1e-3)

    # Output
    p.add_argument("--out-png",  default="full_chr_tv_tads.png")
    p.add_argument("--out-bed",  default=None,
                   help="Optional output BED file for TAD domains")
    args = p.parse_args()

    t_wall   = time.perf_counter()
    start_bin = args.start_bp // BINSIZE
    if args.start_bp > 0:
        print(f"\nSkipping first {args.start_bp:,} bp  (start_bin = {start_bin})")

    print(f"\nLoading {args.hic}")
    M_sparse, n_bins = load_chromosome_sparse(
        args.hic, args.chrom, binsize=BINSIZE, normalization="KR"
    )

    tads, tile_log = run_tiled_tv_tad(
        M_sparse, n_bins,
        start_bin=start_bin,
        window=args.window,
        step=args.step,
        edge_margin=args.edge_margin,
        boundary_tol=args.boundary_tol,
        vote_frac=args.vote_frac,
        min_tad_bins=args.min_tad_bins,
        norm=args.norm,
        lambda_e=args.lambda_e,
        lambda_z=args.lambda_z,
        gamma=args.gamma,
        mu=args.mu, sigma=args.sigma,
        max_iter=args.max_iter, tol=args.tol,
    )

    stats = tad_statistics(tads)
    n_tiles_run = sum(1 for r in tile_log if not r["skipped"])
    t_total = time.perf_counter() - t_wall

    print(f"\n{'═' * 56}")
    print(f"  Chromosome           : {args.chrom}")
    print(f"  Region start         : {args.start_bp:,} bp  (bin {start_bin})")
    print(f"  Resolution           : {BINSIZE:,} bp  ({n_bins:,} bins total)")
    print(f"  Tiles processed      : {n_tiles_run} / {len(tile_log)}")
    print(f"  ── TAD statistics ─────────────────────────────────")
    print(f"  TADs detected        : {stats['n_tads']}")
    print(f"  Internal boundaries  : {stats['n_boundaries']}")
    print(f"  Median TAD size      : {stats['median_kbp']:.0f} kbp")
    print(f"  Mean TAD size        : {stats['mean_kbp']:.0f} kbp")
    print(f"  Total wall time      : {t_total:.1f} s")
    print(f"{'═' * 56}\n")

    if args.out_bed:
        write_bed(tads, args.chrom, args.out_bed)

    plot_results(
        M_sparse, n_bins, tads, stats,
        args.chrom, args.out_png, max_vis_bins=3000,
    )


if __name__ == "__main__":
    print("starting")
    main()
