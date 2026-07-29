"""
Compute and compare insulation scores at predicted vs ground-truth TAD boundaries.

Both sets of boundaries are scored on the **same** Crane diamond insulation
track computed once from the Hi-C file, so the comparison is fully apples-to-
apples.

    I(b; w) = mean( M[b-w : b,  b : b+w] )      (raw Crane score)
    log2_I  = log2( I(b) / median_chrom(I) )     (chromosome-normalised)

Negative log2_I → fewer contacts crossing the locus → stronger insulation.

Inputs
------
  --pred   domain BED produced by full_chr_tv_experiment.py
  --gt     Arrowhead BEDPE  (chr1 x1 x2 chr2 y1 y2 ...)
           or any domain BED/BEDPE — boundaries extracted automatically
  --hic    path to .hic file
  --chrom  chromosome

Usage
-----
  python compare_tad_boundaries.py \\
      --pred chr1_tads.bed \\
      --gt   arrowhead_output/10000_blocks.bedpe \\
      --hic  /path/to/lateG1.hic \\
      --chrom 1 \\
      --out-bed chr1_pred_insulation.bed \\
      --plot   insulation_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

BINSIZE = 10_000


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
    straw_fn = _import_straw()
    print(
        f"  Querying straw:  chrom={chrom}  binsize={binsize} bp  "
        f"norm={normalization} ...",
        flush=True,
    )
    records = straw_fn(
        "observed", normalization, hic_path, chrom, chrom, "BP", binsize
    )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    max_bin = 0
    for rec in records:
        if hasattr(rec, "binX"):
            bx = int(rec.binX) // binsize
            by = int(rec.binY) // binsize
            v = float(rec.counts)
        else:
            bx = int(rec[0]) // binsize
            by = int(rec[1]) // binsize
            v = float(rec[2])
        if v <= 0 or not np.isfinite(v):
            continue
        rows.append(bx); cols.append(by); vals.append(v)
        if bx != by:
            rows.append(by); cols.append(bx); vals.append(v)
        max_bin = max(max_bin, bx, by)
    n_bins = max_bin + 1
    M = csr_matrix(
        (vals, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float64
    )
    print(f"  Loaded: {n_bins} bins ({n_bins * binsize / 1e6:.1f} Mbp)  nnz={M.nnz:,}")
    return M, n_bins


# ── Domain / BEDPE loading ────────────────────────────────────────────────────

def _is_header(line: str) -> bool:
    s = line.lstrip()
    if not s or s.startswith("#") or s.startswith("track"):
        return True
    first = s.split()[0].lower()
    return first in {"chrom", "chr", "#chrom", "#chr1"}


def load_domains(path: str, chrom: str | None = None) -> list[tuple[int, int]]:
    """
    Load (start, end) domain intervals from a BED or BEDPE file.

    For BEDPE (≥6 columns where col[0]==col[3] numerically/chromatically),
    cols 1–2 (x1, x2) are used as the domain span.
    """
    chrom_key = chrom.removeprefix("chr") if chrom else None
    domains: list[tuple[int, int]] = []
    with open(path) as f:
        for line in f:
            if _is_header(line):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            c = parts[0].removeprefix("chr")
            if chrom_key is not None and c != chrom_key:
                continue
            s, e = int(parts[1]), int(parts[2])
            # BEDPE: skip the on-diagonal self-pair; x1/x2 is the domain
            if e < s:
                s, e = e, s
            domains.append((s, e))
    return sorted(domains)


def domains_to_boundary_bins(
    domains: list[tuple[int, int]],
    binsize: int = BINSIZE,
    include_endpoints: bool = False,
) -> np.ndarray:
    """
    Extract boundary positions (bin indices) from a sorted domain list.

    For a domain BED that forms a contiguous partition (pred), only internal
    junctions (domain starts after the first) are emitted.
    For a sparse/filtered set (GT from BEDPE), both ends of every domain
    are boundary candidates.
    """
    if not domains:
        return np.asarray([], dtype=int)

    # Detect whether domains form a gap-free partition
    contiguous = all(
        domains[i][1] == domains[i + 1][0] for i in range(len(domains) - 1)
    )

    bps: set[int] = set()
    if contiguous:
        # Internal junctions only
        for i in range(1, len(domains)):
            bps.add(domains[i][0])
        if include_endpoints:
            bps.add(domains[0][0])
            bps.add(domains[-1][1])
    else:
        # Sparse (e.g. Arrowhead): both ends of every domain
        for s, e in domains:
            bps.add(s)
            bps.add(e)

    return np.asarray(sorted(bps // binsize for bps in bps), dtype=int)


# ── Insulation track ──────────────────────────────────────────────────────────

def insulation_track(M: csr_matrix, window_bins: int) -> np.ndarray:
    """
    Crane insulation track across the chromosome.

    Uses a difference-array accumulation over upper-triangle contacts so the
    full n×n matrix is never materialised (O(nnz) memory).
    """
    n = M.shape[0]
    w = int(window_bins)
    scores = np.full(n, np.nan, dtype=np.float64)
    if w < 1 or n <= 2 * w:
        return scores

    coo = M.tocoo()
    i = coo.row.astype(np.int64)
    j = coo.col.astype(np.int64)
    v = coo.data.astype(np.float64)
    upper = i < j
    i, j, v = i[upper], j[upper], v[upper]

    # (i,j) ∈ diamond(b)  ⟺  b ∈ (i, i+w] ∩ (j-w, j]
    lo = np.maximum(i + 1, j - w + 1)
    hi = np.minimum(i + w, j) + 1
    valid = lo < hi
    lo, hi, v = lo[valid], hi[valid], v[valid]

    diff = np.zeros(n + 1, dtype=np.float64)
    np.add.at(diff, lo, v)
    np.add.at(diff, hi, -v)
    accum = np.cumsum(diff[:-1])

    scores[w : n - w] = accum[w : n - w] / float(w * w)
    return scores


def log2_normalize(raw: np.ndarray) -> np.ndarray:
    """
    log2(I / median_chrom_I).

    Bins with I < 1% of the median (gap/centromere) are left as NaN so they
    don't bias the chromosome summary.
    """
    out = np.full_like(raw, np.nan)
    valid = np.isfinite(raw) & (raw > 0)
    if not np.any(valid):
        return out
    med = float(np.median(raw[valid]))
    if med <= 0:
        return out
    usable = valid & (raw >= 0.01 * med)
    out[usable] = np.log2(raw[usable] / med)
    return out


# ── Stats & plotting ──────────────────────────────────────────────────────────

def summarize(label: str, scores: np.ndarray) -> None:
    x = scores[np.isfinite(scores)]
    if x.size == 0:
        print(f"  {label}: no finite scores")
        return
    print(
        f"  {label}: n={x.size}  "
        f"median={np.median(x):.4f}  mean={x.mean():.4f}  "
        f"p10={np.percentile(x, 10):.4f}  p90={np.percentile(x, 90):.4f}  "
        f"min={x.min():.4f}  max={x.max():.4f}"
    )


def plot_comparison(
    pred_scores: np.ndarray,
    gt_scores: np.ndarray,
    chrom_scores: np.ndarray,
    window_bp: int,
    chrom: str,
    out_png: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_fin = pred_scores[np.isfinite(pred_scores)]
    gt_fin   = gt_scores[np.isfinite(gt_scores)]
    chrom_fin = chrom_scores[np.isfinite(chrom_scores)]

    lo = min(pred_fin.min() if pred_fin.size else 0,
             gt_fin.min()   if gt_fin.size   else 0) - 0.2
    hi = max(pred_fin.max() if pred_fin.size else 1,
             gt_fin.max()   if gt_fin.size   else 1) + 0.2
    bins = np.linspace(lo, hi, 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: overlapping histograms ──────────────────────────────────────────
    ax = axes[0]
    ax.hist(chrom_fin, bins=bins, density=True, alpha=0.25,
            color="grey", label=f"chrom background (n={chrom_fin.size:,})")
    ax.hist(gt_fin,   bins=bins, density=True, alpha=0.55,
            color="steelblue", label=f"GT (Arrowhead) boundaries (n={gt_fin.size})")
    ax.hist(pred_fin, bins=bins, density=True, alpha=0.65,
            color="tomato",    label=f"Predicted boundaries (n={pred_fin.size})")

    for arr, color, ls in [
        (pred_fin, "tomato",     "-"),
        (gt_fin,   "steelblue",  "--"),
    ]:
        if arr.size:
            ax.axvline(np.median(arr), color=color, ls=ls, lw=1.8,
                       label=f"median {'pred' if color=='tomato' else 'GT'} = {np.median(arr):.3f}")

    ax.axvline(0, color="black", ls=":", lw=1, label="chrom median (0)")
    ax.set_xlabel("log₂ insulation score", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"Insulation distribution  (chr{chrom}, w={window_bp//1000} kb)", fontsize=11)
    ax.legend(fontsize=8)

    # ── right: box/violin ────────────────────────────────────────────────────
    ax2 = axes[1]
    parts = ax2.violinplot(
        [pred_fin, gt_fin],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], ["tomato", "steelblue"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Predicted\n(TV-SSC)", "GT\n(Arrowhead)"], fontsize=11)
    ax2.axhline(0, color="grey", ls=":", lw=1)
    ax2.set_ylabel("log₂ insulation score", fontsize=11)
    ax2.set_title("Insulation strength at boundaries", fontsize=11)

    # annotate medians
    for x_pos, arr, color in [(1, pred_fin, "tomato"), (2, gt_fin, "steelblue")]:
        if arr.size:
            med = np.median(arr)
            ax2.text(x_pos, med + 0.04, f"{med:.3f}",
                     ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  Saved plot → {out_png}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Score predicted and GT TAD boundaries with the Crane insulation "
            "metric from a .hic file and compare distributions."
        )
    )
    p.add_argument("--pred",  required=True, help="Predicted domain BED")
    p.add_argument("--gt",    required=True, help="GT domain BED or Arrowhead BEDPE")
    p.add_argument("--hic",   required=True, help="Path to .hic file")
    p.add_argument("--chrom", required=True, help="Chromosome (1 or chr1)")
    p.add_argument(
        "--binsize", type=int, default=BINSIZE,
        help="Resolution in bp (default 10000)",
    )
    p.add_argument(
        "--window-bp", type=int, default=500_000,
        help="Insulation diamond half-width in bp (default 500000)",
    )
    p.add_argument(
        "--norm", default="KR",
        help="Hi-C normalisation for straw (default KR)",
    )
    p.add_argument(
        "--include-endpoints", action="store_true",
        help="Also score the first/last domain endpoints for predicted TADs",
    )
    p.add_argument(
        "--out-bed", default=None,
        help="Write predicted-boundary insulation BED (chrom start end log2_I raw_I)",
    )
    p.add_argument(
        "--plot", default=None,
        help="Save comparison plot to this PNG path",
    )
    args = p.parse_args()

    chrom = args.chrom.removeprefix("chr")
    w_bins = max(1, args.window_bp // args.binsize)

    # ── load domain boundaries ────────────────────────────────────────────────
    print(f"Loading predicted domains from {args.pred} ...")
    pred_domains = load_domains(args.pred, chrom=chrom)
    if not pred_domains:
        raise SystemExit(f"No domains found for chrom {chrom} in {args.pred}")
    pred_bins = domains_to_boundary_bins(
        pred_domains, binsize=args.binsize,
        include_endpoints=args.include_endpoints,
    )
    print(f"  {len(pred_domains)} domains → {len(pred_bins)} boundaries")

    print(f"Loading GT domains from {args.gt} ...")
    gt_domains = load_domains(args.gt, chrom=chrom)
    if not gt_domains:
        raise SystemExit(f"No domains found for chrom {chrom} in {args.gt}")
    gt_bins = domains_to_boundary_bins(gt_domains, binsize=args.binsize)
    print(f"  {len(gt_domains)} domains → {len(gt_bins)} boundaries")

    # ── build insulation track ────────────────────────────────────────────────
    print(f"\nLoading Hi-C from {args.hic} ...")
    M, n_bins = load_chromosome_sparse(
        args.hic, chrom, binsize=args.binsize, normalization=args.norm
    )

    print(
        f"Computing chromosome-wide insulation track  "
        f"(window = {args.window_bp // 1000} kb = {w_bins} bins) ..."
    )
    raw_track  = insulation_track(M, w_bins)
    log2_track = log2_normalize(raw_track)
    chrom_log2 = log2_track[np.isfinite(log2_track)]
    print(
        f"  chrom: n={chrom_log2.size:,}  "
        f"median={np.median(chrom_log2):.4f}  mean={chrom_log2.mean():.4f}"
    )

    # ── look up scores ────────────────────────────────────────────────────────
    def fetch(bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = (bins >= 0) & (bins < n_bins)
        b = bins[keep]
        return raw_track[b], log2_track[b]

    raw_pred,  log2_pred  = fetch(pred_bins)
    raw_gt,    log2_gt    = fetch(gt_bins)

    # ── report ────────────────────────────────────────────────────────────────
    print()
    print("═" * 58)
    print("  PREDICTED boundaries (TV-SSC)")
    summarize("  raw I  ", raw_pred)
    summarize("  log2_I ", log2_pred)
    n_neg = int(np.sum(np.isfinite(log2_pred) & (log2_pred < 0)))
    n_fin = int(np.sum(np.isfinite(log2_pred)))
    if n_fin:
        print(f"  log2_I < 0 (stronger than chrom median): {n_neg}/{n_fin} ({100*n_neg/n_fin:.1f}%)")

    print()
    print("  GROUND-TRUTH boundaries (Arrowhead / GT)")
    summarize("  raw I  ", raw_gt)
    summarize("  log2_I ", log2_gt)
    n_neg_gt = int(np.sum(np.isfinite(log2_gt) & (log2_gt < 0)))
    n_fin_gt = int(np.sum(np.isfinite(log2_gt)))
    if n_fin_gt:
        print(f"  log2_I < 0 (stronger than chrom median): {n_neg_gt}/{n_fin_gt} ({100*n_neg_gt/n_fin_gt:.1f}%)")

    pred_fin = log2_pred[np.isfinite(log2_pred)]
    gt_fin   = log2_gt[np.isfinite(log2_gt)]
    if pred_fin.size and gt_fin.size:
        delta = np.median(pred_fin) - np.median(gt_fin)
        print()
        print("  COMPARISON")
        print(f"  Δ median log2_I  (pred − GT) = {delta:+.4f}")
        print(
            f"  {'▲ pred is WEAKER' if delta > 0 else '▼ pred is STRONGER'} than GT "
            f"({abs(delta):.4f} log2 units)"
        )
        print("═" * 58)

    # ── optional outputs ──────────────────────────────────────────────────────
    if args.out_bed:
        out = Path(args.out_bed)
        keep = (pred_bins >= 0) & (pred_bins < n_bins)
        b_out = pred_bins[keep]
        with open(out, "w") as f:
            f.write("chrom\tstart\tend\tlog2_insulation_score\tinsulation_raw\n")
            for b, lg, raw in zip(b_out, log2_pred, raw_pred):
                start = int(b) * args.binsize
                end   = start + args.binsize
                lg_s  = f"{lg:.6f}"  if np.isfinite(lg)  else "nan"
                raw_s = f"{raw:.6f}" if np.isfinite(raw) else "nan"
                f.write(f"{chrom}\t{start}\t{end}\t{lg_s}\t{raw_s}\n")
        print(f"\nWrote {out}  ({len(b_out)} boundaries)")

    if args.plot:
        plot_comparison(
            log2_pred, log2_gt, log2_track,
            window_bp=args.window_bp,
            chrom=chrom,
            out_png=args.plot,
        )


if __name__ == "__main__":
    main()
