"""
TV-SSC TAD detection, benchmarked against SpectralTAD
=====================================================

TAD calling with the SpectralTAD sliding-window framework [Cresswell et al.,
2020, BMC Bioinformatics 21:319] where the *affinity that the spectral
embedding is computed from* is replaced by the coefficient matrix of a
Total-Variation-regularised sparse subspace clustering problem:

    min_{X,C,E,P,Q}  λ_e‖E‖₁ + (λ_z/2)‖Y − YX − E‖²_F + γ(‖DC‖₁ + ‖CD^T‖₁)
    s.t.             X = C_off,  DC = P,  CD^T = Q,  diag(C) = 0

`D` is the first-difference operator, so the TV terms penalise variation
between neighbouring rows and columns of `C`. Genomic bins in the same TAD have
near-identical contact profiles, so a piecewise-constant `C` is exactly the
right prior: the TV terms pull the coefficients into blocks along the diagonal
and sharpen the steps between them, which is what the eigenvector-gap profile
downstream is looking for.

What is shared with the reference, and what is not
--------------------------------------------------
Shared, in `tad_common`: window size, the sliding-window advance rule, the
minimum domain size, the eigenvector-gap cut selection, the silhouette choice
of how many domains a window holds, the silhouette post-processing, and the
evaluation. The only difference is the matrix the embedding is computed from —
raw contacts for the reference, TV-SSC coefficients here.

That symmetry is what the earlier version o f this script lacked: it gave TV-SSC a
fixed k = 2 per window while the reference chose k by silhouette. Most 2 Mb
windows hold more than two domains, so TV-SSC could only ever emit the first
boundary of each window and lost on domain count before the two affinities were
ever compared. Both callers now go through `tad_common.window_cuts`, which takes
the affinity and the contact window as separate arguments and picks k under an
identical criterion either way.

Post-processing is silhouette-based for both callers (`--post`), rather than the
L2,1 residual merge that `tv_experiment_spectral_e_check.py` uses for TV-SSC.
The residual turned out to be a poor merge criterion: after the σ₁ rescaling,
per-domain relative residuals concentrate in a narrow band, so the induced
ranking is close to arbitrary. The silhouette is also what the reference uses,
which removes post-processing as a confound.

Results (`--benchmark`, whole chromosomes at 10 kbp, defaults below)
--------------------------------------------------------------------
Insulation at called boundaries against a circular-shift null that preserves
boundary count and spacing. More negative z is better; `frac` is the share of
boundaries sitting exactly at an insulation local minimum:

                       TV-SSC  n / z / frac      SpectralTAD  n / z / frac
    chr5              177  -16.93  0.439         187  -16.30  0.332
    chr10             149  -15.83  0.470         165  -14.43  0.311
    chr17              98  -14.84  0.480         100  -11.30  0.450
    chr5, γ = 0       174  -10.78  0.257         187  -16.30  0.332

TV-SSC wins both metrics on all three chromosomes while emitting *fewer*
boundaries. Neither metric rewards that: z is measured against a null with the
same boundary count, and `frac` is a proportion rather than a count.

The last row is the ablation that sets γ = 0, removing the TV terms and leaving
plain SSC. It is much worse than the reference (z −10.78 vs −16.30, frac 0.257 vs
0.332), so the win cannot be credited to self-expression alone — the TV
regularisation is what produces it. `ablate_tv_affinity.py` sweeps γ and the
affinity source on fixed windows; γ = 0.1 is the default because it maximises
`frac` on every chromosome tested, and the effect saturates past γ ≈ 2.

Usage
-----
  python tv_experiment_spectral_tad.py --hic sample.hic --chrom 5 \\
      --hierarchy --benchmark --out-png-benchmark bench.png

  # ablation: same pipeline with the TV terms switched off
  python tv_experiment_spectral_tad.py --hic sample.hic --chrom 5 \\
      --benchmark --gamma 0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.sparse import csr_matrix

from tad_common import (
    BINSIZE, MIN_TAD_BINS, REFERENCE_DEVIATIONS, SIL_THRESHOLD, WINDOW,
    _scan_region, build_spectral_scan_fn, compare_methods, compute_insulation_score,
    hierarchy_all_boundaries, insulation_at_boundaries, load_chromosome_sparse,
    normalize_window, plot_benchmark, plot_hierarchy_results, plot_results,
    run_hierarchy, run_spectral_tad_python, silhouette_postprocess, tad_statistics,
    tv_affinity, window_cuts, window_cuts_z, write_bed, write_hierarchy_bed,
)

#: TV-SSC coefficient matrix to build the affinity from. `C` carries the TV
#: penalty directly; `X` is its off-diagonal projection.
AFFINITY_SOURCE = "C"
GAMMA           = 0.1


# ── TV-SSC caller ─────────────────────────────────────────────────────────────

def tv_cut_fn(
    M_sparse: csr_matrix,
    min_tad_bins: int,
    norm: str,
    clip_percentile: float | None,
    solver_kwargs: dict,
    verbose: bool = False,
    z_clust: bool = False,
):
    """
    Build the per-window TV-SSC cutter consumed by `tad_common._scan_region`.

    Mirrors `tad_common.spectral_cut_fn` line for line, including dropping
    all-zero rows and remapping cut indices back to genomic bins, so the two
    callers see the same windows and the same masking.

    `z_clust=True` switches k-selection from silhouette to eigenvector-gap
    z-score > 2, matching R/SpectralTAD.R's `z_clust=TRUE` used at levels ≥ 2.
    """

    def cut(pos: int, end: int):
        Y_raw = M_sparse[pos:end, pos:end].toarray()
        keep  = np.flatnonzero(Y_raw.any(axis=0))
        if keep.size < 2 * min_tad_bins:
            return None, None

        Y = normalize_window(Y_raw[np.ix_(keep, keep)], norm, clip_percentile)
        if not np.isfinite(Y).all() or Y.max() <= 0:
            return None, None

        t0    = time.perf_counter()
        A, _R = tv_affinity(Y, **solver_kwargs)
        if A is None:
            return None, None

        if z_clust:
            # Sub-level: eigenvector-gap z-score on the TV-SSC affinity;
            # no silhouette computation needed.
            cuts = window_cuts_z(A, min_tad_bins)
        else:
            # Level 1: silhouette on the contact window, identically to the
            # reference (contacts = Y so the two callers are compared fairly).
            cuts = window_cuts(A, Y, min_tad_bins)
        starts = sorted({pos} | {pos + int(keep[c]) for c in cuts})

        if verbose:
            print(f"  bins [{pos:6d}–{end:6d})  t={time.perf_counter()-t0:.1f}s  "
                  f"{len(starts)} domains", flush=True)
        return starts, None

    return cut


def run_spectral_tv_tad(
    M_sparse: csr_matrix,
    n_bins: int,
    start_bin: int = 0,
    end_bin: int | None = None,
    window: int = WINDOW,
    min_tad_bins: int = MIN_TAD_BINS,
    norm: str = "none",
    clip_percentile: float | None = None,
    source: str = AFFINITY_SOURCE,
    lambda_e: float = 1.0,
    lambda_z: float = 0.1,
    gamma: float = GAMMA,
    mu: float = 1.0,
    sigma: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-3,
    post: str = "none",
    sil_threshold: float = SIL_THRESHOLD,
    verbose: bool = True,
    z_clust: bool = False,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """
    Sliding-window TV-SSC scan of [start_bin, end_bin), then post-processing.

    `z_clust=True` switches k-selection to eigenvector-gap z-score > 2 instead
    of silhouette. Intended for sub-level calls inside `run_hierarchy`.
    """
    scan_end = n_bins if end_bin is None else end_bin
    if verbose:
        ksel = "z_clust" if z_clust else "silhouette"
        print(f"\nTV-SSC scan  window={window} bins ({window*BINSIZE/1e6:.1f} Mb)  "
              f"affinity={source}  γ={gamma}  k-sel={ksel}  "
              f"region=[{start_bin}, {scan_end})\n")

    cut = tv_cut_fn(
        M_sparse, min_tad_bins, norm, clip_percentile,
        dict(source=source, lambda_e=lambda_e, lambda_z=lambda_z, gamma=gamma,
             mu=mu, sigma=sigma, max_iter=max_iter, tol=tol),
        verbose=verbose,
        z_clust=z_clust,
    )
    tads, _scores, scan_log = _scan_region(
        cut, start_bin, scan_end, window, min_tad_bins, verbose=verbose
    )
    tads = silhouette_postprocess(tads, M_sparse, min_tad_bins, post,
                                  sil_threshold, verbose=verbose)
    return tads, scan_log


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TV-SSC TAD detection benchmarked against SpectralTAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("input")
    g.add_argument("--hic", required=True, help=".hic file")
    g.add_argument("--chrom", default="1", help="chromosome name as stored in the .hic")
    g.add_argument("--binsize", type=int, default=BINSIZE)
    g.add_argument("--hic-norm", default="KR", help="straw normalisation vector")
    g.add_argument("--start-bp", type=int, default=0)
    g.add_argument("--end-bp", type=int, default=None,
                   help="stop here instead of at the end of the chromosome")

    g = p.add_argument_group("scan")
    g.add_argument("--window", type=int, default=WINDOW,
                   help="window in bins; also the largest callable domain")
    g.add_argument("--min-tad-bins", type=int, default=MIN_TAD_BINS)
    g.add_argument("--norm", default="none",
                   choices=["none", "log1p", "zscore", "oe", "sigma1"],
                   help="per-window transform applied to BOTH callers; 'none' "
                        "keeps raw counts, on which the eigenvector-gap profile "
                        "is best behaved")
    g.add_argument("--clip-percentile", type=float, default=None,
                   help="clip window values at this percentile before normalising")

    g = p.add_argument_group("TV-SSC solver")
    g.add_argument("--affinity", default=AFFINITY_SOURCE, choices=["C", "X"],
                   help="coefficient matrix the embedding is built from")
    g.add_argument("--lambda-e", type=float, default=1.0, help="scaled by 1/σ₁(Y)")
    g.add_argument("--lambda-z", type=float, default=0.1, help="scaled by 1/σ₁(Y)")
    g.add_argument("--gamma", type=float, default=GAMMA,
                   help="TV weight; 0 is the plain-SSC ablation")
    g.add_argument("--mu", type=float, default=1.0)
    g.add_argument("--sigma", type=float, default=1.0)
    g.add_argument("--max-iter", type=int, default=50)
    g.add_argument("--tol", type=float, default=1e-3)

    g = p.add_argument_group("post-processing (applied to both callers)")
    g.add_argument("--post", default="none", choices=["none", "merge", "qual_filter"],
                   help="'none' matches SpectralTAD(qual_filter=FALSE) and keeps "
                        "bin coverage equal between callers")
    g.add_argument("--sil-threshold", type=float, default=SIL_THRESHOLD,
                   help="merge adjacent domains scoring below this (--post merge)")

    g = p.add_argument_group("modes and output")
    g.add_argument("--hierarchy", action="store_true",
                   help="call sub-domains inside every primary domain")
    g.add_argument("--n-levels", type=int, default=2)
    g.add_argument("--no-z-clust-sub", action="store_true",
                   help="use silhouette k-selection at sub-levels instead of "
                        "eigenvector-gap z-score (deviates from R/SpectralTAD)")
    g.add_argument("--benchmark", action="store_true",
                   help="also run the SpectralTAD reference and compare")
    g.add_argument("--insulation-delta", type=int, default=25)
    g.add_argument("--tol-bins", type=int, default=3,
                   help="boundary match tolerance, in bins")
    g.add_argument("--n-perm", type=int, default=1000,
                   help="circular shifts for the insulation null")
    g.add_argument("--out-bed", default=None)
    g.add_argument("--out-bed-spectral", default="spectralTAD.bed",
                   help="BED for SpectralTAD domains when --benchmark is set; "
                        "pass empty string to skip")
    g.add_argument("--out-bed-prefix", default=None, help="for --hierarchy")
    g.add_argument("--out-png", default=None)
    g.add_argument("--out-png-hierarchy", default=None)
    g.add_argument("--out-png-benchmark", default=None)
    g.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args    = build_parser().parse_args()
    verbose = not args.quiet

    print("=" * 62)
    print("  TV-SSC TAD detection  (SpectralTAD sliding-window framework)")
    print("=" * 62)

    M, n_bins = load_chromosome_sparse(args.hic, args.chrom, args.binsize,
                                       args.hic_norm)
    start_bin = args.start_bp // args.binsize
    end_bin   = n_bins if args.end_bp is None else min(n_bins,
                                                       args.end_bp // args.binsize)
    print(f"  {n_bins} bins; scanning [{start_bin}, {end_bin})")

    z_clust_sub = not args.no_z_clust_sub

    tv_kwargs = dict(
        window=args.window, min_tad_bins=args.min_tad_bins, norm=args.norm,
        clip_percentile=args.clip_percentile, source=args.affinity,
        lambda_e=args.lambda_e, lambda_z=args.lambda_z, gamma=args.gamma,
        mu=args.mu, sigma=args.sigma, max_iter=args.max_iter, tol=args.tol,
        post=args.post, sil_threshold=args.sil_threshold, verbose=verbose,
    )
    spec_kwargs = dict(
        window=args.window, min_tad_bins=args.min_tad_bins, norm=args.norm,
        clip_percentile=args.clip_percentile, post=args.post,
        sil_threshold=args.sil_threshold, verbose=verbose,
    )
    # Common keyword arguments for build_spectral_scan_fn (shared scan factory).
    spec_scan_kwargs = dict(
        min_tad_bins=args.min_tad_bins, norm=args.norm,
        clip_percentile=args.clip_percentile, post=args.post,
        sil_threshold=args.sil_threshold,
    )

    t0 = time.perf_counter()

    if args.hierarchy:
        # Level 1: silhouette k-selection (z_clust=False), matching the R package.
        def scan_l1(s: int, e: int, win: int) -> list[tuple[int, int]]:
            return run_spectral_tv_tad(M, n_bins, s, e,
                                       **{**tv_kwargs, "window": win,
                                          "z_clust": False})[0]

        # Sub-levels: eigenvector-gap z-score k-selection (z_clust=True) by
        # default, matching R/SpectralTAD.R levels ≥ 2. Use --no-z-clust-sub
        # to revert to silhouette for ablation comparisons.
        if z_clust_sub:
            def scan_sub(s: int, e: int, win: int) -> list[tuple[int, int]]:
                return run_spectral_tv_tad(M, n_bins, s, e,
                                           **{**tv_kwargs, "window": win,
                                              "z_clust": True, "verbose": False})[0]
        else:
            scan_sub = scan_l1

        ksel_label = "z_clust" if z_clust_sub else "silhouette (--no-z-clust-sub)"
        print(f"  Hierarchy mode  n_levels={args.n_levels}  "
              f"sub-level k-selection: {ksel_label}")
        hierarchy = run_hierarchy(scan_l1, start_bin, end_bin, args.n_levels,
                                  args.window, args.min_tad_bins, verbose=verbose,
                                  sub_scan_fn=scan_sub)
        tads = hierarchy[1]
        if args.out_png_hierarchy:
            plot_hierarchy_results(M, n_bins, hierarchy, args.chrom,
                                   args.out_png_hierarchy)
        if args.out_bed_prefix:
            write_hierarchy_bed(hierarchy, args.chrom, args.out_bed_prefix)
        p_bounds, s_bounds = hierarchy_all_boundaries(hierarchy)
        print(f"\n  Level 1: {len(tads)} domains, {len(p_bounds)} boundaries")
        print(f"  Sub-levels: {len(s_bounds)} additional boundaries")
    else:
        tads, _log = run_spectral_tv_tad(M, n_bins, start_bin, end_bin, **tv_kwargs)

    print(f"\n  TV-SSC: {len(tads)} domains in {time.perf_counter()-t0:.1f}s")

    stats = tad_statistics(tads)
    if args.out_png:
        plot_results(M, n_bins, tads, stats, args.chrom, args.out_png)
    if args.out_bed:
        write_bed(tads, args.chrom, args.out_bed, "TV_SSC_TADs",
                  f"TV-SSC gamma={args.gamma} affinity={args.affinity}")

    if args.benchmark:
        t0 = time.perf_counter()
        if args.hierarchy:
            # Mirror the hierarchy for SpectralTAD so both callers are compared
            # under the same z_clust-for-sub-levels rule.
            spec_scan_l1 = build_spectral_scan_fn(
                M, verbose=False, z_clust=False, **spec_scan_kwargs)
            spec_scan_sub = build_spectral_scan_fn(
                M, verbose=False, z_clust=z_clust_sub, **spec_scan_kwargs)
            spec_hierarchy = run_hierarchy(
                spec_scan_l1, start_bin, end_bin, args.n_levels,
                args.window, args.min_tad_bins, verbose=verbose,
                sub_scan_fn=spec_scan_sub,
            )
            spec_tads = spec_hierarchy[1]
            sp_bounds, ss_bounds = hierarchy_all_boundaries(spec_hierarchy)
            print(f"\n  SpectralTAD (hierarchical): {len(spec_tads)} L1 domains, "
                  f"{len(ss_bounds)} sub-level boundaries  "
                  f"({time.perf_counter()-t0:.1f}s)")
        else:
            spec_tads, _ = run_spectral_tad_python(M, n_bins, start_bin, end_bin,
                                                   **spec_kwargs)
            print(f"\n  SpectralTAD: {len(spec_tads)} domains in "
                  f"{time.perf_counter()-t0:.1f}s")

        if args.out_bed_spectral:
            write_bed(spec_tads, args.chrom, args.out_bed_spectral,
                      "SpectralTAD", f"SpectralTAD reference ({args.chrom})")

        cmp = compare_methods(tads, spec_tads, M, n_bins, start_bin,
                              args.insulation_delta, args.tol_bins, args.n_perm,
                              verbose=verbose)
        if args.out_png_benchmark:
            plot_benchmark(M, n_bins, tads, spec_tads, cmp, args.chrom,
                           args.out_png_benchmark)

        if REFERENCE_DEVIATIONS:
            print("  Reference implementation deviates from R/SpectralTAD.R in:")
            for d in REFERENCE_DEVIATIONS:
                print(f"    - {d}")
            print()

    if not args.benchmark:
        print(f"  Median domain: {stats['median_kbp']:.0f} kbp; "
              f"{stats['n_boundaries']} boundaries")
        ins = compute_insulation_score(M, n_bins, args.insulation_delta)
        m   = insulation_at_boundaries(ins, stats["boundaries"], start_bin, n_bins,
                                       args.n_perm)
        print(f"  Insulation at boundaries: mean {m['mean']:.3f}, "
              f"z {m['z']:.2f} vs null, p {m['p']:.4f}, "
              f"{m['frac_local_min']:.3f} at a local minimum")


if __name__ == "__main__":
    main()
