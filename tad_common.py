"""
Shared infrastructure for the TV-SSC / SpectralTAD TAD-calling experiments
=========================================================================

Everything here is method-agnostic: Hi-C loading, window normalisation, the
sliding-window driver, the hierarchy driver, the eigenvector-gap cut selector,
silhouette post-processing, insulation scoring with a null model, boundary
agreement, figures and BED output.

The two experiment entry points import from here so that a comparison between
them can only differ where they intend to:

  tv_experiment_spectral_tad.py     silhouette post-processing for both callers
  tv_experiment_spectral_e_check.py L2,1 residual post-processing for TV-SSC

Sliding-window driver (SpectralTAD §"Sliding window")
-----------------------------------------------------
Extract the W×W submatrix at `pos`, cut it into domains, accept every domain
except the last, and restart the window at the last domain's start so that
domain is re-cut with full right-hand context. Repeat to the end of the region.

Per-window cut selection (`window_cuts`)
----------------------------------------
Follows R/SpectralTAD.R (Cresswell et al., 2020, BMC Bioinformatics 21:319)
`.windowedSpec` with z_clust = FALSE:

  1. L̄ = D^{-½} A D^{-½} with D from rowSums(|A|), for an affinity A
  2. the 2 largest-magnitude eigenpairs (PRIMME `which = "LM"`)
  3. project rows onto the unit circle; distances between consecutive rows
  4. walk those distances downwards, keeping a cut only if it is more than
     `min_size` from every cut already kept
  5. for k = 2, 3, … score the induced partition's silhouette under
     d = 1/(1+Y) and take the first k that beats k+1

The affinity A and the distance matrix Y are separate arguments. The reference
caller passes the contact window for both; the TV-SSC caller passes its own
self-expression coefficients as A and the same contact window as Y, so the
number of domains is chosen by an identical criterion either way and only the
embedding differs.

Documented deviations from the R package are listed in REFERENCE_DEVIATIONS.

Evaluation
----------
Insulation score (Crane et al., 2015): score(i) = Σ M[i−δ:i, i:i+δ], reported
as log₂(score / mean). Mean insulation at a boundary set improves monotonically
as boundaries are removed, so it is always reported against a circular-shift
null that preserves the boundary count and spacing, as a z-score and an
empirical p-value. Agreement between two callers is reported symmetrically;
neither is ground truth.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings('ignore')

from ssc_admm_nuc_total_var import ssc_admm_nuc_tv

BINSIZE       = 10_000   # 10 kbp per bin
WINDOW        = 200      # 200 bins = 2 Mb, SpectralTAD's max-TAD-size window
MIN_TAD_BINS  = 5        # SpectralTAD min_size
SIL_THRESHOLD = 0.0      # merge a pair scoring below this
QUAL_SIL      = 0.25     # paper/suppl gap filter (R package code used 0.15)

#: Ways this code knowingly departs from R/SpectralTAD.R, so the benchmark
#: cannot be mistaken for a drop-in replication of the published package.
REFERENCE_DEVIATIONS = [
    "Boundaries are found on the KR-normalised matrix straw returns, not on a "
    "user-supplied n x n matrix, so bin masking differs in detail.",
    "The R package evaluates each candidate k with cluster::silhouette; we use "
    "sklearn on the same 1/(1+Y) distances.",
]

# ── Hi-C loading ───────────────────────────────────────────────────────────────

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
    """Load a full intra-chromosomal Hi-C contact map as a scipy sparse matrix."""
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

    for rec in records:
        if hasattr(rec, "binX"):
            bx, by, v = int(rec.binX) // binsize, int(rec.binY) // binsize, float(rec.counts)
        else:
            bx, by, v = int(rec[0]) // binsize, int(rec[1]) // binsize, float(rec[2])

        if v <= 0 or not np.isfinite(v):
            continue

        rows.append(bx); cols.append(by); vals.append(v)
        if bx != by:
            rows.append(by); cols.append(bx); vals.append(v)
        max_bin = max(max_bin, bx, by)

    n_bins = max_bin + 1
    M = csr_matrix((vals, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float64)
    print(f"  Loaded:  {n_bins} bins  ({n_bins * binsize / 1e6:.1f} Mbp)  nnz={M.nnz:,}")
    return M, n_bins


# ── Window normalisation ──────────────────────────────────────────────────────

def normalize_window(
    Y: np.ndarray,
    method: str = "none",
    clip_percentile: float | None = None,
) -> np.ndarray:
    """
    Transform one window before boundary finding.

    `method="none"` (the default) leaves the KR-normalised counts alone. This
    matters: TV-SSC is not scale-invariant, because scaling Y scales the data
    term ‖Y−YX−E‖²_F quadratically while γ(‖DC‖₁+‖CD^T‖₁) is untouched. Under
    `log1p` the TV term dominates, the recovered split stops tracking the
    contact pattern and settles near the window midpoint (see
    `diagnose_boundary_recovery.py`). `zscore` has the same problem and also
    makes Y signed, which is meaningless for the spectral reference's degree
    normalisation.

    `clip_percentile` winsorises positive entries at that percentile. It is off
    by default; at 95 it removes most of the near-diagonal dynamic range, which
    is where the domain signal lives.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if clip_percentile:
        pos = Y[Y > 0]
        if pos.size:
            Y = np.minimum(Y, float(np.percentile(pos, clip_percentile)))
    if method == "log1p":
        return np.log1p(Y)
    if method == "zscore":
        return (Y - Y.mean()) / (Y.std() + 1e-12)
    return Y


# ── Segment / boundary utilities ──────────────────────────────────────────────

def labels_to_segments(pred: np.ndarray) -> list[tuple[int, int]]:
    """Convert a label array to (start, end) half-open runs of equal labels."""
    pred = np.asarray(pred)
    segs: list[tuple[int, int]] = []
    i = 0
    while i < len(pred):
        j = i + 1
        while j < len(pred) and pred[j] == pred[i]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def find_first_boundary(labels: np.ndarray, min_tad_bins: int = MIN_TAD_BINS) -> int | None:
    """
    Local position of the first label transition that leaves at least
    `min_tad_bins` on both sides, or None if there is no such transition.

    Spectral clustering can return non-contiguous labels, so a very short
    leading run is skipped rather than treated as "no boundary in this window".
    """
    W = len(labels)
    for start, _end in labels_to_segments(labels)[1:]:
        if min_tad_bins <= start <= W - min_tad_bins:
            return start
    return None


def tad_boundaries(tads: list[tuple[int, int]]) -> list[int]:
    """
    Internal boundary positions of a TAD list.

    Every interval edge counts, minus the two outermost. For a contiguous
    partition this is the usual "start of every TAD but the first". It also
    stays correct when a caller drops domains and leaves gaps, which the
    SpectralTAD reference does when a domain fails its size or silhouette
    filter.
    """
    if not tads:
        return []
    edges = sorted({t[0] for t in tads} | {t[1] for t in tads})
    return edges[1:-1]


# ── Shared sliding-window driver ──────────────────────────────────────────────

def _scan_region(
    cut_fn,
    start_bin: int,
    end_bin: int,
    window: int,
    min_tad_bins: int = MIN_TAD_BINS,
    verbose: bool = False,
) -> tuple[list[tuple[int, int]], list[float], list[dict]]:
    """
    SpectralTAD's sliding window, shared by both callers.

    `cut_fn(pos, end)` returns `(starts, scores)` where `starts` are the global
    positions at which a domain begins inside the window (always beginning with
    `pos`) and `scores` carries one number per domain for the caller's own
    post-processing. Returning `(None, None)` marks the window unusable.

    Every domain but the last is accepted and the window restarts at the last
    domain's start, so that domain is re-cut with full right-hand context. A
    window that reaches the end of the region keeps all of its domains.
    """
    pos      = start_bin
    tads:     list[tuple[int, int]] = []
    scores:   list[float]           = []
    scan_log: list[dict]            = []
    win_idx  = 0

    while pos < end_bin - min_tad_bins:
        win_idx += 1
        end = min(pos + window, end_bin)

        if end - pos < 2 * min_tad_bins:
            tads.append((pos, end_bin)); scores.append(0.0)
            scan_log.append(dict(pos=pos, end=end_bin, reason="too_small"))
            break

        starts, cut_scores = cut_fn(pos, end)

        if starts is None:
            scan_log.append(dict(pos=pos, end=end, reason="unusable"))
            pos += max(window // 4, 1)
            continue

        # Keep every domain but the last, unless the window is at the region
        # end or found only one domain (in which case there is nothing to hold
        # back and dropping it would stall the scan).
        at_end  = end >= end_bin
        n_keep  = len(starts) if (at_end or len(starts) < 2) else len(starts) - 1
        bounds  = list(starts) + [end]

        for i in range(n_keep):
            tads.append((bounds[i], bounds[i + 1]))
            scores.append(cut_scores[i] if cut_scores else 0.0)

        scan_log.append(dict(pos=pos, end=end, n_domains=len(starts),
                             n_kept=n_keep, reason="cut"))
        pos = bounds[n_keep] if bounds[n_keep] > pos else end

    if not tads:
        tads.append((start_bin, end_bin)); scores.append(0.0)
    elif tads[-1][1] < end_bin:
        tads.append((tads[-1][1], end_bin)); scores.append(0.0)

    if verbose:
        print(f"\n  Scan complete: {len(tads)} initial TADs  ({win_idx} windows)")
    return tads, scores, scan_log


def run_hierarchy(
    scan_fn,
    start_bin: int,
    end_bin: int,
    n_levels: int = 2,
    window: int = WINDOW,
    min_tad_bins: int = MIN_TAD_BINS,
    verbose: bool = True,
    sub_scan_fn=None,
) -> dict:
    """
    Apply `scan_fn(start, end, window) -> list[tad]` recursively.

    Level 1 covers the whole region; level L+1 re-scans inside each level-L
    domain at least `2 * min_tad_bins` wide, matching the R package's
    separability test.

    `sub_scan_fn`, if provided, is used for levels ≥ 2 in place of `scan_fn`.
    This is the correct hook to pass a z_clust=TRUE cutter for sub-levels,
    matching R/SpectralTAD.R which switches to eigenvector-gap z-score > 2 at
    levels ≥ 2.

    Returns `{1: list[tad], 2: {parent: list[tad]}, ...}`; a parent appears only
    if it actually split.
    """
    hierarchy: dict = {1: scan_fn(start_bin, end_bin, window)}
    _sub_fn = sub_scan_fn if sub_scan_fn is not None else scan_fn

    for level in range(2, n_levels + 1):
        parents = (
            hierarchy[1] if level == 2
            else [t for subs in hierarchy[level - 1].values() for t in subs]
        )
        children: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for idx, (s, e) in enumerate(parents):
            if e - s < 2 * min_tad_bins:
                continue
            if verbose:
                print(f"  [L{level}  {idx+1:4d}/{len(parents)}]  "
                      f"[{s}–{e})  {e-s} bins = {(e-s)*BINSIZE//1000} kbp", flush=True)
            subs = _sub_fn(s, e, min(window, e - s))
            if len(subs) > 1:
                children[(s, e)] = subs

        hierarchy[level] = children
        if verbose:
            n_sub = sum(len(v) for v in children.values())
            print(f"  Level {level}: {n_sub} domains inside "
                  f"{len(children)}/{len(parents)} parents\n")

    return hierarchy


def hierarchy_all_boundaries(hierarchy: dict) -> tuple[list[int], list[int]]:
    """Boundaries of level 1, and the de-duplicated boundaries of all sub-levels."""
    primary = tad_boundaries(hierarchy[1])
    secondary: set[int] = set()
    for level in sorted(k for k in hierarchy if k >= 2):
        for subs in hierarchy[level].values():
            secondary.update(tad_boundaries(subs))
    return primary, sorted(secondary - set(primary))


# ── Per-window cut selection (shared by both callers) ─────────────────────────

def unit_circle_gaps(A: np.ndarray) -> np.ndarray:
    """
    SpectralTAD's eigenvector gap profile for one window's affinity A.

    L̄ = D^{-½} A D^{-½} with D from rowSums(|A|); take the two
    largest-magnitude eigenpairs (PRIMME's `which = "LM"` default, which the R
    package relies on); project rows onto the unit circle; return the Euclidean
    distance between consecutive rows. `gaps[i]` is the distance between rows
    i-1 and i, so a cut at i starts a new domain at i. `gaps[0]` is NaN.

    R rescales each eigenvector to norm sqrt(n) before projecting. numpy's
    eigenvectors are already unit-norm, so that step is a common scalar and the
    row projection cancels it.
    """
    d   = np.abs(A).sum(axis=1)
    inv = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    L   = inv[:, None] * A * inv[None, :]
    L[~np.isfinite(L)] = 0.0

    try:
        vals, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.full(A.shape[0], np.nan)

    V = vecs[:, np.argsort(-np.abs(vals))[:2]]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    gaps = np.full(A.shape[0], np.nan)
    gaps[1:] = np.linalg.norm(np.diff(V, axis=0), axis=1)
    return gaps


def greedy_cut_candidates(gaps: np.ndarray, min_size: int, k_max: int) -> list[int]:
    """
    Walk the gap profile from largest to smallest, keeping a candidate cut only
    when it is more than `min_size` from every cut already kept. This is the R
    package's iterative accept/reject over `order(-point_dist)`, which does not
    require cuts to clear the window edges — undersized edge domains are removed
    later by the size filter instead.
    """
    order = np.argsort(np.where(np.isfinite(gaps), -gaps, np.inf))
    cuts: list[int] = []
    for idx in order:
        if not np.isfinite(gaps[idx]):
            break
        idx = int(idx)
        if any(abs(idx - c) <= min_size for c in cuts):
            continue
        cuts.append(idx)
        if len(cuts) >= k_max:
            break
    return cuts


def window_cuts(
    affinity: np.ndarray,
    contacts: np.ndarray,
    min_size: int = MIN_TAD_BINS,
) -> list[int]:
    """
    Local cut positions for one window.

    The embedding comes from `affinity`; the number of domains is chosen by
    silhouette on d = 1/(1+`contacts`), exactly as `.windowedSpec` does with
    `z_clust = FALSE` — evaluate k = 2, 3, … and stop at the first k that scores
    higher than k+1.

    Splitting the two arguments is what makes the benchmark fair: the reference
    passes the contact window as both, while the TV-SSC caller passes its
    self-expression coefficients as `affinity` and the same contact window as
    `contacts`, so both callers pick k under an identical criterion on identical
    data and only the embedding differs.
    """
    W    = affinity.shape[0]
    cand = greedy_cut_candidates(unit_circle_gaps(affinity), min_size,
                                 int(np.ceil(W / max(min_size, 1))))
    if not cand:
        return []

    D = 1.0 / (1.0 + contacts)
    np.fill_diagonal(D, 0.0)

    best_sil, best_cuts = None, []
    for k in range(2, len(cand) + 2):
        cuts   = sorted(cand[:k - 1])
        labels = np.zeros(W, dtype=int)
        for c in cuts:
            labels[c:] += 1
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        if best_sil is not None and sil < best_sil:
            break
        best_sil, best_cuts = sil, cuts

    return best_cuts


def window_cuts_z(
    affinity: np.ndarray,
    min_size: int = MIN_TAD_BINS,
    z_threshold: float = 2.0,
) -> list[int]:
    """
    Local cut positions using the z_clust=TRUE criterion (SpectralTAD levels ≥ 2).

    Positions where the eigenvector-gap z-score exceeds `z_threshold` are cut
    candidates, mirroring R/SpectralTAD.R's `.windowedSpec(z_clust = TRUE)`
    used at hierarchy levels ≥ 2. This replaces the silhouette sweep: it is
    faster and appropriate for sub-levels where the parent domain is already
    known to be divisible, so the silhouette's "is there any structure at all?"
    guard is less necessary.

    The same greedy min-size filter as `greedy_cut_candidates` is applied so
    spurious adjacent cuts are suppressed, walking in descending z-score order
    so the highest-confidence cuts are protected.
    """
    gaps = unit_circle_gaps(affinity)
    finite_gaps = gaps[np.isfinite(gaps)]
    if finite_gaps.size < 2:
        return []

    mu, sd = float(finite_gaps.mean()), float(finite_gaps.std())
    if sd < 1e-12:
        return []

    z = (gaps - mu) / sd
    cands = [i for i in range(1, len(gaps))
             if np.isfinite(z[i]) and z[i] > z_threshold]
    if not cands:
        return []

    cands_sorted = sorted(cands, key=lambda i: -float(z[i]))
    cuts: list[int] = []
    for c in cands_sorted:
        if any(abs(c - kept) <= min_size for kept in cuts):
            continue
        cuts.append(c)
    return cuts


# ── TV-SSC affinity ───────────────────────────────────────────────────────────

def tv_affinity(
    Y: np.ndarray,
    source: str = "C",
    lambda_e: float = 1.0,
    lambda_z: float = 0.1,
    gamma: float = 0.5,
    mu: float = 1.0,
    sigma: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Solve the TV-SSC problem on one window and return
    `(|A| + |A|ᵀ, R)` — the symmetrised coefficient affinity to embed, and the
    reconstruction residual R = Y − YX. `(None, None)` if the window is empty.

    `source` selects which coefficient matrix becomes the affinity: "C" carries
    the TV penalty directly, "X" is its off-diagonal projection.

    λ_e and λ_z are divided by σ₁(Y) so the soft-threshold they induce on E means
    the same thing in windows of differing sequencing depth; without it the
    solver's behaviour tracks read depth rather than structure.
    """
    sigma1 = float(randomized_svd(Y, n_components=1, random_state=0)[1][0])
    if sigma1 < 1e-10:
        return None, None

    X, C, _E = ssc_admm_nuc_tv(
        Y, lambda_e=lambda_e / sigma1, lambda_z=lambda_z / sigma1, gamma=gamma,
        mu=mu, sigma=sigma, max_iter=max_iter, tol=tol,
    )
    A = C if source == "C" else X
    return np.abs(A) + np.abs(A.T), Y - Y @ X


# ── SpectralTAD reference caller ──────────────────────────────────────────────

def spectral_cut_fn(M_sparse: csr_matrix, min_tad_bins: int, norm: str,
                    clip_percentile: float | None, verbose: bool = False,
                    z_clust: bool = False):
    """
    Build the per-window SpectralTAD cutter consumed by `_scan_region`.

    `z_clust=True` switches k-selection from silhouette to eigenvector-gap
    z-score > 2, matching R/SpectralTAD.R's `z_clust=TRUE` used at levels ≥ 2.
    """

    def cut(pos: int, end: int):
        Y_raw = M_sparse[pos:end, pos:end].toarray()
        # gap_threshold = 1: drop rows/columns that are entirely zero.
        keep = np.flatnonzero(Y_raw.any(axis=0))
        if keep.size < 2 * min_tad_bins:
            return None, None

        Y = normalize_window(Y_raw[np.ix_(keep, keep)], norm, clip_percentile)
        if not np.isfinite(Y).all() or Y.max() <= 0:
            return None, None

        if z_clust:
            cuts = window_cuts_z(Y, min_tad_bins)
        else:
            cuts = window_cuts(Y, Y, min_tad_bins)
        starts = sorted({pos} | {pos + int(keep[c]) for c in cuts})
        if verbose:
            print(f"  bins [{pos:6d}–{end:6d})  {len(starts)} domains", flush=True)
        return starts, None

    return cut


def silhouette_postprocess(
    tads: list[tuple[int, int]],
    M_sparse: csr_matrix,
    min_tad_bins: int = MIN_TAD_BINS,
    mode: str = "none",
    sil_threshold: float = SIL_THRESHOLD,
    verbose: bool = True,
) -> list[tuple[int, int]]:
    """
    Silhouette post-processing, applied to whichever caller produced `tads`.

    mode="none"
        The package default, `SpectralTAD(qual_filter = FALSE)`. Keeps the scan's
        contiguous partition, so bin coverage is identical between callers.
    mode="qual_filter"
        What `qual_filter = TRUE` does: score every domain by its mean silhouette
        over the whole partition under d = 1/(1+M) and *discard* domains scoring
        at or below `QUAL_SIL` (0.25 in Cresswell et al. Additional file 1;
        the R package filters at 0.15). Discarding leaves gaps, so the result
        is a domain list rather than a partition — that is the real package's
        output shape too, but it also removes bins from the comparison.
    mode="merge"
        Merge adjacent domains whose pairwise 2-cluster silhouette is below
        `sil_threshold`, repeating until a sweep changes nothing.

    The minimum-size filter is applied in all modes, as in the R package.
    """
    n_in = len(tads)

    if mode == "merge" and len(tads) > 1:
        n_merges, passes, changed = 0, 0, True
        while changed:
            changed, passes = False, passes + 1
            merged = [tads[0]]
            for B in tads[1:]:
                A = merged[-1]
                if _pair_silhouette(M_sparse, A, B) < sil_threshold:
                    merged[-1] = (A[0], B[1])
                    changed, n_merges = True, n_merges + 1
                else:
                    merged.append(B)
            tads = merged
        if verbose:
            print(f"  silhouette merge ({passes} pass(es)):  {n_merges} merges")

    elif mode == "qual_filter" and len(tads) > 1:
        keep = [t for t, s in zip(tads, _domain_silhouettes(M_sparse, tads))
                if s > QUAL_SIL]
        if verbose:
            print(f"  qual_filter (mean silhouette > {QUAL_SIL}):  "
                  f"{len(tads) - len(keep)} domains dropped")
        tads = keep

    tads = [t for t in tads if t[1] - t[0] >= min_tad_bins]
    if verbose:
        print(f"  min-size filter (≥ {min_tad_bins} bins):  {n_in} → {len(tads)} domains")
    return tads


def _pair_silhouette(M_sparse: csr_matrix, tad_a: tuple[int, int],
                     tad_b: tuple[int, int]) -> float:
    """Mean 2-cluster silhouette for A ∪ B under d(i,j) = 1/(1+M_ij)."""
    (a0, a1), (b0, b1) = tad_a, tad_b
    n_a, n_b = a1 - a0, b1 - b0
    if n_a < 2 or n_b < 2:
        return 0.0

    D = 1.0 / (1.0 + M_sparse[a0:b1, a0:b1].toarray())
    np.fill_diagonal(D, 0.0)
    labels = np.array([0] * n_a + [1] * n_b, dtype=int)
    try:
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return 0.0


def _domain_silhouettes(
    M_sparse: csr_matrix,
    tads: list[tuple[int, int]],
    block: int = 512,
) -> list[float]:
    """
    Per-domain mean silhouette over the whole partition under d = 1/(1+M), as
    the R package's `qual_filter` computes it, restricted to the bins the
    domains cover.

    The full distance matrix is dense (1/(1+0) = 1), so a chromosome-wide
    partition would need ~n² entries at once. Domains are contiguous runs of
    bins, so the per-cluster distance sums each silhouette needs are accumulated
    with `reduceat` over row blocks instead of materialising the matrix.
    """
    if len(tads) < 2:
        return [1.0] * len(tads)

    bins    = np.concatenate([np.arange(s, e) for s, e in tads])
    sizes   = np.array([e - s for s, e in tads])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    labels  = np.repeat(np.arange(len(tads)), sizes)

    sums = np.empty((len(bins), len(tads)), dtype=np.float64)
    for lo in range(0, len(bins), block):
        hi   = min(lo + block, len(bins))
        rows = M_sparse[bins[lo:hi], :][:, bins].toarray()
        sums[lo:hi] = np.add.reduceat(1.0 / (1.0 + rows), offsets, axis=1)

    # The diagonal of 1/(1+M) is not zero, so remove each bin's own term before
    # averaging over the rest of its domain.
    self_d = 1.0 / (1.0 + np.asarray(M_sparse[bins, bins]).ravel())
    own    = sums[np.arange(len(bins)), labels] - self_d
    n_own  = sizes[labels]
    a      = np.where(n_own > 1, own / np.maximum(n_own - 1, 1), 0.0)
    others = sums / sizes[None, :]
    others[np.arange(len(bins)), labels] = np.inf
    b      = others.min(axis=1)

    s = np.where(n_own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return [float(s[labels == i].mean()) for i in range(len(tads))]


def run_spectral_tad_python(
    M_sparse: csr_matrix,
    n_bins: int,
    start_bin: int = 0,
    end_bin: int | None = None,
    window: int = WINDOW,
    min_tad_bins: int = MIN_TAD_BINS,
    norm: str = "none",
    clip_percentile: float | None = None,
    post: str = "none",
    sil_threshold: float = SIL_THRESHOLD,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """
    Python reference implementation of SpectralTAD, run through the same
    sliding-window driver as the TV-SSC caller. See `REFERENCE_DEVIATIONS`.
    """
    scan_end = n_bins if end_bin is None else end_bin
    if verbose:
        print(f"\nSpectralTAD reference scan  window={window} bins  "
              f"region=[{start_bin}, {scan_end})\n")

    cut = spectral_cut_fn(M_sparse, min_tad_bins, norm, clip_percentile,
                          verbose=verbose)
    tads, _scores, scan_log = _scan_region(
        cut, start_bin, scan_end, window, min_tad_bins, verbose=verbose
    )
    tads = silhouette_postprocess(tads, M_sparse, min_tad_bins, post,
                                  sil_threshold, verbose=verbose)
    return tads, scan_log


def build_spectral_scan_fn(
    M_sparse: csr_matrix,
    min_tad_bins: int = MIN_TAD_BINS,
    norm: str = "none",
    clip_percentile: float | None = None,
    post: str = "none",
    sil_threshold: float = SIL_THRESHOLD,
    verbose: bool = False,
    z_clust: bool = False,
):
    """
    Return a `scan_fn(start, end, window) -> list[tad]` for SpectralTAD.

    The returned closure wraps `spectral_cut_fn` + `_scan_region` +
    `silhouette_postprocess` and is suitable as either the primary or the
    `sub_scan_fn` argument to `run_hierarchy`. Pass `z_clust=True` for the
    sub-level scanner to match R/SpectralTAD.R's levels ≥ 2 behaviour.
    """
    def scan(start: int, end: int, window: int) -> list[tuple[int, int]]:
        cut = spectral_cut_fn(M_sparse, min_tad_bins, norm, clip_percentile,
                              verbose=verbose, z_clust=z_clust)
        tads, _, _ = _scan_region(cut, start, end, window, min_tad_bins)
        tads = silhouette_postprocess(tads, M_sparse, min_tad_bins, post,
                                      sil_threshold, verbose=False)
        return tads

    return scan


# ── Insulation score ──────────────────────────────────────────────────────────

def compute_insulation_score(
    M_sparse: csr_matrix,
    n_bins: int,
    delta: int = 25,
) -> np.ndarray:
    """
    Crane et al. (2015) insulation score, reported as log₂(score / mean).

    score(i) = Σ_{j∈[i−δ,i), k∈[i,i+δ)} M[j,k]; minima mark boundaries. Each
    upper-triangular contact (j,k) within the 2δ band contributes to every i in
    [max(j+1, k−δ+1), min(j+δ, k)], so the sum is accumulated on a difference
    array in one vectorised pass. Bins with no signal come back NaN.
    """
    coo  = M_sparse.tocoo()
    dist = coo.col.astype(np.int64) - coo.row.astype(np.int64)
    band = (dist > 0) & (dist <= 2 * delta)
    j, k, v = coo.row[band].astype(np.int64), coo.col[band].astype(np.int64), coo.data[band]

    i_lo  = np.maximum(j + 1, k - delta + 1)
    i_hi  = np.minimum(j + delta, k)
    valid = (i_lo <= i_hi) & (i_lo < n_bins)
    i_lo, i_hi, v = i_lo[valid], np.minimum(i_hi[valid], n_bins - 1), v[valid]

    diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff, i_lo, v)
    np.add.at(diff, i_hi + 1, -v)
    scores = np.cumsum(diff[:-1])

    pos = scores > 0
    if not pos.any():
        return np.full(n_bins, np.nan)
    out = np.full(n_bins, np.nan)
    out[pos] = np.log2(scores[pos] / scores[pos].mean())
    return out


def insulation_at_boundaries(
    insulation: np.ndarray,
    bounds: list[int],
    lo: int,
    hi: int,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Score a boundary set by insulation, against a circular-shift null.

    Mean insulation at a boundary set falls monotonically as boundaries are
    removed (only the deepest valleys survive), so the raw mean cannot be
    compared between callers that disagree on how many boundaries to call.
    Shifting the whole set circularly inside [lo, hi) preserves both the count
    and the spacing distribution while destroying the alignment to real
    structure, which gives a comparable z-score and empirical p-value.

    Also reports `frac_local_min`: the share of boundaries sitting at the
    minimum of their own ±`delta_local` neighbourhood.
    """
    span   = hi - lo
    scored = [b for b in bounds if lo <= b < hi and np.isfinite(insulation[b])]
    empty  = dict(n_bounds=len(bounds), n_scored=0, mean=np.nan, median=np.nan,
                  null_mean=np.nan, null_sd=np.nan, z=np.nan, p=np.nan,
                  frac_local_min=np.nan)
    if not scored or span <= 1:
        return empty

    obs  = float(np.mean([insulation[b] for b in scored]))
    barr = np.asarray(bounds)
    rng  = np.random.default_rng(seed)

    null = []
    for shift in rng.integers(1, span, n_perm):
        shifted = (barr - lo + shift) % span + lo
        vals    = insulation[shifted]
        vals    = vals[np.isfinite(vals)]
        if vals.size:
            null.append(vals.mean())
    if not null:
        return {**empty, "n_scored": len(scored), "mean": obs}

    null = np.asarray(null)
    sd   = float(null.std())

    delta_local = 3
    local = [
        insulation[b] <= np.nanmin(insulation[max(b - delta_local, 0):b + delta_local + 1])
        for b in scored
    ]

    return dict(
        n_bounds=len(bounds),
        n_scored=len(scored),
        mean=obs,
        median=float(np.median([insulation[b] for b in scored])),
        null_mean=float(null.mean()),
        null_sd=sd,
        z=float((obs - null.mean()) / sd) if sd > 0 else np.nan,
        p=float((null <= obs).mean()),
        frac_local_min=float(np.mean(local)),
    )


# ── Boundary agreement ────────────────────────────────────────────────────────

def boundary_agreement(bounds_a: list[int], bounds_b: list[int], tol: int = 3) -> dict:
    """
    Symmetric agreement between two boundary sets, matched within `tol` bins.

    Reported in both directions because neither caller is ground truth:
    `frac_a_matched` is the share of A's boundaries with a partner in B and
    `frac_b_matched` the reverse. Jaccard uses the unmatched counts from both
    sides, so it is not inflated when one caller emits far more boundaries.
    """
    n_a, n_b = len(bounds_a), len(bounds_b)
    if not n_a and not n_b:
        return dict(jaccard=1.0, frac_a_matched=1.0, frac_b_matched=1.0, f1=1.0,
                    n_a=0, n_b=0, matched_a=0, matched_b=0)
    if not n_a or not n_b:
        return dict(jaccard=0.0, frac_a_matched=0.0, frac_b_matched=0.0, f1=0.0,
                    n_a=n_a, n_b=n_b, matched_a=0, matched_b=0)

    a_arr, b_arr = np.array(sorted(bounds_a)), np.array(sorted(bounds_b))
    matched_a = int(np.sum([np.any(np.abs(b_arr - a) <= tol) for a in a_arr]))
    matched_b = int(np.sum([np.any(np.abs(a_arr - b) <= tol) for b in b_arr]))

    frac_a = matched_a / n_a
    frac_b = matched_b / n_b
    union  = matched_a + (n_a - matched_a) + (n_b - matched_b)
    return dict(
        jaccard=matched_a / union if union else 0.0,
        frac_a_matched=frac_a,
        frac_b_matched=frac_b,
        f1=2 * frac_a * frac_b / (frac_a + frac_b) if (frac_a + frac_b) else 0.0,
        n_a=n_a, n_b=n_b, matched_a=matched_a, matched_b=matched_b,
    )


def tad_statistics(tads: list[tuple[int, int]]) -> dict:
    bounds    = tad_boundaries(tads)
    sizes_kbp = [(e - s) * BINSIZE / 1000.0 for s, e in tads]
    covered   = sum(e - s for s, e in tads)
    return dict(
        n_tads       = len(tads),
        n_boundaries = len(bounds),
        median_kbp   = float(np.median(sizes_kbp)) if sizes_kbp else 0.0,
        mean_kbp     = float(np.mean(sizes_kbp))   if sizes_kbp else 0.0,
        covered_bins = covered,
        boundaries   = bounds,
    )


def compare_methods(
    tv_tads: list[tuple[int, int]],
    spec_tads: list[tuple[int, int]],
    M_sparse: csr_matrix,
    n_bins: int,
    start_bin: int = 0,
    insulation_delta: int = 25,
    tol: int = 3,
    n_perm: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Compare the two callers' boundary sets: symmetric agreement plus insulation
    enrichment against a circular-shift null for each.
    """
    tv_stats, spec_stats = tad_statistics(tv_tads), tad_statistics(spec_tads)
    agree = boundary_agreement(tv_stats["boundaries"], spec_stats["boundaries"], tol)

    print(f"  Computing insulation score (δ={insulation_delta} bins) ...", flush=True)
    insulation = compute_insulation_score(M_sparse, n_bins, insulation_delta)
    tv_ins   = insulation_at_boundaries(insulation, tv_stats["boundaries"],
                                        start_bin, n_bins, n_perm)
    spec_ins = insulation_at_boundaries(insulation, spec_stats["boundaries"],
                                        start_bin, n_bins, n_perm)

    if verbose:
        def row(label, a, b, fmt="{:>12}"):
            print(f"  {label:<32} {fmt.format(a):>12}  {fmt.format(b):>12}")

        print(f"\n{'═'*62}")
        print(f"  Benchmark  (boundary tolerance {tol} bins = {tol*BINSIZE//1000} kbp)")
        print(f"  {'':<32} {'TV-SSC':>12}  {'SpectralTAD':>12}")
        print(f"  {'─'*58}")
        row("Domains", tv_stats['n_tads'], spec_stats['n_tads'])
        row("Boundaries", tv_stats['n_boundaries'], spec_stats['n_boundaries'])
        row("Median domain (kbp)", f"{tv_stats['median_kbp']:.0f}",
            f"{spec_stats['median_kbp']:.0f}")
        row("Mean domain (kbp)", f"{tv_stats['mean_kbp']:.0f}",
            f"{spec_stats['mean_kbp']:.0f}")
        row("Bins covered", tv_stats['covered_bins'], spec_stats['covered_bins'])
        print(f"  {'─'*58}")
        print("  Insulation at boundaries vs circular-shift null")
        row("  mean log₂ insulation", f"{tv_ins['mean']:.3f}", f"{spec_ins['mean']:.3f}")
        row("  null mean", f"{tv_ins['null_mean']:.3f}", f"{spec_ins['null_mean']:.3f}")
        row("  z-score", f"{tv_ins['z']:.2f}", f"{spec_ins['z']:.2f}")
        row("  empirical p", f"{tv_ins['p']:.4f}", f"{spec_ins['p']:.4f}")
        row("  frac at local minimum", f"{tv_ins['frac_local_min']:.3f}",
            f"{spec_ins['frac_local_min']:.3f}")
        print(f"  {'─'*58}")
        print(f"  Boundary agreement (symmetric, neither is ground truth)")
        print(f"    Jaccard                        {agree['jaccard']:.3f}")
        print(f"    TV boundaries matched in Spec   {agree['frac_a_matched']:.3f}"
              f"  ({agree['matched_a']}/{agree['n_a']})")
        print(f"    Spec boundaries matched in TV   {agree['frac_b_matched']:.3f}"
              f"  ({agree['matched_b']}/{agree['n_b']})")
        print(f"    F1 of the two match rates       {agree['f1']:.3f}")
        print(f"{'═'*62}\n")
        print("  Caveat: mean insulation improves automatically as boundaries are")
        print("  dropped, so compare the z-scores (count-matched) rather than the")
        print("  raw means when the two callers disagree on boundary count.\n")

    return dict(
        agreement=agree,
        tv_stats=tv_stats,
        spec_stats=spec_stats,
        insulation=insulation,
        tv_insulation=tv_ins,
        spec_insulation=spec_ins,
    )


# ── Visualisation ─────────────────────────────────────────────────────────────

def _heatmap_panel(
    ax: plt.Axes,
    Y_log: np.ndarray,
    tads: list[tuple[int, int]],
    vis_end: int,
    color: str = "#3498db",
    alt_color: str = "#2ecc71",
    label: str = "",
) -> None:
    """Hi-C heatmap with TAD blocks overlaid on the diagonal."""
    pos  = Y_log[Y_log > 0]
    vmin = float(np.percentile(pos, 5))  if pos.size else 0.0
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    ax.imshow(Y_log, cmap="Reds", aspect="auto", interpolation="nearest",
              vmin=vmin, vmax=vmax)

    palette = [color, alt_color]
    for i, (s, e) in enumerate(tads):
        ec  = min(e, vis_end)
        col = palette[i % 2]
        ax.add_patch(mpatches.Rectangle((s - 0.5, s - 0.5), ec - s, ec - s,
                                        linewidth=1.2, edgecolor=col,
                                        facecolor=col, alpha=0.12))
        for b in (s, ec):
            if 0 < b < vis_end:
                ax.axhline(b - 0.5, color="navy", lw=0.5, alpha=0.5)
                ax.axvline(b - 0.5, color="navy", lw=0.5, alpha=0.5)

    ax.set_xlim(-0.5, vis_end - 0.5)
    ax.set_ylim(vis_end - 0.5, -0.5)
    ax.set_xlabel("Genomic bin (10 kbp)")
    ax.set_ylabel("Genomic bin (10 kbp)")
    if label:
        ax.set_title(label, fontsize=10)


def _coverage_panel(ax, M_sparse, vis_end, tracks, title):
    """Coverage track with one vertical-line series per (bounds, colour, label)."""
    coverage = np.asarray(M_sparse[:vis_end, :].sum(axis=1)).ravel()
    ax.fill_between(range(vis_end), coverage, color="lightgray", alpha=0.7, lw=0)
    handles = []
    for bounds, color, label, lw in tracks:
        for b in bounds:
            ax.axvline(b, color=color, lw=lw, alpha=0.65)
        handles.append(mpatches.Patch(color=color, label=f"{label} ({len(bounds)})"))
    ax.set_xlim(0, vis_end)
    ax.set_xlabel("Genomic bin (10 kbp)")
    ax.set_ylabel("Sum of contacts")
    ax.set_title(title, fontsize=10)
    ax.legend(handles=handles, fontsize=9)


def plot_results(M_sparse, n_bins, tads, stats, chrom, out_png, max_vis_bins=3000):
    """Coverage track plus heatmap with the TAD partition."""
    vis_end = min(n_bins, max_vis_bins)
    Y_log   = np.log1p(M_sparse[:vis_end, :vis_end].toarray())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    _coverage_panel(
        axes[0], M_sparse, vis_end,
        [([b for b in stats["boundaries"] if b < vis_end], "#e74c3c", "TV-SSC boundary", 0.8)],
        f"Coverage + boundaries  ({chrom}, first {vis_end*BINSIZE//1_000_000} Mbp)",
    )
    _heatmap_panel(axes[1], Y_log, [t for t in tads if t[0] < vis_end], vis_end,
                   label=f"Hi-C (log1p) + TV-SSC partition  ({chrom})")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"TV-SSC TAD detection  |  {stats['n_tads']} domains  "
                 f"{stats['n_boundaries']} boundaries  "
                 f"median={stats['median_kbp']:.0f} kbp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_png}")
    plt.close(fig)


def plot_hierarchy_results(M_sparse, n_bins, hierarchy, chrom, out_png,
                           max_vis_bins=2000):
    """Level-1 heatmap, sub-level heatmap, and a coverage track with both."""
    vis_end = min(n_bins, max_vis_bins)
    Y_log   = np.log1p(M_sparse[:vis_end, :vis_end].toarray())

    primary = [t for t in hierarchy[1] if t[0] < vis_end]
    all_sub = [t for level in sorted(k for k in hierarchy if k >= 2)
               for subs in hierarchy[level].values()
               for t in subs if t[0] < vis_end]
    p_bounds, s_bounds = hierarchy_all_boundaries(hierarchy)
    vis_p = [b for b in p_bounds if b < vis_end]
    vis_s = [b for b in s_bounds if b < vis_end]

    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    _heatmap_panel(axes[0], Y_log, primary, vis_end, "#e74c3c", "#e67e22",
                   f"Level 1 – primary domains (n={len(hierarchy[1])})")
    _heatmap_panel(axes[1], Y_log, all_sub, vis_end, "#3498db", "#2ecc71",
                   f"Sub-levels – domains (n={len(all_sub)})")
    for b in vis_p:
        axes[1].axhline(b - 0.5, color="red", lw=1.0, alpha=0.7)
        axes[1].axvline(b - 0.5, color="red", lw=1.0, alpha=0.7)

    _coverage_panel(
        axes[2], M_sparse, vis_end,
        [(vis_p, "red", "Level 1", 1.2), (vis_s, "#3498db", "Sub-levels", 0.7)],
        f"Coverage + hierarchy boundaries  ({chrom}, first {vis_end*BINSIZE//1_000_000} Mbp)",
    )

    n_sub = sum(len(v) for k in hierarchy if k >= 2 for v in hierarchy[k].values())
    fig.suptitle(f"Hierarchical TV-SSC  |  {chrom}  |  "
                 f"L1={len(hierarchy[1])} domains  sub={n_sub}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved hierarchy figure → {out_png}")
    plt.close(fig)


def plot_benchmark(M_sparse, n_bins, tv_tads, spec_tads, comparison, chrom,
                   out_png, max_vis_bins=2000):
    """Insulation track with both boundary sets, plus one heatmap per caller."""
    vis_end    = min(n_bins, max_vis_bins)
    Y_log      = np.log1p(M_sparse[:vis_end, :vis_end].toarray())
    insulation = comparison["insulation"]
    agree      = comparison["agreement"]
    tv_i, sp_i = comparison["tv_insulation"], comparison["spec_insulation"]

    tv_bounds   = [b for b in comparison["tv_stats"]["boundaries"]   if b < vis_end]
    spec_bounds = [b for b in comparison["spec_stats"]["boundaries"] if b < vis_end]

    fig, axes = plt.subplots(1, 3, figsize=(30, 9))

    ax      = axes[0]
    ins_vis = insulation[:vis_end]
    ax.plot(np.arange(vis_end), ins_vis, color="black", lw=0.8, alpha=0.9)
    for b in tv_bounds:
        ax.axvline(b, color="#e74c3c", lw=0.8, alpha=0.6)
    for b in spec_bounds:
        ax.axvline(b, color="#3498db", lw=0.8, alpha=0.6, ls="--")
    finite = ins_vis[np.isfinite(ins_vis)]
    if finite.size:
        pad = 0.1 * (finite.max() - finite.min() + 1e-9)
        ax.set_ylim(finite.min() - pad, finite.max() + pad)
    ax.set_xlim(0, vis_end)
    ax.set_xlabel("Genomic bin (10 kbp)")
    ax.set_ylabel("Insulation score (log₂)")
    ax.set_title(f"Insulation  ({chrom})\n"
                 f"TV z={tv_i['z']:.1f}  Spectral z={sp_i['z']:.1f}"
                 f"   (vs circular-shift null)", fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label=f"TV-SSC ({len(tv_bounds)} bdry)"),
        mpatches.Patch(color="#3498db", label=f"SpectralTAD ({len(spec_bounds)} bdry)"),
    ], fontsize=9)

    for ax, tads, colors, stats, name in (
        (axes[1], tv_tads,   ("#e74c3c", "#e67e22"), comparison["tv_stats"],   "TV-SSC"),
        (axes[2], spec_tads, ("#3498db", "#2ecc71"), comparison["spec_stats"], "SpectralTAD"),
    ):
        _heatmap_panel(ax, Y_log, [t for t in tads if t[0] < vis_end], vis_end,
                       *colors,
                       f"{name}  |  {stats['n_tads']} domains  "
                       f"median={stats['median_kbp']:.0f} kbp")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"TV-SSC vs SpectralTAD  |  {chrom}  |  "
                 f"Jaccard={agree['jaccard']:.3f}  "
                 f"TV matched={agree['frac_a_matched']:.3f}  "
                 f"Spec matched={agree['frac_b_matched']:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved benchmark figure → {out_png}")
    plt.close(fig)


# ── BED export ────────────────────────────────────────────────────────────────

def write_bed(tads, chrom, out_bed, track_name="TV_SSC_TADs", description=""):
    """Write a domain list as a BED file."""
    desc = description or f"TV-SSC TAD domains ({chrom})"
    with open(out_bed, "w") as f:
        f.write(f"track name='{track_name}' description='{desc}'\n")
        for i, (s, e) in enumerate(tads):
            f.write(f"{chrom}\t{s * BINSIZE}\t{e * BINSIZE}\tTAD_{i+1}\n")
    print(f"Saved BED  → {out_bed}  ({len(tads)} domains)")


def write_hierarchy_bed(hierarchy, chrom, out_prefix):
    """One BED per hierarchy level."""
    write_bed(hierarchy[1], chrom, f"{out_prefix}_L1.bed", "TV_SSC_L1",
              f"TV-SSC level-1 domains ({chrom})")
    for level in sorted(k for k in hierarchy if k >= 2):
        subs = sorted(t for v in hierarchy[level].values() for t in v)
        if subs:
            write_bed(subs, chrom, f"{out_prefix}_L{level}.bed", f"TV_SSC_L{level}",
                      f"TV-SSC level-{level} domains ({chrom})")


