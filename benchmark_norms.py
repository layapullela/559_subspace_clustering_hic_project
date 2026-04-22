"""
Benchmark: L1-norm vs Nuclear-norm vs Nuclear+Laplacian SSC
===========================================================
Generates a sweep of random block-diagonal adjacency matrices with varying
difficulty (cluster count, size, within/between edge probabilities) and
compares three ADMM formulations on subspace clustering quality (ARI).
"""

import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

from ssc_admm import (
    generate_block_diagonal_matrix as gen_block,
    ssc_admm as ssc_l1,
    cluster_from_C,
)
from ssc_admm_nuc import ssc_admm as ssc_nuc
from ssc_admm_nuc_lap import ssc_admm_nuc_lap

# ── Test-case definitions ─────────────────────────────────────────────────────

TEST_CASES = [
    # --- Vary separation (p_in fixed, p_out increasing → harder) ---
    {
        "name": "Easy separation",
        "cluster_sizes": [20, 25, 15, 20],
        "p_in": 0.70,
        "p_out": 0.02,
    },
    {
        "name": "Moderate separation",
        "cluster_sizes": [20, 25, 15, 20],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    {
        "name": "Hard separation",
        "cluster_sizes": [20, 25, 15, 20],
        "p_in": 0.35,
        "p_out": 0.10,
    },
    {
        "name": "Very hard separation",
        "cluster_sizes": [20, 25, 15, 20],
        "p_in": 0.30,
        "p_out": 0.15,
    },
    # --- Vary number of clusters (fixed total N ≈ 80) ---
    {
        "name": "2 clusters",
        "cluster_sizes": [40, 40],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    {
        "name": "4 clusters",
        "cluster_sizes": [20, 20, 20, 20],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    {
        "name": "6 clusters",
        "cluster_sizes": [14, 13, 14, 13, 13, 13],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    {
        "name": "8 clusters",
        "cluster_sizes": [10] * 8,
        "p_in": 0.50,
        "p_out": 0.05,
    },
    # --- Imbalanced clusters ---
    {
        "name": "Imbalanced (mild)",
        "cluster_sizes": [30, 20, 15, 15],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    {
        "name": "Imbalanced (severe)",
        "cluster_sizes": [50, 15, 10, 5],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    # --- Larger matrix ---
    {
        "name": "Large (N=160)",
        "cluster_sizes": [40, 40, 40, 40],
        "p_in": 0.50,
        "p_out": 0.05,
    },
    # --- Dense within-cluster ---
    {
        "name": "Dense blocks, low noise",
        "cluster_sizes": [20, 20, 20, 20],
        "p_in": 0.90,
        "p_out": 0.02,
    },
]

# Shared ADMM hyper-parameters
L1_PARAMS      = dict(lambda_e=1.0, lambda_z=10.0, mu=1.0, max_iter=500, tol=1e-4)
NUC_PARAMS     = dict(lambda_e=1.0, lambda_z=0.1,  mu=1.0, max_iter=500, tol=1e-4)
NUC_LAP_PARAMS = dict(lambda_e=1.0, lambda_z=0.1, gamma=0.1, mu=1.0, rho=1.0, max_iter=500, tol=1e-4)

METHODS = ["l1", "nuc", "nuc_lap"]
METHOD_LABELS = {"l1": "L1", "nuc": "Nuc", "nuc_lap": "Nuc+Lap"}

NUM_SEEDS = 5  # random repetitions per test case


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(Y, true_labels, k, method="l1"):
    """Run a single solver, return (ARI, elapsed_seconds, sparsity)."""
    t0 = time.perf_counter()
    if method == "l1":
        X, _C, _E = ssc_l1(Y, **L1_PARAMS)
    elif method == "nuc":
        X, _C, _J, _E = ssc_nuc(Y, **NUC_PARAMS)
    elif method == "nuc_lap":
        X, _J, _C, _E = ssc_admm_nuc_lap(Y, **NUC_LAP_PARAMS)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.perf_counter() - t0

    pred = cluster_from_C(X, k)
    ari = adjusted_rand_score(true_labels, pred)
    sparsity = np.mean(np.abs(X) < 1e-6)
    return ari, elapsed, sparsity


def run_all_cases(verbose=True):
    """Run every test case × seed × method.  Returns list of result dicts."""
    results = []
    for tc in TEST_CASES:
        k = len(tc["cluster_sizes"])
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {tc['name']}  (k={k}, N={sum(tc['cluster_sizes'])}, "
                  f"p_in={tc['p_in']}, p_out={tc['p_out']})")
            print(f"{'='*60}")

        for seed in range(NUM_SEEDS):
            Y, labels = gen_block(
                tc["cluster_sizes"],
                p_in=tc["p_in"],
                p_out=tc["p_out"],
                seed=seed,
            )
            for method in METHODS:
                ari, elapsed, sparsity = run_one(Y, labels, k, method)
                results.append(dict(
                    case=tc["name"],
                    k=k,
                    N=sum(tc["cluster_sizes"]),
                    p_in=tc["p_in"],
                    p_out=tc["p_out"],
                    seed=seed,
                    method=method,
                    ari=ari,
                    time=elapsed,
                    sparsity=sparsity,
                ))
                if verbose:
                    tag = METHOD_LABELS[method].ljust(7)
                    print(f"  seed={seed}  {tag}  ARI={ari:.3f}  "
                          f"time={elapsed:.2f}s  sparsity={sparsity:.2%}")
    return results


# ── Summary & plotting ────────────────────────────────────────────────────────

def summarize(results):
    """Aggregate ARI across seeds for each (case, method)."""
    cases = list(dict.fromkeys(r["case"] for r in results))
    rows = []
    for case in cases:
        for method in METHODS:
            aris = [r["ari"] for r in results
                    if r["case"] == case and r["method"] == method]
            times = [r["time"] for r in results
                     if r["case"] == case and r["method"] == method]
            rows.append(dict(
                case=case,
                method=method,
                ari_mean=np.mean(aris),
                ari_std=np.std(aris),
                time_mean=np.mean(times),
            ))
    return cases, rows


def print_table(cases, rows):
    hdr = (f"{'Test case':<28} {'L1 ARI':>12} {'Nuc ARI':>12} "
           f"{'Nuc+Lap ARI':>12} {'Winner':>8}")
    print(f"\n{'─'*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")
    for case in cases:
        by_method = {m: next(r for r in rows
                             if r["case"] == case and r["method"] == m)
                     for m in METHODS}
        strs = {m: f"{by_method[m]['ari_mean']:.3f}±{by_method[m]['ari_std']:.3f}"
                for m in METHODS}
        best_m = max(METHODS, key=lambda m: by_method[m]["ari_mean"])
        runner_up = max((m for m in METHODS if m != best_m),
                        key=lambda m: by_method[m]["ari_mean"])
        if abs(by_method[best_m]["ari_mean"] - by_method[runner_up]["ari_mean"]) < 0.01:
            winner = "tie"
        else:
            winner = METHOD_LABELS[best_m]
        print(f"{case:<28} {strs['l1']:>12} {strs['nuc']:>12} "
              f"{strs['nuc_lap']:>12} {winner:>8}")
    print(f"{'─'*len(hdr)}")


def plot_comparison(cases, rows, save_path="benchmark_norms.png"):
    stats = {}
    for m in METHODS:
        stats[m] = {
            "means": [next(r for r in rows if r["case"] == c and r["method"] == m)["ari_mean"]
                      for c in cases],
            "stds":  [next(r for r in rows if r["case"] == c and r["method"] == m)["ari_std"]
                      for c in cases],
        }

    x = np.arange(len(cases))
    n_methods = len(METHODS)
    w = 0.25
    offsets = np.linspace(-(n_methods - 1) / 2 * w, (n_methods - 1) / 2 * w, n_methods)
    colors = {"l1": "#4C72B0", "nuc": "#DD8452", "nuc_lap": "#55A868"}

    fig, ax = plt.subplots(figsize=(16, 6))
    for m, off in zip(METHODS, offsets):
        ax.bar(x + off, stats[m]["means"], w, yerr=stats[m]["stds"],
               label=f"{METHOD_LABELS[m]}-norm SSC", capsize=3,
               color=colors[m], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, 1.05)
    ax.set_title("L1  vs  Nuclear  vs  Nuclear+Laplacian SSC  "
                 f"({NUM_SEEDS} seeds per case)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {save_path}")
    plt.show()


def plot_time_comparison(cases, rows, save_path="benchmark_norms_time.png"):
    times = {}
    for m in METHODS:
        times[m] = [next(r for r in rows if r["case"] == c and r["method"] == m)["time_mean"]
                    for c in cases]

    x = np.arange(len(cases))
    n_methods = len(METHODS)
    w = 0.25
    offsets = np.linspace(-(n_methods - 1) / 2 * w, (n_methods - 1) / 2 * w, n_methods)
    colors = {"l1": "#4C72B0", "nuc": "#DD8452", "nuc_lap": "#55A868"}

    fig, ax = plt.subplots(figsize=(16, 5))
    for m, off in zip(METHODS, offsets):
        ax.bar(x + off, times[m], w, label=f"{METHOD_LABELS[m]}-norm SSC",
               color=colors[m], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime: L1  vs  Nuclear  vs  Nuclear+Laplacian SSC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {save_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_cases(verbose=True)
    cases, rows = summarize(results)
    print_table(cases, rows)
    plot_comparison(cases, rows)
    plot_time_comparison(cases, rows)
