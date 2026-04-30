"""
Lambda sweep: ARI heatmap over lambda_lr x lambda_sp
=====================================================
Runs SSC-ADMM Nuc+L1 on the synthetic block-diagonal dataset for a grid of
(lambda_lr, lambda_sp) values and visualises the resulting ARI as a heatmap.

For each (lambda_lr, lambda_sp) pair the ARI is averaged over EXPERIMENTS,
each with a different cluster structure, p_in, and p_out, to get a more
robust estimate of performance.

All other hyper-parameters (lambda_e, lambda_z, mu, rho_sp, rho_e) are fixed
to the defaults used in the main demo.
"""

import itertools
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend so plt.show() is a no-op
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from ssc_admm_nuc_l1 import (
    generate_block_diagonal_matrix,
    ssc_admm_nuc_l1,
    cluster_from_C,
)

warnings.filterwarnings("ignore")


# ── Sweep configuration ────────────────────────────────────────────────────────

LAMBDA_LR_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
#LAMBDA_SP_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
LAMBDA_SP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

FIXED_PARAMS = dict(
    lambda_e=1.0,
    lambda_z=0.1,
    mu=1.0,
    rho_sp=1.0,
    rho_e=1.0,
    max_iter=500,
    tol=1e-4,
)

# Each entry: (cluster_sizes, p_in, p_out, seed)
EXPERIMENTS = [
    ([20, 25, 15, 20], 0.50, 0.05, 42),   # original demo
    ([30, 20, 30],     0.45, 0.08, 7),    # 3 clusters, moderate separation
    ([15, 15, 15, 15, 15], 0.55, 0.04, 99),  # 5 equal clusters, tight signal
]


# ── Run the sweep ──────────────────────────────────────────────────────────────

def run_one(Y, true_labels, k, lr, sp):
    """Run solver + spectral clustering for a single dataset and λ pair."""
    _, C, _, _, _ = ssc_admm_nuc_l1(
        Y, lambda_lr=lr, lambda_sp=sp, **FIXED_PARAMS
    )
    pred_labels = cluster_from_C(C, k)
    return adjusted_rand_score(true_labels, pred_labels)


def run_sweep():
    # Pre-generate all datasets once
    datasets = []
    for cluster_sizes, p_in, p_out, seed in EXPERIMENTS:
        Y, true_labels = generate_block_diagonal_matrix(
            cluster_sizes, p_in=p_in, p_out=p_out, seed=seed
        )
        k = len(cluster_sizes)
        datasets.append((Y, true_labels, k, cluster_sizes, p_in, p_out))
        print(f"  Experiment: cluster_sizes={cluster_sizes}, p_in={p_in}, "
              f"p_out={p_out}, seed={seed}, N={Y.shape[0]}")
    print()

    n_lr  = len(LAMBDA_LR_VALUES)
    n_sp  = len(LAMBDA_SP_VALUES)
    # Store per-experiment ARIs so we can report mean ± std
    ari_runs = np.full((len(EXPERIMENTS), n_lr, n_sp), np.nan)

    total = n_lr * n_sp
    for idx, (i, j) in enumerate(itertools.product(range(n_lr), range(n_sp))):
        lr = LAMBDA_LR_VALUES[i]
        sp = LAMBDA_SP_VALUES[j]
        print(
            f"[{idx + 1:3d}/{total}]  lambda_lr={lr:.3f}  lambda_sp={sp:.3f}",
            flush=True,
        )
        for e, (Y, true_labels, k, cluster_sizes, p_in, p_out) in enumerate(datasets):
            try:
                ari = run_one(Y, true_labels, k, lr, sp)
            except Exception as exc:
                print(f"    experiment {e+1} ERROR: {exc}")
                ari = np.nan
            ari_runs[e, i, j] = ari
            print(f"    exp {e+1} (k={k}, p_in={p_in}, p_out={p_out})  ARI={ari:.4f}")

        mean_ari = np.nanmean(ari_runs[:, i, j])
        print(f"  → mean ARI = {mean_ari:.4f}\n")

    ari_mean = np.nanmean(ari_runs, axis=0)
    ari_std  = np.nanstd(ari_runs,  axis=0)
    return ari_mean, ari_std


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_heatmap(ari_mean, ari_std, save_path="sweep_lambda_heatmap.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(ari_mean, origin="upper", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)

    # Annotate each cell: mean on top, ±std below
    for i in range(len(LAMBDA_LR_VALUES)):
        for j in range(len(LAMBDA_SP_VALUES)):
            mean = ari_mean[i, j]
            std  = ari_std[i, j]
            if np.isnan(mean):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
            else:
                color = "white" if mean < 0.5 else "black"
                ax.text(j, i - 0.15, f"{mean:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")
                ax.text(j, i + 0.25, f"±{std:.2f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    ax.set_xticks(range(len(LAMBDA_SP_VALUES)))
    ax.set_xticklabels([str(v) for v in LAMBDA_SP_VALUES])
    ax.set_yticks(range(len(LAMBDA_LR_VALUES)))
    ax.set_yticklabels([str(v) for v in LAMBDA_LR_VALUES])

    ax.set_xlabel("lambda_sp  (L1 / sparsity weight)", fontsize=11)
    ax.set_ylabel("lambda_lr  (nuclear-norm / low-rank weight)", fontsize=11)

    n_exp = len(EXPERIMENTS)
    exp_summary = ", ".join(
        f"k={len(cs)}" for cs, *_ in EXPERIMENTS
    )
    ax.set_title(
        f"Mean ARI heatmap: SSC-ADMM Nuc+L1  (avg over {n_exp} experiments)\n"
        f"[{exp_summary}]",
        fontsize=11,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Adjusted Rand Index", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {save_path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running sweep over {len(LAMBDA_LR_VALUES)}×{len(LAMBDA_SP_VALUES)} λ pairs, "
          f"averaged over {len(EXPERIMENTS)} experiments.\n")

    ari_mean, ari_std = run_sweep()

    print("\n── Mean ARI grid (rows=lambda_lr, cols=lambda_sp) ──")
    header = "         " + "  ".join(f"{v:>5}" for v in LAMBDA_SP_VALUES)
    print(f"lambda_sp→ {header}")
    for i, lr in enumerate(LAMBDA_LR_VALUES):
        row = "  ".join(f"{ari_mean[i, j]:5.3f}" for j in range(len(LAMBDA_SP_VALUES)))
        print(f"  lr={lr:>4}  {row}")

    plot_heatmap(ari_mean, ari_std)
