"""
Nuclear-norm SSC with Graph-Laplacian Regularisation via ADMM
=============================================================

Objective
---------
    min_{X,J,C,E}  ||J||_*  +  λ_e ||E||_1  +  (λ_z/2) ||Y - YX - E||_F^2
                    +  γ tr(C^T L C)
    s.t.  X = J,   X = C,   diag(X) = 0

where  L  is the combinatorial graph Laplacian of the affinity
W = |C| + |C|^T,  i.e.  L = diag(W 1) − W.

Variable splitting introduces two copies of X (J for the nuclear-norm
prox, C for the Laplacian term) with dual variables Λ₁ (X = J) and
Λ₂ (X = C).

Because L depends on C, at each iteration we freeze L at the value
implied by the current C and solve the C-subproblem with L fixed
(alternating linearisation).

Augmented Lagrangian
--------------------
    L = ||J||_* + λ_e||E||_1 + (λ_z/2)||Y-YX-E||_F^2 + γ tr(C^T L C)
        + ⟨Λ₁, X-J⟩ + (μ/2)||X-J||_F^2
        + ⟨Λ₂, X-C⟩ + (μ/2)||X-C||_F^2

ADMM updates
------------
  1. J-update  (nuclear-norm prox, unchanged):
        J_{k+1} = SVT_{1/μ}( X_k + Λ_{1,k}/μ )

  2. C-update  (Laplacian-regularised quadratic, L_k fixed):
        ∇_C [ γ tr(C^T L_k C) + (μ/2)||C - V_k||_F^2 ] = 0
        (2γ L_k + μ I) C = μ V_k   where  V_k = X_k + Λ_{2,k}/μ
        C_{k+1} = μ (2γ L_k + μ I)^{-1} V_k ;   diag(C_{k+1}) = 0

  3. X-update  (column-wise normal equations, two penalty terms):
        For column j (rows ≠ j), with Ṽ₁=J-Λ₁/μ, Ṽ₂=C-Λ₂/μ :
        (λ_z Y_{-j}^T Y_{-j} + 2μ I) x_{-j,j}
            = λ_z Y_{-j}^T(y_j - e_j) + μ Ṽ₁_{-j,j} + μ Ṽ₂_{-j,j}

  4. E-update  (unchanged):
        E_{k+1} = S_{λ_e/λ_z}( Y - Y X_{k+1} )

  5. Dual ascent:
        Λ₁ += μ (X - J)
        Λ₂ += μ (X - C)

  6. Laplacian refresh:
        W = |C_{k+1}| + |C_{k+1}|^T,   L_{k+1} = diag(W 1) − W
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Data generation (shared) ──────────────────────────────────────────────────

def generate_block_diagonal_matrix(cluster_sizes, p_in=0.30, p_out=0.05, seed=None):
    rng = np.random.default_rng(seed)
    N = sum(cluster_sizes)
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    same = (labels[:, None] == labels[None, :])
    probs = np.where(same, p_in, p_out)
    upper = np.triu(rng.random((N, N)) < probs, k=1)
    Y = (upper + upper.T).astype(float)
    return Y, labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def graph_laplacian(C):
    """Combinatorial Laplacian of the symmetrised affinity W = |C|+|C|^T."""
    W = np.abs(C) + np.abs(C).T
    D = np.diag(W.sum(axis=1))
    return D - W


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_nuc_lap(Y, lambda_e=1.0, lambda_z=0.1, gamma=0.1, mu=1.0,
                     max_iter=500, tol=1e-4):
    """
    Nuclear-norm SSC + γ tr(C^T L C) Laplacian regularisation.

    Parameters
    ----------
    Y        : ndarray (n, N)
    lambda_e : float   weight on ||E||_1
    lambda_z : float   weight on reconstruction loss
    gamma    : float   weight on Laplacian smoothness  tr(C^T L C)
    mu       : float   ADMM penalty parameter
    max_iter : int
    tol      : float   convergence tolerance (Frobenius)

    Returns
    -------
    X, J, C, E : ndarrays
    """
    n, N = Y.shape

    X       = np.zeros((N, N))
    J       = np.zeros((N, N))
    C       = np.zeros((N, N))
    E       = np.zeros((n, N))
    Lambda1 = np.zeros((N, N))
    Lambda2 = np.zeros((N, N))

    In1 = np.eye(N - 1)
    I_N = np.eye(N)

    for it in range(max_iter):
        J_prev = J.copy()
        C_prev = C.copy()

        # ── 1. J-update: SVT_{1/μ}(X + Λ₁/μ) ───────────────────────────
        U, s, Vt = np.linalg.svd(X + Lambda1 / mu, full_matrices=False)
        J = (U * soft_threshold(s, 1.0 / mu)) @ Vt

        # ── 2. C-update: (2γ L + μI) C = μ V,  V = X + Λ₂/μ ────────────
        L = graph_laplacian(C_prev)
        V = X + Lambda2 / mu
        C = np.linalg.solve(2.0 * gamma * L + mu * I_N, mu * V)
        np.fill_diagonal(C, 0.0)

        # ── 3. X-update: column-wise least squares ───────────────────────
        Vtilde1 = J - Lambda1 / mu
        Vtilde2 = C - Lambda2 / mu
        for j in range(N):
            mask = np.ones(N, dtype=bool)
            mask[j] = False
            idx = np.flatnonzero(mask)
            Ymj = Y[:, idx]
            A_j = lambda_z * (Ymj.T @ Ymj) + 2.0 * mu * In1
            b_j = (lambda_z * Ymj.T @ (Y[:, j] - E[:, j])
                   + mu * Vtilde1[idx, j]
                   + mu * Vtilde2[idx, j])
            X[idx, j] = np.linalg.solve(A_j, b_j)
        np.fill_diagonal(X, 0.0)

        # ── 4. E-update: soft-threshold ──────────────────────────────────
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # ── 5. Dual ascent ───────────────────────────────────────────────
        Lambda1 += mu * (X - J)
        Lambda2 += mu * (X - C)

        # ── Convergence ──────────────────────────────────────────────────
        p_res = max(np.linalg.norm(X - J, 'fro'),
                    np.linalg.norm(X - C, 'fro'))
        d_res = max(mu * np.linalg.norm(J - J_prev, 'fro'),
                    mu * np.linalg.norm(C - C_prev, 'fro'))
        if (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}  primal={p_res:.2e}  dual={d_res:.2e}")
        if p_res < tol and d_res < tol:
            print(f"  Converged at iter {it + 1}.")
            break

    return X, J, C, E


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_from_C(C, k):
    W = np.abs(C) + np.abs(C.T)
    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                            assign_labels='kmeans', random_state=0)
    return sc.fit_predict(W)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_results(Y, true_labels, pred_labels, C, title_prefix='', save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def _show(ax, M, title, cmap='Blues'):
        im = ax.imshow(M, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ord_t = np.argsort(true_labels)
    ord_p = np.argsort(pred_labels)

    _show(axes[0], Y,                        'Y  (original order)')
    _show(axes[1], Y[np.ix_(ord_t, ord_t)],  'Y  (true cluster order)')
    _show(axes[2], np.abs(C),                'Learned |C|', cmap='hot')
    _show(axes[3], Y[np.ix_(ord_p, ord_p)],  'Y  (predicted order)')

    ari = adjusted_rand_score(true_labels, pred_labels)
    fig.suptitle(f'{title_prefix}SSC-ADMM Nuc+Lap  (ARI = {ari:.3f})', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ── Demo / quick test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from ssc_admm_nuc import ssc_admm as ssc_nuc
    from ssc_admm     import ssc_admm as ssc_l1

    k             = 4
    cluster_sizes = [20, 25, 15, 20]

    Y, true_labels = generate_block_diagonal_matrix(
        cluster_sizes, p_in=0.50, p_out=0.05, seed=42
    )
    N = Y.shape[0]
    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")

    # ── Nuclear-norm baseline ────────────────────────────────────────
    print("Running Nuclear-norm SSC ...")
    t0 = time.perf_counter()
    X_nuc, _, _ = ssc_nuc(Y, lambda_e=1.0, lambda_z=0.1, mu=1.0)
    t_nuc = time.perf_counter() - t0
    pred_nuc = cluster_from_C(X_nuc, k)
    ari_nuc = adjusted_rand_score(true_labels, pred_nuc)
    print(f"  ARI = {ari_nuc:.4f}   time = {t_nuc:.2f}s\n")

    # ── L1-norm baseline ─────────────────────────────────────────────
    print("Running L1-norm SSC ...")
    t0 = time.perf_counter()
    X_l1, _, _ = ssc_l1(Y, lambda_e=1.0, lambda_z=10.0, mu=1.0)
    t_l1 = time.perf_counter() - t0
    pred_l1 = cluster_from_C(X_l1, k)
    ari_l1 = adjusted_rand_score(true_labels, pred_l1)
    print(f"  ARI = {ari_l1:.4f}   time = {t_l1:.2f}s\n")

    # ── Nuclear + Laplacian (sweep γ) ────────────────────────────────
    gammas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    aris_lap = []
    times_lap = []
    print("Running Nuclear+Laplacian SSC  (sweeping γ) ...")
    for g in gammas:
        print(f"\n  γ = {g}")
        t0 = time.perf_counter()
        X_lap, _, C_lap, _ = ssc_admm_nuc_lap(
            Y, lambda_e=1.0, lambda_z=0.1, gamma=g, mu=1.0
        )
        elapsed = time.perf_counter() - t0
        pred_lap = cluster_from_C(X_lap, k)
        ari_lap = adjusted_rand_score(true_labels, pred_lap)
        aris_lap.append(ari_lap)
        times_lap.append(elapsed)
        print(f"  ARI = {ari_lap:.4f}   time = {elapsed:.2f}s")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Method':<26} {'ARI':>8} {'Time (s)':>10}")
    print(f"{'─'*50}")
    print(f"{'L1-norm SSC':<26} {ari_l1:>8.4f} {t_l1:>10.2f}")
    print(f"{'Nuclear-norm SSC':<26} {ari_nuc:>8.4f} {t_nuc:>10.2f}")
    for g, a, t in zip(gammas, aris_lap, times_lap):
        print(f"{'Nuc+Lap γ=' + str(g):<26} {a:>8.4f} {t:>10.2f}")
    print(f"{'─'*50}")

    # ── Bar chart ────────────────────────────────────────────────────
    labels_bar = ['L1', 'Nuc'] + [f'Nuc+Lap\nγ={g}' for g in gammas]
    ari_vals   = [ari_l1, ari_nuc] + aris_lap
    colors     = ['#4C72B0', '#DD8452'] + ['#55A868'] * len(gammas)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_bar, ari_vals, color=colors, edgecolor='white')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_ylim(0, 1.05)
    ax.set_title('SSC Comparison: L1  vs  Nuclear  vs  Nuclear + Laplacian')
    ax.axhline(ari_l1,  ls='--', lw=0.8, color='#4C72B0', alpha=0.5)
    ax.axhline(ari_nuc, ls='--', lw=0.8, color='#DD8452', alpha=0.5)
    for b, v in zip(bars, ari_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('ssc_nuc_lap_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved figure → ssc_nuc_lap_comparison.png")
    plt.show()
