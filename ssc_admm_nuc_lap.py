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
    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))
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

def ssc_admm_nuc_lap(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma=0.1,
    mu=1.0,
    rho=1.0,
    max_iter=500,
    tol=1e-4,
):
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

    # Match `ssc_admm_nuc.py` variable naming / constraints:
    #   X - C + diag(C) = 0   and   C - J = 0
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    J = np.zeros((N, N))
    E = np.zeros((n, N))

    Lambda = np.zeros((N, N))
    Gamma = np.zeros((N, N))

    YtY = Y.T @ Y
    A_inv = np.linalg.inv(lambda_z * YtY + mu * np.eye(N))

    I_N = np.eye(N)

    for it in range(max_iter):
        X_prev = X.copy()
        J_prev = J.copy()
        C_prev = C.copy()

        # X-update (same as nuclear-norm objective):
        #   (lambda_z Y^T Y + mu I) X = lambda_z Y^T (Y - E) + mu (C - diag(C)) - Lambda
        C_off = C.copy()
        np.fill_diagonal(C_off, 0.0)
        RHS = lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda
        X = A_inv @ RHS

        # J-update (same as nuclear-norm objective): SVT on (C + Gamma/rho)
        U, s, Vt = np.linalg.svd(C + Gamma / rho, full_matrices=False)
        J = (U * soft_threshold(s, 1.0 / rho)) @ Vt
        #J = soft_threshold(C + Gamma / rho, 1.0 / rho)

        # C-update: column-wise solve per derived formula
        #   c^j = ((μ+ρ)I + 2γL - μ e_j e_j^T)^{-1} (μ(a_j - a_{jj} e_j) + ρ b_j)
        # The rank-1 correction -μ e_j e_j^T removes the μ penalty from the j-th diagonal
        # (constraint 1 is X = C_off, so it does not constrain C_jj), and the -a_{jj} e_j
        # term in the RHS removes the μ contribution from row j of the system.
        A = X + Lambda / mu
        B = J - Gamma / rho

        L = graph_laplacian(C_prev)
        M_base = 2.0 * gamma * L + (mu + rho) * I_N   # shared across columns

        # c^j = ((μ+ρ)I + 2γL - μ e_j e_j^T)^{-1} (μ(a_j - a_{jj} e_j) + ρ b_j)
        for j in range(N):
            e_j    = I_N[:, j]
            M_j    = M_base - mu * np.outer(e_j, e_j)          # (μ+ρ)I + 2γL - μ e_j e_j^T
            rhs_j  = mu * (A[:, j] - A[j, j] * e_j) + rho * B[:, j]  # μ(a_j - a_{jj} e_j) + ρ b_j
            C[:, j] = np.linalg.solve(M_j, rhs_j)

        # E-update (same as nuclear-norm objective)
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # Dual updates (same as nuclear-norm objective)
        Lambda += mu * (X - C + np.diag(np.diag(C)))
        Gamma += rho * (C - J)

        primal1 = np.linalg.norm(X - C + np.diag(np.diag(C)), 'fro')
        primal2 = np.linalg.norm(C - J, 'fro')
        primal_res = max(primal1, primal2)
        dual_res = max(mu * np.linalg.norm(X - X_prev, 'fro'),
                       rho * np.linalg.norm(J - J_prev, 'fro'))
        if (it) % 5 == 0:
            print(f"  iter {it+1:4d}  primal={primal_res:.2e}  dual={dual_res:.2e}")
        if primal_res < tol and dual_res < tol:
            print(f"  Converged at iter {it + 1}.")
            break

    # Return order compatible with benchmark usage: X, J, C, E
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
        cluster_sizes, p_in=0.75, p_out=0.05, seed=42
    )
    N = Y.shape[0]
    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")

    # ── Nuclear-norm baseline ────────────────────────────────────────
    print("Running Nuclear-norm SSC ...")
    t0 = time.perf_counter()
    X_nuc, _C_nuc, _J_nuc, _E_nuc = ssc_nuc(Y, lambda_e=1.0, lambda_z=0.1, mu=1.0)
    t_nuc = time.perf_counter() - t0
    pred_nuc = cluster_from_C(X_nuc, k)
    ari_nuc = adjusted_rand_score(true_labels, pred_nuc)
    print(f"  ARI = {ari_nuc:.4f}   time = {t_nuc:.2f}s\n")

    # ── L1-norm baseline ─────────────────────────────────────────────
    print("Running L1-norm SSC ...")
    t0 = time.perf_counter()
    X_l1, _C_l1, _E_l1 = ssc_l1(Y, lambda_e=1.0, lambda_z=10.0, mu=1.0)
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
