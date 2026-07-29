"""
Nuclear-norm + L1 SSC via ADMM
==============================

Objective
---------
    min_{X,C,S,J,E}  λ_sp ||S||_1  +  λ_lr ||J||_*  +  λ_e ||E||_1
                      +  (λ_z/2) ||Y - YX - E||_F^2
    s.t.  X - C + diag(C) = 0      (dual Λ,  penalty μ)
          C - S = 0                (dual U,  penalty ρ_sp)
          C - J = 0                (dual V,  penalty ρ_e)

Variable splitting
------------------
  S  : sparse proxy for C  (L1 prox)
  J  : low-rank proxy for C  (nuclear-norm prox)
  C  : consensus variable — off-diagonal averaged from X, S, J contributions;
       diagonal from S and J contributions only (X constraint's diagonal vanishes)

Augmented Lagrangian  (unscaled duals Λ, U, V)
----------------------------------------------
  L = λ_sp||S||_1 + λ_lr||J||_* + λ_e||E||_1 + (λ_z/2)||Y-YX-E||_F^2
      + ⟨Λ, X-C+diag(C)⟩  + (μ/2)   ||X-C+diag(C)||_F^2
      + ⟨U, C-S⟩           + (ρ_sp/2)||C-S||_F^2
      + ⟨V, C-J⟩           + (ρ_e/2) ||C-J||_F^2

ADMM updates  (each step uses results from current sub-iteration)
-----------------------------------------------------------------
  1. X-update  (normal equations; diag(C)=0 so C_off = C):
        (λ_z Y^T Y + μ I) X_{k+1} = λ_z Y^T (Y - E_k) + μ C_k - Λ_k

  2. S-update  (soft-thresholding, uses C_k):
        S_{k+1} = S_{λ_sp/ρ_sp}(C_k + ρ_sp^{-1} U_k)

  3. J-update  (singular value thresholding, uses C_k):
        J_{k+1} = SVT_{λ_lr/ρ_e}(C_k + ρ_e^{-1} V_k)

  4. C-update  (consensus, uses X_{k+1}, S_{k+1}, J_{k+1}):
        A = X_{k+1} + μ^{-1} Λ_k
        B = S_{k+1} - ρ_sp^{-1} U_k
        D = J_{k+1} - ρ_e^{-1} V_k

        C_{k+1} = (μ A + ρ_sp B + ρ_e D) / (μ + ρ_sp + ρ_e),  diag(C) = 0
          [uniform average since diag(C) = 0 is enforced; constraint 1 simplifies
           to X - C = 0 and the dual update is just Λ += μ(X - C)]

  5. E-update  (same as ssc_admm_nuc / ssc_admm):
        E_{k+1} = S_{λ_e/λ_z}(Y - Y X_{k+1})

  6. Dual updates  (unscaled, each with its own penalty):
        Λ_{k+1} = Λ_k + μ    (X_{k+1} - C_{k+1})      [diag(C)=0 simplifies this]
        U_{k+1} = U_k + ρ_sp (C_{k+1} - S_{k+1})
        V_{k+1} = V_k + ρ_e  (C_{k+1} - J_{k+1})
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Data generation ────────────────────────────────────────────────────────────

def generate_block_diagonal_matrix(cluster_sizes, p_in=0.30, p_out=0.05, seed=None):
    """
    Generate a symmetric binary matrix with block-diagonal cluster structure.

    Parameters
    ----------
    cluster_sizes : list[int]   number of nodes per cluster
    p_in          : float       within-cluster edge probability
    p_out         : float       between-cluster edge probability
    seed          : int or None

    Returns
    -------
    Y      : ndarray (N, N)  symmetric binary data matrix
    labels : ndarray (N,)    ground-truth cluster label per node
    """
    rng = np.random.default_rng(seed)
    N = sum(cluster_sizes)
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)

    same = (labels[:, None] == labels[None, :])
    probs = np.where(same, p_in, p_out)

    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))
    return Y, labels


# ── Helpers ────────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


# ── ADMM solver ────────────────────────────────────────────────────────────────

def ssc_admm_nuc_l1(
    Y,
    lambda_sp=1.0,
    lambda_lr=1.0,
    lambda_e=1.0,
    lambda_z=10.0,
    mu=1.0,
    rho_sp=1.0,
    rho_e=1.0,
    max_iter=500,
    tol=1e-4,
):
    """
    Combined nuclear-norm + L1 SSC via ADMM with three-way variable splitting
    and separate penalty parameters for each constraint.

    Parameters
    ----------
    Y         : ndarray (n, N)   data matrix
    lambda_sp : float            weight on ||S||_1   (sparsity)
    lambda_lr : float            weight on ||J||_*   (low-rank)
    lambda_e  : float            weight on ||E||_1   (noise)
    lambda_z  : float            weight on reconstruction loss
    mu        : float            ADMM penalty for X - C + diag(C) = 0
    rho_sp    : float            ADMM penalty for C - S = 0
    rho_e     : float            ADMM penalty for C - J = 0
    max_iter  : int
    tol       : float            convergence tolerance (max primal Frobenius norm)

    Returns
    -------
    X, C, S, J, E : ndarrays
    """
    n, N = Y.shape

    X = np.zeros((N, N))
    C = np.zeros((N, N))
    S = np.zeros((N, N))
    J = np.zeros((N, N))
    E = np.zeros((n, N))

    # Unscaled duals, updated with their respective penalty params
    Lambda = np.zeros((N, N))   # for X - C + diag(C) = 0,  step μ
    U      = np.zeros((N, N))   # for C - S = 0,             step ρ_sp
    V      = np.zeros((N, N))   # for C - J = 0,             step ρ_e

    YtY = Y.T @ Y
    A_inv = np.linalg.inv(lambda_z * YtY + mu * np.eye(N))

    for it in range(max_iter):
        X_prev = X.copy()
        C_prev = C.copy()

        # ── 1. X-update ──────────────────────────────────────────────────────
        # diag(C) = 0 is enforced below, so C_off = C and the constraint
        # X - C + diag(C) = 0 simplifies to X - C = 0 (off-diagonal, X_diag=0).
        # (λ_z Y^T Y + μ I) X = λ_z Y^T (Y - E) + μ C - Λ
        RHS = lambda_z * (Y.T @ (Y - E)) + mu * C - Lambda
        X = A_inv @ RHS

        # ── 2. S-update (sparse proxy, uses C_k) ─────────────────────────────
        # S_{k+1} = S_{λ_sp/ρ_sp}(C_k + ρ_sp^{-1} U_k)
        S = soft_threshold(C + U / rho_sp, lambda_sp / rho_sp)

        # ── 3. J-update (low-rank proxy, uses C_k) ───────────────────────────
        # J_{k+1} = SVT_{λ_lr/ρ_e}(C_k + ρ_e^{-1} V_k)
        M = C + V / rho_e
        Usvd, sigma, Vt = np.linalg.svd(M, full_matrices=False)
        J = (Usvd * soft_threshold(sigma, lambda_lr / rho_e)) @ Vt

        # ── 4. C-update (consensus, uses X_{k+1}, S_{k+1}, J_{k+1}) ─────────
        # A = X_{k+1} + μ^{-1} Λ_k
        # B = S_{k+1} - ρ_sp^{-1} U_k
        # D = J_{k+1} - ρ_e^{-1} V_k
        #
        # With diag(C) = 0, all three constraints involve C uniformly,
        # so the consensus average uses the same weights for every entry.
        # C = (μ A + ρ_sp B + ρ_e D) / (μ + ρ_sp + ρ_e),  then zero diagonal.
        A_mat = X + Lambda / mu
        B_mat = S - U / rho_sp
        D_mat = J - V / rho_e

        C = (mu * A_mat + rho_sp * B_mat + rho_e * D_mat) / (mu + rho_sp + rho_e)
        np.fill_diagonal(C, 0.0)

        # ── 5. E-update (same as ssc_admm_nuc / ssc_admm) ───────────────────
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # ── 6. Dual updates (unscaled, each with its own penalty) ─────────────
        # diag(C) = 0, so X - C + diag(C) simplifies to X - C
        Lambda += mu     * (X - C)
        U      += rho_sp * (C - S)
        V      += rho_e  * (C - J)

        # ── Convergence check ─────────────────────────────────────────────────
        primal1 = np.linalg.norm(X - C, 'fro')
        primal2 = np.linalg.norm(C - S, 'fro')
        primal3 = np.linalg.norm(C - J, 'fro')
        primal_res = max(primal1, primal2, primal3)
        dual_res = max(mu * np.linalg.norm(X - X_prev, 'fro'),
                       rho_sp * np.linalg.norm(C - C_prev, 'fro'))

        if it % 50 == 0:
            print(f"  iter {it+1:4d}  primal={primal_res:.2e}  dual={dual_res:.2e}")
        if primal_res < tol and dual_res < tol:
            print(f"  Converged at iter {it + 1}.")
            break

    return X, C, S, J, E


# ── Clustering ─────────────────────────────────────────────────────────────────

def cluster_from_C(C, k=None):
    """Spectral clustering on the symmetric affinity W = |C| + |C|^T.

    If k is None, k is estimated via the eigengap heuristic on the
    symmetrically-normalised affinity D^{-½}WD^{-½}, clipped to [1, 12] TADs.
    TODO: above heuristic not implemented yet.
    """
    W = np.abs(C) + np.abs(C.T)
    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                            assign_labels='kmeans', random_state=0)
    return sc.fit_predict(W)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_results(Y, true_labels, pred_labels, C, title_prefix='', save_path=None):
    """
    Four-panel figure:
      (1) Original Y
      (2) Y reordered by true labels
      (3) Learned |C|
      (4) Y reordered by predicted labels
    """
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
    fig.suptitle(f'{title_prefix}SSC-ADMM Nuc+L1  (ARI = {ari:.3f})', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    k             = 4
    cluster_sizes = [20, 25, 15, 20]

    Y, true_labels = generate_block_diagonal_matrix(
        cluster_sizes, p_in=0.50, p_out=0.05, seed=42
    )
    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")

    print("Running SSC-ADMM Nuc+L1 ...")
    X, C, S, J, E = ssc_admm_nuc_l1(
        Y,
        lambda_sp=0,
        lambda_lr=1,
        lambda_e=1.0,
        lambda_z=0.1,
        mu=1.0,
        rho_sp=1.0,
        rho_e=1.0,
    )

    pred_labels = cluster_from_C(X, k)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"\nAdjusted Rand Index: {ari:.4f}")

    visualize_results(Y, true_labels, pred_labels, X,
                      title_prefix='', save_path='ssc_nuc_l1_results.png')
