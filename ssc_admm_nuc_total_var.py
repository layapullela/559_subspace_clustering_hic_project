"""
Total-Variation Sparse Subspace Clustering via ADMM
====================================================

Objective
---------
    min_{X, C, E, P, Q}   λ_e ||E||_1  +  (λ_z/2) ||Y - YX - E||_F^2
                           +  γ ( ||P||_1 + ||Q||_1 )
    s.t.  X = C_off,   DC = P,   C D^T = Q,   diag(C) = 0

where  D ∈ ℝ^{(N-1)×N}  is the first-order finite-difference operator,
C_off = C − diag(C), and the two anisotropic TV terms penalise the row-
differences (P = DC) and column-differences (Q = CD^T) of the coefficient
matrix C, driving it toward a piecewise-constant (block) structure.

Variable splitting
------------------
  X   – reconstruction variable (self-expression Y ≈ YX + E)
  C   – coefficient matrix, forced block-piecewise-constant by the TV terms
  E   – sparse noise / outliers
  P   – row-difference auxiliary            (constraint: DC = P)
  Q   – column-difference auxiliary         (constraint: CD^T = Q)

Dual variables
--------------
  Λ   – for  X − C_off = 0       (penalty μ)
  Π_P – for  DC − P = 0          (penalty σ)
  Π_Q – for  CD^T − Q = 0        (penalty σ)

ADMM updates
------------
  1. X-update  (normal equations):
        (λ_z Y^T Y + μ I) X = λ_z Y^T(Y − E) + μ C_off − Λ

  2. C-update  (Sylvester equation, diag zeroed afterward):
        Let K = D^T D  (symmetric PSD, N×N)  with  K = V Λ_eig V^T.
        Equation:  μ C + σ (KC + CK) = RHS_C
        where  RHS_C = μ (X + Λ/μ) + σ (D^T P̃ + Q̃ D),
               P̃ = P − Π_P/σ,   Q̃ = Q − Π_Q/σ.
        Solve by diagonalising K:
               C'_{ij} = (V^T RHS_C V)_{ij} / [μ + σ(λ_i + λ_j)]
               C = V C' V^T ;   diag(C) = 0

  3. P-update:  P = S_{γ/σ}( DC + Π_P/σ )        (row-difference soft threshold)
  4. Q-update:  Q = S_{γ/σ}( CD^T + Π_Q/σ )      (col-difference soft threshold)
  5. E-update:  E = S_{λ_e/λ_z}( Y − YX )
  6. Dual ascent:
        Λ   += μ (X − C_off)
        Π_P += σ (DC − P)
        Π_Q += σ (CD^T − Q)
"""

import warnings
import numpy as np
from sklearn.cluster import SpectralClustering

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def finite_diff_matrix(N):
    """First-order finite-difference operator D ∈ ℝ^{(N-1)×N}."""
    D = np.zeros((N - 1, N))
    idx = np.arange(N - 1)
    D[idx, idx]     = -1.0
    D[idx, idx + 1] =  1.0
    return D


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_nuc_tv(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma=0.1,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    Sparse Subspace Clustering with anisotropic Total-Variation regularisation.

    Parameters
    ----------
    Y        : ndarray (n, N)   data matrix (columns = data points)
    lambda_e : float            weight on ||E||_1
    lambda_z : float            weight on reconstruction loss
    gamma    : float            TV regularisation weight  γ(||DC||_1 + ||CD^T||_1)
    mu       : float            ADMM penalty for the X = C_off constraint
    sigma    : float            ADMM penalty for the TV auxiliary constraints
    max_iter : int
    tol      : float            convergence tolerance (max primal Frobenius residual)

    Returns
    -------
    X, C, E : ndarrays
    """
    n, N = Y.shape

    # ── Precompute static quantities ──────────────────────────────────────────
    D = finite_diff_matrix(N)                 # (N-1, N)
    K = D.T @ D                               # (N, N), symmetric PSD
    eigs, V = np.linalg.eigh(K)               # eigs ascending, V orthogonal

    # Sylvester denominator: denom[i,j] = μ + σ(λ_i + λ_j)
    denom = mu + sigma * (eigs[:, None] + eigs[None, :])          # (N, N)
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))  # for X-update

    # ── Initialise primal and dual variables ──────────────────────────────────
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    P = np.zeros((N - 1, N))     # DC   auxiliary
    Q = np.zeros((N, N - 1))     # CD^T auxiliary

    Lambda = np.zeros((N, N))    # dual for X = C_off
    Pi_P   = np.zeros((N - 1, N))   # dual for DC = P
    Pi_Q   = np.zeros((N, N - 1))   # dual for CD^T = Q

    for it in range(max_iter):
        X_prev = X

        # 1. X-update
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        # 2. C-update (Sylvester equation via eigendecomposition of K)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C   = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)

        # 3-4. P- and Q-updates (soft threshold on row/col differences)
        DC  = D @ C
        CDt = C @ D.T
        P = soft_threshold(DC  + Pi_P / sigma, gamma / sigma)
        Q = soft_threshold(CDt + Pi_Q / sigma, gamma / sigma)

        # 5. E-update
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # 6. Dual updates
        C_off = C - np.diag(np.diag(C))
        Lambda += mu    * (X   - C_off)
        Pi_P   += sigma * (DC  - P)
        Pi_Q   += sigma * (CDt - Q)

        # Convergence check
        primal_res = max(
            np.linalg.norm(X   - C_off, 'fro'),
            np.linalg.norm(DC  - P,     'fro'),
            np.linalg.norm(CDt - Q,     'fro'),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        if primal_res < tol and dual_res < tol:
            break

    return X, C, E


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_from_C(C, k=None, k_max=None):
    """Spectral clustering on the symmetric affinity W = |C| + |C|^T.

    If ``k`` is None it is estimated with the eigengap heuristic on the
    symmetrically-normalised affinity D^{-½} W D^{-½}: k is the index of the
    largest drop between consecutive (descending) eigenvalues, clipped to
    ``[1, k_max]``.  ``k_max`` defaults to N // 20.
    """
    W = np.abs(C) + np.abs(C.T)

    if k is None:
        if k_max is None:
            k_max = max(1, C.shape[0] // 20)
        d = np.maximum(W.sum(axis=1), 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        W_norm = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
        eigvals = np.linalg.eigvalsh(W_norm)[::-1]   # descending
        gaps = eigvals[:-1] - eigvals[1:]
        k = int(np.clip(np.argmax(gaps) + 1, 1, k_max))

    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                            assign_labels='kmeans', random_state=0)
    return sc.fit_predict(W)


# ── Synthetic sanity check ──────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from sklearn.metrics import adjusted_rand_score

    cluster_sizes = [20, 25, 15, 20]
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, 0.75, 0.05)
    N = sum(cluster_sizes)
    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))

    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")
    t0 = time.perf_counter()
    X, C, E = ssc_admm_nuc_tv(Y, lambda_e=1.0, lambda_z=0.1, gamma=0.1)
    pred = cluster_from_C(X, k=len(cluster_sizes))
    print(f"ARI = {adjusted_rand_score(labels, pred):.4f}   "
          f"time = {time.perf_counter() - t0:.2f}s")
