"""
Sparse Subspace Clustering (SSC) via ADMM
==========================================
Solves:
    min_{C,E}  ||C||_1  +  λ_e ||E||_1  +  (λ_z/2) ||Y - YC - E||_F^2
    s.t.       diag(C) = 0

Variable splitting  X = C  (X is the reconstruction auxiliary; diag(C) = 0 by construction).
Augmented Lagrangian (μ = penalty, Λ = dual variable, constraint X = C):
    L = ||C||_1 + λ_e||E||_1 + (λ_z/2)||Y-YX-E||_F^2 + <Λ,X-C> + (μ/2)||X-C||_F^2

ADMM updates each iteration:

  X-update  — normal equation of the LS subproblem (min over X):

    (λ_z Y^T Y + μ I) X_{k+1}  =  λ_z Y^T(Y - E_k) + μ C_k - Λ_k
    diag(X_{k+1}) = 0                 ← enforce zero diagonal

  C-update  — prox of ||·||_1, then zero the diagonal:

    J         =  S_{1/μ}( X_{k+1} + μ^{-1} Λ_k )
    C_{k+1}   =  J - diag(J)

  E-update  — prox of λ_e ||·||_1 / λ_z :

    E_{k+1}   =  S_{λ_e/λ_z}( Y - Y X_{k+1} )

  Λ-update  — dual ascent:

    Λ_{k+1}   =  Λ_k + μ ( X_{k+1} - C_{k+1} )
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# numpy 2.x + Apple Accelerate can emit spurious "divide by zero in matmul"
# for float64 matrices even when all values are finite.  Suppress it globally.
warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Data generation ────────────────────────────────────────────────────────────

def generate_block_diagonal_matrix(cluster_sizes, p_in=0.75, p_out=0.05, seed=None):
    """
    Generate a symmetric binary matrix with block-diagonal cluster structure.

    Nodes are ordered so cluster 0 is rows/cols 0..c0-1, cluster 1 is
    c0..c0+c1-1, etc.  Within-cluster edges appear with probability p_in;
    between-cluster edges appear with probability p_out.

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

    same = (labels[:, None] == labels[None, :])          # (N,N) bool
    probs = np.where(same, p_in, p_out)

    # Sample upper triangle INCLUDING diagonal, then mirror to enforce symmetry.
    # This allows probabilistic sampling of diagonal \"self-contact\" entries too.
    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))
    return Y, labels


# ── ADMM solver ────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def ssc_admm(Y, lambda_e=1.0, lambda_z=10.0, mu=1.0, max_iter=500, tol=1e-4):
    """
    Solve the SSC objective via ADMM (see module docstring).

    Parameters
    ----------
    Y        : ndarray (n, N)   data matrix
    lambda_e : float            weight on ||E||_1
    lambda_z : float            weight on reconstruction loss
    mu       : float            ADMM penalty parameter
    max_iter : int
    tol      : float            convergence tolerance (Frobenius norm)

    Returns
    -------
    X, C, E  : ndarrays
    """
    n, N = Y.shape

    X      = np.zeros((N, N))
    C      = np.zeros((N, N))
    E      = np.zeros((n, N))
    Lambda = np.zeros((N, N))

    YtY   = Y.T @ Y
    A_inv = np.linalg.inv(lambda_z * YtY + mu * np.eye(N))

    for it in range(max_iter):
        X_prev = X.copy()

        # X-update: (λ_z Y^T Y + μ I) X = λ_z Y^T(Y - E) + μ C - Λ
        RHS = lambda_z * Y.T @ (Y - E) + mu * C - Lambda
        X   = A_inv @ RHS
        #np.fill_diagonal(X, 0.0)   # enforce diag(X) = 0

        # C-update: J = S_{1/μ}(X + μ^{-1} Λ),  C = J - diag(J)
        C = soft_threshold(X + Lambda / mu, 1.0 / mu)
        np.fill_diagonal(C, 0.0)   # enforce diag(C) = 0

        # E-update: E = S_{λ_e/λ_z}(Y - YX)
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # Λ-update: Λ = Λ + μ(X_{k+1} - C_{k+1} + diag(C_{k+1}))
        # diag(C_{k+1}) = 0 by construction, so this simplifies to X - C
        Lambda += mu * (X - C)

        # Convergence: primal = ||X - C||,  dual = μ||X - X_prev||
        primal_res = np.linalg.norm(X - C, 'fro') # check the condition is met
        dual_res   = mu * np.linalg.norm(X - X_prev, 'fro') # check algo is converging in final values
        if (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}  primal={primal_res:.2e}  dual={dual_res:.2e}")
        if primal_res < tol and dual_res < tol:
            print(f"  Converged at iter {it + 1}.")
            break

    return X, C, E


# ── Clustering from C ──────────────────────────────────────────────────────────

def cluster_from_C(C, k):
    """Spectral clustering on the symmetric affinity W = |C| + |C|^T."""
    W = np.abs(C) + np.abs(C.T)
    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                            assign_labels='kmeans', random_state=0)
    return sc.fit_predict(W)


def cluster_from_X(X, k):
    """Spectral clustering using the auxiliary variable X."""
    return cluster_from_C(X, k)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_results(Y, true_labels, pred_labels, C, save_path=None):
    """
    Four-panel figure:
      (1) Original Y
      (2) Y reordered by true labels   → should be block-diagonal
      (3) Learned |C|
      (4) Y reordered by predicted labels → should also be block-diagonal
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def _show(ax, M, title, cmap='Blues'):
        im = ax.imshow(M, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ord_t = np.argsort(true_labels)
    ord_p = np.argsort(pred_labels)

    _show(axes[0], Y,                           'Y  (original order)')
    _show(axes[1], Y[np.ix_(ord_t, ord_t)],     'Y  (true cluster order)')
    _show(axes[2], np.abs(C),                   'Learned |C|', cmap='hot')
    _show(axes[3], Y[np.ix_(ord_p, ord_p)],     'Y  (predicted order)')

    ari = adjusted_rand_score(true_labels, pred_labels)
    fig.suptitle(f'SSC-ADMM  (ARI = {ari:.3f})', fontsize=13)
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
        cluster_sizes, p_in=0.75, p_out=0.05, seed=42
    )
    print(f"Y: {Y.shape}, clusters: {cluster_sizes}")

    print("\nRunning SSC-ADMM ...")
    X, C, E = ssc_admm(Y, lambda_e=1.0, lambda_z=10.0, mu=1.0)

    # Cluster on X (the representation variable used in Y ≈ YX + E).
    # At convergence X = C, but X converges faster so use it when stopping early.
    pred_labels = cluster_from_C(X, k)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"\nAdjusted Rand Index: {ari:.4f}")

    visualize_results(Y, true_labels, pred_labels, X, save_path='ssc_results.png')
