import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# numpy 2.x + Apple Accelerate can emit spurious "divide by zero in matmul"
# for float64 matrices even when all values are finite.  Suppress it globally.
warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Data generation ────────────────────────────────────────────────────────────

def generate_block_diagonal_matrix(cluster_sizes, p_in=0.30, p_out=0.05, seed=None):
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


def ssc_admm(
    Y,
    lambda_e=1.0,
    lambda_z=10.0,
    mu=1.0,
    rho=1.0,
    max_iter=500,
    tol=1e-4,
):
    """
    Nuclear-norm SSC via ADMM, matching the common splitting:

        min_{X,C,J,E}  ||J||_* + lambda_e ||E||_1 + (lambda_z/2)||Y - YX - E||_F^2
        s.t.           X - C + diag(C) = 0
                       C - J = 0

    With duals Lambda (for X - C + diag(C)=0) and Gamma (for C - J=0),
    and penalties mu, rho. The C-update follows the entrywise formula:

      Let A = X + (1/mu) Lambda,   B = J - (1/rho) Gamma.
      For i != j:  C_ij = (mu * A_ij + rho * B_ij) / (mu + rho)
      For i == j:  C_ii = B_ii

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
    X, J, E  : ndarrays
    """
    n, N = Y.shape

    X = np.zeros((N, N))
    C = np.zeros((N, N))
    J = np.zeros((N, N))
    E = np.zeros((n, N))

    Lambda = np.zeros((N, N))
    Gamma = np.zeros((N, N))

    YtY = Y.T @ Y
    A_inv = np.linalg.inv(lambda_z * YtY + mu * np.eye(N))

    for it in range(max_iter):
        X_prev = X.copy()
        J_prev = J.copy()
        C_prev = C.copy()

        # X-update:
        #   (lambda_z Y^T Y + mu I) X = lambda_z Y^T (Y - E) + mu (C - diag(C)) - Lambda
        C_off = C.copy()
        np.fill_diagonal(C_off, 0.0)
        RHS = lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda
        X = A_inv @ RHS
        #np.fill_diagonal(X, 0.0)  # implied by X - C + diag(C)=0

        # J-update: singular value thresholding (SVT) on (C + Gamma/rho)
        U, S_svd, Vt = np.linalg.svd(C + Gamma / rho, full_matrices=False)
        S_soft = soft_threshold(S_svd, 1.0 / rho)
        J = (U * S_soft) @ Vt

        # C-update: entrywise closed form from the augmented Lagrangian
        A = X + Lambda / mu
        B = J - Gamma / rho
        C = (mu * A + rho * B) / (mu + rho)
        np.fill_diagonal(C, np.diag(B))

        # E_{k+1} = S_{λ_e/λ_z}(Y − Y X_{k+1})
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # Dual updates
        # Lambda for: X - C + diag(C) = 0
        Lambda += mu * (X - C + np.diag(np.diag(C)))
        # Gamma for: C - J = 0
        Gamma += rho * (C - J)

        primal1 = np.linalg.norm(X - C + np.diag(np.diag(C)), 'fro')
        primal2 = np.linalg.norm(C - J, 'fro')
        primal_res = max(primal1, primal2)
        dual_res = max(mu * np.linalg.norm(X - X_prev, 'fro'),
                       rho * np.linalg.norm(J - J_prev, 'fro'))
        if (it) % 50 == 0:
            print(f"  iter {it+1:4d}  primal={primal_res:.2e}  dual={dual_res:.2e}")
        if primal_res < tol and dual_res < tol:
            print(f"  Converged at iter {it + 1}.")
            break

    return X, C, J, E


# ── Clustering from C ──────────────────────────────────────────────────────────

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
        cluster_sizes, p_in=0.50, p_out=0.05, seed=42
    )
    print(f"Y: {Y.shape}, clusters: {cluster_sizes}")

    print("\nRunning SSC-ADMM ...")
    X, C, J, E = ssc_admm(Y, lambda_e=1.0, lambda_z=0.1, mu=1.0, rho=1.0)

    # Cluster on X (Y ≈ YX + E). At convergence X ≈ J; use X if stopping early.
    pred_labels = cluster_from_C(X, k)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"\nAdjusted Rand Index: {ari:.4f}")

    visualize_results(Y, true_labels, pred_labels, X, save_path='ssc_results.png')
