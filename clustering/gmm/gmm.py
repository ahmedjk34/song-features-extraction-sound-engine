
import numpy as np


def safe_log_det(matrix):
    """Safely compute log determinant to avoid overflow."""
    try:
        sign, log_det = np.linalg.slogdet(matrix)
        if sign <= 0 or not np.isfinite(log_det):
            return -700.0  # Very negative log det
        return log_det
    except:
        return -700.0

def regularize_covariance(cov_matrix, reg_coeff=1e-6):
    """Add regularization to prevent singular matrices."""
    return cov_matrix + reg_coeff * np.eye(cov_matrix.shape[0])

def gmm_em(X, n_clusters=5, max_iters=100, tol=1e-6, seed=None, verbose=False):
    """
    FIXED: Numerically stable GMM implementation.
    """
    np.random.seed(seed)
    N, D = X.shape
    K = n_clusters
    
    print(f"Starting GMM with N={N}, D={D}, K={K}")
    print(f"Input data range: [{X.min():.3f}, {X.max():.3f}], std: {X.std():.3f}")
    
    # Initialize parameters more carefully
    # Use K-means++ for better initialization
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=seed, n_init=10)
    kmeans.fit(X)
    means = kmeans.cluster_centers_
    
    # Initialize covariances as regularized versions of data covariance
    data_cov = np.cov(X.T)
    covariances = np.array([regularize_covariance(0.5 * data_cov, reg_coeff=0.1) for _ in range(K)])
    weights = np.ones(K) / K
    
    # Pre-compute some constants
    log_2pi = np.log(2 * np.pi)
    
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iters):
        # E-step with numerical stability
        log_responsibilities = np.zeros((N, K))
        
        for k in range(K):
            # Regularize covariance matrix
            reg_cov = regularize_covariance(covariances[k], reg_coeff=0.01)
            
            try:
                # Use Cholesky decomposition for numerical stability
                L = np.linalg.cholesky(reg_cov)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                
                # Solve triangular system instead of inverting
                diff = X - means[k]
                v = np.linalg.solve(L, diff.T).T
                mahalanobis = np.sum(v * v, axis=1)
                
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(reg_cov)
                eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
                log_det = np.sum(np.log(eigenvals))
                
                diff = X - means[k]
                v = diff @ eigenvecs
                mahalanobis = np.sum((v * v) / eigenvals, axis=1)
            
            # Compute log probabilities with clipping
            mahalanobis = np.clip(mahalanobis, 0, 700)  # Prevent overflow
            log_prob = -0.5 * (D * log_2pi + log_det + mahalanobis)
            log_responsibilities[:, k] = np.log(weights[k] + 1e-10) + log_prob
        
        # Normalize responsibilities using log-sum-exp trick
        max_log_resp = np.max(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities_stable = log_responsibilities - max_log_resp
        responsibilities = np.exp(log_responsibilities_stable)
        resp_sum = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities = responsibilities / (resp_sum + 1e-10)
        
        # M-step
        N_k = np.sum(responsibilities, axis=0)
        N_k = np.maximum(N_k, 1e-10)  # Prevent division by zero
        
        # Update weights
        weights = N_k / N
        
        # Update means
        for k in range(K):
            means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        
        # Update covariances with regularization
        for k in range(K):
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / N_k[k]
            
            # Strong regularization for numerical stability
            reg_strength = max(0.01, 0.1 / N_k[k])
            covariances[k] = regularize_covariance(cov, reg_coeff=reg_strength)
        
        # Compute log-likelihood
        log_likelihood = np.sum(max_log_resp.flatten() + np.log(resp_sum.flatten() + 1e-10))
        
        if verbose and iteration % 10 == 0:
            print(f"EM iter {iteration}: log-likelihood = {log_likelihood:.4f}")
        
        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
        
        prev_log_likelihood = log_likelihood
    
    # Final cluster assignments
    labels = np.argmax(responsibilities, axis=1)
    
    return means, covariances, weights, responsibilities, log_likelihood, labels

def compute_bic_aic(X, log_likelihood, n_clusters):
    """Compute BIC and AIC for model selection."""
    N, D = X.shape
    # Parameters: K means (K*D) + K covariances (K*D*(D+1)/2) + K-1 weights
    n_params = n_clusters * D + n_clusters * D * (D + 1) // 2 + (n_clusters - 1)
    
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(N) * n_params - 2 * log_likelihood
    
    return bic, aic

