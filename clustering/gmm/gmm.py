import numpy as np

def compute_log_likelihood(X, means, covariances, weights):
    """
    Compute the total log-likelihood of the data under the GMM.
    """
    N, D = X.shape
    K = means.shape[0]
    log_likelihood = 0.0
    
    for n in range(N):
        p = 0.0
        for k in range(K):
            diff = X[n] - means[k]
            try:
                inv_cov = np.linalg.inv(covariances[k])
                det_cov = np.linalg.det(covariances[k])
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(covariances[k])
                det_cov = np.linalg.det(covariances[k] + 1e-6 * np.eye(D))
            
            if det_cov <= 0:
                det_cov = 1e-10
                
            norm_const = 1.0 / (np.power(2 * np.pi, D / 2) * np.sqrt(det_cov))
            exp_term = np.exp(-0.5 * diff @ inv_cov @ diff)
            p += weights[k] * norm_const * exp_term
        
        log_likelihood += np.log(p + 1e-10)
    
    return log_likelihood

def initialize_gmm_parameters(X, K, seed=None, method='kmeans++'):
    """
    Initialize GMM means, covariances, and weights.
    """
    np.random.seed(seed)
    N, D = X.shape
    
    if method == 'kmeans++':
        # K-means++ initialization
        means = np.zeros((K, D))
        means[0] = X[np.random.randint(N)]
        
        for k in range(1, K):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in means[:k]]) for x in X])
            probs = distances / distances.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            means[k] = X[np.searchsorted(cumulative_probs, r)]
    else:
        # Random initialization
        indices = np.random.choice(N, K, replace=False)
        means = X[indices]
    
    # Initialize covariances as scaled identity matrices
    global_cov = np.cov(X, rowvar=False)
    covariances = np.array([0.1 * global_cov + 1e-6 * np.eye(D) for _ in range(K)])
    
    weights = np.ones(K) / K
    
    return means, covariances, weights

def e_step(X, means, covariances, weights):
    """
    Compute the responsibilities for each data point and each cluster.
    """
    N, D = X.shape
    K = means.shape[0]
    responsibilities = np.zeros((N, K))
    
    for k in range(K):
        diff = X - means[k]
        try:
            inv_cov = np.linalg.inv(covariances[k])
            det_cov = np.linalg.det(covariances[k])
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(covariances[k])
            det_cov = np.linalg.det(covariances[k] + 1e-6 * np.eye(D))
        
        # Check for negative or zero determinant
        if det_cov <= 0:
            det_cov = 1e-10
            
        norm_const = 1.0 / (np.power(2 * np.pi, D / 2) * np.sqrt(det_cov))
        
        # Fixed matrix multiplication for quadratic form
        quadratic_form = np.sum((diff @ inv_cov) * diff, axis=1)
        
        # Add numerical stability
        quadratic_form = np.clip(quadratic_form, 0, 700)  # Prevent overflow
        
        exp_term = np.exp(-0.5 * quadratic_form)
        responsibilities[:, k] = weights[k] * norm_const * exp_term
    
    # Normalize responsibilities
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities = responsibilities / (responsibilities_sum + 1e-10)
    
    return responsibilities

def m_step(X, responsibilities):
    """
    Update GMM means, covariances, and weights based on responsibilities.
    """
    N, D = X.shape
    K = responsibilities.shape[1]
    
    N_k = responsibilities.sum(axis=0)
    # Prevent division by zero
    N_k = np.maximum(N_k, 1e-10)
    
    means = np.zeros((K, D))
    covariances = np.zeros((K, D, D))
    weights = N_k / N
    
    for k in range(K):
        # Update means
        means[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / N_k[k]
        
        # Update covariances
        diff = X - means[k]
        weighted_diff = responsibilities[:, k][:, np.newaxis] * diff
        covariances[k] = (weighted_diff.T @ diff) / N_k[k]
        
        # Add regularization - more aggressive for small clusters
        reg_strength = max(1e-6, 1e-4 / N_k[k])
        covariances[k] += reg_strength * np.eye(D)
    
    return means, covariances, weights

def gmm_em(X, n_clusters=5, max_iters=100, tol=1e-4, seed=None, verbose=False):
    """
    Perform Gaussian Mixture Model clustering using the EM algorithm.
    """
    means, covariances, weights = initialize_gmm_parameters(X, n_clusters, seed, method='kmeans++')
    
    prev_log_likelihood = None
    
    for iteration in range(max_iters):
        responsibilities = e_step(X, means, covariances, weights)
        means, covariances, weights = m_step(X, responsibilities)
        
        log_likelihood = compute_log_likelihood(X, means, covariances, weights)
        
        if verbose:
            print(f"EM iter {iteration}: log-likelihood = {log_likelihood:.4f}")
        
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
        
        prev_log_likelihood = log_likelihood
    
    return means, covariances, weights, responsibilities, log_likelihood

def assign_labels(responsibilities):
    """
    Assign each data point to the cluster with the highest responsibility.
    """
    return np.argmax(responsibilities, axis=1)