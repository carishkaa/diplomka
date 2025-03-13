import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import bisect
from scipy.stats import multivariate_normal, norm
from statsmodels.stats.correlation_tools import cov_nearest

def assert_positive_semi_definite(matrix):
    if not np.all(np.linalg.eigvals(matrix) > 0):
        raise ValueError("Matrix not positive semi-definite")

def build_cor_mat(nvars, cor_matrix):
    if nvars != cor_matrix.shape[0]:
        raise ValueError("Length of mean vector mismatched with correlation matrix")
    
    if not np.allclose(cor_matrix, cor_matrix.T):
        raise ValueError("Correlation matrix not symmetric")
    
    assert_positive_semi_definite(cor_matrix)
    return cor_matrix

# Function to generate binary data matrix
def gen_bin_ep(n, marginal_probas, corr_matrix):
    # Length of p
    nvars = len(marginal_probas)
    
    # Generate the correlation matrix
    phicorr = get_rho_mat(nvars, marginal_probas, corr_matrix)
    
    # Ensure the correlation matrix is positive semi-definite
    # assert_positive_semi_definite(phicorr)
    if not np.all(np.linalg.eigvals(phicorr) > 0):
        phicorr = cov_nearest(phicorr)
    
    # Generate multivariate normal data with zero mean and the correlation matrix as the covariance matrix
    normvars = np.random.multivariate_normal(np.zeros(nvars), phicorr, size=n)
    
    # Quantile for each probability in p (inverse of the CDF)
    z = np.tile(norm.ppf(marginal_probas), (n, 1))
    
    # Generate binary variables
    binvars = (normvars < z).astype(int)
    
    # Create a DataFrame similar to the `data.table` in R
    dtX = pd.DataFrame(binvars, columns=[f"X{i+1}" for i in range(nvars)])
    # dtX['id'] = dtX.index + 1  # ID from 1 to n
    
    return dtX

def check_bounds_bin(p1, p2, d):
    d = np.ceil(d * 1e12) / 1e12 # Round to 12 decimal places
    l = (p1 * p2) / ((1 - p1) * (1 - p2))
    L = max(-np.sqrt(l), -np.sqrt(1 / l))
    u = (p1 * (1 - p2)) / (p2 * (1 - p1))
    U = min(np.sqrt(u), np.sqrt(1 / u))
    
    if d < L or d > U:
        raise ValueError(f"Specified correlation {d} out of range ({L} ... {U})")

def find_rho_bin(p1, p2, d):
    check_bounds_bin(p1, p2, d)
    
    target = d * np.sqrt(p1 * p2 * (1 - p1) * (1 - p2)) + p1 * p2
    bounds = [norm.ppf(p1), norm.ppf(p2)]
    
    def objective(rho):
        corr = np.identity(2)
        corr[0, 1] = corr[1, 0] = rho
        est = multivariate_normal.cdf(bounds, mean=[0, 0], cov=corr, allow_singular=True)
        return est - target

    # print('[-1] = ', objective(-1))
    rho = bisect(objective, -1, 1)
    return rho

def get_rho_mat(N, P, TCORR):
    PCORR = np.zeros_like(TCORR)
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            p1 = P[i]
            p2 = P[j]
            rho = find_rho_bin(p1, p2, TCORR[i, j])
            PCORR[i, j] = rho
            PCORR[j, i] = rho
    
    np.fill_diagonal(PCORR, 1)
    return PCORR
