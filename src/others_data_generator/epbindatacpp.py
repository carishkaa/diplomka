import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect

findRhoBin

def vec_multinom(probs):
    k = len(probs)
    ans = np.random.multinomial(1, probs)
    total = sum(ans[i] * (i + 1) for i in range(k))
    return total

def check_bounds_bin(p1, p2, d):
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
        est = norm.cdf(bounds[0], bounds[1], rho)
        return est - target
    
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
