import numpy as np
from sympy import primerange

def build_transition_matrix(limit=5_000_000, modulus=30, gap_exponent=0.85):
    primes = list(primerange(2, limit))
    residues = [r for r in range(1, modulus) if np.gcd(r, modulus) == 1]
    res_to_idx = {r: i for i, r in enumerate(residues)}
    dim = len(residues)
    T = np.zeros((dim, dim))
    for i in range(len(primes)-1):
        a = primes[i] % modulus
        b = primes[i+1] % modulus
        if a in res_to_idx and b in res_to_idx:
            ia = res_to_idx[a]
            ib = res_to_idx[b]
            gap = primes[i+1] - primes[i]
            weight = 1.0 / (np.log(gap + 1.7) ** gap_exponent)
            T[ia, ib] += weight
    row_sums = T.sum(axis=1, keepdims=True)
    T_norm = np.divide(T, row_sums, where=row_sums > 0)
    return T_norm, residues, res_to_idx
