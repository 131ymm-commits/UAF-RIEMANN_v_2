import numpy as np
from sympy import primerange, zeta, dirichlet_eta
from scipy.linalg import expm

LIMIT = 4_000_000
MODULUS = 30

primes = list(primerange(2, LIMIT))
residues = [r for r in range(1, MODULUS) if np.gcd(r, MODULUS) == 1]
res_to_idx = {r: i for i, r in enumerate(residues)}
dim = len(residues)

def chi4(n): 
    r = n % 4
    return 1 if r == 1 else (-1 if r == 3 else 0)

def chi5(n): 
    r = n % 5
    return 0 if r == 0 else (1 if r in (1,4) else -1)

# Построение комплексной матрицы переходов
T = np.zeros((dim, dim), dtype=complex)
for i in range(len(primes)-1):
    p, q = primes[i], primes[i+1]
    a, b = p % MODULUS, q % MODULUS
    if a not in res_to_idx or b not in res_to_idx:
        continue
    ia, ib = res_to_idx[a], res_to_idx[b]
    gap = q - p
    base_weight = 1.0 / (np.log(gap + 1.7) ** 0.85)
    c4 = chi4(p) * chi4(q)
    c5 = chi5(p) * chi5(q)
    S = c4 + 1j * c5
    phase = np.angle(S) if S != 0 else 0.0
    weight = base_weight * np.exp(1j * phase)
    T[ia, ib] += weight

# Нормализация и вещественный эрмитов базовый гамильтониан
row_sums = np.sum(np.abs(T), axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1e-12)
T_norm = T / row_sums
H_base = -np.log(np.maximum(np.abs(T_norm), 1e-10))
H_base = (H_base + H_base.T) / 2
H_base = np.real(H_base)

# Параметры (лучшие)
annihilation = 2.1
neutral = 1.5
coupling = 0.7
reg = 1.48
global_scale = 7.1

H_m = H_base.copy()
H_a = H_base.copy()
H_n = np.eye(dim) * neutral

H_full = np.block([
    [H_m,          -annihilation * np.eye(dim),   -coupling * H_n],
    [-annihilation * np.eye(dim),  H_a,           -coupling * H_n],
    [-coupling * H_n, -coupling * H_n,            H_n]
])
H_full += np.eye(3 * dim) * reg
np.random.seed(42)
H_full += np.random.normal(0, 0.03, H_full.shape) * 0.1
H_full = H_full / np.max(np.abs(H_full)) * global_scale

# Статистическая сумма
def Z(beta):
    return np.real(np.trace(expm(-beta * H_full)))

# Целевые функции
betas = np.linspace(1.4, 5.4, 100)
zeta_vals = np.array([float(zeta(1 + b).evalf()) for b in betas])
L4_vals = np.array([float(dirichlet_eta(1 + b).evalf()) for b in betas])

def L5_approx(s, max_n=20000):
    s = complex(s)
    total = 0j
    for n in range(1, max_n):
        chi = chi5(n)
        if chi:
            total += chi / (n ** s)
    return abs(total)

L5_vals = np.array([L5_approx(1 + b) for b in betas])
z_vals = np.array([Z(b) for b in betas])

# Ошибки в диапазоне [2.2, 4.4]
mask = (betas >= 2.2) & (betas <= 4.4)
scale_zeta = np.median(zeta_vals[mask] / z_vals[mask])
err_zeta = np.abs(zeta_vals[mask] - z_vals[mask] * scale_zeta) / zeta_vals[mask] * 100
scale_L4 = np.median(L4_vals[mask] / z_vals[mask])
err_L4 = np.abs(L4_vals[mask] - z_vals[mask] * scale_L4) / L4_vals[mask] * 100
scale_L5 = np.median(L5_vals[mask] / z_vals[mask])
err_L5 = np.abs(L5_vals[mask] - z_vals[mask] * scale_L5) / L5_vals[mask] * 100

print(f"ζ(s)        : {np.mean(err_zeta):.2f}%")
print(f"L(s,χ₄)     : {np.mean(err_L4):.2f}%")
print(f"L(s,χ₅)     : {np.mean(err_L5):.2f}%")
