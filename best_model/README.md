# Лучшая модель: Matter-Antimatter Operator v3

**Достигнутая ошибка:** ~25.6% (среднее на β∈[2.2,4.4])  
**Параметры:** см. `params_best.json`  
**Код:** `matter_antimatter_operator_v3.py`

## Запуск

```bash
python matter_antimatter_operator_v3.py
---

## 5. `best_model/matter_antimatter_operator_v3.py`

Этот код — оптимизированная версия с параметрами, дающими ошибку ~25-35%. Я приведу полный код (он уже был в диалоге, но добавлю небольшие комментарии).

```python
import numpy as np
from sympy import primerange, zeta
import matplotlib.pyplot as plt
from scipy.linalg import expm

LIMIT = 5_000_000
MODULUS = 30

primes = list(primerange(2, LIMIT))
residues = [r for r in range(1, MODULUS) if np.gcd(r, MODULUS) == 1]
res_to_idx = {r: i for i, r in enumerate(residues)}
dim = len(residues)

print(f"Размерность: {dim} → полная {3*dim}")

# Transition matrix с весами 1/log(gap)^0.85
T = np.zeros((dim, dim))
for i in range(len(primes)-1):
    a = primes[i] % MODULUS
    b = primes[i+1] % MODULUS
    if a in res_to_idx and b in res_to_idx:
        ia = res_to_idx[a]
        ib = res_to_idx[b]
        gap = primes[i+1] - primes[i]
        weight = 1.0 / (np.log(gap + 1.7) ** 0.85)
        T[ia, ib] += weight

row_sums = T.sum(axis=1, keepdims=True)
T_norm = np.divide(T, row_sums, where=row_sums > 0)

H_base = -np.log(np.maximum(T_norm, 1e-10))
H_base = (H_base + H_base.T) / 2

# Параметры (лучшая найденная комбинация)
annihilation = 2.3
neutral = 1.55
coupling = 0.85
reg = 1.48
global_scale = 7.1

H_m = H_base.copy()
H_a = H_base.copy()
H_n = np.eye(dim) * neutral

H_full = np.block([
    [H_m,          -annihilation*np.eye(dim),   -coupling*H_n],
    [-annihilation*np.eye(dim),  H_a,          -coupling*H_n],
    [-coupling*H_n, -coupling*H_n,   H_n]
])

H_full += np.eye(3*dim) * reg
# Мягкий шум для стабильности
np.random.seed(42)
H_full += np.random.normal(0, 0.03, H_full.shape) * 0.1
H_full = H_full / np.max(np.abs(H_full)) * global_scale

eig = np.linalg.eigvalsh(H_full)
print(f"Eig min: {eig.min():.4f}   max: {eig.max():.4f}")

def Z(beta):
    return np.real(np.trace(expm(-beta * H_full)))

betas = np.linspace(1.4, 5.4, 140)
zeta_vals = np.array([float(zeta(1 + b).evalf()) for b in betas])
z_vals = np.array([Z(b) for b in betas])

mask = (betas >= 2.2) & (betas <= 4.5)
scale = np.median(zeta_vals[mask] / z_vals[mask])
z_scaled = z_vals * scale

error = np.abs(zeta_vals[mask] - z_scaled[mask]) / zeta_vals[mask] * 100
print(f"Scale: {scale:.4f}")
print(f"Средняя ошибка (2.2-4.5): {np.mean(error):.2f}%")

plt.figure(figsize=(12,6))
plt.plot(betas, zeta_vals, 'b-', label='ζ(1+β)')
plt.plot(betas, z_scaled, 'r--', label='Z scaled')
plt.legend()
plt.grid(True)
plt.title(f"Matter-Antimatter Operator, error = {np.mean(error):.1f}%")
plt.show()
