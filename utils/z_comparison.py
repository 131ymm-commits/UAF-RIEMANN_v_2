import numpy as np
from scipy.linalg import expm

def compute_Z(beta, H_full):
    return np.real(np.trace(expm(-beta * H_full)))

def compare_with_zeta(betas, H_full, zeta_vals, mask_range=(2.2,4.4)):
    z_vals = np.array([compute_Z(b, H_full) for b in betas])
    mask = (betas >= mask_range[0]) & (betas <= mask_range[1])
    scale = np.median(zeta_vals[mask] / z_vals[mask])
    z_scaled = z_vals * scale
    error = np.abs(zeta_vals[mask] - z_scaled[mask]) / zeta_vals[mask] * 100
    return z_scaled, error.mean(), error.min(), error.max(), scale
