"""
estimators.py
Plug-and-play channel estimators: LS, MMSE, DL (CNN/MLP).

All estimators share the same interface:
    estimate(rx_freq, ofdm, snr_db=None) → H_est  (n_sym, N_sub) complex
"""

import numpy as np
import torch
import torch.nn as nn

from .ofdm_system import OFDMSystem


class BaseEstimator:
    """Abstract base class for channel estimators."""
    label: str = "Base"

    def estimate(self, rx_freq: np.ndarray,
                 ofdm: OFDMSystem,
                 snr_db: float = 20.0) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------ #
#  Least Squares                                                       #
# ------------------------------------------------------------------ #

class LSEstimator(BaseEstimator):
    """
    Least-Squares estimator.
    H_LS[pilot] = Y[pilot] / X[pilot], then linear interpolation.
    """
    label = "LS"

    def estimate(self, rx_freq: np.ndarray,
                 ofdm: OFDMSystem,
                 snr_db: float = 20.0) -> np.ndarray:
        H_pilots = ofdm.ls_at_pilots(rx_freq)        # (n_sym, Np)
        return ofdm.interpolate_channel(H_pilots)     # (n_sym, N)


# ------------------------------------------------------------------ #
#  MMSE (diagonal Wiener filter)                                       #
# ------------------------------------------------------------------ #

class MMSEEstimator(BaseEstimator):
    """
    Simplified MMSE (per-subcarrier Wiener filter).
    Starts from the LS estimate and applies the Wiener correction.

    H_MMSE = SNR / (SNR + 1) * H_LS     [unit channel power assumption]
    """
    label = "MMSE"

    def estimate(self, rx_freq: np.ndarray,
                 ofdm: OFDMSystem,
                 snr_db: float = 20.0) -> np.ndarray:
        H_pilots = ofdm.ls_at_pilots(rx_freq)
        H_ls     = ofdm.interpolate_channel(H_pilots)
        return ofdm.mmse_estimate(H_ls, snr_db)


# ------------------------------------------------------------------ #
#  Deep Learning estimator                                             #
# ------------------------------------------------------------------ #

class DLEstimator(BaseEstimator):
    """
    Wraps a trained PyTorch model as a plug-and-play estimator.

    The model takes the interpolated LS estimate in (2, n_sym, N) format
    and returns a refined channel estimate.
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 arch_label: str = "DL-CNN"):
        self.model  = model.to(device).eval()
        self.device = device
        self.label  = arch_label

    def estimate(self, rx_freq: np.ndarray,
                 ofdm: OFDMSystem,
                 snr_db: float = 20.0) -> np.ndarray:
        # Step 1: LS + interpolation (same as LS estimator)
        H_pilots = ofdm.ls_at_pilots(rx_freq)
        H_ls     = ofdm.interpolate_channel(H_pilots)

        # Step 2: model refinement
        x_real = ofdm.to_real(H_ls)                       # (2, n_sym, N)
        tensor  = torch.tensor(x_real, dtype=torch.float32
                               ).unsqueeze(0).to(self.device)  # (1,2,ns,N)

        with torch.no_grad():
            pred = self.model(tensor).cpu().numpy()[0]     # (2, n_sym, N)

        return ofdm.to_complex(pred)                       # (n_sym, N)


# ------------------------------------------------------------------ #
#  Perfect (Oracle) estimator  — for benchmarking only                #
# ------------------------------------------------------------------ #

class PerfectEstimator(BaseEstimator):
    """
    Returns the true channel (oracle upper bound).
    Requires true_H to be injected at construction time.
    """
    label = "Perfect"

    def __init__(self, true_H: np.ndarray):
        self._H = true_H

    def estimate(self, rx_freq: np.ndarray,
                 ofdm: OFDMSystem,
                 snr_db: float = 20.0) -> np.ndarray:
        return self._H
