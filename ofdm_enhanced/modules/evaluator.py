"""
evaluator.py
Evaluation metrics: NMSE, BER vs SNR, Channel Capacity vs SNR.
"""

import numpy as np
from typing import Dict, List

from .ofdm_system    import OFDMSystem
from .channel_models import ChannelModel
from .estimators     import BaseEstimator


# ------------------------------------------------------------------ #
#  Per-sample metrics                                                  #
# ------------------------------------------------------------------ #

def nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Normalised Mean Square Error (linear scale)."""
    num = np.mean(np.abs(H_est - H_true) ** 2)
    den = np.mean(np.abs(H_true) ** 2) + 1e-12
    return float(num / den)


def ber_one_frame(H_est: np.ndarray, rx_freq: np.ndarray,
                  tx_freq: np.ndarray, bits: np.ndarray,
                  ofdm: OFDMSystem) -> float:
    """BER for a single OFDM frame using equalised symbols."""
    # Zero-forcing equalisation
    H_safe = H_est + 1e-10                         # avoid divide-by-zero
    eq_syms = rx_freq[:, ofdm.data_idx] / H_safe[:, ofdm.data_idx]

    bits_hat = ofdm.qpsk_demod(eq_syms.flatten())
    bits_ref = bits[: len(bits_hat)]
    errors   = np.sum(bits_hat != bits_ref)
    return errors / len(bits_ref)


def channel_capacity(H: np.ndarray, snr_db: float) -> float:
    """
    Average Shannon capacity over all subcarriers (bits/s/Hz).
    C = mean_k [ log2(1 + SNR * |H[k]|²) ]
    """
    snr_lin = 10 ** (snr_db / 10)
    gain    = np.abs(H) ** 2
    return float(np.mean(np.log2(1 + snr_lin * gain)))


# ------------------------------------------------------------------ #
#  SNR sweep evaluation                                                #
# ------------------------------------------------------------------ #

class Evaluator:
    """
    Run Monte-Carlo BER, NMSE, and capacity evaluations over a SNR range
    for multiple estimators simultaneously.

    Parameters
    ----------
    ofdm        : OFDMSystem
    channel     : ChannelModel
    estimators  : dict  { label: BaseEstimator }
    n_trials    : Monte-Carlo trials per SNR point
    seed        : random seed
    """

    def __init__(self, ofdm: OFDMSystem, channel: ChannelModel,
                 estimators: Dict[str, BaseEstimator],
                 n_trials: int = 300,
                 seed: int = 0):
        self.ofdm       = ofdm
        self.channel    = channel
        self.estimators = estimators
        self.n_trials   = n_trials
        self.rng        = np.random.default_rng(seed)

    # ---------------------------------------------------------------- #

    def run(self, snr_range: List[float],
            progress_callback=None) -> Dict:
        """
        Evaluate all estimators across snr_range.

        Returns a dict with keys:
            snr_range, ber, nmse_db, capacity
        Each value is itself a dict {label: [values per SNR point]}.
        """
        labels  = list(self.estimators.keys())
        results = {
            "snr_range": snr_range,
            "ber":       {lb: [] for lb in labels},
            "nmse_db":   {lb: [] for lb in labels},
            "capacity":  {lb: [] for lb in labels},
        }

        n_snr = len(snr_range)
        for i, snr_db in enumerate(snr_range):
            print(f"  SNR={snr_db:5.1f} dB  ({i+1}/{n_snr})", end="  ")

            # Accumulators
            ber_acc  = {lb: 0.0 for lb in labels}
            nmse_acc = {lb: 0.0 for lb in labels}
            cap_acc  = {lb: 0.0 for lb in labels}

            for _ in range(self.n_trials):
                # Generate frame
                bits = self.rng.integers(0, 2, self.ofdm.bits_per_frame()).astype(int)
                freq_tx, time_tx = self.ofdm.build_tx_frame(bits)

                # Random channel
                h = self.channel.generate()
                H_true = np.tile(
                    self.channel.freq_response(h, self.ofdm.N),
                    (self.ofdm.n_sym, 1)
                )

                # Propagate + noise
                rx_time = np.zeros_like(time_tx)
                for s in range(self.ofdm.n_sym):
                    rx_s = self.channel.apply(time_tx[s], h)
                    rx_time[s] = self.ofdm.add_awgn(rx_s, snr_db)

                rx_freq = self.ofdm.rx_frame(rx_time)

                # Evaluate each estimator
                for lb, est in self.estimators.items():
                    H_est = est.estimate(rx_freq, self.ofdm, snr_db)
                    ber_acc[lb]  += ber_one_frame(H_est, rx_freq, freq_tx,
                                                   bits, self.ofdm)
                    nmse_acc[lb] += nmse(H_est, H_true)
                    cap_acc[lb]  += channel_capacity(H_est, snr_db)

            # Average over trials
            for lb in labels:
                ber_val = ber_acc[lb]  / self.n_trials
                nmse_val = nmse_acc[lb] / self.n_trials
                cap_val  = cap_acc[lb]  / self.n_trials

                results["ber"][lb].append(max(ber_val, 1e-6))   # floor for log plot
                results["nmse_db"][lb].append(
                    10 * np.log10(nmse_val + 1e-12)
                )
                results["capacity"][lb].append(cap_val)
                print(f"{lb}: BER={ber_val:.4f}", end="  ")
            print()

            if progress_callback:
                progress_callback(i + 1, n_snr)

        return results
