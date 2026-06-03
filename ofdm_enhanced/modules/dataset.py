"""
dataset.py
Synthetic OFDM dataset generation for channel estimation.

Pipeline per sample
-------------------
1. Generate random channel taps h ~ chosen model
2. Build OFDM Tx frame (bits → QPSK → pilot embed → IFFT+CP)
3. Apply channel + AWGN at random SNR in [snr_min, snr_max]
4. Receive frame (remove CP → FFT)
5. LS estimate at pilots → linear interpolation to full grid
6. Input  X : (2, n_sym, N)  interpolated LS estimate [real | imag]
7. Target Y : (2, n_sym, N)  true channel response    [real | imag]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from .ofdm_system   import OFDMSystem
from .channel_models import ChannelModel


# ------------------------------------------------------------------ #
#  PyTorch Dataset                                                     #
# ------------------------------------------------------------------ #

class OFDMChannelDataset(Dataset):
    """Pre-generated in-memory dataset for channel estimation."""

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 snr_list: np.ndarray, chan_list: list):
        # X,Y : (n_samples, 2, n_sym, N)
        self.X        = torch.tensor(X, dtype=torch.float32)
        self.Y        = torch.tensor(Y, dtype=torch.float32)
        self.snr_list  = snr_list
        self.chan_list = chan_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ------------------------------------------------------------------ #
#  Generator                                                           #
# ------------------------------------------------------------------ #

class DatasetGenerator:
    """
    Generate and split the OFDM channel estimation dataset.

    Parameters
    ----------
    ofdm        : OFDMSystem instance
    channel     : ChannelModel instance
    n_samples   : total number of frames to generate
    snr_min/max : SNR drawn uniformly from [snr_min, snr_max] dB
    seed        : random seed for reproducibility
    """

    def __init__(self, ofdm: OFDMSystem, channel: ChannelModel,
                 n_samples: int = 5000,
                 snr_min: float = 0.0, snr_max: float = 30.0,
                 seed: int = 42):
        self.ofdm      = ofdm
        self.channel   = channel
        self.n_samples = n_samples
        self.snr_min   = snr_min
        self.snr_max   = snr_max
        self.seed      = seed

    # ---------------------------------------------------------------- #

    def generate(self, verbose: bool = True):
        """
        Generate the full dataset.

        Returns
        -------
        dataset : OFDMChannelDataset
        """
        rng = np.random.default_rng(self.seed)
        ofdm = self.ofdm
        ch   = self.channel
        N    = ofdm.N
        ns   = ofdm.n_sym

        X_all    = np.zeros((self.n_samples, 2, ns, N), dtype=np.float32)
        Y_all    = np.zeros((self.n_samples, 2, ns, N), dtype=np.float32)
        snr_arr  = np.zeros(self.n_samples)
        chan_arr  = []

        for i in range(self.n_samples):
            if verbose and (i % 500 == 0):
                print(f"  Generating sample {i}/{self.n_samples} ...")

            # Random SNR for this sample
            snr_db = float(rng.uniform(self.snr_min, self.snr_max))
            snr_arr[i] = snr_db

            # Random channel
            h = ch.generate()
            H_true_freq = ch.freq_response(h, N)  # (N,) complex
            # Replicate across all symbols (quasi-static assumption)
            H_true = np.tile(H_true_freq, (ns, 1))  # (ns, N)

            # Generate random bits and build Tx frame
            bits = rng.integers(0, 2, ofdm.bits_per_frame()).astype(int)
            freq_tx, time_tx = ofdm.build_tx_frame(bits)

            # Pass through channel (per OFDM symbol, in time domain)
            rx_time = np.zeros_like(time_tx)
            for s in range(ns):
                rx_s = ch.apply(time_tx[s], h)
                rx_time[s] = ofdm.add_awgn(rx_s, snr_db)

            # Receive: remove CP + FFT
            rx_freq = ofdm.rx_frame(rx_time)

            # LS estimate + interpolation
            H_ls_pilots = ofdm.ls_at_pilots(rx_freq)       # (ns, Np)
            H_ls_full   = ofdm.interpolate_channel(H_ls_pilots)  # (ns, N)

            # Store real/imag format
            X_all[i] = ofdm.to_real(H_ls_full)
            Y_all[i] = ofdm.to_real(H_true)
            chan_arr.append(ch.model)

        print(f"  Dataset generation complete: {self.n_samples} samples.")
        return OFDMChannelDataset(X_all, Y_all, snr_arr, chan_arr)

    # ---------------------------------------------------------------- #

    @staticmethod
    def split(dataset: OFDMChannelDataset,
              train_frac: float = 0.70,
              val_frac:   float = 0.15,
              seed: int   = 42):
        """
        Split dataset into train / validation / test subsets.

        Returns (train_ds, val_ds, test_ds)
        """
        n      = len(dataset)
        n_tr   = int(n * train_frac)
        n_val  = int(n * val_frac)
        n_test = n - n_tr - n_val

        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, [n_tr, n_val, n_test], generator=generator)
