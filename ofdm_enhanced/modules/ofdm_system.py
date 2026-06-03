"""
ofdm_system.py
Core OFDM signal processing: modulation, pilot handling, IFFT/FFT, CP, noise.
Uses comb-type pilot structure for 2D time-frequency channel estimation.
"""

import numpy as np


class OFDMSystem:
    """
    OFDM system with comb-type pilot structure.

    Frame layout:
      - n_sym OFDM symbols per frame
      - Every symbol has Np pilots evenly spaced across N subcarriers
      - Remaining subcarriers carry QPSK data

    Parameters
    ----------
    n_sub    : number of subcarriers (FFT size)
    cp_len   : cyclic prefix length
    n_pilots : pilots per OFDM symbol
    n_sym    : OFDM symbols per frame
    """

    def __init__(self, n_sub=64, cp_len=16, n_pilots=8, n_sym=14):
        self.N      = n_sub
        self.CP     = cp_len
        self.Np     = n_pilots
        self.n_sym  = n_sym

        # --- Pilot / data index sets ---
        self.spacing   = n_sub // n_pilots
        self.pilot_idx = np.arange(0, n_sub, self.spacing)[:n_pilots]
        self.data_idx  = np.setdiff1d(np.arange(n_sub), self.pilot_idx)
        self.n_data    = len(self.data_idx)

        # Known pilot symbols (unit-power BPSK → all 1+0j)
        self.pilot_syms = np.ones(n_pilots, dtype=complex)

    # ------------------------------------------------------------------ #
    #  Modulation                                                          #
    # ------------------------------------------------------------------ #

    def qpsk_mod(self, bits: np.ndarray) -> np.ndarray:
        """Map pairs of bits to QPSK symbols (Gray coded, normalised)."""
        bits = np.asarray(bits, dtype=int).reshape(-1, 2)
        s = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
        return s / np.sqrt(2)

    def qpsk_demod(self, symbols: np.ndarray) -> np.ndarray:
        """Hard-decision QPSK demodulation → bit vector."""
        bits = np.zeros((len(symbols), 2), dtype=int)
        bits[:, 0] = (np.real(symbols) > 0).astype(int)
        bits[:, 1] = (np.imag(symbols) > 0).astype(int)
        return bits.reshape(-1)

    # ------------------------------------------------------------------ #
    #  Transmitter                                                         #
    # ------------------------------------------------------------------ #

    def build_tx_frame(self, data_bits: np.ndarray):
        """
        Construct an OFDM frame from data bits.

        Returns
        -------
        freq_frame : (n_sym, N) complex  — frequency-domain frame
        time_frame : (n_sym, N+CP) complex — time-domain frame (with CP)
        """
        n_need = self.n_sym * self.n_data * 2
        bits = np.asarray(data_bits[:n_need], dtype=int)
        data_syms = self.qpsk_mod(bits).reshape(self.n_sym, self.n_data)

        freq_frame = np.zeros((self.n_sym, self.N), dtype=complex)
        for s in range(self.n_sym):
            freq_frame[s, self.pilot_idx] = self.pilot_syms
            freq_frame[s, self.data_idx]  = data_syms[s]

        time_frame = np.array([
            self._add_cp(np.fft.ifft(freq_frame[s]))
            for s in range(self.n_sym)
        ])
        return freq_frame, time_frame

    # ------------------------------------------------------------------ #
    #  Receiver                                                            #
    # ------------------------------------------------------------------ #

    def rx_frame(self, rx_time: np.ndarray) -> np.ndarray:
        """Remove CP and apply FFT to each received symbol. → (n_sym, N)"""
        return np.array([
            np.fft.fft(self._rem_cp(rx_time[s]))
            for s in range(self.n_sym)
        ])

    # ------------------------------------------------------------------ #
    #  Channel Estimation helpers                                          #
    # ------------------------------------------------------------------ #

    def ls_at_pilots(self, rx_freq: np.ndarray) -> np.ndarray:
        """
        LS estimate at pilot positions.
        H_LS = Y_p / X_p  →  (n_sym, Np)
        """
        return rx_freq[:, self.pilot_idx] / self.pilot_syms[np.newaxis, :]

    def interpolate_channel(self, H_pilots: np.ndarray) -> np.ndarray:
        """
        Linear interpolation from pilot positions to all N subcarriers.
        H_pilots : (n_sym, Np)  →  (n_sym, N)
        """
        H_full = np.zeros((self.n_sym, self.N), dtype=complex)
        x_all = np.arange(self.N)
        for s in range(self.n_sym):
            H_full[s] = (
                np.interp(x_all, self.pilot_idx, H_pilots[s].real)
                + 1j * np.interp(x_all, self.pilot_idx, H_pilots[s].imag)
            )
        return H_full

    def mmse_estimate(self, H_ls: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Per-subcarrier Wiener filter (diagonal-MMSE approximation).
        Assumes unit channel power → σ_H² = 1.
        H_MMSE = SNR / (SNR + 1) * H_LS
        """
        snr_lin = 10 ** (snr_db / 10)
        alpha   = snr_lin / (snr_lin + 1)
        return alpha * H_ls

    # ------------------------------------------------------------------ #
    #  Noise                                                               #
    # ------------------------------------------------------------------ #

    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add complex AWGN at the specified SNR (dB)."""
        snr_lin = 10 ** (snr_db / 10)
        power   = np.mean(np.abs(signal) ** 2)
        n_var   = power / snr_lin
        noise   = np.sqrt(n_var / 2) * (
            np.random.randn(*signal.shape)
            + 1j * np.random.randn(*signal.shape)
        )
        return signal + noise

    # ------------------------------------------------------------------ #
    #  Format conversion                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_real(H: np.ndarray) -> np.ndarray:
        """(n_sym, N) complex  →  (2, n_sym, N) float32  [real | imag]"""
        return np.stack([H.real, H.imag], axis=0).astype(np.float32)

    @staticmethod
    def to_complex(H_real: np.ndarray) -> np.ndarray:
        """(2, n_sym, N) float32  →  (n_sym, N) complex"""
        return H_real[0] + 1j * H_real[1]

    # ------------------------------------------------------------------ #
    #  Misc                                                                #
    # ------------------------------------------------------------------ #

    def bits_per_frame(self) -> int:
        return self.n_sym * self.n_data * 2

    def _add_cp(self, s: np.ndarray) -> np.ndarray:
        return np.concatenate([s[-self.CP:], s])

    def _rem_cp(self, s: np.ndarray) -> np.ndarray:
        return s[self.CP:]
