"""
channel_models.py
Channel impulse response generators: Rayleigh, Rician, ETU multipath.
"""

import numpy as np


class ChannelModel:
    """
    Factory for wireless channel models.

    Supported types
    ---------------
    'rayleigh'  – i.i.d. Rayleigh fading taps
    'rician'    – Rician fading with LOS component
    'etu'       – Fixed ETU-like 4-tap multipath profile
    """

    def __init__(self, model: str = "rayleigh", n_taps: int = 4,
                 k_factor: float = 3.0, rng: np.random.Generator = None):
        model = model.lower()
        if model not in ("rayleigh", "rician", "etu"):
            raise ValueError(f"Unknown channel model: {model!r}")
        self.model    = model
        self.n_taps   = n_taps
        self.K        = k_factor
        self.rng      = rng or np.random.default_rng()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self) -> np.ndarray:
        """Return a random channel impulse response (complex, length n_taps)."""
        if self.model == "rayleigh":
            return self._rayleigh()
        elif self.model == "rician":
            return self._rician()
        else:
            return self._etu()

    def freq_response(self, h: np.ndarray, n_fft: int) -> np.ndarray:
        """
        Compute the frequency-domain channel from taps h.
        Zero-pads h to n_fft and applies FFT.
        """
        h_pad = np.zeros(n_fft, dtype=complex)
        h_pad[:len(h)] = h
        return np.fft.fft(h_pad)

    def apply(self, tx_time: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Pass a single time-domain OFDM symbol through the channel.
        Uses linear convolution, output truncated to input length.
        """
        return np.convolve(tx_time, h, mode="full")[: len(tx_time)]

    # ------------------------------------------------------------------ #
    #  Private generators                                                  #
    # ------------------------------------------------------------------ #

    def _rayleigh(self) -> np.ndarray:
        """i.i.d. CN(0, 1/L) taps; normalised to unit total power."""
        real = self.rng.standard_normal(self.n_taps)
        imag = self.rng.standard_normal(self.n_taps)
        h = (real + 1j * imag) / np.sqrt(2 * self.n_taps)
        return h / np.sqrt(np.sum(np.abs(h) ** 2))

    def _rician(self) -> np.ndarray:
        """
        Rician: LOS component + scattered component.
        K-factor controls LOS dominance.
        """
        los_amp  = np.sqrt(self.K / (self.K + 1))
        scat_amp = np.sqrt(1 / (self.K + 1))

        phases = self.rng.uniform(0, 2 * np.pi, self.n_taps)
        h_los  = los_amp  * np.exp(1j * phases)
        h_scat = scat_amp * (
            self.rng.standard_normal(self.n_taps)
            + 1j * self.rng.standard_normal(self.n_taps)
        ) / np.sqrt(2)
        h = h_los + h_scat
        return h / np.sqrt(np.sum(np.abs(h) ** 2))

    def _etu(self) -> np.ndarray:
        """
        Fixed ETU-inspired 4-tap power delay profile
        with small random phase variations for diversity.
        """
        pdp_amp = np.array([0.9, 0.45, 0.25, 0.12])  # power-delay profile
        phases  = self.rng.uniform(0, 2 * np.pi, self.n_taps)
        h = pdp_amp * np.exp(1j * phases)
        return h / np.sqrt(np.sum(np.abs(h) ** 2))
