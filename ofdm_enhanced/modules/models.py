"""
models.py
Neural network architectures for OFDM channel estimation.

Models
------
ChannelNetCNN  – Two-stage CNN: SRCNN (super-resolution) + DnCNN (denoising)
                 Input/Output: (B, 2, n_sym, N_sub)  [real/imag channels]

ChannelNetMLP  – Simple MLP baseline.
                 Flattens the input, passes through FC layers, reshapes output.
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  ChannelNet CNN  (SRCNN + DnCNN)                                    #
# ------------------------------------------------------------------ #

class _SRBlock(nn.Module):
    """Super-Resolution sub-network (channel interpolation refinement)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32,  2, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class _DnBlock(nn.Module):
    """Denoising sub-network with residual learning (DnCNN-style)."""
    def __init__(self, n_layers: int = 5):
        super().__init__()
        layers = [nn.Conv2d(2, 32, 3, padding=1), nn.PReLU()]
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
            ]
        layers.append(nn.Conv2d(32, 2, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)          # residual connection


class ChannelNetCNN(nn.Module):
    """
    Two-stage CNN for channel estimation.

    Stage 1 (SR block)  : Enhances pilot-interpolated LS estimate
    Stage 2 (Dn block)  : Removes noise via learned residual

    Input  : (B, 2, n_sym, N_sub)  — interpolated LS estimate [real, imag]
    Output : (B, 2, n_sym, N_sub)  — refined channel estimate  [real, imag]
    """

    def __init__(self, dn_layers: int = 5):
        super().__init__()
        self.sr = _SRBlock()
        self.dn = _DnBlock(n_layers=dn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dn(self.sr(x))

    @staticmethod
    def name() -> str:
        return "ChannelNet-CNN"


# ------------------------------------------------------------------ #
#  ChannelNet MLP  (baseline)                                          #
# ------------------------------------------------------------------ #

class ChannelNetMLP(nn.Module):
    """
    Simple MLP baseline.

    Input  : (B, 2, n_sym, N_sub)  → flattened to (B, 2*n_sym*N_sub)
    Output : same shape as input
    """

    def __init__(self, n_sym: int = 14, n_sub: int = 64,
                 hidden: list = None):
        super().__init__()
        self.n_sym = n_sym
        self.n_sub = n_sub
        in_dim = 2 * n_sym * n_sub

        if hidden is None:
            hidden = [1024, 512, 256]

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        flat = x.view(B, -1)
        out  = self.net(flat)
        return out.view(B, 2, self.n_sym, self.n_sub)

    @staticmethod
    def name() -> str:
        return "ChannelNet-MLP"


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

def build_model(arch: str, n_sym: int = 14, n_sub: int = 64,
                device: str = "cpu") -> nn.Module:
    """
    Convenience factory.

    Parameters
    ----------
    arch   : 'cnn' or 'mlp'
    n_sym  : OFDM symbols per frame
    n_sub  : subcarriers per symbol
    device : 'cpu' or 'cuda'
    """
    arch = arch.lower()
    if arch == "cnn":
        model = ChannelNetCNN()
    elif arch == "mlp":
        model = ChannelNetMLP(n_sym=n_sym, n_sub=n_sub)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Use 'cnn' or 'mlp'.")
    return model.to(device)
