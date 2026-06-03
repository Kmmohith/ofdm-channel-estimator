"""
visualizer.py
Publication-ready interactive plots using Plotly.

All functions return a plotly.graph_objects.Figure.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ #
#  Colour palette  (consistent across all plots)                       #
# ------------------------------------------------------------------ #

PALETTE = {
    "LS":      "#EF553B",
    "MMSE":    "#00CC96",
    "DL-CNN":  "#636EFA",
    "DL-MLP":  "#AB63FA",
    "Perfect": "#FFA15A",
    "default": "#19D3F3",
}
DASH_MAP = {"LS": "dot", "MMSE": "dash", "DL-CNN": "solid",
            "DL-MLP": "dashdot", "Perfect": "longdash"}


def _color(label: str) -> str:
    return PALETTE.get(label, PALETTE["default"])


def _dash(label: str) -> str:
    return DASH_MAP.get(label, "solid")


# ------------------------------------------------------------------ #
#  BER vs SNR                                                          #
# ------------------------------------------------------------------ #

def plot_ber_vs_snr(results: dict) -> go.Figure:
    snr = results["snr_range"]
    fig = go.Figure()
    for label, ber in results["ber"].items():
        fig.add_trace(go.Scatter(
            x=snr, y=ber,
            mode="lines+markers",
            name=label,
            line=dict(color=_color(label), dash=_dash(label), width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title="BER vs SNR",
        xaxis_title="SNR (dB)",
        yaxis_title="Bit Error Rate",
        yaxis_type="log",
        yaxis=dict(exponentformat="e"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ------------------------------------------------------------------ #
#  NMSE vs SNR                                                         #
# ------------------------------------------------------------------ #

def plot_nmse_vs_snr(results: dict) -> go.Figure:
    snr = results["snr_range"]
    fig = go.Figure()
    for label, nmse in results["nmse_db"].items():
        fig.add_trace(go.Scatter(
            x=snr, y=nmse,
            mode="lines+markers",
            name=label,
            line=dict(color=_color(label), dash=_dash(label), width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title="NMSE vs SNR",
        xaxis_title="SNR (dB)",
        yaxis_title="NMSE (dB)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ------------------------------------------------------------------ #
#  Channel Capacity vs SNR                                             #
# ------------------------------------------------------------------ #

def plot_capacity_vs_snr(results: dict) -> go.Figure:
    snr = results["snr_range"]
    fig = go.Figure()
    for label, cap in results["capacity"].items():
        fig.add_trace(go.Scatter(
            x=snr, y=cap,
            mode="lines+markers",
            name=label,
            line=dict(color=_color(label), dash=_dash(label), width=2),
            marker=dict(size=7),
            fill="tonexty" if label != list(results["capacity"].keys())[0] else None,
            fillcolor=_color(label).replace(")", ",0.05)").replace("rgb(", "rgba("),
        ))
    fig.update_layout(
        title="Achievable Channel Capacity vs SNR",
        xaxis_title="SNR (dB)",
        yaxis_title="Capacity (bits/s/Hz)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ------------------------------------------------------------------ #
#  Channel response: magnitude & phase                                 #
# ------------------------------------------------------------------ #

def plot_channel_response(H_true: np.ndarray,
                          H_estimates: dict,
                          sym_idx: int = 0) -> go.Figure:
    """
    Plot true vs estimated channel magnitude and phase for one OFDM symbol.

    H_true       : (n_sym, N) complex
    H_estimates  : {label: (n_sym, N) complex}
    """
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Channel Magnitude |H[k]|",
                                        "Channel Phase ∠H[k]  (deg)"))
    N = H_true.shape[1]
    k = np.arange(N)

    # True channel
    fig.add_trace(go.Scatter(
        x=k, y=np.abs(H_true[sym_idx]),
        name="True", line=dict(color="white", width=2.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=k, y=np.angle(H_true[sym_idx], deg=True),
        name="True", line=dict(color="white", width=2.5),
        showlegend=False,
    ), row=2, col=1)

    for label, H_est in H_estimates.items():
        fig.add_trace(go.Scatter(
            x=k, y=np.abs(H_est[sym_idx]),
            name=label,
            line=dict(color=_color(label), dash=_dash(label), width=1.8)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=k, y=np.angle(H_est[sym_idx], deg=True),
            name=label,
            line=dict(color=_color(label), dash=_dash(label), width=1.8),
            showlegend=False,
        ), row=2, col=1)

    fig.update_xaxes(title_text="Subcarrier index k")
    fig.update_yaxes(title_text="|H[k]|", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)",  row=2, col=1)
    fig.update_layout(height=550, template="plotly_dark",
                      hovermode="x unified", title="Channel Frequency Response")
    return fig


# ------------------------------------------------------------------ #
#  Channel impulse response (CIR)                                      #
# ------------------------------------------------------------------ #

def plot_cir(h: np.ndarray) -> go.Figure:
    n = np.arange(len(h))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("CIR Magnitude", "CIR Phase (deg)"))
    fig.add_trace(go.Bar(x=n, y=np.abs(h),
                         marker_color="#636EFA", name="|h[n]|"), row=1, col=1)
    fig.add_trace(go.Bar(x=n, y=np.angle(h, deg=True),
                         marker_color="#EF553B", name="∠h[n]"), row=1, col=2)
    fig.update_layout(height=380, template="plotly_dark",
                      title="Channel Impulse Response",
                      showlegend=False)
    fig.update_xaxes(title_text="Tap index n")
    return fig


# ------------------------------------------------------------------ #
#  Constellation diagram                                               #
# ------------------------------------------------------------------ #

def plot_constellation(rx_eq: np.ndarray, title: str = "Constellation") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.real(rx_eq).flatten(),
        y=np.imag(rx_eq).flatten(),
        mode="markers",
        marker=dict(size=3, color="#636EFA", opacity=0.6),
    ))
    # Reference QPSK points
    ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    fig.add_trace(go.Scatter(
        x=np.real(ref), y=np.imag(ref),
        mode="markers",
        marker=dict(size=14, color="red", symbol="x"),
        name="QPSK ideal",
    ))
    fig.update_layout(
        title=title, xaxis_title="In-Phase", yaxis_title="Quadrature",
        template="plotly_dark", height=430,
        xaxis=dict(range=[-2, 2]), yaxis=dict(range=[-2, 2]),
        xaxis_scaleanchor="y",
    )
    return fig


# ------------------------------------------------------------------ #
#  Training loss curves                                                #
# ------------------------------------------------------------------ #

def plot_training_curves(train_losses: list, val_losses: list) -> go.Figure:
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines",
                             name="Train Loss",
                             line=dict(color="#636EFA", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode="lines",
                             name="Val Loss",
                             line=dict(color="#EF553B", width=2, dash="dash")))
    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        yaxis_type="log",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


# ------------------------------------------------------------------ #
#  Channel heatmap (time-frequency grid)                               #
# ------------------------------------------------------------------ #

def plot_channel_heatmap(H: np.ndarray, title: str = "Channel |H|") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=np.abs(H),
        colorscale="Viridis",
        colorbar=dict(title="|H|"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Subcarrier index",
        yaxis_title="OFDM symbol",
        template="plotly_dark",
        height=350,
    )
    return fig


# ------------------------------------------------------------------ #
#  Error distribution histogram                                        #
# ------------------------------------------------------------------ #

def plot_error_histogram(H_est: np.ndarray, H_true: np.ndarray,
                         label: str = "Estimator") -> go.Figure:
    err_mag = np.abs(H_est - H_true).flatten()
    fig = go.Figure(go.Histogram(
        x=err_mag,
        nbinsx=60,
        marker_color=_color(label),
        opacity=0.8,
        name=label,
    ))
    fig.update_layout(
        title=f"Channel Estimation Error Distribution — {label}",
        xaxis_title="|H_est - H_true|",
        yaxis_title="Count",
        template="plotly_dark",
        height=380,
    )
    return fig


# ------------------------------------------------------------------ #
#  Capacity gap analysis                                               #
# ------------------------------------------------------------------ #

def plot_capacity_gap(results: dict) -> go.Figure:
    snr = results["snr_range"]
    cap = results["capacity"]
    labels = [lb for lb in cap.keys() if lb != "Perfect"]
    perfect = np.array(cap.get("Perfect", [np.nan] * len(snr)))

    fig = go.Figure()
    for lb in labels:
        gap = perfect - np.array(cap[lb])
        fig.add_trace(go.Bar(
            x=snr, y=gap, name=lb,
            marker_color=_color(lb), opacity=0.8,
        ))
    fig.update_layout(
        title="Capacity Gap vs Perfect Channel (bits/s/Hz)",
        xaxis_title="SNR (dB)",
        yaxis_title="Capacity Gap",
        barmode="group",
        template="plotly_dark",
    )
    return fig
