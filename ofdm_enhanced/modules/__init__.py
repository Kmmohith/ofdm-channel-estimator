"""OFDM Channel Estimation — modules package."""
from .ofdm_system    import OFDMSystem
from .channel_models import ChannelModel
from .dataset        import DatasetGenerator, OFDMChannelDataset
from .models         import ChannelNetCNN, ChannelNetMLP, build_model
from .estimators     import LSEstimator, MMSEEstimator, DLEstimator, PerfectEstimator
from .trainer        import Trainer
from .evaluator      import Evaluator, nmse, ber_one_frame, channel_capacity
from .visualizer     import (
    plot_ber_vs_snr, plot_nmse_vs_snr, plot_capacity_vs_snr,
    plot_channel_response, plot_cir, plot_constellation,
    plot_training_curves, plot_channel_heatmap,
    plot_error_histogram, plot_capacity_gap,
)

__all__ = [
    "OFDMSystem", "ChannelModel",
    "DatasetGenerator", "OFDMChannelDataset",
    "ChannelNetCNN", "ChannelNetMLP", "build_model",
    "LSEstimator", "MMSEEstimator", "DLEstimator", "PerfectEstimator",
    "Trainer", "Evaluator",
    "nmse", "ber_one_frame", "channel_capacity",
    "plot_ber_vs_snr", "plot_nmse_vs_snr", "plot_capacity_vs_snr",
    "plot_channel_response", "plot_cir", "plot_constellation",
    "plot_training_curves", "plot_channel_heatmap",
    "plot_error_histogram", "plot_capacity_gap",
]
