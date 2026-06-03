# ofdm-channel-estimator
# 📡 OFDM Channel Estimation — Enhanced

A modular, GPU-ready framework for OFDM channel estimation comparing
**LS**, **MMSE**, and **Deep Learning (CNN/MLP)** methods with a clean Streamlit GUI.

---

## Features

| Feature | Details |
|---------|---------|
| **Channel models** | Rayleigh fading, Rician fading, ETU multipath |
| **Estimators** | LS, MMSE (Wiener filter), DL-CNN (ChannelNet), DL-MLP |
| **Metrics** | NMSE (dB), BER vs SNR, Shannon Capacity vs SNR |
| **Training** | Adam + CosineAnnealingLR, best-model checkpointing, GPU-ready |
| **GUI** | Streamlit dashboard with real-time training curves & interactive Plotly charts |

---

## Project Structure

```
ofdm_enhanced/
├── app.py                  ← Main Streamlit GUI
├── config.yaml             ← Default experiment parameters
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── ofdm_system.py      ← OFDMSystem (modulation, CP, pilots, noise)
│   ├── channel_models.py   ← ChannelModel (Rayleigh / Rician / ETU)
│   ├── dataset.py          ← DatasetGenerator + OFDMChannelDataset
│   ├── models.py           ← ChannelNetCNN + ChannelNetMLP
│   ├── estimators.py       ← LSEstimator / MMSEEstimator / DLEstimator
│   ├── trainer.py          ← Trainer (GPU-ready training loop)
│   ├── evaluator.py        ← Evaluator (BER / NMSE / Capacity)
│   └── visualizer.py       ← Plotly visualisation functions
└── models_saved/           ← Model checkpoints saved here
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the GUI
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Google Colab (GPU)

```python
# Install
!pip install streamlit torch numpy scipy plotly pandas pyyaml -q

# Upload the project zip and unzip
from google.colab import files
# upload ofdm_enhanced.zip, then:
!unzip ofdm_enhanced.zip

# Run with ngrok or localtunnel
!pip install pyngrok -q
from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen(["streamlit", "run", "ofdm_enhanced/app.py",
                          "--server.port", "8501"])
time.sleep(3)
tunnel = ngrok.connect(8501)
print("Public URL:", tunnel.public_url)
```

---

## GUI Workflow

### Tab 1 – Dataset & Channel
1. Set OFDM / channel / dataset parameters in the **sidebar**
2. Click **Generate Dataset** — builds train/val/test splits
3. Preview true channel, CIR, and LS estimate quality

### Tab 2 – Train Model
1. Choose architecture (`cnn` or `mlp`) in the sidebar
2. Click **Start Training** — watch live loss curves update
3. Best model is auto-saved to `models_saved/`

### Tab 3 – Evaluation
1. Click **Run Evaluation** — runs Monte-Carlo BER/NMSE/Capacity sweep
2. View interactive BER vs SNR, NMSE vs SNR, Capacity vs SNR plots
3. Examine the capacity gap bar chart and performance summary table

### Tab 4 – Visualise
1. Select an SNR value and click **Refresh Snapshot**
2. View channel magnitude/phase, time-frequency heatmaps, error histograms,
   and ZF-equalised constellation diagrams

---

## DL Architecture — ChannelNet

```
Input (2, n_sym, N_sub)  ← interpolated LS estimate [real|imag]
        │
    ┌───┴──────────────────────────────┐
    │  SRCNN Block (Super-Resolution)   │
    │  Conv(2→64, 9×9) → PReLU          │
    │  Conv(64→32, 1×1) → PReLU         │
    │  Conv(32→2,  5×5)                 │
    └───────────────┬──────────────────┘
                    │
    ┌───────────────┴──────────────────┐
    │  DnCNN Block (Denoising)          │
    │  Conv → PReLU → [BN+Conv+PReLU]×n │
    │  Residual: out = in - noise        │
    └───────────────┬──────────────────┘
                    │
Output (2, n_sym, N_sub)  ← refined channel estimate [real|imag]
```

---

## Key Equations

| Estimator | Formula |
|-----------|---------|
| **LS** | `H_LS[k] = Y_p[k] / X_p[k]` then linear interp |
| **MMSE** | `H_MMSE = SNR/(SNR+1) · H_LS` |
| **Capacity** | `C = mean_k log₂(1 + SNR·\|H[k]\|²)` bits/s/Hz |
| **NMSE** | `‖H_est - H_true‖² / ‖H_true‖²` |

---

## Improvements vs Original Code

| Original | Enhanced |
|----------|---------|
| Single flat file | Modular package with 8 modules |
| Fixed SNR=15 dB | Random SNR in configurable range |
| Fixed Rayleigh channel | Rayleigh / Rician / ETU selectable |
| No pilot structure | Comb-type pilots with interpolation |
| No MMSE | MMSE (Wiener filter) added |
| No BER evaluation loop | Full Monte-Carlo BER/NMSE/Capacity sweep |
| matplotlib static plots | Interactive Plotly figures |
| No GUI | Full Streamlit dashboard |
| No capacity analysis | Shannon capacity + gap analysis |
| No model loading | Checkpoint save/load with best-val tracking |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Streamlit ≥ 1.28
- NumPy, SciPy, Plotly, Pandas, PyYAML
