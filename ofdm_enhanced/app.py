"""
app.py
OFDM Channel Estimation — Streamlit GUI
Run: streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import torch
import streamlit as st
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from modules import (
    OFDMSystem, ChannelModel,
    DatasetGenerator,
    build_model, ChannelNetCNN, ChannelNetMLP,
    LSEstimator, MMSEEstimator, DLEstimator, PerfectEstimator,
    Trainer, Evaluator,
    plot_ber_vs_snr, plot_nmse_vs_snr, plot_capacity_vs_snr,
    plot_channel_response, plot_cir, plot_constellation,
    plot_training_curves, plot_channel_heatmap,
    plot_error_histogram, plot_capacity_gap,
)

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="OFDM Channel Estimator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS                                                          #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    h1 { color: #636EFA; letter-spacing: 1px; }
    h2 { color: #00CC96; }
    h3 { color: #AB63FA; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 4px 0;
        border-left: 4px solid #636EFA;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetric"] {
        background: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Session state initialisation                                        #
# ------------------------------------------------------------------ #

for key, default in {
    "dataset":        None,
    "train_ds":       None,
    "val_ds":         None,
    "test_ds":        None,
    "model":          None,
    "arch":           None,
    "train_losses":   [],
    "val_losses":     [],
    "eval_results":   None,
    "ofdm":           None,
    "channel":        None,
    "sample_channel": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------ #
#  Sidebar — System Parameters                                         #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("📡 System Parameters")

    st.subheader("OFDM")
    n_sub     = st.select_slider("Subcarriers (N)", [32, 64, 128], value=64)
    cp_len    = st.select_slider("CP Length", [8, 16, 32], value=16)
    n_pilots  = st.select_slider("Pilots per symbol", [4, 8, 16], value=8)
    n_sym     = st.select_slider("Symbols per frame", [7, 14, 28], value=14)

    st.subheader("Channel")
    chan_type = st.selectbox("Model", ["rayleigh", "rician", "etu"])
    n_taps    = st.slider("Channel taps", 2, 8, 4)
    k_factor  = st.slider("Rician K-factor", 0.5, 10.0, 3.0, 0.5,
                           disabled=(chan_type != "rician"))

    st.subheader("Dataset")
    n_samples = st.slider("Total samples", 1000, 10000, 4000, 500)
    snr_min   = st.slider("SNR min (dB)", -5, 10, 0)
    snr_max   = st.slider("SNR max (dB)", 15, 40, 30)
    seed      = st.number_input("Random seed", 0, 9999, 42)

    st.subheader("Training")
    arch      = st.selectbox("Architecture", ["cnn", "mlp"])
    epochs    = st.slider("Epochs", 10, 200, 50, 5)
    batch_sz  = st.select_slider("Batch size", [16, 32, 64, 128], value=32)
    lr        = st.select_slider("Learning rate",
                                  [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)

    st.subheader("Evaluation")
    snr_eval_min = st.slider("Eval SNR min (dB)", -5, 5, 0)
    snr_eval_max = st.slider("Eval SNR max (dB)", 20, 40, 30)
    snr_step     = st.slider("SNR step (dB)", 1, 5, 3)
    n_trials     = st.slider("MC trials per SNR", 50, 500, 200, 50)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"🖥 Device: **{device_str.upper()}**")

# ------------------------------------------------------------------ #
#  Header                                                              #
# ------------------------------------------------------------------ #

st.title("📡 OFDM Channel Estimation Dashboard")
st.caption("LS · MMSE · Deep Learning (CNN / MLP) · BER · NMSE · Shannon Capacity")
st.divider()

# ------------------------------------------------------------------ #
#  Tabs                                                                #
# ------------------------------------------------------------------ #

tab_ds, tab_tr, tab_ev, tab_vis, tab_ab = st.tabs([
    "📊 Dataset & Channel",
    "🧠 Train Model",
    "📈 Evaluation",
    "🔬 Visualise",
    "ℹ️ About",
])

# ==================================================================== #
#  TAB 1 – Dataset & Channel                                            #
# ==================================================================== #

with tab_ds:
    st.header("1 · Dataset Generation & Channel Preview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Generates synthetic OFDM frames under randomised channel conditions.
        Each sample includes an **interpolated LS estimate** as model input
        and the **true channel response** as the target.
        """)
    with col2:
        gen_btn = st.button("⚡ Generate Dataset", type="primary",
                            use_container_width=True)

    if gen_btn:
        with st.spinner("Generating dataset…"):
            ofdm = OFDMSystem(n_sub=n_sub, cp_len=cp_len,
                              n_pilots=n_pilots, n_sym=n_sym)
            ch   = ChannelModel(model=chan_type, n_taps=n_taps,
                                k_factor=k_factor)
            gen  = DatasetGenerator(ofdm, ch, n_samples=n_samples,
                                    snr_min=snr_min, snr_max=snr_max, seed=seed)
            dataset = gen.generate(verbose=False)
            train_ds, val_ds, test_ds = DatasetGenerator.split(dataset, seed=seed)

        st.session_state.update({
            "dataset":  dataset,
            "train_ds": train_ds,
            "val_ds":   val_ds,
            "test_ds":  test_ds,
            "ofdm":     ofdm,
            "channel":  ch,
        })
        st.success(f"✅ Generated {n_samples:,} samples "
                   f"({len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test)")

    # ---- Dataset stats ----
    if st.session_state.dataset is not None:
        ds   = st.session_state.dataset
        ofdm = st.session_state.ofdm
        ch   = st.session_state.channel

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total samples",  f"{len(ds):,}")
        m2.metric("Train",          f"{len(st.session_state.train_ds):,}")
        m3.metric("Val",            f"{len(st.session_state.val_ds):,}")
        m4.metric("Test",           f"{len(st.session_state.test_ds):,}")

        st.divider()
        st.subheader("Channel Preview")

        # Sample a random channel
        rng = np.random.default_rng()
        h_sample = ch.generate()
        H_freq   = ch.freq_response(h_sample, ofdm.N)
        H_2d     = np.tile(H_freq, (ofdm.n_sym, 1))

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_cir(h_sample), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_channel_heatmap(H_2d,
                            title=f"{chan_type.title()} Channel |H| (Time-Frequency)"),
                            use_container_width=True)

        # LS vs true channel
        st.subheader("LS Estimate Preview (SNR = 15 dB)")
        bits      = rng.integers(0, 2, ofdm.bits_per_frame()).astype(int)
        freq_tx, time_tx = ofdm.build_tx_frame(bits)
        rx_time   = np.array([
            ofdm.add_awgn(ch.apply(time_tx[s], h_sample), 15)
            for s in range(ofdm.n_sym)
        ])
        rx_freq   = ofdm.rx_frame(rx_time)
        H_ls_p    = ofdm.ls_at_pilots(rx_freq)
        H_ls      = ofdm.interpolate_channel(H_ls_p)
        H_mmse    = ofdm.mmse_estimate(H_ls, 15)

        st.plotly_chart(
            plot_channel_response(H_2d, {"LS": H_ls, "MMSE": H_mmse}),
            use_container_width=True
        )
    else:
        st.info("👆 Click **Generate Dataset** to begin.")


# ==================================================================== #
#  TAB 2 – Train Model                                                  #
# ==================================================================== #

with tab_tr:
    st.header("2 · Train Deep Learning Estimator")

    if st.session_state.dataset is None:
        st.warning("⚠️ Please generate a dataset first (Tab 1).")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Architecture:** `{arch.upper()}`  ·  
            **Epochs:** `{epochs}`  ·  
            **Batch:** `{batch_sz}`  ·  
            **LR:** `{lr}`  ·  
            **Device:** `{device_str.upper()}`
            """)
        with col2:
            train_btn = st.button("🚀 Start Training", type="primary",
                                  use_container_width=True)

        # ---- Training ----
        if train_btn:
            ofdm = st.session_state.ofdm
            model = build_model(arch, n_sym=ofdm.n_sym, n_sub=ofdm.N,
                                 device=device_str)
            save_path = f"models_saved/channelnet_{arch}.pth"

            trainer = Trainer(model, lr=lr, batch_size=batch_sz,
                              device=device_str, save_path=save_path)

            progress_bar   = st.progress(0, text="Training…")
            metrics_ph     = st.empty()
            chart_ph       = st.empty()

            train_losses, val_losses = [], []

            def epoch_cb(ep, tr_l, v_l):
                train_losses.append(tr_l)
                val_losses.append(v_l)
                pct = int(ep / epochs * 100)
                progress_bar.progress(pct,
                    text=f"Epoch {ep}/{epochs} · Train: {tr_l:.5f} · Val: {v_l:.5f}")
                with metrics_ph.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Epoch",      f"{ep}/{epochs}")
                    c2.metric("Train Loss", f"{tr_l:.5f}")
                    c3.metric("Val Loss",   f"{v_l:.5f}",
                              delta=f"{v_l-train_losses[-2]:.5f}"
                                    if len(train_losses) > 1 else None)
                if len(train_losses) >= 2:
                    chart_ph.plotly_chart(
                        plot_training_curves(train_losses, val_losses),
                        use_container_width=True
                    )

            trainer.train(
                st.session_state.train_ds,
                st.session_state.val_ds,
                epochs=epochs,
                progress_callback=epoch_cb,
            )

            progress_bar.progress(100, text="Training complete ✅")
            st.session_state.update({
                "model":        model,
                "arch":         arch,
                "train_losses": train_losses,
                "val_losses":   val_losses,
            })
            st.success(f"✅ Model saved to `{save_path}`")

        # ---- Show previous training curves ----
        elif st.session_state.train_losses:
            st.subheader("Previous Training Run")
            st.plotly_chart(
                plot_training_curves(
                    st.session_state.train_losses,
                    st.session_state.val_losses,
                ),
                use_container_width=True
            )
            tl = st.session_state.train_losses
            vl = st.session_state.val_losses
            c1, c2, c3 = st.columns(3)
            c1.metric("Epochs trained", len(tl))
            c2.metric("Final train loss", f"{tl[-1]:.5f}")
            c3.metric("Best val loss",    f"{min(vl):.5f}")


# ==================================================================== #
#  TAB 3 – Evaluation                                                   #
# ==================================================================== #

with tab_ev:
    st.header("3 · Performance Evaluation")

    if st.session_state.ofdm is None:
        st.warning("⚠️ Generate a dataset first (Tab 1).")
    else:
        ofdm = st.session_state.ofdm
        ch   = st.session_state.channel

        # Build estimator dict
        estimators = {
            "LS":      LSEstimator(),
            "MMSE":    MMSEEstimator(),
        }
        if st.session_state.model is not None:
            arch_lbl = f"DL-{st.session_state.arch.upper()}"
            estimators[arch_lbl] = DLEstimator(
                st.session_state.model, device=device_str, arch_label=arch_lbl
            )
        estimators["Perfect"] = PerfectEstimator(
            np.ones((ofdm.n_sym, ofdm.N), dtype=complex)
        )

        snr_range = list(range(snr_eval_min, snr_eval_max + 1, snr_step))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **SNR range:** `{snr_eval_min} → {snr_eval_max}` dB  (step `{snr_step}`)  ·  
            **MC trials per point:** `{n_trials}`  ·  
            **Channel model:** `{chan_type.title()}`
            """)
            st.markdown("_Estimators: " +
                        " · ".join(f"`{k}`" for k in estimators.keys()) + "_")
        with col2:
            eval_btn = st.button("▶ Run Evaluation", type="primary",
                                 use_container_width=True)

        if eval_btn:
            evaluator = Evaluator(ofdm, ch, estimators,
                                  n_trials=n_trials, seed=int(seed))
            ev_prog = st.progress(0, "Evaluating…")

            def ev_cb(done, total):
                ev_prog.progress(int(done/total*100),
                                 f"SNR point {done}/{total}…")

            with st.spinner("Running Monte-Carlo simulation…"):
                results = evaluator.run(snr_range, progress_callback=ev_cb)

            ev_prog.progress(100, "Done ✅")
            st.session_state.eval_results = results
            st.success("Evaluation complete!")

        # ---- Show results ----
        if st.session_state.eval_results is not None:
            res = st.session_state.eval_results

            st.subheader("BER vs SNR")
            st.plotly_chart(plot_ber_vs_snr(res), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("NMSE vs SNR")
                st.plotly_chart(plot_nmse_vs_snr(res), use_container_width=True)
            with c2:
                st.subheader("Channel Capacity vs SNR")
                st.plotly_chart(plot_capacity_vs_snr(res), use_container_width=True)

            st.subheader("Capacity Gap Analysis")
            st.plotly_chart(plot_capacity_gap(res), use_container_width=True)

            # Summary table
            st.subheader("Performance Summary  (at highest SNR)")
            snr_vals = res["snr_range"]
            last_idx = -1
            rows = []
            for lb in res["ber"]:
                rows.append({
                    "Estimator":     lb,
                    "BER":           f"{res['ber'][lb][last_idx]:.2e}",
                    "NMSE (dB)":     f"{res['nmse_db'][lb][last_idx]:.2f}",
                    "Capacity (b/s/Hz)": f"{res['capacity'][lb][last_idx]:.3f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==================================================================== #
#  TAB 4 – Visualise                                                    #
# ==================================================================== #

with tab_vis:
    st.header("4 · Visualise Estimates & Constellations")

    if st.session_state.ofdm is None:
        st.warning("⚠️ Generate a dataset first (Tab 1).")
    else:
        ofdm = st.session_state.ofdm
        ch   = st.session_state.channel

        snr_vis = st.slider("Visualisation SNR (dB)", snr_min, snr_max, 15)
        vis_btn = st.button("🔄 Refresh Snapshot", use_container_width=False)

        if vis_btn or "vis_data" not in st.session_state:
            rng  = np.random.default_rng()
            h_v  = ch.generate()
            H_true_freq = ch.freq_response(h_v, ofdm.N)
            H_true = np.tile(H_true_freq, (ofdm.n_sym, 1))

            bits = rng.integers(0, 2, ofdm.bits_per_frame()).astype(int)
            freq_tx, time_tx = ofdm.build_tx_frame(bits)
            rx_time = np.array([
                ofdm.add_awgn(ch.apply(time_tx[s], h_v), snr_vis)
                for s in range(ofdm.n_sym)
            ])
            rx_freq = ofdm.rx_frame(rx_time)

            H_ls_p = ofdm.ls_at_pilots(rx_freq)
            H_ls   = ofdm.interpolate_channel(H_ls_p)
            H_mmse = ofdm.mmse_estimate(H_ls, snr_vis)

            ests   = {"LS": H_ls, "MMSE": H_mmse}
            if st.session_state.model is not None:
                lbl = f"DL-{st.session_state.arch.upper()}"
                dl_est = DLEstimator(st.session_state.model,
                                     device=device_str, arch_label=lbl)
                ests[lbl] = dl_est.estimate(rx_freq, ofdm, snr_vis)

            st.session_state.vis_data = {
                "H_true": H_true, "H_ests": ests,
                "rx_freq": rx_freq, "h_v": h_v, "bits": bits,
            }

        vd = st.session_state.vis_data

        # Channel response
        st.subheader("Channel Frequency Response")
        st.plotly_chart(
            plot_channel_response(vd["H_true"], vd["H_ests"]),
            use_container_width=True
        )

        # Heatmaps
        st.subheader("Time-Frequency Channel Maps")
        cols = st.columns(len(vd["H_ests"]) + 1)
        cols[0].plotly_chart(
            plot_channel_heatmap(vd["H_true"], "True Channel"),
            use_container_width=True
        )
        for i, (lb, He) in enumerate(vd["H_ests"].items()):
            cols[i+1].plotly_chart(
                plot_channel_heatmap(He, f"{lb} Estimate"),
                use_container_width=True
            )

        # Error histograms
        st.subheader("Estimation Error Distribution")
        err_cols = st.columns(len(vd["H_ests"]))
        for i, (lb, He) in enumerate(vd["H_ests"].items()):
            err_cols[i].plotly_chart(
                plot_error_histogram(He, vd["H_true"], label=lb),
                use_container_width=True
            )

        # Constellations (ZF equalization)
        st.subheader("Constellation Diagrams (ZF Equalised)")
        con_cols = st.columns(len(vd["H_ests"]))
        for i, (lb, He) in enumerate(vd["H_ests"].items()):
            H_safe = He + 1e-10
            eq = vd["rx_freq"][:, ofdm.data_idx] / H_safe[:, ofdm.data_idx]
            con_cols[i].plotly_chart(
                plot_constellation(eq, title=f"Constellation — {lb} ({snr_vis} dB)"),
                use_container_width=True
            )


# ==================================================================== #
#  TAB 5 – About                                                        #
# ==================================================================== #

with tab_ab:
    st.header("ℹ️ About This Project")
    st.markdown("""
    ## OFDM Channel Estimation with Deep Learning

    This dashboard implements and compares **three channel estimation methods** for OFDM systems:

    | Method | Description |
    |--------|-------------|
    | **LS** | Least Squares — pilot-based, linear interpolation |
    | **MMSE** | Minimum Mean Square Error — diagonal Wiener filter |
    | **DL-CNN** | Deep CNN — SRCNN super-resolution + DnCNN denoising |
    | **DL-MLP** | MLP baseline — fully-connected network |

    ## System Architecture

    ```
    Bits → QPSK → Pilot embed → IFFT → CP add → Channel (Rayleigh/Rician/ETU) → AWGN
         ← Demod ← ZF-Equalize ← Estimator ← CP remove ← FFT
    ```

    ## DL Model (ChannelNet)

    The CNN follows the ChannelNet architecture:
    - **Stage 1 (SRCNN):** 9×9 → 1×1 → 5×5 convolutions with PReLU activations  
    - **Stage 2 (DnCNN):** Residual denoising with BatchNorm layers  
    - Input/Output: `(2, n_sym, N_sub)` real/imag stacked tensors

    ## Metrics

    | Metric | Formula |
    |--------|---------|
    | **NMSE** | `‖H_est - H_true‖² / ‖H_true‖²` |
    | **BER** | Bit errors after ZF equalisation |
    | **Capacity** | `mean_k log₂(1 + SNR·\|H[k]\|²)` bits/s/Hz |

    ## Running on Google Colab (GPU)

    ```python
    # Install dependencies
    !pip install streamlit torch numpy scipy plotly pandas

    # Clone / upload project, then:
    !streamlit run app.py &
    
    # Use localtunnel to expose:
    !npx localtunnel --port 8501
    ```

    ## Project Structure

    ```
    ofdm_enhanced/
    ├── app.py                  ← Streamlit GUI (this file)
    ├── config.yaml             ← Default experiment config
    ├── requirements.txt
    ├── modules/
    │   ├── ofdm_system.py      ← OFDM signal processing
    │   ├── channel_models.py   ← Rayleigh / Rician / ETU
    │   ├── dataset.py          ← Dataset generation & splitting
    │   ├── models.py           ← CNN + MLP architectures
    │   ├── estimators.py       ← LS / MMSE / DL estimators
    │   ├── trainer.py          ← GPU-ready training loop
    │   ├── evaluator.py        ← BER / NMSE / Capacity metrics
    │   └── visualizer.py       ← Interactive Plotly figures
    └── models_saved/           ← Checkpoints saved here
    ```
    """)

    st.divider()
    st.caption("Built with PyTorch · Streamlit · Plotly")
