import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from src.ui.styling import apply_custom_styling, get_plotly_layout
from src.core.forecaster import (
    OilProductionForecaster,
    run_arps_dca_baseline,
    run_arima_baseline,
    run_prophet_baseline,
    run_xgboost_baseline,
    run_gru_baseline,
    run_multi_well_analysis,
    run_ablation_study,
)
from src.core.data import (
    calculate_metrics,
    generate_sample_data,
    preprocess_production,
    diebold_mariano_test,
)

warnings.filterwarnings('ignore')

# --- PRE-CONFIG ---
st.set_page_config(
    page_title="PetroAI Suite | Unified Energy Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply External Premium Styling
apply_custom_styling()

# --- UI MAIN ---

def auto_detect_columns(df):
    """Smartly detect date and oil-production columns from a DataFrame."""
    cols = df.columns.tolist()
    cols_lower = [c.lower() for c in cols]

    # ── Date column ──────────────────────────────────────────────────────────
    date_priority = ['dateprd', 'date_prd', 'proddate', 'production_date', 'date']
    date_col = None
    for pat in date_priority:
        match = next((c for c, cl in zip(cols, cols_lower) if pat == cl), None)
        if match:
            date_col = match
            break
    if date_col is None:
        # fall back: first column whose values parse as dates
        for c in cols:
            try:
                pd.to_datetime(df[c].dropna().head(10))
                date_col = c
                break
            except Exception:
                pass
    if date_col is None:
        date_col = cols[0]

    # ── Production column ────────────────────────────────────────────────────
    # Positive priority patterns (oil volume)
    prod_priority = ['bore_oil_vol', 'oil_vol', 'oilvol', 'bore_oil', 'oil_production',
                     'prod_oil', 'production_oil', 'qo', 'oil_rate', 'oilrate']
    # Terms that indicate NOT an oil production column
    prod_exclude = ['day', 'hr', 'hour', 'water', 'wat', 'gas', 'inject', 'choke',
                    'pressure', 'temp', 'wc', 'gor', 'status', 'well', 'on_stream']

    prod_col = None
    for pat in prod_priority:
        match = next((c for c, cl in zip(cols, cols_lower) if pat in cl), None)
        if match:
            prod_col = match
            break

    if prod_col is None:
        # Fallback: any numeric col whose name contains 'vol' or 'oil',
        # but is not date_col and not in exclude list
        for c, cl in zip(cols, cols_lower):
            if c == date_col:
                continue
            if any(ex in cl for ex in prod_exclude):
                continue
            if any(k in cl for k in ['oil', 'vol', 'bore']):
                if pd.api.types.is_numeric_dtype(df[c]):
                    prod_col = c
                    break

    if prod_col is None:
        # Last resort: first numeric column that isn't the date col
        for c in cols:
            if c != date_col and pd.api.types.is_numeric_dtype(df[c]):
                prod_col = c
                break

    if prod_col is None:
        prod_col = cols[1] if len(cols) > 1 else cols[0]

    return date_col, prod_col


def run_pipeline(raw_df, date_col, prod_col, lookback_days, forecast_days, epochs,
                 lstm_units=64, learning_rate=0.001,
                 forecast_mode='univariate', feature_cols=None,
                 run_baselines=False, mc_iterations=0):
    """
    Full LSTM pipeline:
      1. preprocess_production  → daily rate, smoothed
      2. Chronological train/test split (80/20)
      3. Fit scaler on train only  → no data leakage
      4. Train LSTM
      5. Evaluate on test set
      6. Auto-regressive future forecast
      7. Optionally run baseline models & DM tests
    """
    if feature_cols is None:
        feature_cols = []

    # ── 1. Preprocess (strict order: diff → dropna → drop_zeros → smooth) ──
    pipeline_df, was_cumulative = preprocess_production(
        raw_df, date_col=date_col, prod_col=prod_col, smooth_window=14
    )

    if pipeline_df['BORE_OIL_VOL'].sum() == 0:
        raise ValueError(
            f"Column '{prod_col}' contains only zeros after preprocessing. "
            "Check that this is an oil production column."
        )

    # Guard: if std/mean < 1% the series is nearly flat → cumulative data slipped through
    series_vals = pipeline_df['BORE_OIL_VOL'].values
    coeff_var   = float(series_vals.std() / series_vals.mean()) if series_vals.mean() > 0 else 0.0
    if coeff_var < 0.01:
        raise ValueError(
            f"Post-preprocessing variance is suspiciously low (CV={coeff_var:.4f}). "
            "The production column may still be cumulative. Check your CSV column mapping."
        )

    # Use at least lookback+10 rows
    if len(pipeline_df) < lookback_days + 10:
        raise ValueError(
            f"Not enough usable data rows ({len(pipeline_df)}) for lookback "
            f"window ({lookback_days}). Try reducing the Lookback Window slider."
        )

    # ── 2. Prepare sequences (raw, unscaled) ──────────────────────────────
    target_col = 'BORE_OIL_VOL'
    forecaster = OilProductionForecaster(lstm_units=lstm_units,
                                         learning_rate=learning_rate)
    X, y = forecaster.prepare_data(
        pipeline_df, target_col, [target_col], lookback_days, 'univariate'
    )

    # ── 3. Chronological split ────────────────────────────────────────────
    split   = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # ── 4. Train (scaler fitted inside train_model on train only) ─────────────
    forecaster.train_model(X_tr, y_tr, epochs=epochs, batch_size=16,
                           validation_split=0.05)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    actual   = y_te.flatten()
    pred_raw = forecaster.predict(X_te).flatten()
    metrics  = calculate_metrics(actual, pred_raw)

    # ── 6. Future forecast ────────────────────────────────────────────────
    future_fc    = forecaster.forecast_future(X[-1], forecast_days, [target_col], pipeline_df)
    last_date    = pipeline_df['DATEPRD'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    result = {
        'pipeline_df':    pipeline_df,
        'forecaster':     forecaster,
        'actual':         actual,
        'pred':           pred_raw,
        'metrics':        metrics,
        'future_dates':   future_dates,
        'future_fc':      future_fc,
        'last_sequence':  X[-1],
        'was_cumulative': was_cumulative,
    }

    # ── 7. Optional baselines & DM tests ─────────────────────────────────
    if run_baselines:
        dca_m       = run_arps_dca_baseline(pipeline_df)
        arima_m     = run_arima_baseline(pipeline_df)
        prophet_m   = run_prophet_baseline(pipeline_df)
        xgb_m       = run_xgboost_baseline(pipeline_df)
        gru_m       = run_gru_baseline(pipeline_df, lookback=min(30, lookback_days),
                                        epochs=min(30, epochs))

        result['dca_metrics']    = dca_m
        result['arima_metrics']  = arima_m
        result['prophet_metrics']= prophet_m
        result['xgb_metrics']    = xgb_m
        result['gru_metrics']    = gru_m

        # Reconstruct baseline errors on common test set for DM test
        n_te = len(actual)
        dca_pred   = pipeline_df['BORE_OIL_VOL'].values[-n_te:] * 0  # placeholder zeros
        # For DM tests we need aligned error arrays — compute simple DCA predictions
        q0  = float(pipeline_df['BORE_OIL_VOL'].values[split - 1])
        Di  = max(dca_m['RMSE'] / (q0 + 1e-6) * 0.001, 1e-4)  # rough Di proxy
        t_arr = np.arange(1, n_te + 1, dtype=np.float64)
        dca_aligned = q0 * np.exp(-Di * t_arr)

        e_hybrid = actual - pred_raw
        e_dca    = actual - dca_aligned

        result['dm_vs_dca']    = diebold_mariano_test(e_hybrid, e_dca)
        result['dm_vs_arima']  = {'dm_stat': np.nan, 'p_value': np.nan,
                                   'significant': False,
                                   'interpretation': 'Requires aligned test set'}
        result['dm_vs_xgb']    = result['dm_vs_arima'].copy()
        result['dm_vs_gru']    = result['dm_vs_arima'].copy()
        result['dm_vs_prophet']= result['dm_vs_arima'].copy()

    return result


def main():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        try:
            st.image("assets/logo.png", width=120)
        except:
            st.markdown("### 🛢️")

    with col_title:
        st.markdown("<h1>PETROAI SUITE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: -1rem;'>Advanced Reservoir Analytics &amp; Neural Forecasting</p>", unsafe_allow_html=True)

    st.write("---")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ MISSION CONTROL")
    st.sidebar.write("---")

    forecast_mode = st.sidebar.selectbox(
        "🧠 AI Forecasting Mode",
        ["univariate", "multivariate"],
        help="Univariate: History only. Multivariate: External drivers."
    )

    lookback_days = st.sidebar.slider("🔍 Lookback Window", 7, 180, 90)
    forecast_days = st.sidebar.slider("🔮 Prediction Horizon", 1, 180, 45)

    with st.sidebar.expander("🛠️ NEURAL ENGINE CONFIG"):
        epochs     = st.number_input("Training Cycles (Epochs)", 10, 500, 100)
        lstm_units = st.slider("Neural Density (LSTM Units)", 10, 200, 64)
        lr         = st.selectbox("Intelligence Rate", [0.0001, 0.001, 0.01], index=1)

    run_baselines = False
    mc_iters      = 0

    # ── Sidebar feature selection (multivariate) ──────────────────────────
    sidebar_feature_cols = []

    # ── DATA INGESTION ────────────────────────────────────────────────────────
    st.sidebar.write("---")
    st.sidebar.markdown("### 📥 DATA INGESTION")

    uploaded_file = st.sidebar.file_uploader("Upload Field Production CSV", type=['csv'])

    # Track which file is loaded so we can detect new uploads
    if uploaded_file is not None:
        file_id = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.get('_csv_file_id') != file_id:
            try:
                raw = pd.read_csv(uploaded_file)
                st.session_state.raw_df = raw
                st.session_state._csv_file_id = file_id
                # Reset any prior results when a new file lands
                for k in ['df', 'forecaster', 'metrics', 'test_data', 'last_sequence',
                          'run_results', 'run_forecast_csv', '_date_col', '_prod_col']:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.sidebar.error(f"Could not read CSV: {e}")

    # ── RUN FILE button — shown immediately after upload, no column selectors ──
    if 'raw_df' in st.session_state:
        raw_df = st.session_state.raw_df

        # Auto-detect columns silently
        date_col, prod_col = auto_detect_columns(raw_df)

        # Show what was detected as a small hint
        st.sidebar.markdown(
            f"<small style='color:#64748b;'>📅 <b>{date_col}</b> &nbsp;|&nbsp; 🛢️ <b>{prod_col}</b></small>",
            unsafe_allow_html=True
        )
        st.sidebar.write("")

        # ── THE BIG RUN BUTTON ────────────────────────────────────────────────
        if st.sidebar.button("▶ RUN FILE", type="primary", use_container_width=True, key="run_file_btn"):
            with st.sidebar:
                with st.spinner("Running pipeline…"):
                    try:
                        result = run_pipeline(
                            raw_df, date_col, prod_col,
                            lookback_days, forecast_days, epochs,
                            lstm_units=lstm_units,
                            learning_rate=lr,
                            forecast_mode=forecast_mode,
                            feature_cols=sidebar_feature_cols,
                            run_baselines=run_baselines,
                            mc_iterations=mc_iters,
                        )
                        st.session_state.df            = result['pipeline_df']
                        st.session_state.forecaster    = result['forecaster']
                        st.session_state.metrics       = result['metrics']
                        st.session_state.last_sequence = result['last_sequence']
                        st.session_state.run_results   = result
                        exp_df = pd.DataFrame({
                            'Date': result['future_dates'],
                            'Forecasted_Volume_bbl': result['future_fc']
                        })
                        st.session_state.run_forecast_csv = exp_df.to_csv(index=False)
                        st.success("✅ Pipeline complete!")
                    except Exception as ex:
                        st.error(f"❌ {ex}")

        if st.sidebar.button("🔄 Reset CSV", use_container_width=True, key="reset_csv_btn"):
            for k in ['raw_df', '_csv_file_id', 'df', 'forecaster',
                      'metrics', 'test_data', 'last_sequence', 'run_results', 'run_forecast_csv']:
                st.session_state.pop(k, None)
            st.rerun()

    st.sidebar.write("---")
    if st.sidebar.button("✨ BOOTSTRAP DEMO DATA", type="secondary"):
        df_demo = generate_sample_data()
        st.session_state.df = df_demo
        for k in ['raw_df', '_csv_file_id', 'run_results', 'run_forecast_csv',
                  'forecaster', 'metrics', 'test_data', 'last_sequence']:
            st.session_state.pop(k, None)
        st.toast("Pipeline Initialized with Synthetic Field Data!", icon="⚡")

    # ── MAIN DASHBOARD ────────────────────────────────────────────────────────
    if 'df' in st.session_state:
        df         = st.session_state.df
        target_col = 'BORE_OIL_VOL'

        # ── Result banner ─────────────────────────────────────────────────────
        if 'run_results' in st.session_state:
            res = st.session_state.run_results
            m   = res['metrics']
            cumul_note = " (converted from cumulative)" if res.get('was_cumulative') else ""
            r2_color   = "#4ade80" if m['R2'] >= 0.7 else "#fb923c" if m['R2'] >= 0.4 else "#f87171"
            st.markdown("""
                <div style="background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.3);
                            border-radius:14px;padding:1rem 1.5rem;margin-bottom:1rem;
                            display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
                    <span style="color:#38bdf8;font-size:1.1rem;font-weight:700;">✅ Pipeline Results{note}</span>
                    <span style="color:#94a3b8;">RMSE <b style="color:#f1f5f9">{rmse:.1f}</b> bbl/d</span>
                    <span style="color:#94a3b8;">MAE  <b style="color:#f1f5f9">{mae:.1f}</b> bbl/d</span>
                    <span style="color:#94a3b8;">R²   <b style="color:{r2c}">{r2:.3f}</b></span>
                    <span style="color:#94a3b8;">MAPE <b style="color:#f1f5f9">{mape:.1f}%</b></span>
                    <span style="color:#94a3b8;">NSE  <b style="color:#f1f5f9">{nse:.3f}</b></span>
                    <span style="color:#94a3b8;">Avg Forecast <b style="color:#f43f5e">{avg:.0f}</b> bbl/d</span>
                </div>
            """.format(
                note=cumul_note,
                rmse=m['RMSE'], mae=m['MAE'],
                r2=m['R2'],     r2c=r2_color,
                mape=m['MAPE'], nse=m.get('NSE', float('nan')),
                avg=float(res['future_fc'].mean())
            ), unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            "📊 ASSET PERFORMANCE",
            "🧠 NEURAL TRAINING",
            "🔮 FUTURE OUTLOOK",
        ])

        # ── TAB 1: ASSET PERFORMANCE ──────────────────────────────────────────
        with tab1:
            st.subheader("🌐 Real-time Asset Intelligence")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Field Life", f"{(df['DATEPRD'].max() - df['DATEPRD'].min()).days} Days")

            # Sum of daily rates = estimated total production (bbl)
            total_prod_mmbbl = df['BORE_OIL_VOL'].sum() / 1_000_000
            c2.metric("Est. Cumulative (MMbbl)", f"{total_prod_mmbbl:.3f}")

            # Current Flow Rate = last valid non-zero daily rate
            nz            = df[df['BORE_OIL_VOL'] > 0]['BORE_OIL_VOL']
            current_rate  = float(nz.iloc[-1]) if len(nz) >= 1 else 0.0
            prev_rate     = float(nz.iloc[-2]) if len(nz) >= 2 else current_rate
            c3.metric("Current Flow Rate",
                      f"{current_rate:,.0f} bbl/d",
                      delta=f"{current_rate - prev_rate:+.0f} bbl/d")
            c4.metric("Zero-Flow Events", f"{int((df['BORE_OIL_VOL'] == 0).sum())}")

            # If we have a forecast result, show history + forecast together
            if 'run_results' in st.session_state:
                res = st.session_state.run_results
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(
                    x=df['DATEPRD'], y=df['BORE_OIL_VOL'],
                    name="Historical Flow", line=dict(color='#38bdf8', width=2),
                    fill='tozeroy', fillcolor='rgba(56,189,248,0.08)'
                ))
                fig_main.add_trace(go.Scatter(
                    x=res['future_dates'], y=res['future_fc'],
                    name=f"{forecast_days}-Day Forecast",
                    line=dict(color='#f43f5e', width=3, dash='dash')
                ))
                fig_main.update_layout(**get_plotly_layout("Production History + Forecast", "Timeline", "Volume (bbl/day)"))
                st.plotly_chart(fig_main, use_container_width=True)

                # Forecast download
                st.download_button(
                    "📥 DOWNLOAD FORECAST CSV",
                    data=st.session_state.run_forecast_csv,
                    file_name="petroai_forecast.csv",
                    mime="text/csv",
                    key="dl_asset_tab"
                )
            else:
                fig = px.line(df, x='DATEPRD', y='BORE_OIL_VOL', color_discrete_sequence=['#38bdf8'])
                fig.update_traces(line=dict(width=3, color='#38bdf8'),
                                  fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)')
                fig.update_layout(**get_plotly_layout("Asset Production History", "Timeline", "Volume (bbl/day)"))
                st.plotly_chart(fig, use_container_width=True)

            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                with st.expander("🔍 Subsurface correlations"):
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    fig_corr = px.imshow(df[num_cols].corr(), text_auto=".2f", aspect="auto",
                                         color_continuous_scale='Purpor')
                    fig_corr.update_layout(**get_plotly_layout("Variable Sensitivity Matrix"))
                    st.plotly_chart(fig_corr, use_container_width=True)
            with col_eda2:
                with st.expander("📈 Distribution Analysis"):
                    fig_hist = px.histogram(df, x='BORE_OIL_VOL', nbins=50, color_discrete_sequence=['#818cf8'])
                    fig_hist.update_layout(**get_plotly_layout("Production Distribution"))
                    st.plotly_chart(fig_hist, use_container_width=True)

        # ── TAB 2: NEURAL TRAINING ────────────────────────────────────────────
        with tab2:
            st.subheader("🛠️ Deep Learning Model Studio")

            feature_cols = st.multiselect(
                "📡 Select Neural Input Signals",
                options=[c for c in df.columns if c != 'DATEPRD' and c != target_col],
                default=[]
            )

            st.markdown("""
                <div style="background: rgba(56, 189, 248, 0.05); padding: 1rem; border-radius: 10px;
                            margin-bottom: 1rem; border-left: 4px solid #38bdf8;">
                    <b>Model Architecture:</b> Recurrent LSTM Network with Dense Output.
                    Optimized for non-linear petroleum decline curves.
                </div>
            """, unsafe_allow_html=True)

            # Show results from sidebar RUN FILE if available
            if 'run_results' in st.session_state:
                res = st.session_state.run_results
                m = res['metrics']
                st.info("✅ Model trained via **▶ RUN FILE** — showing those results below.")
                st.write("#### 📡 Intelligence Validation Results")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Precision (RMSE)", f"{m['RMSE']:.1f} bbl/d",
                           delta=f"{m['RMSE']/df[target_col].mean()*100:.1f}% Margin")
                cc2.metric("Mean Deviation (MAE)", f"{m['MAE']:.1f} bbl/d")
                cc3.metric("Coefficient of Determination", f"{m['R2']:.3f}")

                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(y=res['actual'], name="Ground Truth",
                                             line=dict(color="#64748b", width=2)))
                fig_res.add_trace(go.Scatter(y=res['pred'], name="AI Prediction",
                                             line=dict(color="#38bdf8", dash="dash", width=3)))
                fig_res.update_layout(**get_plotly_layout("Temporal Pattern Match (Test Set)", "Sample Index", "Flow Rate"))
                st.plotly_chart(fig_res, use_container_width=True)

            if st.button("🚀 INITIATE NEURAL TRAINING", type="primary"):
                with st.status("Propagating Neural Weights...", expanded=True) as status:
                    st.write("Initializing sequence buffers...")
                    f2   = OilProductionForecaster(lstm_units=lstm_units,
                                                   learning_rate=lr)
                    X2, y2 = f2.prepare_data(
                        df, target_col, [target_col] + feature_cols,
                        lookback_days, forecast_mode
                    )
                    split2        = int(len(X2) * 0.8)
                    X2_tr, X2_te  = X2[:split2], X2[split2:]
                    y2_tr, y2_te  = y2[:split2], y2[split2:]

                    st.write(f"Fitting layers with {epochs} epochs...")
                    f2.train_model(X2_tr, y2_tr, epochs=epochs, batch_size=64)

                    st.write("Cross-validating...")
                    pred2 = f2.predict(X2_te).flatten()
                    metrics2 = calculate_metrics(y2_te.flatten(), pred2)

                    st.session_state.forecaster      = f2
                    st.session_state.metrics_manual  = metrics2
                    st.session_state.test_data       = (y2_te, pred2)
                    st.session_state.last_sequence   = X2[-1]

                    status.update(label="Training Synchronized!", state="complete", expanded=False)

            if 'metrics_manual' in st.session_state:
                m = st.session_state.metrics_manual
                st.write("#### 📡 Manual Training Results")
                cc1, cc2, cc3 = st.columns(3)
                r2c = "#4ade80" if m['R2'] >= 0.7 else "#fb923c" if m['R2'] >= 0.4 else "#f87171"
                cc1.metric("RMSE", f"{m['RMSE']:.1f} bbl/d")
                cc2.metric("MAE",  f"{m['MAE']:.1f} bbl/d")
                cc3.metric("R²",   f"{m['R2']:.3f}")

                y_te_raw, pred_vals = st.session_state.test_data
                fig_res2 = go.Figure()
                fig_res2.add_trace(go.Scatter(
                    y=y_te_raw.flatten(), name="Ground Truth",
                    line=dict(color="#64748b", width=2)))
                fig_res2.add_trace(go.Scatter(
                    y=pred_vals.flatten(), name="LSTM Prediction",
                    line=dict(color="#38bdf8", dash="dash", width=3)))
                fig_res2.update_layout(**get_plotly_layout(
                    "Temporal Pattern Match (Test Set)", "Sample Index", "Flow Rate (bbl/d)"))
                st.plotly_chart(fig_res2, use_container_width=True)

        # ── TAB 3: FUTURE OUTLOOK ─────────────────────────────────────────────
        with tab3:
            # Use RUN FILE forecast if available
            if 'run_results' in st.session_state:
                res = st.session_state.run_results
                st.subheader("🔮 Predictive Field Trajectory  •  Hybrid LSTM + Arps DCA")

                fc           = res['future_fc']
                future_dates = res['future_dates']
                last_actual  = float(df[df['BORE_OIL_VOL'] > 0]['BORE_OIL_VOL'].iloc[-1])

                # Re-compute the pure DCA envelope for display
                from src.core.forecaster import OilProductionForecaster as _F
                Di, q0 = _F._fit_exponential_decline(
                    df['BORE_OIL_VOL'].values.astype(float)
                )
                t_arr    = np.arange(1, len(future_dates) + 1, dtype=float)
                dca_curve = q0 * np.exp(-Di * t_arr)

                fig_final = go.Figure()
                # Historical
                recent = df.tail(150)
                fig_final.add_trace(go.Scatter(
                    x=recent['DATEPRD'], y=recent[target_col],
                    name="Historical (daily rate)", line=dict(color="#64748b", width=2)
                ))
                # Pure Arps DCA envelope
                fig_final.add_trace(go.Scatter(
                    x=future_dates, y=dca_curve,
                    name=f"Arps DCA (Di={Di*100:.2f}%/day)",
                    line=dict(color="#4ade80", width=2, dash="dot"),
                ))
                # Hybrid LSTM + DCA forecast
                fig_final.add_trace(go.Scatter(
                    x=future_dates, y=fc,
                    name="Hybrid Forecast (LSTM+DCA)",
                    line=dict(color="#f43f5e", width=3),
                    fill='tozeroy', fillcolor='rgba(244,63,94,0.05)'
                ))
                fig_final.update_layout(**get_plotly_layout(
                    "Future Production Outlook", "Timeline", "Production (bbl/day)"))
                st.plotly_chart(fig_final, use_container_width=True)

                cc4, cc5, cc6 = st.columns(3)
                total_decline_pct = (fc[-1] - last_actual) / last_actual * 100
                cc4.metric("Avg Forecasted Rate",  f"{fc.mean():.0f} bbl/d")
                cc5.metric("Final Forecast Value", f"{fc[-1]:.0f} bbl/d")
                cc6.metric("Total Decline",
                           f"{total_decline_pct:+.1f}%",
                           delta_color="inverse")

                exp_df = pd.DataFrame({
                    'Target Date':      future_dates,
                    'Hybrid_Forecast':  fc,
                    'DCA_Envelope':     dca_curve,
                })
                st.download_button("📥 EXPORT INTELLIGENCE REPORT (CSV)",
                                   data=exp_df.to_csv(index=False),
                                   file_name="petroai_forecast_report.csv",
                                   key="dl_outlook_tab")

            elif 'forecaster' not in st.session_state:
                st.warning("⚠️ No forecast yet. Upload a CSV and click **▶ RUN FILE** in the sidebar, "
                           "or use the Neural Training tab.")
            else:
                st.subheader("🔮 Predictive Field Trajectory")
                feature_cols_out = []
                with st.spinner("Crunching future probabilities..."):
                    future_forecast = st.session_state.forecaster.forecast_future(
                        st.session_state.last_sequence,
                        forecast_days,
                        [target_col] + (feature_cols_out if forecast_mode == "multivariate" else []),
                        df
                    )

                last_date = df['DATEPRD'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

                fig_final2 = go.Figure()
                recent2 = df.tail(120)
                fig_final2.add_trace(go.Scatter(x=recent2['DATEPRD'], y=recent2[target_col],
                                                name="Historical Flow", line=dict(color="#64748b", width=2)))
                fig_final2.add_trace(go.Scatter(x=future_dates, y=future_forecast,
                                                name="AI Projection", line=dict(color="#f43f5e", width=4)))
                fig_final2.update_layout(**get_plotly_layout("Future Production Outlook", "Timeline", "Production (bbl/day)"))
                st.plotly_chart(fig_final2, use_container_width=True)

                cc4, cc5, cc6 = st.columns(3)
                cc4.metric("Avg Forecasted Rate", f"{future_forecast.mean():.0f} bbl/d")
                cc5.metric("Peak Projected Flow", f"{future_forecast.max():.0f} bbl/d")
                cc6.metric("Decline Sensitivity", f"{(future_forecast[-1]-future_forecast[0])/len(future_forecast):.2f} bbl/d²")

                exp_df2 = pd.DataFrame({'Target Date': future_dates, 'Forecasted Volume': future_forecast})
                st.download_button("📥 EXPORT INTELLIGENCE REPORT (CSV)",
                                   data=exp_df2.to_csv(index=False),
                                   file_name="petroai_forecast_report.csv",
                                   key="dl_outlook_manual")

    else:
        # ── Hero landing page ─────────────────────────────────────────────────
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: rgba(30, 41, 59, 0.4);
                        border-radius: 30px; border: 1px solid rgba(56, 189, 248, 0.2); margin-top: 2rem;">
                <h2 style="font-size: 2.5rem; color: #f1f5f9; margin-bottom: 1rem;">Bridge the gap between data and strategy.</h2>
                <p style="font-size: 1.2rem; color: #94a3b8; max-width: 800px; margin: 0 auto;">
                    PetroAI Suite utilizes advanced <b>Long Short-Term Memory (LSTM)</b> architectures to decode complex
                    subsurface production patterns, providing the precision needed for modern reservoir management.
                </p>
                <div style="margin-top: 2rem; color: #64748b; font-size: 1rem;">
                    👈 Upload a CSV in the sidebar and click <b style="color:#38bdf8;">▶ RUN FILE</b> to start.
                </div>
                <div style="margin-top: 3rem; display: flex; justify-content: center; gap: 2rem;">
                    <div style="background: rgba(56, 189, 248, 0.05); padding: 1.5rem; border-radius: 20px;
                                border: 1px solid rgba(56, 189, 248, 0.1); width: 250px;">
                        <h4 style="color: #38bdf8;">🧠 Neural Logic</h4>
                        <p style="color: #64748b; font-size: 0.9rem;">Adaptive time-series learning from historical reservoir behavior.</p>
                    </div>
                    <div style="background: rgba(129, 140, 248, 0.05); padding: 1.5rem; border-radius: 20px;
                                border: 1px solid rgba(129, 140, 248, 0.1); width: 250px;">
                        <h4 style="color: #818cf8;">📊 Multi-Variable</h4>
                        <p style="color: #64748b; font-size: 0.9rem;">Integrate Pressure, Temperature, and Injection signals.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.image("https://images.unsplash.com/photo-1544383835-bda2bc66a55d?auto=format&fit=crop&q=80&w=1200",
                 caption="Autonomous Energy Intelligence Portal", use_container_width=True)


if __name__ == "__main__":
    main()
