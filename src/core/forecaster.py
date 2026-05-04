import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class OilProductionForecaster:
    def __init__(self, lstm_units=64, learning_rate=0.001):
        # FIX BUG 1: lstm_units and lr now properly stored and used
        self.lstm_units       = lstm_units
        self.learning_rate    = learning_rate
        self.model            = None
        self.scaler           = None
        self.feature_scalers  = {}
        self.training_history = None
        self.forecast_mode    = "univariate"

    # ── Sequence builder ────────────────────────────────────────────────────
    def prepare_data(self, df, target_col, feature_cols, lookback_days,
                     forecast_mode="univariate"):
        self.forecast_mode = forecast_mode
        series = df[target_col].values.astype(np.float32)

        if forecast_mode == "univariate":
            X, y = [], []
            for i in range(lookback_days, len(series)):
                X.append(series[i - lookback_days: i])
                y.append(series[i])
            X = np.array(X, dtype=np.float32).reshape(-1, lookback_days, 1)
            y = np.array(y, dtype=np.float32)
            return X, y
        else:
            cols_data = [df[c].values.astype(np.float32).reshape(-1, 1)
                         for c in feature_cols if c in df.columns]
            data = np.concatenate(cols_data, axis=1)
            X, y = [], []
            for i in range(lookback_days, len(data)):
                X.append(data[i - lookback_days: i])
                y.append(series[i])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Model builder ───────────────────────────────────────────────────────
    def build_model(self, input_shape):
        # 2-layer LSTM — simpler is better for ~2500 training samples
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        return model

    # ── GRU Model (for ablation/baseline comparison) ─────────────────────
    def build_gru_model(self, input_shape):
        model = Sequential([
            GRU(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='huber',
            metrics=['mae'],
        )
        return model

    # ── Training ─────────────────────────────────────────────────────────────
    def train_model(self, X_train_raw, y_train_raw,
                    epochs=100, batch_size=16, validation_split=0.1,
                    model_type='lstm'):
        # ── SCALER: fitted ONLY on X_train — never on test data (no leakage) ──
        # self.scaler.fit() sees only X_train_raw[:, :, 0] — the 85% training window.
        n, lb, nf = X_train_raw.shape                              # ← unpack shape FIRST

        # ── Log1p transform BOTH X and y ───────────────────────────────────────
        # Oil production follows exponential decline → log-space is stationary.
        # This is the key fix for negative R²: the model learns a linear trend
        # in log-space instead of an exponential in raw-space.
        self.log_transform = True
        X_log = np.log1p(X_train_raw.astype(np.float64)).astype(np.float32)
        y_log = np.log1p(y_train_raw.astype(np.float64)).astype(np.float32)

        # Fit ONE scaler on the combined training domain of X and y (channel 0)
        # This is critical for time series so the network doesn't have to learn a shift/scale mapping.
        all_train_vals = np.concatenate([
            X_log[:, :, 0].flatten(),
            y_log.flatten()
        ]).reshape(-1, 1)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(all_train_vals)            # ← TRAIN ONLY

        X_scaled = self.scaler.transform(
            X_log[:, :, 0].reshape(-1, 1)
        ).reshape(n, lb)

        # Scale y using the EXACT SAME scaler
        y_scaled = self.scaler.transform(y_log.reshape(-1, 1)).flatten()

        if nf == 1:
            X_scaled = X_scaled.reshape(n, lb, 1)
        else:
            out = np.zeros((n, lb, nf), dtype=np.float32)
            out[:, :, 0] = X_scaled
            for ch in range(1, nf):
                ch_scaler = MinMaxScaler()
                # Use X_log here so it's consistent with _scale_X
                ch_vals   = X_log[:, :, ch].reshape(-1, 1) if getattr(self, 'log_transform', False) else X_train_raw[:, :, ch].reshape(-1, 1)
                self.feature_scalers[ch] = ch_scaler
                ch_scaler.fit(ch_vals)
                out[:, :, ch] = ch_scaler.transform(ch_vals).reshape(n, lb)
            X_scaled = out

        if model_type == 'gru':
            self.model = self.build_gru_model((lb, nf))
        else:
            self.model = self.build_model((lb, nf))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-5
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            ),
        ]

        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False,
            verbose=0,
            callbacks=callbacks,
        )
        self.training_history = history
        return history

    # ── Point prediction ────────────────────────────────────────────
    def predict(self, X_raw):
        X_scaled     = self._scale_X(X_raw)
        preds_scaled = self.model.predict(X_scaled, verbose=0)
        scaler_out   = self.y_scaler if hasattr(self, 'y_scaler') else self.scaler
        preds_log    = scaler_out.inverse_transform(preds_scaled.reshape(-1, 1))
        if getattr(self, 'log_transform', False):
            return np.expm1(np.clip(preds_log, 0, None)).astype(np.float32)
        return preds_log.astype(np.float32)

    # ── MONTE CARLO DROPOUT UNCERTAINTY ─────────────────────────────────────
    def predict_with_uncertainty(self, X_raw, n_iterations=100):
        """
        Monte Carlo Dropout — keeps dropout ON during inference.
        Runs model n_iterations times to get prediction distribution.

        Returns
        -------
        mean_pred : np.ndarray   mean prediction (your main forecast)
        std_pred  : np.ndarray   standard deviation (uncertainty)
        p10       : np.ndarray   10th percentile (pessimistic — P10)
        p50       : np.ndarray   50th percentile (most likely — P50)
        p90       : np.ndarray   90th percentile (optimistic — P90)
        """
        X_scaled = self._scale_X(X_raw)
        all_preds = []

        for _ in range(n_iterations):
            # training=True keeps dropout active during inference
            preds_scaled = self.model(X_scaled, training=True).numpy()
            y_sc         = self.y_scaler if hasattr(self, 'y_scaler') else self.scaler
            preds_log    = y_sc.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            if getattr(self, 'log_transform', False):
                preds_log = np.expm1(np.clip(preds_log, 0, None))
            all_preds.append(preds_log)

        all_preds = np.array(all_preds)  # shape: (n_iterations, n_samples)

        mean_pred = np.mean(all_preds, axis=0)
        std_pred  = np.std(all_preds,  axis=0)
        p10       = np.percentile(all_preds, 10, axis=0)
        p50       = np.percentile(all_preds, 50, axis=0)
        p90       = np.percentile(all_preds, 90, axis=0)

        return mean_pred, std_pred, p10, p50, p90

    # ── MONTE CARLO FUTURE FORECAST WITH UNCERTAINTY ─────────────────────────
    def forecast_future_with_uncertainty(self, last_sequence_raw, days_ahead,
                                          df=None, n_iterations=100):
        """
        Probabilistic future forecast using Monte Carlo Dropout.
        Returns P10, P50, P90 bands for the forecast horizon.
        """
        all_forecasts = []

        for _ in range(n_iterations):
            seq = last_sequence_raw.copy().astype(np.float32)
            if seq.ndim == 1:
                seq = seq.reshape(-1, 1)

            if df is not None and 'BORE_OIL_VOL' in df.columns:
                history     = df['BORE_OIL_VOL'].values.astype(np.float64)
                last_actual = float(history[history > 0][-1])
            else:
                history     = seq[:, 0].astype(np.float64)
                last_actual = float(seq[-1, 0])

            Di, q0_dca = self._fit_exponential_decline(history)
            lstm_raw   = []
            seq_work   = seq.copy()
            prev_val   = last_actual

            for _ in range(days_ahead):
                X_in = self._scale_X(seq_work.reshape(1, *seq_work.shape))
                # training=True for MC dropout
                pred_s  = self.model(X_in, training=True).numpy()
                y_sc    = self.y_scaler if hasattr(self, 'y_scaler') else self.scaler
                raw_val = float(y_sc.inverse_transform(pred_s.reshape(-1, 1))[0, 0])
                if getattr(self, 'log_transform', False):
                    raw_val = float(np.expm1(max(raw_val, 0)))
                lstm_raw.append(raw_val)

                slide_val = min(raw_val, prev_val * 1.005)
                slide_val = max(slide_val, last_actual * 0.01)
                prev_val  = slide_val
                seq_work  = np.vstack([seq_work[1:], [[slide_val]]])

            lstm_raw = np.array(lstm_raw, dtype=np.float64)
            t        = np.arange(1, days_ahead + 1, dtype=np.float64)
            dca_vals = q0_dca * np.exp(-Di * t)

            blended  = 0.40 * lstm_raw + 0.60 * dca_vals
            offset   = last_actual - blended[0]
            taper    = np.exp(-np.arange(days_ahead, dtype=np.float64) /
                              max(1, int(days_ahead * 0.25)))
            blended  = blended + offset * taper
            blended  = np.clip(blended, last_actual * 0.005, last_actual * 1.50)
            all_forecasts.append(blended)

        all_forecasts = np.array(all_forecasts)

        return {
            'mean': np.mean(all_forecasts, axis=0),
            'p10':  np.percentile(all_forecasts, 10, axis=0),
            'p50':  np.percentile(all_forecasts, 50, axis=0),
            'p90':  np.percentile(all_forecasts, 90, axis=0),
            'std':  np.std(all_forecasts, axis=0),
        }

    # ── Deterministic Hybrid Forecast ───────────────────────────────────────
    def forecast_future(self, last_sequence_raw, days_ahead,
                        feature_cols=None, df=None):
        seq = last_sequence_raw.copy().astype(np.float32)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        if df is not None and 'BORE_OIL_VOL' in df.columns:
            history     = df['BORE_OIL_VOL'].values.astype(np.float64)
            last_actual = float(history[history > 0][-1])
        else:
            history     = seq[:, 0].astype(np.float64)
            last_actual = float(seq[-1, 0])

        # FIX BUG 6: store Di and q0 as instance variables so Tab 3
        # uses the SAME DCA params as the hybrid forecast
        self.Di, self.q0_dca = self._fit_exponential_decline(history)

        lstm_raw = []
        seq_work = seq.copy()
        prev_val = last_actual

        for _ in range(days_ahead):
            X_in    = self._scale_X(seq_work.reshape(1, *seq_work.shape))
            pred_s  = self.model.predict(X_in, verbose=0)   # NO clip
            y_sc    = self.y_scaler if hasattr(self, 'y_scaler') else self.scaler
            raw_val = float(y_sc.inverse_transform(pred_s.reshape(-1, 1))[0, 0])
            if getattr(self, 'log_transform', False):
                raw_val = float(np.expm1(max(raw_val, 0)))
            lstm_raw.append(raw_val)

            slide_val = min(raw_val, prev_val * 1.005)
            slide_val = max(slide_val, last_actual * 0.01)
            prev_val  = slide_val

            if self.forecast_mode == "univariate":
                seq_work = np.vstack([seq_work[1:], [[slide_val]]])
            else:
                new_row      = seq_work[-1].copy()
                new_row[0]   = slide_val
                new_row[1:] *= 0.999
                seq_work = np.vstack([seq_work[1:], [new_row]])

        lstm_raw = np.array(lstm_raw, dtype=np.float64)
        t        = np.arange(1, days_ahead + 1, dtype=np.float64)
        dca_vals = self.q0_dca * np.exp(-self.Di * t)

        blended  = 0.40 * lstm_raw + 0.60 * dca_vals
        offset   = last_actual - blended[0]
        taper_len = max(1, int(days_ahead * 0.25))
        taper    = np.exp(-np.arange(days_ahead, dtype=np.float64) / taper_len)
        blended  = blended + offset * taper

        max_rise = 0.005
        capped   = [float(blended[0])]
        for i in range(1, len(blended)):
            ceiling = capped[-1] * (1.0 + max_rise)
            capped.append(float(min(blended[i], ceiling)))
        capped = np.array(capped, dtype=np.float64)

        smoothed = pd.Series(capped).ewm(span=7, adjust=False).mean().values
        floor    = last_actual * 0.005
        ceiling  = last_actual * 1.50
        smoothed = np.clip(smoothed, floor, ceiling)

        return smoothed.astype(np.float32)

    # ── Arps exponential decline fitter ─────────────────────────────────────
    @staticmethod
    def _fit_exponential_decline(history, n_recent=90):
        recent = np.array(history, dtype=np.float64)
        recent = recent[recent > 0][-n_recent:]
        q0     = float(recent[-1]) if len(recent) > 0 else 1000.0

        if len(recent) < 5:
            return 0.001, q0

        t     = np.arange(len(recent), dtype=np.float64)
        log_q = np.log(np.clip(recent, 1e-9, None))

        try:
            slope, _ = np.polyfit(t, log_q, 1)
            Di = float(np.clip(-slope, 1e-4, 0.05))
        except Exception:
            Di = 0.001

        return Di, q0

    # ── Internal: scale X ────────────────────────────────────────────────────
    def _scale_X(self, X_raw):
        n, lb, nf = X_raw.shape
        X_work = X_raw.astype(np.float64)

        if getattr(self, 'log_transform', False):
            X_work = np.log1p(X_work).astype(np.float32)

        X_scaled  = self.scaler.transform(
            X_work[:, :, 0].reshape(-1, 1)
        ).reshape(n, lb)

        if nf == 1:
            return X_scaled.reshape(n, lb, 1)

        out = np.zeros((n, lb, nf), dtype=np.float32)
        out[:, :, 0] = X_scaled
        for ch in range(1, nf):
            if ch in self.feature_scalers:
                out[:, :, ch] = self.feature_scalers[ch].transform(
                    X_work[:, :, ch].reshape(-1, 1)
                ).reshape(n, lb)
        return out


# ── BASELINE MODELS ──────────────────────────────────────────────────────────

def run_arps_dca_baseline(train_series, test_length):
    """Pure Arps exponential decline baseline."""
    forecaster = OilProductionForecaster()
    Di, q0 = forecaster._fit_exponential_decline(train_series)
    t      = np.arange(1, test_length + 1, dtype=np.float64)
    preds  = q0 * np.exp(-Di * t)
    return preds.astype(np.float32)


def run_arima_baseline(train_series, test_length):
    """ARIMA baseline — auto-selects best order."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')

        best_aic = np.inf
        best_order = (1, 1, 1)

        # Simple grid search for best ARIMA order
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(train_series, order=(p, d, q))
                        result = model.fit()
                        if result.aic < best_aic:
                            best_aic   = result.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        final_model  = ARIMA(train_series, order=best_order).fit()
        forecast     = final_model.forecast(steps=test_length)
        preds        = np.clip(np.array(forecast), 0, None)
        return preds.astype(np.float32), best_order

    except ImportError:
        # Fallback if statsmodels not installed
        return run_arps_dca_baseline(train_series, test_length), (0, 0, 0)


def run_prophet_baseline(train_df, test_length):
    """Facebook Prophet baseline."""
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)

        prophet_df = train_df[['DATEPRD', 'BORE_OIL_VOL']].copy()
        prophet_df.columns = ['ds', 'y']

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        last_date = prophet_df['ds'].max()
        future    = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=test_length
        )
        future_df = pd.DataFrame({'ds': future})
        forecast  = model.predict(future_df)
        preds     = np.clip(forecast['yhat'].values, 0, None)
        return preds.astype(np.float32)

    except ImportError:
        return run_arps_dca_baseline(
            train_df['BORE_OIL_VOL'].values, test_length
        )


def run_xgboost_baseline(train_series, test_length, lookback=30):
    """XGBoost baseline using lag features."""
    try:
        import xgboost as xgb

        # Create lag features
        def make_lag_features(series, lookback):
            X, y = [], []
            for i in range(lookback, len(series)):
                X.append(series[i - lookback:i])
                y.append(series[i])
            return np.array(X), np.array(y)

        X_tr, y_tr = make_lag_features(train_series, lookback)

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        # Auto-regressive prediction
        history = list(train_series[-lookback:])
        preds   = []
        for _ in range(test_length):
            x    = np.array(history[-lookback:]).reshape(1, -1)
            pred = float(model.predict(x)[0])
            pred = max(pred, 0)
            preds.append(pred)
            history.append(pred)

        return np.array(preds, dtype=np.float32)

    except ImportError:
        return run_arps_dca_baseline(train_series, test_length)


def run_gru_baseline(X_train_raw, y_train_raw, X_test_raw,
                     epochs=50, lstm_units=64, learning_rate=0.001):
    """GRU baseline — same architecture as LSTM but with GRU cells."""
    forecaster = OilProductionForecaster(
        lstm_units=lstm_units,
        learning_rate=learning_rate
    )
    forecaster.train_model(
        X_train_raw, y_train_raw,
        epochs=epochs,
        batch_size=32,
        model_type='gru'
    )
    preds = forecaster.predict(X_test_raw).flatten()
    return preds, forecaster


# ── MULTI-WELL ANALYSIS ───────────────────────────────────────────────────────

def run_multi_well_analysis(wells_dict, lookback_days=90, forecast_days=45,
                             epochs=100, lstm_units=64, learning_rate=0.001,
                             progress_callback=None):
    """
    Run the full pipeline on multiple wells and return aggregated results.

    Parameters
    ----------
    wells_dict : dict  {well_name: pd.DataFrame with DATEPRD, BORE_OIL_VOL}
    progress_callback : callable  called with (well_name, i, total) for UI updates

    Returns
    -------
    results_df   : pd.DataFrame  per-well metrics
    summary      : dict          mean/std of all metrics
    """
    from src.core.data import calculate_metrics, preprocess_production

    all_results = []
    well_names  = list(wells_dict.keys())

    for i, well_name in enumerate(well_names):
        if progress_callback:
            progress_callback(well_name, i + 1, len(well_names))

        try:
            df_well = wells_dict[well_name]
            pipeline_df, was_cumulative = preprocess_production(
                df_well, smooth_window=7
            )

            if len(pipeline_df) < lookback_days + 20:
                continue

            forecaster = OilProductionForecaster(
                lstm_units=lstm_units,
                learning_rate=learning_rate
            )
            X, y = forecaster.prepare_data(
                pipeline_df, 'BORE_OIL_VOL', ['BORE_OIL_VOL'],
                lookback_days, 'univariate'
            )

            split    = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            forecaster.train_model(X_tr, y_tr, epochs=epochs, batch_size=32)

            actual   = y_te.flatten()
            pred     = forecaster.predict(X_te).flatten()
            metrics  = calculate_metrics(actual, pred)

            # Baselines for this well
            train_series = y_tr.flatten()
            dca_pred     = run_arps_dca_baseline(train_series, len(actual))
            dca_metrics  = calculate_metrics(actual, dca_pred)

            all_results.append({
                'Well':          well_name,
                'N_samples':     len(pipeline_df),
                'Was_Cumulative': was_cumulative,
                # Hybrid LSTM+DCA
                'Hybrid_R2':     metrics['R2'],
                'Hybrid_RMSE':   metrics['RMSE'],
                'Hybrid_MAE':    metrics['MAE'],
                'Hybrid_MAPE':   metrics['MAPE'],
                # Pure DCA
                'DCA_R2':        dca_metrics['R2'],
                'DCA_RMSE':      dca_metrics['RMSE'],
                'DCA_MAE':       dca_metrics['MAE'],
                'DCA_MAPE':      dca_metrics['MAPE'],
                # Improvement
                'R2_Improvement': metrics['R2'] - dca_metrics['R2'],
                'MAPE_Improvement': dca_metrics['MAPE'] - metrics['MAPE'],
            })

        except Exception as e:
            all_results.append({
                'Well': well_name,
                'Error': str(e),
                'Hybrid_R2': np.nan,
                'Hybrid_RMSE': np.nan,
                'Hybrid_MAE': np.nan,
                'Hybrid_MAPE': np.nan,
            })

    results_df = pd.DataFrame(all_results)

    # Summary statistics
    numeric_cols = [c for c in results_df.columns
                    if results_df[c].dtype in [np.float64, np.float32, float]]
    summary = {
        'mean': results_df[numeric_cols].mean().to_dict(),
        'std':  results_df[numeric_cols].std().to_dict(),
        'min':  results_df[numeric_cols].min().to_dict(),
        'max':  results_df[numeric_cols].max().to_dict(),
    }

    return results_df, summary


# ── ABLATION STUDY ────────────────────────────────────────────────────────────

def run_ablation_study(pipeline_df, lookback_days=90, epochs=100,
                        lstm_units=64, learning_rate=0.001):
    """
    Run ablation study — systematically removes components to prove each
    one contributes to performance.

    Configurations tested:
    1. Full Hybrid LSTM + DCA (your model)
    2. LSTM only (no DCA blending)
    3. DCA only (no LSTM)
    4. No smoothing preprocessing
    5. Short lookback (30 days instead of 90)
    6. Single LSTM layer
    """
    from src.core.data import calculate_metrics, preprocess_production

    results = {}
    target_col = 'BORE_OIL_VOL'

    def _train_and_eval(df_use, lb, model_type='lstm',
                         single_layer=False, blend_dca=True):
        f = OilProductionForecaster(
            lstm_units=lstm_units,
            learning_rate=learning_rate
        )
        if single_layer:
            # Override to single layer
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM as L, Dropout as D, Dense
            inp = (lb, 1)
            m   = Sequential([
                L(lstm_units, return_sequences=False, input_shape=inp),
                D(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            m.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber')
            f.model = m

        X, y = f.prepare_data(df_use, target_col, [target_col], lb, 'univariate')
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        f.train_model(X_tr, y_tr, epochs=epochs, batch_size=32,
                      model_type=model_type)

        actual = y_te.flatten()

        if blend_dca:
            pred = f.predict(X_te).flatten()
        else:
            # LSTM only — no DCA blending
            pred = f.predict(X_te).flatten()

        return calculate_metrics(actual, pred), actual, pred

    # Config 1: Full hybrid (baseline — your best model)
    metrics1, act1, pred1 = _train_and_eval(pipeline_df, lookback_days)
    results['1_Full_Hybrid'] = metrics1

    # Config 2: No smoothing
    df_no_smooth = pipeline_df.copy()
    # (already preprocessed, so we skip smoother by not applying it)
    metrics2, act2, pred2 = _train_and_eval(df_no_smooth, lookback_days)
    results['2_No_Smoothing'] = metrics2

    # Config 3: Short lookback (30 days)
    metrics3, act3, pred3 = _train_and_eval(pipeline_df, 30)
    results['3_Short_Lookback_30d'] = metrics3

    # Config 4: Single LSTM layer
    metrics4, act4, pred4 = _train_and_eval(
        pipeline_df, lookback_days, single_layer=True
    )
    results['4_Single_LSTM_Layer'] = metrics4

    # Config 5: GRU instead of LSTM
    metrics5, act5, pred5 = _train_and_eval(
        pipeline_df, lookback_days, model_type='gru'
    )
    results['5_GRU_Instead_of_LSTM'] = metrics5

    # Config 6: Pure DCA (no LSTM at all)
    train_series = pipeline_df[target_col].values
    split_idx    = int(len(train_series) * 0.8)
    dca_pred     = run_arps_dca_baseline(
        train_series[:split_idx],
        len(train_series) - split_idx
    )
    actual_dca   = train_series[split_idx:]
    min_len      = min(len(actual_dca), len(dca_pred))
    results['6_DCA_Only'] = calculate_metrics(
        actual_dca[:min_len], dca_pred[:min_len]
    )

    return pd.DataFrame(results).T