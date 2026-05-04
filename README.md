# 🛢️ PetroAI Suite
### Advanced Reservoir Analytics & Neural Production Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![R2 Score](https://img.shields.io/badge/R²-0.924-brightgreen?style=flat-square)
![MAPE](https://img.shields.io/badge/MAPE-6.1%25-brightgreen?style=flat-square)

> **PetroAI Suite** is an end-to-end oil production forecasting dashboard that combines **Hybrid LSTM + Arps Decline Curve Analysis (DCA)** to predict future field production rates from historical data. Built for petroleum engineers who need fast, data-driven reservoir insights.

---

## 📸 Demo

![PetroAI Dashboard](assets/dashboard_preview.png)

---

## 🚀 Features

- **Hybrid LSTM + Arps DCA Forecasting** — Blends deep learning with classical petroleum engineering decline curves for robust predictions
- **Auto-detect Cumulative vs Daily Rate** — Automatically identifies and converts cumulative volume columns to daily production rates
- **Smart Column Detection** — No manual column mapping needed; the app detects date and production columns from any CSV format
- **Interactive Dashboard** — Three-tab UI: Asset Performance, Neural Training Studio, and Future Outlook
- **Baseline Model Comparison** — Compare against ARIMA, Prophet, XGBoost, GRU, and pure Arps DCA
- **Monte Carlo Uncertainty** — P10/P50/P90 probabilistic forecast bands
- **Multi-Well Analysis** — Batch process multiple wells and compare metrics
- **Ablation Study** — Systematically prove each model component contributes to performance
- **CSV Export** — Download forecast reports directly from the dashboard

---

## 🧠 Model Architecture

```
Raw CSV
   │
   ▼
preprocess_production()
   ├── Parse & sort chronologically
   ├── Auto-detect cumulative → .diff() → daily rate
   ├── Drop NaN, zeros, negatives
   └── Rolling mean smoother (14-day causal window)
   │
   ▼
LSTM Sequence Builder (lookback window)
   │
   ▼
Train/Test Split (70/30 chronological)
   │
   ├── log1p transform (handles exponential decline)
   ├── MinMaxScaler on X_train only (no data leakage)
   └── MinMaxScaler on y_train only (no data leakage)
   │
   ▼
2-Layer LSTM → Huber Loss → EarlyStopping
   │
   ▼
Hybrid Blend: 40% LSTM + 60% Arps DCA
   │
   ▼
Future Forecast (1–180 days)
```

---

## 📁 Project Structure

```
petroai-suite/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── assets/
│   └── logo.png                  # App logo
│
└── src/
    ├── core/
    │   ├── forecaster.py         # LSTM model, GRU, baseline models, multi-well
    │   └── data.py               # Preprocessing, metrics, sample data generator
    └── ui/
        └── styling.py            # Custom CSS and Plotly layout helpers
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/adimishr2-eng/PetroAI-Suite.git
cd PetroAI-Suite
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
tensorflow>=2.12.0
scikit-learn>=1.2.0
plotly>=5.14.0
statsmodels>=0.14.0
prophet>=1.1.4
xgboost>=1.7.0
scipy>=1.10.0
```

---

## 📊 Supported CSV Formats

The app accepts any CSV with a **date column** and an **oil production column**. It auto-detects both.

| Column Type | Accepted Names (examples) |
|---|---|
| Date | `DATE`, `DATEPRD`, `date_prd`, `production_date` |
| Oil Volume | `OIL_VOLUME`, `BORE_OIL_VOL`, `oil_vol`, `qo`, `oil_rate` |

**Cumulative OR daily rate** — both are handled automatically.

### Example CSV structure
```csv
DATE,OIL_VOLUME,GAS_VOLUME,WATER_VOLUME
1997-11-06,474954,12300,8900
1997-11-07,949752,24500,17800
...
```

---

## 🛠️ Usage

1. **Upload your CSV** using the sidebar file uploader
2. The app auto-detects your date and production columns
3. Configure the neural engine settings (lookback window, forecast horizon, epochs)
4. Click **▶ RUN FILE** to train the model and generate forecasts
5. Explore results across the three dashboard tabs
6. **Download** the forecast CSV for reporting

### Bootstrap Demo
No CSV? Click **✨ BOOTSTRAP DEMO DATA** in the sidebar to run on synthetic Arps-decline field data instantly.

---

## 📈 Metrics Explained

| Metric | Description | Good Range |
|---|---|---|
| **R²** | Coefficient of determination — how well model explains variance | > 0.70 |
| **RMSE** | Root Mean Squared Error in bbl/d | Lower is better |
| **MAE** | Mean Absolute Error in bbl/d | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 15% |
| **NSE** | Nash-Sutcliffe Efficiency (hydrology standard) | > 0.70 |

---

## 🔬 Baseline Models

PetroAI Suite benchmarks the Hybrid LSTM+DCA against:

| Model | Description |
|---|---|
| **Arps DCA** | Classical exponential decline curve (petroleum standard) |
| **ARIMA** | Auto-selected order via AIC grid search |
| **Prophet** | Facebook Prophet with yearly seasonality |
| **XGBoost** | Gradient boosting with lag features |
| **GRU** | Gated Recurrent Unit (same architecture, different cell) |

---

## 🏆 Results on Norne Field Dataset

| Model | R² | RMSE (bbl/d) | MAE (bbl/d) | MAPE | NSE |
|---|---|---|---|---|---|
| **Hybrid LSTM+DCA** | **0.924** | **720.7** | **583.4** | **6.1%** | **0.924** |
| Arps DCA only | 0.61 | 1,891 | — | 14.7% | — |
| ARIMA | 0.54 | 2,103 | — | 16.3% | — |
| XGBoost | 0.69 | 1,612 | — | 11.1% | — |

*Results using 3,299 daily rows (Norne Field, 1997–2006), 70/30 chronological train/test split, 90-day lookback window, 100 epochs, Huber loss, MinMaxScaler (no data leakage).*

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| Deep Learning | TensorFlow / Keras (LSTM, GRU) |
| Classical ML | XGBoost, scikit-learn |
| Time Series | statsmodels (ARIMA), Prophet |
| Data | Pandas, NumPy |
| Scaling | MinMaxScaler (train-only, no leakage) |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Adarsh Mishra**
B.Tech Petroleum Engineering 

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Norne Field Dataset](https://github.com/OPM/opm-data) — Open Porous Media (OPM) Initiative
- [TensorFlow](https://tensorflow.org) — Neural network framework
- [Streamlit](https://streamlit.io) — Dashboard framework
- Arps, J.J. (1945) — *Analysis of Decline Curves*, SPE-945228)
