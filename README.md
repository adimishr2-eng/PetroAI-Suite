# PetroAI Suite - Production Forecasting 🛢️

Next-gen oil production forecasting tool using Long Short-Term Memory (LSTM) neural networks. Built with Streamlit, TensorFlow, and Plotly.

## ✨ Features
*   **Deep Learning (LSTM)**: Sophisticated time-series forecasting capable of capturing non-linear reservoir dynamics.
*   **Univariate & Multivariate**: Support for single-variable historical trends or multi-variable models (e.g., Pressure, Temperature, Water Injection).
*   **Interactive EDA**: Real-time correlation analysis and field performance monitoring.
*   **Premium Dashboard**: Sleek, modern design with glassmorphism and performance-optimized visuals.
*   **Synthetic Data Generation**: Instant testing capabilities for quick prototyping.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

## 📊 Data Specifications
The application expects a CSV file with at least two columns:
*   `DATEPRD`: Formatting can be `DD-MMM-YY` (e.g., 01-Jan-20) or standard ISO dates.
*   `BORE_OIL_VOL`: Daily production volume in barrels.

Optional columns for multivariate analysis:
*   `AVG_DOWNHOLE_PRESSURE`
*   `AVG_WHP_P`
*   `Total Water Injection`
*   `AVG_DOWNHOLE_TEMPERATURE`

---
*Created as part of the PetroAI-Suite port for gullar.*
