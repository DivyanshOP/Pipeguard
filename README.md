# PipeGuard: Fluid Flow Anomaly Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Isolation_Forest-orange)
![Physics](https://img.shields.io/badge/Physics-Fluid_Mechanics-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Project Overview

PipeGuard is a real-time monitoring dashboard that integrates standard fluid mechanics with unsupervised machine learning to detect pipeline anomalies (leaks, blockages, or sensor failures).

Unlike standard ML projects that rely purely on historical data trends, PipeGuard calculates the mathematical ground truth of the system using the **Darcy-Weisbach equation** and uses an **Isolation Forest** algorithm to detect real-time physical deviations.

## How It Works

1. **Physics Engine:** Calculates the expected system pressure drop and classifies the flow regime (Laminar vs. Turbulent) based on the **Reynolds Number**.
2. **Data Simulation:** Generates synthetic, physics-informed sensor telemetry with injected hazard anomalies.
3. **Machine Learning:** An Isolation Forest model evaluates the difference between expected physical baselines and actual sensor readings to isolate and flag hazards.
4. **Control Room UI:** A live Streamlit dashboard visualizes system telemetry and logs real-time hazard alerts.

## Tech Stack

| Layer | Tools |
|---|---|
| Backend Engine | Python, NumPy, Pandas |
| Machine Learning | scikit-learn (Isolation Forest) |
| Frontend Dashboard | Streamlit, Plotly |

## How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/YourUsername/Pipeguard.git
cd Pipeguard
```

**2. Install dependencies:**

```bash
pip install pandas numpy scikit-learn streamlit plotly joblib
```

**3. Generate the synthetic dataset:**

```bash
cd src
python data_gen.py
```

**4. Train the ML model:**

```bash
python anomaly_model.py
cd ..
```

**5. Launch the Streamlit dashboard:**

```bash
streamlit run dashboard/app.py
```