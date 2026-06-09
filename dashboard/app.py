import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="PipeGuard Dashboard", page_icon="🚰", layout="wide")

st.title("PipeGuard: Fluid Flow Anomaly Detection")
st.markdown("Real-time monitoring dashboard powered by Darcy-Weisbach physics and Isolation Forest ML.")

@st.cache_data
def load_data():
    data_path = "data/synthetic_pipeline_data.csv"
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please run src/data_gen.py first.")
        return None
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    model_path = "src/isolation_forest.joblib"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run src/anomaly_model.py first.")
        return None
    return joblib.load(model_path)

df = load_data()
model = load_model()

if df is not None and model is not None:
    st.markdown("### ⏱️ Simulation Controls")
    st.write("Use the slider below to scrub through time and simulate live sensor readings.")
    
    max_idx = len(df) - 1
    
    current_idx = st.slider("Time (Data Row)", 0, max_idx, 0)
    
    st.divider()
    
    current_reading = df.iloc[[current_idx]]
    
    features = ['velocity', 'reynolds_number', 'expected_pressure_drop', 'actual_pressure_drop']
    X_current = current_reading[features]
    
    prediction = model.predict(X_current)[0]
    is_anomaly = True if prediction == -1 else False
    
    if is_anomaly:
        st.error("**HAZARD DETECTED:** Fluid flow anomaly isolated! Potential leak or blockage detected based on physical deviations. 🚨", icon="⚠️")
    else:
        st.success("**STATUS NORMAL:** Pipeline operating within standard physical parameters.", icon="🟢")

    st.markdown("### Live Sensor Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    
    re_val = current_reading['reynolds_number'].values[0]
    regime = "Laminar" if re_val < 2000 else "Transitional" if re_val <= 4000 else "Turbulent"
    
    col1.metric("Velocity", f"{current_reading['velocity'].values[0]:.2f} m/s")
    col2.metric("Reynolds Number", f"{re_val:.0f}", regime)
    col3.metric("Expected Pressure Drop", f"{current_reading['expected_pressure_drop'].values[0]:.2f} Pa")
    
    actual_p = current_reading['actual_pressure_drop'].values[0]
    delta_p = actual_p - current_reading['expected_pressure_drop'].values[0]
    col4.metric("Actual Pressure Drop", f"{actual_p:.2f} Pa", f"{delta_p:.2f} Pa diff", delta_color="inverse")

    st.divider()

    st.markdown("### Pressure Drop History (Last 100 Readings)")
    
    start_idx = max(0, current_idx - 100)
    history_df = df.iloc[start_idx:current_idx+1].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['expected_pressure_drop'], 
                             mode='lines', name='Expected (Physics)', line=dict(color='blue', dash='dash')))
    
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['actual_pressure_drop'], 
                             mode='lines', name='Actual (Sensors)', line=dict(color='orange')))
    
    history_X = history_df[features]
    history_preds = model.predict(history_X)
    history_df['is_ml_anomaly'] = history_preds
    
    anomalies_df = history_df[history_df['is_ml_anomaly'] == -1]
    fig.add_trace(go.Scatter(x=anomalies_df.index, y=anomalies_df['actual_pressure_drop'],
                             mode='markers', name='ML Alert', marker=dict(color='red', size=10, symbol='x')))

    fig.update_layout(
        xaxis_title="Time (Simulation Step)", 
        yaxis_title="Pressure Drop (Pa)", 
        height=400, 
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown("### 📋 System Diagnostics & Incident Log")
    col_health, col_log = st.columns([1, 2])
    
    total_recent = len(history_df)
    total_anomalies = len(anomalies_df)
    health_score = ((total_recent - total_anomalies) / total_recent) * 100 if total_recent > 0 else 100.0
    
    with col_health:
        st.metric("System Health (Last 100)", f"{health_score:.1f}%")
        
        if health_score < 90:
            st.warning("Maintenance recommended. High frequency of physical deviations.")
        else:
            st.success("System operating within acceptable bounds.")

    with col_log:
        st.metric("Hazards Detected (Current Window)", f"{total_anomalies}")
        
        if not anomalies_df.empty:
            display_log = anomalies_df[['velocity', 'expected_pressure_drop', 'actual_pressure_drop']].copy()
            display_log.rename(columns={
                'velocity': 'Velocity (m/s)',
                'expected_pressure_drop': 'Expected Drop (Pa)',
                'actual_pressure_drop': 'Actual Drop (Pa)'
            }, inplace=True)
            
            st.dataframe(display_log, use_container_width=True)
        else:
            st.info("No anomalies detected in the current time window.")