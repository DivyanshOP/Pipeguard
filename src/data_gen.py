import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from physics import (
    calculate_reynolds_number,
    calculate_friction_factor,
    calculate_pressure_drop
)

def generate_pipeline_data(num_samples: int = 2000, anomaly_fraction: float = 0.05) -> pd.DataFrame:
    """
    Generates synthetic pipeline sensor data with injected anomalies.
    """
    np.random.seed(42)
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(num_samples)]
    
    density = np.random.normal(998.0, 2.0, num_samples)
    viscosity = np.random.normal(0.001, 0.00005, num_samples)
    diameter = np.full(num_samples, 0.5)
    length = np.full(num_samples, 1000.0)
    velocity = np.random.normal(2.0, 0.1, num_samples)
    
    num_anomalies = int(num_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    velocity[anomaly_indices] *= np.random.uniform(0.4, 0.7, num_anomalies)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'density': density,
        'viscosity': viscosity,
        'diameter': diameter,
        'length': length,
        'velocity': velocity,
        'is_anomaly': 0
    })
    
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    
    df['reynolds_number'] = df.apply(
        lambda row: calculate_reynolds_number(row.density, row.velocity, row.diameter, row.viscosity), axis=1
    )
    df['friction_factor'] = df['reynolds_number'].apply(calculate_friction_factor)
    df['expected_pressure_drop'] = df.apply(
        lambda row: calculate_pressure_drop(row.friction_factor, row.length, row.diameter, row.density, row.velocity), axis=1
    )
    
    df['actual_pressure_drop'] = df['expected_pressure_drop'] + np.random.normal(0, 100, num_samples)
    
    df.loc[anomaly_indices, 'actual_pressure_drop'] *= np.random.uniform(1.8, 3.5, num_anomalies)

    return df

if __name__ == "__main__":
    print("Generating synthetic pipeline dataset...")
    dataset = generate_pipeline_data(num_samples=2500, anomaly_fraction=0.05)
    
    os.makedirs('../data', exist_ok=True)
    
    file_path = '../data/synthetic_pipeline_data.csv'
    dataset.to_csv(file_path, index=False)
    
    print(f"Dataset generated successfully! Saved to {file_path}")
    print(f"Total records: {dataset.shape[0]}")
    print(f"Total anomalies injected: {dataset['is_anomaly'].sum()}")