import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def train_anomaly_detector(data_path: str, model_save_path: str):
    """
    Trains an Isolation Forest model on pipeline data and saves it for deployment.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    features = [
        'velocity', 
        'reynolds_number', 
        'expected_pressure_drop', 
        'actual_pressure_drop'
    ]
    
    X = df[features]
    y_true = df['is_anomaly']
    
    print("Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    model.fit(X)
    
    predictions = model.predict(X)
    
    df['ml_prediction'] = [1 if pred == -1 else 0 for pred in predictions]
    
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_true, df['ml_prediction'], target_names=['Normal (0)', 'Anomaly (1)']))
    
    joblib.dump(model, model_save_path)
    print(f"Model successfully saved to {model_save_path}")

if __name__ == "__main__":
    DATA_FILE = '../data/synthetic_pipeline_data.csv'
    MODEL_FILE = 'isolation_forest.joblib'
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Could not find {DATA_FILE}. Did you run data_gen.py yet?")
    else:
        train_anomaly_detector(DATA_FILE, MODEL_FILE)