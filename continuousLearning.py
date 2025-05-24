import psutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Data Collection
def collect_data(interval=2):

    print("Collecting...")
    
    data = []
    #while True:
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            
            info = proc.info

            print(info)

            data.append({
                'pid': info['pid'],
                'name': info['name'],
                'cpu_percent': info['cpu_percent'],
                'memory_percent': info['memory_percent']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    time.sleep(interval)

# Data Preprocessing
def preprocess_data(data):
    # Clean and preprocess data
    df = pd.DataFrame(data)
    # Extract relevant features
    features = df[['cpu_percent', 'memory_percent']]
    return features

# Model Training
def train_model(features):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, [0] * len(features), test_size=0.2, random_state=42)
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Anomaly Detection
def detect_anomalies(model, features):
    # Predict likelihood of a process being malicious or benign
    predictions = model.predict(features)
    # Set threshold to determine when a process is considered suspicious
    threshold = 0.5
    suspicious_processes = []
    for i, prediction in enumerate(predictions):
        if prediction < threshold:
            suspicious_processes.append(features.iloc[i])
    return suspicious_processes

# Alerting
def alert_user(suspicious_processes):
    # Create a system to alert the user
    print("Suspicious processes detected:")
    for process in suspicious_processes:
        print(process)

# Main loop
while True:
    data = collect_data()
    features = preprocess_data(data)
    model = train_model(features)
    accuracy = evaluate_model(model, features, [0] * len(features))
    suspicious_processes = detect_anomalies(model, features)
    alert_user(suspicious_processes)
