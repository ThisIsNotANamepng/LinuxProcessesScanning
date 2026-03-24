import joblib


loaded_model = joblib.load("ProcessAnalyses.pkl")
print(loaded_model.version)  # "1.0.3"