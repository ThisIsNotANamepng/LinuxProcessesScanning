import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#from skl2onnx import convert_sklearn, update_registered_converter
#from skl2onnx.common.data_types import FloatTensorType

# Load collected features; assume last column is label (0=benign, 1=malicious).
data = pd.read_csv('proc_features.csv')
features = [c for c in data.columns if c not in ['timestamp','pid','name','status','label']]
X = data[features].fillna(0)
y = data['label']

# Split train/test for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train a lightweight classifier (XGBoost here for example)
model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate performance (accuracy, recall, etc.)
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Convert to ONNX (requires skl2onnx or similar, omitted install here).
# Here we sketch the idea; actual conversion requires the skl2onnx package.
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType

# Example conversion (adjust input size as needed).
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("malware_detector.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
