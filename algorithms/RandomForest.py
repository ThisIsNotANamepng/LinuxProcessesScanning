# Trains a Random Forest Model based on malicious linux processes

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
data = pd.read_csv('CombinedSets.csv')

data = data.drop('type', axis=1)
data = data.drop('CPUNR', axis=1)
data = data.drop('EXC', axis=1)
data = data.drop('PID', axis=1)
data = data.drop('ts', axis=1)

#data = data.drop('ts', axis=1) I don't think we need to drop this, I assume ts is something like time to live but it may be the id of the process or some other unique value

#data = data.drop('attack_cat', axis=1) # For now just train for label attack or not, train for type of attack later


# Separate features and target variable
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

# Encode categorical variables
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.version = "0.1.0"  # Custom version tag

# Train the classifier
rf_classifier.fit(X, y)

# Save the trained model to a file
joblib.dump(rf_classifier, 'ProcessAnalyses.pkl')
