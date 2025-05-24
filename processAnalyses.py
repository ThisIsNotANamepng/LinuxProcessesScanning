# Creates model and tests it base don linux processes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import time

start=time.time()
# Load the data
df = pd.read_csv('CombinedSets.csv')

#df = df.drop('type', axis=1)
cols_to_drop = [
    'ts',
    'EXC',
    'type',
    'CPUNR',
    'PID',
]

## psutil can't get POLI, RTPR, TSLPI, or TSLPU but you can get it another way (see chatgpt)

df.drop(columns=cols_to_drop, inplace=True)

#print("Columns:", df.columns.tolist())

# Split data into features and target variable
X = df.iloc[:, :-1]  # Features (all columns except the last one)
y = df.iloc[:, -1]   # Target variable (last column)

#print(X.head())

# Encode categorical variables
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Time taken:  ", time.time()-start)

joblib.dump(rf_classifier, 'ProcessAnalyses.pkl')
