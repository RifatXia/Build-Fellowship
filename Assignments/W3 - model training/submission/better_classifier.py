import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort']
X = merged[features]
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

window_size = 2  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)
y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Store classification reports and track the best model
report_lines = []
best_model_name = None
best_accuracy = 0
best_y_pred = None

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Format the report for easy visibility
    report_lines.append(f"Model: {model_name}\n")
    report_lines.append(f"Class 0 - Precision: {report['0']['precision']:.2f}, Recall: {report['0']['recall']:.2f}, F1-Score: {report['0']['f1-score']:.2f}, Support: {report['0']['support']}\n")
    report_lines.append(f"Class 1 - Precision: {report['1']['precision']:.2f}, Recall: {report['1']['recall']:.2f}, F1-Score: {report['1']['f1-score']:.2f}, Support: {report['1']['support']}\n")
    report_lines.append(f"Accuracy: {report['accuracy']:.2f}\n")
    report_lines.append(f"Macro Avg - Precision: {report['macro avg']['precision']:.2f}, Recall: {report['macro avg']['recall']:.2f}, F1-Score: {report['macro avg']['f1-score']:.2f}\n")
    report_lines.append(f"Weighted Avg - Precision: {report['weighted avg']['precision']:.2f}, Recall: {report['weighted avg']['recall']:.2f}, F1-Score: {report['weighted avg']['f1-score']:.2f}\n")
    report_lines.append("\n")
    
    # Check if this model is the best based on accuracy
    if report['accuracy'] > best_accuracy:
        best_accuracy = report['accuracy']
        best_model_name = model_name
        best_y_pred = y_pred

# Write the comparison report to a text file
with open('submission/report.txt', 'w') as f:
    f.writelines(report_lines)

# Write predictions to CSV based on the best model
predictions_df = pd.DataFrame({'frame': frames_test, 'value': best_y_pred})
predictions_df.to_csv('submission/predictions.csv', index=False)

print(f"Predictions saved based on the best model: {best_model_name} with Accuracy: {best_accuracy:.2f}")

