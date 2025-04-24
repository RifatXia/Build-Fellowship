import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ray

# Load updated_provided_data.csv that includes motion parameters
data = pd.read_csv('updated_provided_data.csv')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv (from parent directory)
target = pd.read_csv('../target.csv')

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target
features = [col for col in data.columns if col != 'frame']
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

@ray.remote
def create_model(X, y, window_size, C):
    X_lagged = create_lag_features(X, window_size)
    y_lagged = y.iloc[window_size - 1:]
    
    # Align indices
    y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
    X_lagged = X_lagged.reset_index(drop=True)
    
    # Split into train and test sets
    split_index = int(len(X_lagged) * 0.7)
    X_train = X_lagged.iloc[:split_index]
    X_test = X_lagged.iloc[split_index:]
    y_train = y_lagged.iloc[:split_index]
    y_test = y_lagged.iloc[split_index:]
    
    # Train and evaluate the model using Logistic Regression
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_train, y_train)
    return window_size, C, clf.score(X_test, y_test)

# Define parameter grid
param_grid = {
    'window_size': [1, 2, 5, 10, 50],
    'C': [0.1, 0.5, 1, 2, 10]
}

# Initialize Ray
ray.init(num_cpus=2)

# Perform grid search
results = []
total_iterations = len(param_grid['window_size']) * len(param_grid['C'])
futures = []

for window_size in param_grid['window_size']:
    for C in param_grid['C']:
        futures.append(create_model.remote(X_scaled, y, window_size, C))

with tqdm(total=total_iterations, desc="Parameter Search") as pbar:
    while futures:
        done, futures = ray.wait(futures)
        results.extend(ray.get(done))
        pbar.update(len(done))

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['window_size', 'C', 'score'])

# Create heatmap
plt.figure(figsize=(12, 8))
pivot_table = results_df.pivot_table(index='window_size', columns='C', values='score')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Model Performance: Window Size vs. Logistic Regression C Parameter')
plt.xlabel('C Parameter')
plt.ylabel('Window Size')
plt.tight_layout()
plt.savefig('parameter_search_heatmap.png')
plt.close()

# Find best parameters
best_result = results_df.loc[results_df['score'].idxmax()]
print(f"Best parameters: Window Size = {best_result['window_size']}, "
      f"C = {best_result['C']}")
print(f"Best score: {best_result['score']:.3f}")

# Train final model with best parameters
best_window_size = int(best_result['window_size'])
best_C = best_result['C']

X_lagged = create_lag_features(X_scaled, best_window_size)
y_lagged = y.iloc[best_window_size - 1:]
frames_lagged = merged['frame'].iloc[best_window_size - 1:]

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]

# Train final model
clf = LogisticRegression(C=best_C, max_iter=1000)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

# Shut down Ray
ray.shutdown()
