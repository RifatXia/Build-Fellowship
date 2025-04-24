import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

def train_with_original_data():
    """Train models using the original data, similar to better_classifier.py"""
    print("Training models using original data...")
    
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
    
    # Save test sets for comparison
    original_test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'frames_test': frames_test
    }
    
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
    best_model = None

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Compute classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Format the report for easy visibility
        report_lines.append(f"Model (Original Data): {model_name}\n")
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
            best_model = model

    # Write predictions to CSV based on the best model
    predictions_df = pd.DataFrame({'frame': frames_test, 'value': best_y_pred})
    
    # Save to both the original location and a renamed file
    predictions_df.to_csv('original_data_predictions.csv', index=False)
    
    # Also save to submission directory with a different name
    os.makedirs('submission', exist_ok=True)
    predictions_df.to_csv('submission/predictions_original_data.csv', index=False)

    print(f"Best model using original data: {best_model_name} with Accuracy: {best_accuracy:.2f}")
    print(f"Predictions saved to original_data_predictions.csv and submission/predictions_original_data.csv")
    
    return original_test_data, best_model, best_model_name, best_accuracy, predictions_df, report_lines

def train_with_enhanced_data():
    """Train models using the enhanced data with motion parameters"""
    print("\nTraining models using enhanced data with motion parameters...")
    
    # Load updated_provided_data.csv that includes motion parameters
    data = pd.read_csv('updated_provided_data.csv')

    # Ensure 'frame' is integer type for merging
    data['frame'] = data['frame'].astype(int)

    # Load target.csv
    target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

    # Ensure 'frame' is integer type for merging
    target['frame'] = target['frame'].astype(int)

    # Merge data and target on 'frame'
    merged = pd.merge(data, target, on='frame', how='inner')

    # Original features + additional motion parameters as features
    # Include all calculated motion parameters
    features = [col for col in data.columns if col != 'frame']
    X = merged[features]
    y = merged['value']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    
    # Save test sets for comparison
    enhanced_test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'frames_test': frames_test
    }
    
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
    best_model = None

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Compute classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Format the report for easy visibility
        report_lines.append(f"Model (Enhanced Data): {model_name}\n")
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
            best_model = model

    # Write predictions to CSV based on the best model
    predictions_df = pd.DataFrame({'frame': frames_test, 'value': best_y_pred})
    
    # Save to both the original location and a renamed file
    predictions_df.to_csv('enhanced_data_predictions.csv', index=False)
    
    # Also save to submission directory with a different name
    os.makedirs('submission', exist_ok=True)
    predictions_df.to_csv('submission/predictions_enhanced_data.csv', index=False)

    print(f"Best model using enhanced data: {best_model_name} with Accuracy: {best_accuracy:.2f}")
    print(f"Predictions saved to enhanced_data_predictions.csv and submission/predictions_enhanced_data.csv")
    
    return enhanced_test_data, best_model, best_model_name, best_accuracy, predictions_df, report_lines

def compare_predictions(original_test_data, enhanced_test_data, 
                       original_pred_df, enhanced_pred_df, 
                       original_best, enhanced_best,
                       original_model_name, enhanced_model_name):
    """Compare predictions from original and enhanced data models"""
    print("\nComparing predictions from original and enhanced data models...")
    
    # Create a directory for comparison plots
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Compare overall accuracy
    accuracy_comparison = {
        'Original Data': original_best,
        'Enhanced Data': enhanced_best
    }
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_comparison.keys(), accuracy_comparison.values(), color=['blue', 'green'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1 for better visualization
    for i, v in enumerate(accuracy_comparison.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.savefig('comparison_plots/accuracy_comparison.png')
    plt.close()
    
    # Compare predictions by looking at differences
    original_frames = set(original_pred_df['frame'])
    enhanced_frames = set(enhanced_pred_df['frame'])
    
    # Find common frames
    common_frames = original_frames.intersection(enhanced_frames)
    
    # Create merged dataframe for comparison
    original_pred_df_common = original_pred_df[original_pred_df['frame'].isin(common_frames)]
    enhanced_pred_df_common = enhanced_pred_df[enhanced_pred_df['frame'].isin(common_frames)]
    
    # Merge on frame
    comparison_df = pd.merge(
        original_pred_df_common, 
        enhanced_pred_df_common, 
        on='frame', 
        suffixes=('_original', '_enhanced')
    )
    
    # Calculate agreement percentage
    agreement = (comparison_df['value_original'] == comparison_df['value_enhanced']).mean() * 100
    disagreement = 100 - agreement
    
    print(f"Agreement between models: {agreement:.2f}%")
    print(f"Disagreement between models: {disagreement:.2f}%")
    
    # Plot disagreement
    plt.figure(figsize=(8, 8))
    plt.pie([agreement, disagreement], 
            labels=['Agreement', 'Disagreement'], 
            autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral'])
    plt.title('Prediction Agreement Between Models')
    plt.savefig('comparison_plots/prediction_agreement.png')
    plt.close()
    
    # Analyze when models disagree
    disagreement_df = comparison_df[comparison_df['value_original'] != comparison_df['value_enhanced']]
    
    # Analyze which frames have disagreements
    plt.figure(figsize=(12, 6))
    plt.scatter(comparison_df['frame'], comparison_df['value_original'], 
                label='Original Predictions', alpha=0.5, s=10)
    plt.scatter(disagreement_df['frame'], disagreement_df['value_enhanced'], 
                label='Enhanced Predictions (where different)', color='red', s=20)
    plt.xlabel('Frame Number')
    plt.ylabel('Prediction (0 or 1)')
    plt.title('Prediction Differences by Frame')
    plt.legend()
    plt.savefig('comparison_plots/prediction_differences.png')
    plt.close()
    
    # Write comparison results to file
    with open('comparison_results.txt', 'w') as f:
        f.write(f"Model Comparison: Original Data vs. Enhanced Data\n\n")
        f.write(f"Best model using original data: {original_model_name} with Accuracy: {original_best:.4f}\n")
        f.write(f"Best model using enhanced data: {enhanced_model_name} with Accuracy: {enhanced_best:.4f}\n\n")
        f.write(f"Accuracy improvement: {enhanced_best - original_best:.4f} ({(enhanced_best - original_best) * 100:.2f}%)\n\n")
        f.write(f"Agreement between models: {agreement:.2f}%\n")
        f.write(f"Disagreement between models: {disagreement:.2f}%\n\n")
        f.write(f"Number of frames with different predictions: {len(disagreement_df)}\n")
    
    # Return the comparison results
    return {
        'agreement': agreement,
        'disagreement': disagreement,
        'disagreement_frames': len(disagreement_df),
        'accuracy_improvement': enhanced_best - original_best
    }

def combined_report(original_report_lines, enhanced_report_lines, comparison_results,
                  original_model_name, enhanced_model_name, original_best, enhanced_best):
    """Generate a combined report with all results"""
    with open('final_classifier_report.txt', 'w') as f:
        f.write("=== ORIGINAL DATA MODELS ===\n\n")
        f.writelines(original_report_lines)
        f.write("\n=== ENHANCED DATA MODELS ===\n\n")
        f.writelines(enhanced_report_lines)
        f.write("\n=== COMPARISON SUMMARY ===\n\n")
        f.write(f"Original Data Best Model: {original_model_name} with Accuracy: {original_best:.4f}\n")
        f.write(f"Enhanced Data Best Model: {enhanced_model_name} with Accuracy: {enhanced_best:.4f}\n\n")
        f.write(f"Accuracy improvement: {comparison_results['accuracy_improvement']:.4f} ({(comparison_results['accuracy_improvement'] * 100):.2f}%)\n\n")
        f.write(f"Agreement between models: {comparison_results['agreement']:.2f}%\n")
        f.write(f"Disagreement between models: {comparison_results['disagreement']:.2f}%\n")
        f.write(f"Number of frames with different predictions: {comparison_results['disagreement_frames']}\n")
        
    print(f"Combined report saved to final_classifier_report.txt")

if __name__ == "__main__":
    # Process original data
    original_test_data, original_model, original_model_name, original_best, original_pred_df, original_report_lines = train_with_original_data()
    
    # Process enhanced data with motion parameters
    enhanced_test_data, enhanced_model, enhanced_model_name, enhanced_best, enhanced_pred_df, enhanced_report_lines = train_with_enhanced_data()
    
    # Compare predictions
    comparison_results = compare_predictions(
        original_test_data, enhanced_test_data,
        original_pred_df, enhanced_pred_df,
        original_best, enhanced_best,
        original_model_name, enhanced_model_name
    )
    
    # Generate combined report
    combined_report(
        original_report_lines, enhanced_report_lines, comparison_results,
        original_model_name, enhanced_model_name, original_best, enhanced_best
    )
    
    # Save the best overall prediction to predictions.csv for submission
    if enhanced_best > original_best:
        print("\nThe enhanced data model had better accuracy!")
        print(f"Best overall model: {enhanced_model_name} with enhanced data, Accuracy: {enhanced_best:.4f}")
        enhanced_pred_df.to_csv('submission/predictions.csv', index=False)
    else:
        print("\nThe original data model had better accuracy!")
        print(f"Best overall model: {original_model_name} with original data, Accuracy: {original_best:.4f}")
        original_pred_df.to_csv('submission/predictions.csv', index=False)
        
    print("\nBoth prediction files have been saved:")
    print("1. Original data predictions: submission/predictions_original_data.csv")
    print("2. Enhanced data predictions: submission/predictions_enhanced_data.csv")
    print("3. Best model's predictions (for submission): submission/predictions.csv") 