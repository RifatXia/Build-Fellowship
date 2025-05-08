import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

class Classifier:
    def __init__(self, output_dir):
        """Initialize the classifier with output directory for results"""
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store results and models
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None

    def create_lag_features(self, X, window_size=2):
        """Create time-series lag features"""
        X_lagged = pd.DataFrame()
        for i in range(window_size):
            X_shifted = pd.DataFrame(X).shift(i)
            X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
            X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
        return X_lagged.dropna()

    def prepare_data(self, features_df, target_df=None):
        """Prepare data for training/prediction"""
        # Ensure all numeric columns are float
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].astype(float)
        
        # Drop non-feature columns
        features = [col for col in features_df.columns if col not in ['frame']]
        X = features_df[features]
        frames = features_df['frame'].values

        if target_df is not None:
            # Merge and handle any missing values
            merged = pd.merge(features_df, target_df, on='frame', how='inner')
            print(f"Merged data shape: {merged.shape}")
            
            if merged.shape[0] == 0:
                raise ValueError("No matching frames between features and target data")
                
            y = merged['value']
            frames = merged['frame'].values
            X = merged[features]
        else:
            y = None

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Create lag features
        X_lagged = self.create_lag_features(X_scaled)
        
        if y is not None:
            # Align y with lagged features
            y_lagged = pd.Series(y).iloc[2:].reset_index(drop=True)
            frames_lagged = pd.Series(frames).iloc[2:].reset_index(drop=True)
            
            # Ensure all data has the same length
            min_length = min(len(X_lagged), len(y_lagged), len(frames_lagged))
            X_lagged = X_lagged.iloc[:min_length]
            y_lagged = y_lagged.iloc[:min_length]
            frames_lagged = frames_lagged.iloc[:min_length]
        else:
            y_lagged = None
            frames_lagged = pd.Series(frames).iloc[2:].reset_index(drop=True)
            
        print(f"Final prepared data shapes - X: {X_lagged.shape}, frames: {len(frames_lagged)}")
        if y_lagged is not None:
            print(f"y: {len(y_lagged)}")

        return X_lagged, y_lagged, frames_lagged

    def train_and_evaluate(self, X, y, frames, data_type="original"):
        """Train multiple models and evaluate their performance"""
        # Ensure all data is properly aligned
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        frames = frames.reset_index(drop=True)
        
        # Split data
        split_idx = int(len(X) * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        frames_test = frames.iloc[split_idx:]

        # Define models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        results = []
        best_accuracy = 0
        best_model = None
        best_model_name = None
        best_predictions = None
        report_lines = []

        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Ensure predictions are aligned with test data
            y_pred = y_pred[:len(y_test)]
            
            # Compute classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Format the report for easy visibility
            report_lines.append(f"Model ({data_type} Data): {name}\n")
            report_lines.append(f"Class 0 - Precision: {report['0']['precision']:.2f}, Recall: {report['0']['recall']:.2f}, F1-Score: {report['0']['f1-score']:.2f}, Support: {report['0']['support']}\n")
            report_lines.append(f"Class 1 - Precision: {report['1']['precision']:.2f}, Recall: {report['1']['recall']:.2f}, F1-Score: {report['1']['f1-score']:.2f}, Support: {report['1']['support']}\n")
            report_lines.append(f"Accuracy: {report['accuracy']:.2f}\n")
            report_lines.append(f"Macro Avg - Precision: {report['macro avg']['precision']:.2f}, Recall: {report['macro avg']['recall']:.2f}, F1-Score: {report['macro avg']['f1-score']:.2f}\n")
            report_lines.append(f"Weighted Avg - Precision: {report['weighted avg']['precision']:.2f}, Recall: {report['weighted avg']['recall']:.2f}, F1-Score: {report['weighted avg']['f1-score']:.2f}\n")
            report_lines.append("\n")
            
            result = {
                'name': name,
                'accuracy': report['accuracy'],
                'model': model,
                'predictions': y_pred,
                'y_test': y_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'report': report
            }
            results.append(result)

            if report['accuracy'] > best_accuracy:
                best_accuracy = report['accuracy']
                best_model = model
                best_model_name = name
                best_predictions = y_pred

        # Save results
        self._save_model_results(results, frames_test, data_type)
        
        # Save detailed report
        with open(os.path.join(self.output_dir, f'{data_type}_model_report.txt'), 'w') as f:
            f.writelines(report_lines)
        
        return {
            'model': best_model,
            'name': best_model_name,
            'accuracy': best_accuracy,
            'predictions': pd.DataFrame({'frame': frames_test, 'value': best_predictions}),
            'all_results': results,
            'report_lines': report_lines
        }

    def _save_model_results(self, results, frames, data_type):
        """Save detailed results and visualizations"""
        # Create comparison plots
        plt.figure(figsize=(10, 6))
        accuracies = [r['accuracy'] for r in results]
        names = [r['name'] for r in results]
        
        # Bar plot of accuracies
        plt.bar(names, accuracies)
        plt.title(f'Model Accuracies ({data_type} data)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{data_type}_accuracies.png'))
        plt.close()

        # Confusion matrices
        for result in results:
            plt.figure(figsize=(6, 6))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'{result["name"]} Confusion Matrix ({data_type} data)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{data_type}_{result["name"]}_confusion.png'))
            plt.close()

    def predict(self, tracking_csv, target_csv):
        """Main prediction method that handles both original and updated data"""
        # Load original data with correct column names
        print("Processing original tracking data...")
        tracking_df = pd.read_csv(tracking_csv, header=None, 
                                names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
        
        # Handle missing values in effort column
        tracking_df['effort'] = pd.to_numeric(tracking_df['effort'], errors='coerce')
        tracking_df['effort'] = tracking_df['effort'].ffill()  # Forward fill
        tracking_df['effort'] = tracking_df['effort'].bfill()  # Backward fill for any remaining NAs
        
        # Load target data (has headers: frame,value)
        print("Loading target data...")
        target_df = pd.read_csv(target_csv)
        
        # Ensure frame columns are integers for proper merging
        tracking_df['frame'] = tracking_df['frame'].astype(int)
        target_df['frame'] = target_df['frame'].astype(int)
        
        # Verify data after merging
        print(f"Original tracking data shape: {tracking_df.shape}")
        print(f"Target data shape: {target_df.shape}")
        
        # Prepare and evaluate original data
        X_orig, y_orig, frames_orig = self.prepare_data(tracking_df, target_df)
        
        # Verify prepared data
        print(f"Prepared original data shape: {X_orig.shape}")
        if y_orig is not None:
            print(f"Prepared target data shape: {y_orig.shape}")
        
        if X_orig.shape[0] == 0:
            raise ValueError("No valid data after preparation. Please check the data alignment between tracking and target files.")
        
        original_results = self.train_and_evaluate(X_orig, y_orig, frames_orig, "original")
        
        # Check for updated tracking data
        updated_tracking_csv = os.path.join(os.path.dirname(tracking_csv), "updated_tracking.csv")
        if os.path.exists(updated_tracking_csv):
            print("Processing updated tracking data...")
            # updated_tracking.csv has headers because we saved it with headers
            updated_df = pd.read_csv(updated_tracking_csv)
            
            # Ensure frame column is integer
            updated_df['frame'] = updated_df['frame'].astype(int)
            
            # Verify updated data
            print(f"Updated tracking data shape: {updated_df.shape}")
            
            X_updated, y_updated, frames_updated = self.prepare_data(updated_df, target_df)
            
            # Verify prepared updated data
            print(f"Prepared updated data shape: {X_updated.shape}")
            if y_updated is not None:
                print(f"Prepared updated target data shape: {y_updated.shape}")
            
            if X_updated.shape[0] == 0:
                print("Warning: No valid data after preparing updated tracking data. Using original data only.")
                return original_results['predictions'], None, original_results['predictions']
            
            updated_results = self.train_and_evaluate(X_updated, y_updated, frames_updated, "updated")
            
            # Compare results and choose best model
            if updated_results['accuracy'] > original_results['accuracy']:
                print(f"Using updated data model ({updated_results['name']}) with accuracy: {updated_results['accuracy']:.3f}")
                best_predictions = updated_results['predictions']
                self.best_model = updated_results['model']
                self.best_model_name = f"Updated_{updated_results['name']}"
            else:
                print(f"Using original data model ({original_results['name']}) with accuracy: {original_results['accuracy']:.3f}")
                best_predictions = original_results['predictions']
                self.best_model = original_results['model']
                self.best_model_name = f"Original_{original_results['name']}"
            
            # Plot comparison between original and updated
            self._plot_model_comparison(original_results, updated_results)
            
            # Save comparison report
            self._save_comparison_report(original_results, updated_results)
            
            return original_results['predictions'], updated_results['predictions'], best_predictions
        else:
            print("No updated tracking data found. Using only original data.")
            self.best_model = original_results['model']
            self.best_model_name = f"Original_{original_results['name']}"
            return original_results['predictions'], None, original_results['predictions']

    def _plot_model_comparison(self, original_results, updated_results):
        """Create comparison plots between original and updated data results"""
        # Accuracy comparison
        plt.figure(figsize=(8, 6))
        plt.bar(['Original', 'Updated'], 
                [original_results['accuracy'], updated_results['accuracy']],
                color=['blue', 'green'])
        plt.title('Accuracy Comparison: Original vs Updated Data')
        plt.ylabel('Accuracy')
        for i, v in enumerate([original_results['accuracy'], updated_results['accuracy']]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_comparison.png'))
        plt.close()

    def _save_comparison_report(self, original_results, updated_results):
        """Save detailed comparison report"""
        with open(os.path.join(self.output_dir, 'model_comparison.txt'), 'w') as f:
            f.write("Model Comparison Report\n")
            f.write("======================\n\n")
            f.write(f"Original Data Best Model: {original_results['name']}\n")
            f.write(f"Original Data Accuracy: {original_results['accuracy']:.3f}\n\n")
            f.write(f"Updated Data Best Model: {updated_results['name']}\n")
            f.write(f"Updated Data Accuracy: {updated_results['accuracy']:.3f}\n\n")
            f.write(f"Improvement: {(updated_results['accuracy'] - original_results['accuracy']):.3f}\n")
