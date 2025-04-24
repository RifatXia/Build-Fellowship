import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt

@dataclass
class SmoothingParams:
    window_size: int
    min_duration: int
    hysteresis: float
    
    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'min_duration': self.min_duration,
            'hysteresis': self.hysteresis
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SmoothingParams':
        return cls(
            window_size=d['window_size'],
            min_duration=d['min_duration'],
            hysteresis=d['hysteresis']
        )

class PredictionSmoother:
    """
    A class that smooths frame-by-frame predictions and can optimize its parameters
    to match target sequences.
    """
    
    def __init__(self):
        # Default parameter search spaces
        self.default_param_grid = {
            'window_size': [3, 5, 7, 9, 11],
            'min_duration': [1, 2, 3, 4, 5, 10, 30, 60, 120],
            'hysteresis': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        }
    
    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Smooth frame-by-frame predictions using specified parameters.
        """
        if not predictions:
            return []
        
        # Convert input predictions to integers, just in case
        predictions = [int(pred) for pred in predictions]
        
        # Ensure window_size is odd
        window_size = max(3, params.window_size if params.window_size % 2 == 1 
                         else params.window_size + 1)
        half_window = window_size // 2
        
        # Step 1: Apply sliding window majority voting
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            
            active_ratio = sum(window) / len(window)
            
            threshold = 0.5
            if smoothed:
                threshold = params.hysteresis if smoothed[-1] else 1 - params.hysteresis
                
            smoothed.append(1 if active_ratio >= threshold else 0)
        
        # Step 2: Remove short duration state changes
        if params.min_duration > 1:
            final = []
            current_state = smoothed[0]
            current_duration = 1
            
            for pred in smoothed[1:]:
                if pred == current_state:
                    current_duration += 1
                else:
                    if current_duration >= params.min_duration:
                        final.extend([current_state] * current_duration)
                    else:
                        final.extend([not current_state] * current_duration)
                    current_state = pred
                    current_duration = 1
            
            # Handle the last sequence
            if current_duration >= params.min_duration:
                final.extend([current_state] * current_duration)
            else:
                final.extend([not current_state] * current_duration)
                
            return final
        
        return [int(pred) for pred in smoothed]
    
    def calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """
        Calculate various metrics comparing predictions to targets.
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
            
        # Convert to numpy arrays for easier computation
        pred_arr = np.array(predictions)
        target_arr = np.array(targets)
        
        # Calculate basic metrics
        accuracy = np.mean(pred_arr == target_arr)
        
        # Calculate state change metrics
        pred_changes = np.sum(np.abs(np.diff(pred_arr)))
        target_changes = np.sum(np.abs(np.diff(target_arr)))
        
        # F1 score components
        true_pos = np.sum((pred_arr == 1) & (target_arr == 1))
        false_pos = np.sum((pred_arr == 1) & (target_arr == 0))
        false_neg = np.sum((pred_arr == 0) & (target_arr == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'state_change_difference': float(abs(pred_changes - target_changes))
        }
    
    def optimize_parameters(self, 
                          predictions: List[bool], 
                          targets: List[bool],
                          param_grid: Optional[Dict] = None,
                          metric: str = 'f1_score') -> Tuple[SmoothingParams, Dict[str, float]]:
        """
        Find optimal parameters by grid search to match target sequence.
        
        Args:
            predictions: Original predictions to smooth
            targets: Target sequence to match
            param_grid: Dictionary of parameter ranges to search
            metric: Metric to optimize ('accuracy', 'f1_score', etc.)
            
        Returns:
            Tuple of (best parameters, best metrics)
        """
        if param_grid is None:
            param_grid = self.default_param_grid
            
        best_params = None
        best_metrics = None
        best_score = -float('inf')
        
        # Generate all parameter combinations
        param_combinations = product(
            param_grid['window_size'],
            param_grid['min_duration'],
            param_grid['hysteresis']
        )
        
        for window_size, min_duration, hysteresis in param_combinations:
            params = SmoothingParams(window_size, min_duration, hysteresis)
            smoothed = self.smooth_predictions(predictions, params)
            metrics = self.calculate_metrics(smoothed, targets)
            
            # Update best parameters if we found better results
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def save_params(self, params: SmoothingParams, filepath: str):
        """Save parameters to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def load_params(self, filepath: str) -> SmoothingParams:
        """Load parameters from a JSON file."""
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        return SmoothingParams.from_dict(params_dict)

def plot_predictions(raw_predictions: List[int], smoothed_predictions: List[int], 
                    target_sequence: List[int], train_size: int, filepath: str):
    """
    Create a plot comparing raw predictions, smoothed predictions, and target sequence,
    with train/test split visualization. Each sequence is shown as a horizontal bar
    where filled sections represent 1s and empty sections represent 0s.
    
    Args:
        raw_predictions: List of raw predictions (0 or 1)
        smoothed_predictions: List of smoothed predictions (0 or 1)
        target_sequence: List of target values (0 or 1)
        train_size: Number of frames in training set
        filepath: Path to save the plot
    """
    plt.figure(figsize=(15, 4))
    
    # Create the horizontal bars
    sequences = [
        ('Target Sequence', target_sequence),
        ('Smoothed Predictions', smoothed_predictions),
        ('Raw Predictions', raw_predictions)
    ]
    
    for idx, (label, sequence) in enumerate(sequences):
        # Plot filled rectangles for 1s
        for i in range(len(sequence)):
            if sequence[i]:
                plt.fill_between([i, i+1], idx-0.4, idx+0.4, color='blue', alpha=0.6)
        
        # Plot the outline of the entire bar
        plt.plot([0, len(sequence)], [idx, idx], 'k-', linewidth=0.5)
        plt.fill_between([0, len(sequence)], idx-0.4, idx+0.4, 
                        color='none', edgecolor='black', linewidth=0.5)
        
        # Add label on the left
        plt.text(-0.01 * len(sequence), idx, label, 
                horizontalalignment='right', verticalalignment='center')
    
    # Add train/test split line
    plt.axvline(x=train_size, color='r', linestyle='--', label='Train/Test Split')
    
    # Add train/test annotations
    plt.text(train_size/2, -1, 'Training Set', horizontalalignment='center')
    plt.text((len(raw_predictions) + train_size)/2, -1, 'Test Set', horizontalalignment='center')
    
    # Customize the plot
    plt.ylim(-1.5, len(sequences)-0.5)
    plt.xlabel('Frame')
    plt.title('Comparison of Predictions (filled = active)')
    
    # Remove y-axis ticks and labels since we have custom labels
    plt.yticks([])
    
    # Add grid for x-axis only
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Read CSV files
    predictions_df = pd.read_csv('predictions.csv')
    target_df = pd.read_csv('target.csv')

    # Merge dataframes on 'frame' column and sort by frame
    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    # Convert values to integers (0 and 1)
    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()

    # Create train/test split (80% train, 20% test)
    train_size = int(len(raw_predictions) * 0.8)
    
    train_predictions = raw_predictions[:train_size]
    train_targets = target_sequence[:train_size]
    test_predictions = raw_predictions[train_size:]
    test_targets = target_sequence[train_size:]

    # Create smoother and optimize parameters on training data only
    smoother = PredictionSmoother()
    best_params, train_metrics = smoother.optimize_parameters(
        train_predictions,
        train_targets,
        metric='f1_score'
    )

    # Apply smoothing with best parameters to both train and test sets
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params)
    
    # Calculate metrics for test set
    test_metrics = smoother.calculate_metrics(
        smoothed_predictions[train_size:],
        test_targets
    )

    # Print results
    print("\nBest parameters found (using training data):")
    print(json.dumps(best_params.to_dict(), indent=2))
    print("\nTraining metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Create a new dataframe with smoothed predictions
    result_df = pd.DataFrame({
        'frame': merged_df['frame'],
        'value': [int(pred) for pred in smoothed_predictions]  # Explicitly convert to int
    })

    # Write the result to a new CSV file
    result_df.to_csv('smoothed_predictions.csv', index=False)

    print("\nSmoothed predictions have been written to 'smoothed_predictions.csv'")

    # Create and save the plot
    plot_predictions(raw_predictions, smoothed_predictions, target_sequence, train_size, 'predictions_comparison.png')
    print("\nPredictions comparison plot has been saved to 'predictions_comparison.png'")

    # Example of saving and loading parameters
    smoother.save_params(best_params, "best_smoothing_params.json")
    loaded_params = smoother.load_params("best_smoothing_params.json")
