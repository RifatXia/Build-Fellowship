import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Callable
import json
import matplotlib.pyplot as plt
import argparse

@dataclass
class SmoothingParams:
    window_size: int               # Size of the sliding window for initial smoothing
    min_duration: int              # Minimum duration of an activation state to keep
    hysteresis: float              # Hysteresis threshold for state changes
    median_window: int = 5         # Window size for median filtering
    adaptive_threshold: float = 0  # Threshold for adaptive filtering (0 to disable)
    decay_factor: float = 0.95     # Decay factor for adaptive threshold
    neighbor_weight: float = 0.3   # Weight for neighboring frame influence
    filter_chain: List[str] = None # Chain of filters to apply in sequence
    
    def __post_init__(self):
        # Default filter chain if none provided
        if self.filter_chain is None:
            self.filter_chain = ['median', 'majority', 'hysteresis', 'duration']
    
    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'min_duration': self.min_duration,
            'hysteresis': self.hysteresis,
            'median_window': self.median_window,
            'adaptive_threshold': self.adaptive_threshold,
            'decay_factor': self.decay_factor,
            'neighbor_weight': self.neighbor_weight,
            'filter_chain': self.filter_chain
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SmoothingParams':
        return cls(
            window_size=d['window_size'],
            min_duration=d['min_duration'],
            hysteresis=d['hysteresis'],
            median_window=d.get('median_window', 5),
            adaptive_threshold=d.get('adaptive_threshold', 0),
            decay_factor=d.get('decay_factor', 0.95),
            neighbor_weight=d.get('neighbor_weight', 0.3),
            filter_chain=d.get('filter_chain', ['median', 'majority', 'hysteresis', 'duration'])
        )

class PredictionSmoother:
    """
    An enhanced class that smooths frame-by-frame predictions with multiple filtering stages
    and optimizes parameters for various F-beta scores.
    """
    
    def __init__(self):
        # Default parameter search spaces
        self.default_param_grid = {
            'window_size': [3, 5, 7, 9, 11, 15],
            'min_duration': [1, 2, 3, 5, 10, 15, 30, 60],
            'hysteresis': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            'median_window': [3, 5, 7, 9],
            'adaptive_threshold': [0, 0.05, 0.1, 0.15, 0.2],
            'decay_factor': [0.9, 0.95, 0.98],
            'neighbor_weight': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'filter_chain': [
                ['median', 'majority', 'hysteresis', 'duration'],
                ['median', 'adaptive', 'duration'],
                ['majority', 'hysteresis', 'neighbor', 'duration'],
                ['adaptive', 'median', 'duration'],
                ['neighbor', 'median', 'hysteresis', 'duration']
            ]
        }
        
        # Dictionary of filter functions for the filter chain
        self.filter_functions = {
            'median': self._apply_median_filter,
            'majority': self._apply_majority_voting,
            'hysteresis': self._apply_hysteresis,
            'duration': self._apply_duration_filter,
            'adaptive': self._apply_adaptive_threshold,
            'neighbor': self._apply_neighbor_influence
        }
    
    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Apply a chain of filters to smooth predictions based on specified filter chain.
        """
        if not predictions:
            return []
        
        # Convert input predictions to integers, just in case
        current_preds = [int(pred) for pred in predictions]
        
        # Apply each filter in the chain
        for filter_name in params.filter_chain:
            if filter_name in self.filter_functions:
                current_preds = self.filter_functions[filter_name](current_preds, params)
            else:
                print(f"Warning: Unknown filter '{filter_name}' specified in filter chain")
        
        return [int(pred) for pred in current_preds]
    
    def _apply_median_filter(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Apply median filtering to reduce noise in predictions.
        """
        window_size = params.median_window
        half_window = window_size // 2
        smoothed = []
        
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            # Median filtering - take the median value in the window
            smoothed.append(int(np.median(window)))
            
        return smoothed
    
    def _apply_majority_voting(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Apply sliding window majority voting filter.
        """
        # Ensure window_size is odd
        window_size = max(3, params.window_size if params.window_size % 2 == 1 
                         else params.window_size + 1)
        half_window = window_size // 2
        
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            
            # Simple majority voting - if more than half are 1s, output 1
            active_ratio = sum(window) / len(window)
            smoothed.append(1 if active_ratio >= 0.5 else 0)
            
        return smoothed
    
    def _apply_hysteresis(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Apply hysteresis to filter, making it harder to switch states.
        """
        smoothed = []
        
        for i, pred in enumerate(predictions):
            if i == 0:
                smoothed.append(pred)
                continue
                
            # Apply hysteresis threshold based on previous state
            threshold = params.hysteresis if smoothed[-1] else 1 - params.hysteresis
            
            # Calculate ratio of 1s in a window centered at current position
            window_size = params.window_size
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            active_ratio = sum(window) / len(window)
            
            # Apply threshold with hysteresis
            smoothed.append(1 if active_ratio >= threshold else 0)
            
        return smoothed
    
    def _apply_duration_filter(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Filter out state changes that are shorter than the minimum duration.
        """
        if params.min_duration <= 1:
            return predictions
            
        final = []
        current_state = predictions[0]
        current_duration = 1
        
        for pred in predictions[1:]:
            if pred == current_state:
                current_duration += 1
            else:
                if current_duration >= params.min_duration:
                    final.extend([current_state] * current_duration)
                else:
                    # If duration is too short, flip the state to maintain continuity
                    final.extend([1 - current_state] * current_duration)
                current_state = pred
                current_duration = 1
        
        # Handle the last sequence
        if current_duration >= params.min_duration:
            final.extend([current_state] * current_duration)
        else:
            final.extend([1 - current_state] * current_duration)
            
        return final
    
    def _apply_adaptive_threshold(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Apply an adaptive threshold that changes based on recent activity level.
        This is especially useful for handling sequences with varying activity patterns.
        """
        if params.adaptive_threshold <= 0:
            return predictions
            
        window_size = params.window_size
        half_window = window_size // 2
        smoothed = []
        
        # Start with base threshold
        threshold = 0.5
        
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            
            active_ratio = sum(window) / len(window)
            
            # Adjust threshold based on recent activity - makes it harder to activate
            # when there's been a lot of recent activity, and easier when there hasn't
            if i > 0 and smoothed[-1] == 1:
                # If previous was active, increase threshold (harder to stay active)
                threshold = min(0.8, threshold + params.adaptive_threshold)
            else:
                # If previous was inactive, decrease threshold (easier to become active)
                threshold = max(0.2, threshold * params.decay_factor)
                
            smoothed.append(1 if active_ratio >= threshold else 0)
            
        return smoothed
    
    def _apply_neighbor_influence(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Consider the influence of neighboring frames beyond simple window averaging.
        This gives more weight to frames that are part of consistent sequences.
        """
        if params.neighbor_weight <= 0:
            return predictions
            
        smoothed = []
        window_size = params.window_size
        half_window = window_size // 2
        
        for i in range(len(predictions)):
            # Base prediction
            base_pred = predictions[i]
            
            # Look at trend of neighbors
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            
            # Calculate weighted influence based on distance from current frame
            neighbor_sum = 0
            weight_sum = 0
            
            for j in range(start, end):
                if j == i:
                    continue
                    
                distance = abs(j - i)
                weight = 1 / (distance + 1)  # Weight decreases with distance
                neighbor_sum += predictions[j] * weight
                weight_sum += weight
            
            neighbor_influence = neighbor_sum / weight_sum if weight_sum > 0 else 0
            
            # Combine current prediction with neighbor influence
            combined = (1 - params.neighbor_weight) * base_pred + params.neighbor_weight * neighbor_influence
            smoothed.append(1 if combined >= 0.5 else 0)
            
        return smoothed
    
    def calculate_metrics(self, predictions: List[int], targets: List[int], beta: float = 1.0) -> Dict[str, float]:
        """
        Calculate various metrics comparing predictions to targets, including F-beta score.
        
        Args:
            predictions: List of predicted values (0 or 1)
            targets: List of target values (0 or 1)
            beta: Beta parameter for F-beta score (beta > 1 favors recall, beta < 1 favors precision)
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
        
        # F-beta score components
        true_pos = np.sum((pred_arr == 1) & (target_arr == 1))
        false_pos = np.sum((pred_arr == 1) & (target_arr == 0))
        false_neg = np.sum((pred_arr == 0) & (target_arr == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        # Calculate F-beta score: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        beta_squared = beta ** 2
        f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate traditional F1 score for compatibility
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'f_beta_score': float(f_beta),
            'precision': float(precision),
            'recall': float(recall),
            'state_change_difference': float(abs(pred_changes - target_changes)),
            'beta': float(beta)
        }
    
    def optimize_parameters(self, 
                          predictions: List[bool], 
                          targets: List[bool],
                          param_grid: Optional[Dict] = None,
                          metric: str = 'f_beta_score',
                          beta: float = 1.0,
                          max_combinations: int = 100) -> Tuple[SmoothingParams, Dict[str, float]]:
        """
        Find optimal parameters by grid search to match target sequence.
        
        Args:
            predictions: Original predictions to smooth
            targets: Target sequence to match
            param_grid: Dictionary of parameter ranges to search
            metric: Metric to optimize ('accuracy', 'f1_score', 'f_beta_score', etc.)
            beta: Beta value for F-beta score optimization
            max_combinations: Maximum number of parameter combinations to try
            
        Returns:
            Tuple of (best parameters, best metrics)
        """
        if param_grid is None:
            param_grid = self.default_param_grid
            
        # For better searching, we can use a staged approach:
        # 1. First try different filter chains with default parameters
        # 2. Then optimize specific parameters for the best filter chain
        
        # Simplified parameter combinations for stage 1
        filter_chains = param_grid.get('filter_chain', [['median', 'majority', 'hysteresis', 'duration']])
        best_filter_chain = filter_chains[0]
        
        # Initialize with default parameters using the best filter chain
        best_params = SmoothingParams(
            window_size=5, 
            min_duration=5, 
            hysteresis=0.6,
            median_window=5,
            filter_chain=best_filter_chain
        )
        
        # Calculate initial best metrics
        smoothed = self.smooth_predictions(predictions, best_params)
        best_metrics = self.calculate_metrics(smoothed, targets, beta=beta)
        best_score = best_metrics[metric]
        
        print("Stage 1: Finding best filter chain...")
        for filter_chain in filter_chains:
            # Use a default set of parameters for comparing filter chains
            params = SmoothingParams(
                window_size=5, 
                min_duration=5, 
                hysteresis=0.6,
                median_window=5,
                filter_chain=filter_chain
            )
            
            smoothed = self.smooth_predictions(predictions, params)
            metrics = self.calculate_metrics(smoothed, targets, beta=beta)
            
            score = metrics[metric]
            print(f"  Filter chain {filter_chain}: {metric} = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
                best_filter_chain = filter_chain
        
        print(f"Best filter chain: {best_filter_chain}")
        
        # Now optimize parameters using the best filter chain
        print("Stage 2: Optimizing parameters for best filter chain...")
        
        # Remove filter_chain from param_grid for stage 2
        stage2_param_grid = {k: v for k, v in param_grid.items() if k != 'filter_chain'}
        
        # Generate parameter combinations to test
        param_keys = list(stage2_param_grid.keys())
        param_values = [stage2_param_grid[k] for k in param_keys]
        
        # Calculate total possible combinations
        total_possible = np.prod([len(v) for v in param_values])
        
        # Limit to max_combinations
        total_combinations = min(total_possible, max_combinations)
        print(f"Testing {total_combinations} parameter combinations (out of {total_possible} possible)...")
        
        # Initialize counter
        counter = 0
        
        # For limiting combinations, we can either:
        # 1. Randomly sample from all possible combinations, or
        # 2. Take a subset of parameter values
        # Here we'll use approach #2 for better coverage with fewer combinations
        
        # Reduce parameter space for each parameter
        reduced_param_grid = {}
        for key, values in stage2_param_grid.items():
            if len(values) > 3:
                # Take a subset: first, middle, last
                step = max(1, (len(values) - 1) // 2)
                reduced_values = [values[0], values[step], values[-1]]
            else:
                reduced_values = values
            reduced_param_grid[key] = reduced_values
        
        # Get reduced parameter values
        reduced_param_keys = list(reduced_param_grid.keys())
        reduced_param_values = [reduced_param_grid[k] for k in reduced_param_keys]
        
        # Generate combinations and test
        for values in product(*reduced_param_values):
            if counter >= total_combinations:
                break
                
            param_dict = {reduced_param_keys[i]: values[i] for i in range(len(reduced_param_keys))}
            param_dict['filter_chain'] = best_filter_chain
            
            params = SmoothingParams(**param_dict)
            smoothed = self.smooth_predictions(predictions, params)
            metrics = self.calculate_metrics(smoothed, targets, beta=beta)
            
            score = metrics[metric]
            
            # Update progress
            counter += 1
            if counter % 10 == 0:
                print(f"  Processed {counter}/{total_combinations} combinations. Current best: {best_score:.4f}")
            
            # Update best parameters if we found better results
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

def fbeta_score(y_true, y_pred, beta: float = 1.0):
    """
    Compute Fβ score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter - β > 1 prioritizes recall, β < 1 prioritizes precision
        
    Returns:
        Fβ score
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate precision and recall
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    # Calculate Fβ score
    if precision == 0 and recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

def plot_fbeta_scores(raw_predictions: List[int], targets: List[int], 
                     smoother: PredictionSmoother, params: SmoothingParams,
                     betas: List[float], filepath: str):
    """
    Plot Fβ scores for various β values to help choose the optimal β for the project.
    
    Args:
        raw_predictions: Raw model predictions
        targets: Target values
        smoother: Prediction smoother instance
        params: Current best smoothing parameters
        betas: List of beta values to evaluate
        filepath: Path to save the plot
    """
    scores = []
    
    # Apply smoothing once using current best parameters
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, params)
    
    # Calculate Fβ scores for each beta value
    for beta in betas:
        score = fbeta_score(targets, smoothed_predictions, beta)
        scores.append(score)
        print(f"F{beta} score: {score:.4f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(betas, scores, 'o-', linewidth=2, markersize=8)
    
    # Add annotations for each data point
    for i, beta in enumerate(betas):
        plt.annotate(f"β={beta}\nF{beta}={scores[i]:.4f}", 
                    (beta, scores[i]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Customize plot
    plt.xlabel('β value')
    plt.ylabel('Fβ score')
    plt.title('Fβ scores for different β values')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add precision-recall trade-off explanation
    plt.text(0.02, 0.02, 
            "β < 1: Prioritize precision\nβ > 1: Prioritize recall", 
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return scores

def generate_highlight_reel(predictions: List[int], frame_indices: List[int], output_path: str, video_path: str = None):
    """
    Generate a highlight reel video from the positive predictions.
    
    Args:
        predictions: Filtered predictions (0 or 1)
        frame_indices: List of frame indices corresponding to predictions
        output_path: Path to save the highlight reel video
        video_path: Path to the source video file (default: looks for 'video.mp4')
    """
    try:
        import cv2
        print("OpenCV imported successfully for video processing")
    except ImportError:
        print("OpenCV (cv2) is required for video processing but not installed.")
        print("Please install it with: pip install opencv-python")
        return

    if video_path is None:
        # Look for video file in current directory
        import os
        video_files = [f for f in os.listdir('.') if f.endswith('.mp4') and f != output_path]
        if not video_files:
            print("No source video files found. Please provide a video_path.")
            return
        video_path = video_files[0]
        print(f"Using {video_path} as source video")
    
    print("Generating highlight reel...")
    
    # Find continuous segments of positive predictions
    segments = []
    current_segment = []
    
    for idx, pred in enumerate(predictions):
        if pred == 1:
            current_segment.append(frame_indices[idx])
        elif current_segment:
            # End of a segment
            segments.append((min(current_segment), max(current_segment)))
            current_segment = []
    
    # Don't forget the last segment if it ends with a positive prediction
    if current_segment:
        segments.append((min(current_segment), max(current_segment)))
    
    if not segments:
        print("No positive segments found in predictions")
        return
    
    print(f"Found {len(segments)} positive segments")
    
    # Open the source video
    source_video = cv2.VideoCapture(video_path)
    if not source_video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source_video.get(cv2.CAP_PROP_FPS)
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not output_video.isOpened():
        print(f"Error: Could not create output video {output_path}")
        source_video.release()
        return
    
    # Process each segment
    total_frames_processed = 0
    highlight_frames = 0
    
    for segment_start, segment_end in segments:
        # Set position to segment start frame
        source_video.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
        
        # Read and write frames for this segment
        segment_length = segment_end - segment_start + 1
        for _ in range(segment_length):
            ret, frame = source_video.read()
            if not ret:
                print(f"Warning: Could not read frame in segment {segment_start}-{segment_end}")
                break
            
            # Add text overlay showing this is a highlight
            # cv2.putText(frame, "HIGHLIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            #            1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Write frame to output video
            output_video.write(frame)
            highlight_frames += 1
        
        total_frames_processed += segment_length
    
    # Release video objects
    source_video.release()
    output_video.release()
    
    print(f"Highlight reel generated with {highlight_frames} frames across {len(segments)} segments")
    print(f"Saved to {output_path}")
    
    return segments

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter predictions and generate highlight reel')
    parser.add_argument('--video', type=str, help='Path to source video file')
    args = parser.parse_args()

    # Read CSV files
    predictions_df = pd.read_csv('predictions.csv')
    target_df = pd.read_csv('target.csv')

    # Merge dataframes on 'frame' column and sort by frame
    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    # Convert values to integers (0 and 1)
    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()
    frame_indices = merged_df['frame'].tolist()

    # Create train/validation/test split (60% train, 20% validation, 20% test)
    train_size = int(len(raw_predictions) * 0.6)
    val_size = int(len(raw_predictions) * 0.2)
    
    train_predictions = raw_predictions[:train_size]
    train_targets = target_sequence[:train_size]
    
    val_predictions = raw_predictions[train_size:train_size+val_size]
    val_targets = target_sequence[train_size:train_size+val_size]
    
    test_predictions = raw_predictions[train_size+val_size:]
    test_targets = target_sequence[train_size+val_size:]
    test_frames = frame_indices[train_size+val_size:]

    print(f"Data split: {len(train_predictions)} training, {len(val_predictions)} validation, {len(test_predictions)} test samples")

    # Create smoother instance
    smoother = PredictionSmoother()
    
    # First, optimize parameters using F1 score (β=1) on training data
    print("\n=== Phase 1: Parameter Optimization with F1 Score ===")
    initial_params, train_metrics = smoother.optimize_parameters(
        train_predictions,
        train_targets,
        metric='f_beta_score',
        beta=1.0,
        max_combinations=100
    )
    
    # Plot Fβ scores for different β values on validation set
    print("\n=== Phase 2: Evaluating Different β Values on Validation Set ===")
    beta_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    fbeta_scores = plot_fbeta_scores(
        val_predictions, 
        val_targets,
        smoother,
        initial_params,
        beta_values,
        'fbeta_comparison.png'
    )
    
    # Select the β value that best aligns with our project goals
    # For highlight-reel discovery, we prioritize recall (β > 1)
    # Here we select β = 2.0 as it balances recall priority while maintaining decent precision
    selected_beta = 2.0
    print(f"\nSelected β = {selected_beta} for final optimization (prioritizing recall for highlight detection)")
    
    # Re-optimize parameters using the selected β value
    print("\n=== Phase 3: Re-optimizing Parameters with Selected β ===")
    best_params, final_train_metrics = smoother.optimize_parameters(
        train_predictions,
        train_targets,
        metric='f_beta_score',
        beta=selected_beta,
        max_combinations=100
    )
    
    # Apply smoothing with best parameters to the entire dataset for visualization
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params)
    
    # Calculate metrics for validation and test sets
    val_metrics = smoother.calculate_metrics(
        smoothed_predictions[train_size:train_size+val_size],
        val_targets,
        beta=selected_beta
    )
    
    test_metrics = smoother.calculate_metrics(
        smoothed_predictions[train_size+val_size:],
        test_targets,
        beta=selected_beta
    )

    # Print results
    print("\nBest parameters found (optimized for β =", selected_beta, "):")
    print(json.dumps(best_params.to_dict(), indent=2))
    print("\nTraining metrics:")
    print(json.dumps(final_train_metrics, indent=2))
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Plot predictions for visual comparison
    plot_predictions(
        raw_predictions, 
        smoothed_predictions, 
        target_sequence, 
        train_size + val_size,  # Show train+val as one section
        'predictions_comparison.png'
    )
    print("\nPredictions comparison plot has been saved to 'predictions_comparison.png'")

    # Create a new dataframe with smoothed predictions
    result_df = pd.DataFrame({
        'frame': merged_df['frame'],
        'value': [int(pred) for pred in smoothed_predictions]  # Explicitly convert to int
    })

    # Write the result to a new CSV file
    result_df.to_csv('smoothed_predictions.csv', index=False)
    print("\nSmoothed predictions have been written to 'smoothed_predictions.csv'")

    # Save parameters for future use
    smoother.save_params(best_params, "best_smoothing_params.json")
    
    # Generate highlight reel from test set predictions
    print("\n=== Phase 4: Generating Highlight Reel ===")
    test_smoothed_predictions = smoothed_predictions[train_size+val_size:]
    segments = generate_highlight_reel(
        test_smoothed_predictions,
        test_frames,
        'highlight_test.mp4',
        video_path=args.video
    )
    
    # Explain choice of β
    print("\n=== Explanation of β Choice ===")
    print(f"Selected β = {selected_beta} because:")
    print("1. For highlight-reel discovery, we prioritize recall over precision")
    print("   (Better to include a few false positives than miss important highlights)")
    print("2. This β value provides a good balance between precision and recall")
    print("   while giving recall more weight in the F-score calculation")
    print("3. Higher β values (like 4.0) might include too many false positives,")
    print("   while lower values might miss important action moments")
