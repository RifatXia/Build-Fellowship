import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class SmoothingParams:
    window_size: int = 5               # Size of the sliding window for initial smoothing
    min_duration: int = 5              # Minimum duration of an activation state to keep
    hysteresis: float = 0.6            # Hysteresis threshold for state changes
    median_window: int = 5             # Window size for median filtering
    adaptive_threshold: float = 0      # Threshold for adaptive filtering (0 to disable)
    decay_factor: float = 0.95         # Decay factor for adaptive threshold
    neighbor_weight: float = 0.3       # Weight for neighboring frame influence
    filter_chain: List[str] = None     # Chain of filters to apply in sequence
    
    def __post_init__(self):
        if self.filter_chain is None:
            self.filter_chain = ['median', 'majority', 'hysteresis', 'duration']

class OutputFilter:
    def __init__(self):
        self.filter_functions = {
            'median': self._apply_median_filter,
            'majority': self._apply_majority_voting,
            'hysteresis': self._apply_hysteresis,
            'duration': self._apply_duration_filter,
            'adaptive': self._apply_adaptive_threshold,
            'neighbor': self._apply_neighbor_influence
        }
        # Default parameters optimized for highlight detection
        self.params = SmoothingParams()

    def filter(self, predictions):
        """
        Accepts predictions and returns filtered/highlighted predictions.
        """
        print("Filtering predictions...")
        if not isinstance(predictions, pd.DataFrame):
            raise ValueError("Predictions must be a pandas DataFrame with 'frame' and 'value' columns")
        
        # Extract values and ensure they are integers
        pred_values = predictions['value'].astype(int).tolist()
        
        # Apply smoothing
        smoothed = self.smooth_predictions(pred_values, self.params)
        
        # Create new DataFrame with smoothed predictions
        filtered = pd.DataFrame({
            'frame': predictions['frame'],
            'value': smoothed
        })
        
        return filtered

    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Apply a chain of filters to smooth predictions"""
        if not predictions:
            return []
        
        current_preds = [int(pred) for pred in predictions]
        
        for filter_name in params.filter_chain:
            if filter_name in self.filter_functions:
                current_preds = self.filter_functions[filter_name](current_preds, params)
            else:
                print(f"Warning: Unknown filter '{filter_name}' specified in filter chain")
        
        return [int(pred) for pred in current_preds]

    def _apply_median_filter(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Apply median filtering to reduce noise"""
        window_size = params.median_window
        half_window = window_size // 2
        smoothed = []
        
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            smoothed.append(int(np.median(window)))
            
        return smoothed

    def _apply_majority_voting(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Apply sliding window majority voting filter"""
        window_size = max(3, params.window_size if params.window_size % 2 == 1 else params.window_size + 1)
        half_window = window_size // 2
        
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            active_ratio = sum(window) / len(window)
            smoothed.append(1 if active_ratio >= 0.5 else 0)
            
        return smoothed

    def _apply_hysteresis(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Apply hysteresis to filter, making it harder to switch states"""
        smoothed = []
        
        for i, pred in enumerate(predictions):
            if i == 0:
                smoothed.append(pred)
                continue
                
            threshold = params.hysteresis if smoothed[-1] else 1 - params.hysteresis
            
            window_size = params.window_size
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            active_ratio = sum(window) / len(window)
            
            smoothed.append(1 if active_ratio >= threshold else 0)
            
        return smoothed

    def _apply_duration_filter(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Filter out state changes that are shorter than the minimum duration"""
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
                    final.extend([1 - current_state] * current_duration)
                current_state = pred
                current_duration = 1
        
        if current_duration >= params.min_duration:
            final.extend([current_state] * current_duration)
        else:
            final.extend([1 - current_state] * current_duration)
            
        return final

    def _apply_adaptive_threshold(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Apply an adaptive threshold that changes based on recent activity level"""
        if params.adaptive_threshold <= 0:
            return predictions
            
        window_size = params.window_size
        half_window = window_size // 2
        smoothed = []
        threshold = 0.5
        
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            
            active_ratio = sum(window) / len(window)
            
            if i > 0 and smoothed[-1] == 1:
                threshold = min(0.8, threshold + params.adaptive_threshold)
            else:
                threshold = max(0.2, threshold * params.decay_factor)
                
            smoothed.append(1 if active_ratio >= threshold else 0)
            
        return smoothed

    def _apply_neighbor_influence(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """Consider the influence of neighboring frames beyond simple window averaging"""
        if params.neighbor_weight <= 0:
            return predictions
            
        smoothed = []
        window_size = params.window_size
        half_window = window_size // 2
        
        for i in range(len(predictions)):
            base_pred = predictions[i]
            
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            
            neighbor_sum = 0
            weight_sum = 0
            
            for j in range(start, end):
                if j == i:
                    continue
                    
                distance = abs(j - i)
                weight = 1 / (distance + 1)
                neighbor_sum += predictions[j] * weight
                weight_sum += weight
            
            neighbor_influence = neighbor_sum / weight_sum if weight_sum > 0 else 0
            
            combined = (1 - params.neighbor_weight) * base_pred + params.neighbor_weight * neighbor_influence
            smoothed.append(1 if combined >= 0.5 else 0)
            
        return smoothed 