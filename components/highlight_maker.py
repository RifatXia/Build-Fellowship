import cv2
import os
import numpy as np
from typing import List, Tuple, Optional

class HighlightMaker:
    def __init__(self):
        self.max_segments = 30  # Use 30 segments for highlight reel

    def create_highlights(self, predictions, video_path, output_path):
        """
        Creates highlight video from predictions.
        
        Args:
            predictions: DataFrame with 'frame' and 'value' columns
            video_path: Path to source video file
            output_path: Path to save highlight video
        """
        print(f"\nCreating highlights from {video_path} into {output_path}...")
        
        # Find continuous segments of positive predictions
        segments = []
        current_segment = []
        
        for idx, row in predictions.iterrows():
            if row['value'] == 1:
                current_segment.append(row['frame'])
            elif current_segment:
                segments.append((min(current_segment), max(current_segment)))
                current_segment = []
        
        # Don't forget the last segment if it ends with a positive prediction
        if current_segment:
            segments.append((min(current_segment), max(current_segment)))
        
        if not segments:
            print("No positive segments found in predictions")
            return
        
        # Select and prioritize segments
        if len(segments) > self.max_segments:
            print(f"Selecting {self.max_segments} best segments from {len(segments)} total segments")
            
            # Calculate segment duration and spacing
            durations = [(end - start + 1) for start, end in segments]
            
            # Calculate average distance between segments (to ensure spread)
            starts = [start for start, _ in segments]
            
            # Score segments based on multiple factors
            segment_scores = []
            for i, (start, end) in enumerate(segments):
                duration = end - start + 1
                
                # Score longer segments higher
                duration_score = min(1.0, duration / 200)  # Cap at 1.0 for segments of 200+ frames
                
                # Add to scores
                segment_scores.append((i, duration_score))
            
            # Sort by score (highest first)
            segment_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top segments
            selected_indices = [score[0] for score in segment_scores[:self.max_segments]]
            selected_segments = [segments[i] for i in selected_indices]
            
            # Re-sort by frame number for chronological playback
            selected_segments.sort(key=lambda x: x[0])
            segments = selected_segments
        
        print(f"Using {len(segments)} segments for highlights")
        
        # Open the source video
        source_video = cv2.VideoCapture(video_path)
        if not source_video.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source_video.get(cv2.CAP_PROP_FPS)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
            segment_length = int(segment_end - segment_start + 1)
            for _ in range(segment_length):
                ret, frame = source_video.read()
                if not ret:
                    print(f"Warning: Could not read frame in segment {segment_start}-{segment_end}")
                    break
                
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