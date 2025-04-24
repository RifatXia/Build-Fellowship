import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def display_data_summary(df):
    """Display summary information about the dataset"""
    print("First 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nSummary statistics:")
    print(df.describe())

def create_animation(df, output_file='animation.mp4', max_frames=None):
    """
    Create an animation of the ball's trajectory with color indicating motion parameters
    
    Parameters:
    df (DataFrame): DataFrame containing ball motion data
    output_file (str): Output video file name
    max_frames (int, optional): Maximum number of frames to render (for testing)
    """
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (800, 600))

    # Normalize coordinates to fit within the frame
    x_min, x_max = df['xc'].min(), df['xc'].max()
    y_min, y_max = df['yc'].min(), df['yc'].max()
    
    # Create color mapping based on velocity
    velocity_min, velocity_max = df['velocity_magnitude'].min(), df['velocity_magnitude'].max()
    norm = Normalize(vmin=velocity_min, vmax=velocity_max)
    
    # Keep track of past positions for trail effect
    trail_positions = []
    trail_colors = []
    trail_length = 30  # Number of past positions to show
    
    # Process each frame
    for i, (_, row) in enumerate(df.iterrows()):
        # Create blank frame
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Normalize and scale coordinates
        x = int((row['xc'] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row['yc'] - y_min) / (y_max - y_min) * 580 + 10)
        
        # Get color based on velocity (green->yellow->red as velocity increases)
        if pd.notna(row['velocity_magnitude']):
            velocity_ratio = norm(row['velocity_magnitude'])
            # Create color: green (low velocity) to red (high velocity)
            color = (0, int(255 * (1 - velocity_ratio)), int(255 * velocity_ratio))
            trail_positions.append((x, y))
            trail_colors.append(color)
        else:
            color = (0, 255, 0)  # Default green
            trail_positions.append((x, y))
            trail_colors.append(color)
        
        # Draw trail
        if len(trail_positions) > trail_length:
            trail_positions.pop(0)
            trail_colors.pop(0)
        
        for j in range(len(trail_positions) - 1):
            alpha = j / len(trail_positions)  # Fade intensity based on position
            faded_color = tuple(int(c * alpha) for c in trail_colors[j])
            cv2.line(frame, trail_positions[j], trail_positions[j+1], faded_color, 2)
        
        # Draw the current position
        radius = int(max(5, row['area'] * 1000))  # Scale ball size based on area
        cv2.circle(frame, (x, y), radius, color, -1)
        
        # Add frame number and metrics text
        cv2.putText(frame, f"Frame: {int(row['frame'])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if pd.notna(row['velocity_magnitude']):
            cv2.putText(frame, f"Velocity: {row['velocity_magnitude']:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if pd.notna(row['acceleration_magnitude']):
            cv2.putText(frame, f"Accel: {row['acceleration_magnitude']:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if pd.notna(row['kinetic_energy']):
            cv2.putText(frame, f"Energy: {row['kinetic_energy']:.2f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Stop after max_frames if specified
        if max_frames is not None and i >= max_frames:
            break
    
    # Release the video writer
    out.release()
    print(f"Animation saved as '{output_file}'")

def create_analysis_plots(df, output_dir='.'):
    """
    Create analysis plots showing relationships between motion parameters
    
    Parameters:
    df (DataFrame): DataFrame containing ball motion data
    output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Trajectory colored by velocity
    plt.figure(figsize=(10, 8))
    plt.scatter(df['xc'], df['yc'], c=df['velocity_magnitude'], 
                cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Ball Trajectory Colored by Velocity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(os.path.join(output_dir, 'trajectory_plot.png'))
    plt.close()
    
    # Plot 2: Velocity and acceleration over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['velocity_magnitude_smooth'], 'b-', label='Velocity (Smoothed)')
    plt.plot(df['time'], df['acceleration_magnitude_smooth'], 'r-', label='Acceleration (Smoothed)')
    plt.title('Velocity and Acceleration Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'velocity_acceleration_plot.png'))
    plt.close()
    
    # Plot 3: Aspect ratio and area over time
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['area'], 'g-')
    plt.title('Area Over Time')
    plt.ylabel('Area')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['aspect_ratio'], 'm-')
    plt.title('Aspect Ratio Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Aspect Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'area_aspect_ratio_plot.png'))
    plt.close()
    
    # Plot 4: Kinetic energy over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['kinetic_energy'], 'k-')
    plt.title('Kinetic Energy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Kinetic Energy')

    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'kinetic_energy_plot.png'))
    plt.close()
    
    print(f"Analysis plots saved to {output_dir}")

def main():
    # Read the updated data with motion parameters
    input_file = 'updated_provided_data.csv'
    print(f"Reading data from {input_file}...")
    
    df = pd.read_csv(input_file)
    
    # Display summary information
    display_data_summary(df)
    
    # Create visualizations
    print("\nCreating animation...")
    create_animation(df, max_frames=1000)  # Limit to 1000 frames for testing
    
    print("\nGenerating analysis plots...")
    create_analysis_plots(df, output_dir='analysis_plots')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()     main() 
