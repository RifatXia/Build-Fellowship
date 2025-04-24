import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

def calculate_motion_parameters(df):
    """
    Calculate various motion parameters for ball tracking data
    
    Parameters:
    df (DataFrame): DataFrame with columns ['frame', 'xc', 'yc', 'w', 'h', 'effort']
    
    Returns:
    DataFrame: Original DataFrame with added motion parameters (6-7 key properties)
    """
    # Create a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Calculate time and time delta
    fps = 30
    df_result['time'] = df_result['frame'] / fps
    df_result['time_delta'] = 1/fps  # Constant time step
    
    # Calculate displacement
    df_result['displacement_x'] = df_result['xc'].diff()
    df_result['displacement_y'] = df_result['yc'].diff()
    df_result['displacement_total'] = np.sqrt(df_result['displacement_x']**2 + df_result['displacement_y']**2)
    
    # Calculate velocity
    df_result['velocity_x'] = df_result['displacement_x'] / df_result['time_delta']
    df_result['velocity_y'] = df_result['displacement_y'] / df_result['time_delta']
    df_result['velocity_magnitude'] = np.sqrt(df_result['velocity_x']**2 + df_result['velocity_y']**2)
    
    # Calculate acceleration
    df_result['acceleration_x'] = df_result['velocity_x'].diff() / df_result['time_delta']
    df_result['acceleration_y'] = df_result['velocity_y'].diff() / df_result['time_delta']
    df_result['acceleration_magnitude'] = np.sqrt(df_result['acceleration_x']**2 + df_result['acceleration_y']**2)
    
    # Calculate rotational motion
    df_result['angle'] = np.arctan2(df_result['yc'].diff(), df_result['xc'].diff())
    df_result['angular_velocity'] = df_result['angle'].diff() / df_result['time_delta']
    
    # Calculate energy
    df_result['kinetic_energy'] = 0.5 * df_result['velocity_magnitude']**2
    
    # Calculate area and aspect ratio
    df_result['area'] = df_result['w'] * df_result['h']
    df_result['aspect_ratio'] = df_result['w'] / df_result['h']
    
    # Apply Savitzky-Golay filter to smooth key metrics
    # First handle NaN values by forward-filling
    df_result = df_result.fillna(method='ffill')
    
    # Ensure window_length is odd and smaller than data length
    window_length = min(11, len(df_result) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    # Only apply filter if we have enough data
    if window_length > 3 and len(df_result) > window_length:
        # Apply smoothing to key metrics
        df_result['velocity_magnitude_smooth'] = savgol_filter(
            df_result['velocity_magnitude'].fillna(0), 
            window_length=window_length, 
            polyorder=min(3, window_length-1)
        )
        
        df_result['acceleration_magnitude_smooth'] = savgol_filter(
            df_result['acceleration_magnitude'].fillna(0), 
            window_length=window_length, 
            polyorder=min(3, window_length-1)
        )
    
    # Select only the key metrics to keep (6-7 most informative ones)
    columns_to_keep = [
        'frame', 'xc', 'yc', 'w', 'h', 'effort', 'time',
        'velocity_magnitude', 'acceleration_magnitude', 'angle',
        'angular_velocity', 'kinetic_energy', 'area', 'aspect_ratio'
    ]
    
    # Add smoothed columns if they exist
    if 'velocity_magnitude_smooth' in df_result.columns:
        columns_to_keep.append('velocity_magnitude_smooth')
    if 'acceleration_magnitude_smooth' in df_result.columns:
        columns_to_keep.append('acceleration_magnitude_smooth')
    
    # Return only selected columns
    return df_result[columns_to_keep]

def main():
    # Set paths
    input_file = 'provided_data.csv'
    output_file = 'updated_provided_data.csv'
    
    # Read the data with correct column names
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, header=None, 
                    names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
    
    # Convert 'effort' column to numeric; non-numeric entries will be set to NaN
    df['effort'] = pd.to_numeric(df['effort'], errors='coerce')
    
    # Impute missing 'effort' values using linear interpolation
    df['effort'] = df['effort'].interpolate(method='linear')
    
    # Display information about the data
    print("\nData summary:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Calculate motion parameters
    print("\nCalculating motion parameters...")
    df_updated = calculate_motion_parameters(df)
    
    # Print new columns
    original_columns = ['frame', 'xc', 'yc', 'w', 'h', 'effort']
    new_columns = [col for col in df_updated.columns if col not in original_columns]
    print(f"\nNew columns added: {new_columns}")
    
    # Save to CSV
    print(f"\nSaving updated data to {output_file}...")
    df_updated.to_csv(output_file, index=False)
    print("Done!")
    
    # Display summary of new data
    if new_columns:
        print("\nSummary of updated data (new columns):")
        print(df_updated[new_columns].describe())
    else:
        print("\nNo new columns were added.")

if __name__ == "__main__":
    main() 