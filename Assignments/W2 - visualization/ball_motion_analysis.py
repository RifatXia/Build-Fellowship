import pandas as pd
import numpy as np
import os

def calculate_motion_parameters(df):
    """
    Calculate various motion parameters for ball tracking data
    
    Parameters:
    df (DataFrame): DataFrame with columns ['frame', 'xc', 'yc', 'w', 'h', 'effort']
    
    Returns:
    DataFrame: Original DataFrame with added motion parameters
    """
    # Create a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Calculate time assuming constant frame rate (e.g., 30 fps)
    fps = 30
    df_result['time'] = df_result['frame'] / fps
    
    # Calculate velocity components (using central difference method)
    df_result['velocity_x'] = np.gradient(df_result['xc'], df_result['time'])
    df_result['velocity_y'] = np.gradient(df_result['yc'], df_result['time'])
    
    # Calculate speed (magnitude of velocity)
    df_result['speed'] = np.sqrt(df_result['velocity_x']**2 + df_result['velocity_y']**2)
    
    # Calculate acceleration components
    df_result['acceleration_x'] = np.gradient(df_result['velocity_x'], df_result['time'])
    df_result['acceleration_y'] = np.gradient(df_result['velocity_y'], df_result['time'])
    
    # Calculate total acceleration magnitude
    df_result['acceleration'] = np.sqrt(df_result['acceleration_x']**2 + df_result['acceleration_y']**2)
    
    # Calculate direction/angle of motion (in radians)
    df_result['angle'] = np.arctan2(df_result['velocity_y'], df_result['velocity_x'])
    
    # Calculate jerk (rate of change of acceleration)
    df_result['jerk_x'] = np.gradient(df_result['acceleration_x'], df_result['time'])
    df_result['jerk_y'] = np.gradient(df_result['acceleration_y'], df_result['time'])
    df_result['jerk'] = np.sqrt(df_result['jerk_x']**2 + df_result['jerk_y']**2)
    
    # Calculate area of the bounding box
    df_result['area'] = df_result['w'] * df_result['h']
    
    # Calculate change in area over time
    df_result['area_change_rate'] = np.gradient(df_result['area'], df_result['time'])
    
    return df_result

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