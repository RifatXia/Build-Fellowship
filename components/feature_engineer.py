import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, tracking_csv):
        self.tracking_csv = tracking_csv

    def calculate_motion_parameters(self, df):
        """
        Calculate various motion parameters for ball tracking data
        """
        df_result = df.copy()
        fps = 30
        df_result['time'] = df_result['frame'] / fps
        df_result['velocity_x'] = np.gradient(df_result['xc'], df_result['time'])
        df_result['velocity_y'] = np.gradient(df_result['yc'], df_result['time'])
        df_result['speed'] = np.sqrt(df_result['velocity_x']**2 + df_result['velocity_y']**2)
        df_result['acceleration_x'] = np.gradient(df_result['velocity_x'], df_result['time'])
        df_result['acceleration_y'] = np.gradient(df_result['velocity_y'], df_result['time'])
        df_result['acceleration'] = np.sqrt(df_result['acceleration_x']**2 + df_result['acceleration_y']**2)
        df_result['angle'] = np.arctan2(df_result['velocity_y'], df_result['velocity_x'])
        df_result['jerk_x'] = np.gradient(df_result['acceleration_x'], df_result['time'])
        df_result['jerk_y'] = np.gradient(df_result['acceleration_y'], df_result['time'])
        df_result['jerk'] = np.sqrt(df_result['jerk_x']**2 + df_result['jerk_y']**2)
        df_result['area'] = df_result['w'] * df_result['h']
        df_result['area_change_rate'] = np.gradient(df_result['area'], df_result['time'])
        return df_result

    def extract_features(self):
        # Read the data with correct column names
        print(f"Extracting features from {self.tracking_csv}")
        df = pd.read_csv(self.tracking_csv, header=None, 
                         names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
        df['effort'] = pd.to_numeric(df['effort'], errors='coerce')
        df['effort'] = df['effort'].interpolate(method='linear')
        print("\nData summary:")
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nCalculating motion parameters...")
        df_updated = self.calculate_motion_parameters(df)
        original_columns = ['frame', 'xc', 'yc', 'w', 'h', 'effort']
        new_columns = [col for col in df_updated.columns if col not in original_columns]
        print(f"\nNew columns added: {new_columns}")
        df_updated.to_csv('data/updated_tracking.csv', index=False)
        return df_updated 