# Sports Highlight Generation System

![Banner](Assignments/W8%20-%20final%20submission/output/plots/highlight_banner.png)

## Project Overview

This project implements an automated sports highlight detection and extraction system. It leverages machine learning techniques to analyze sports video footage, identify key moments, and automatically generate highlight clips for easier sports content creation and consumption.

## Key Features

- **Automated Ball Tracking**: Computer vision-based ball tracking in sports videos
- **Feature Engineering**: Extracting motion features from ball trajectory data
- **Highlight Classification**: Using multiple ML models to identify exciting moments
- **Video Processing**: Cutting and assembling highlight clips
- **Modular Pipeline**: Well-defined components for easy extension and modification

## Project Structure

```
project/
├── main.py              # Main entry point for the pipeline
├── components/          # Core system components
│   ├── feature_engineer.py    # Motion feature extraction
│   ├── classifier.py          # ML model training and prediction
│   ├── output_filter.py       # Post-processing of predictions
│   ├── highlight_maker.py     # Video clip extraction
├── data/                # Input data directory
├── output/              # Output directory for results
│   └── plots/           # Visualization and performance plots
```

## Installation & Setup

1. Create a virtual environment:
   ```bash
   python -m venv mleng_env
   source mleng_env/bin/activate  # On Windows: mleng_env\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline with the following command:

```bash
python main.py <tracking_csv> <target_csv> <video_raw> <output_dir>
```

Arguments:
- `tracking_csv`: Path to the ball tracking data CSV file
- `target_csv`: Path to the ground truth highlight labels CSV file
- `video_raw`: Path to the raw video file
- `output_dir`: Directory to save outputs and results

Example:
```bash
python main.py data/tracking.csv data/target.csv data/match.mp4 output/
```

## Pipeline Workflow

### 1. Feature Engineering

The system processes raw ball tracking data to calculate advanced motion parameters:
- Velocity and acceleration profiles
- Jerk (rate of change of acceleration)
- Ball trajectory angle and area changes
- Time-based lag features for temporal patterns

![Feature Engineering](Assignments/W8%20-%20final%20submission/output/plots/features_correlation.png)

### 2. Classification

Multiple machine learning models are trained and evaluated:
- Random Forest
- XGBoost
- Logistic Regression
- Support Vector Machines
- Decision Trees

The system automatically selects the best performing model based on accuracy metrics.

![Model Comparison](Assignments/W8%20-%20final%20submission/output/plots/model_comparison.png)

### 3. Post-processing

The raw prediction outputs are filtered to remove noise and ensure smooth highlight segments.

### 4. Highlight Generation

The system extracts video clips based on the filtered predictions and assembles them into a cohesive highlight reel.

## Performance Results

The system achieves:
- Classification accuracy of 85-90% on highlight detection
- Efficient processing time suitable for offline highlight generation
- Balanced precision and recall in highlight detection

![Performance Metrics](Assignments/W8%20-%20final%20submission/output/plots/updated_accuracies.png)

## Design Decisions

### Feature Engineering Approach

Motion features are calculated to identify exciting moments based on ball dynamics. This approach was chosen because:
- Ball movement patterns strongly correlate with exciting moments in sports
- Physics-based features (velocity, acceleration) are more interpretable
- These features work well across different sports and scenarios

### Multi-Model Approach

The system evaluates multiple ML models instead of just one because:
- Different models capture different patterns in the data
- Ensemble-based models (Random Forest, XGBoost) generally perform best on this task
- This approach provides more robust performance across different datasets

### Modular Pipeline Design

The system is built as separate components for:
- Easier maintenance and extension
- Independent improvement of each stage
- Better testability and debugging

## Future Improvements

- Add player tracking data integration
- Implement sound analysis for crowd reactions
- Add real-time processing capabilities
- Support for multiple sports with different highlight criteria
- Deep learning-based feature extraction