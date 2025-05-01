import argparse
import os
from components.feature_engineer import FeatureEngineer
from components.classifier import Classifier
from components.output_filter import OutputFilter
from components.highlight_maker import HighlightMaker

def main():
    parser = argparse.ArgumentParser(description="Highlight Extraction Pipeline")
    parser.add_argument('tracking_csv', type=str, help='Path to tracking CSV file')
    parser.add_argument('target_csv', type=str, help='Path to target CSV file')
    parser.add_argument('video_raw', type=str, help='Path to raw video file')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    args = parser.parse_args()

    # 1. Feature Engineering
    print("Running feature engineering...")
    feature_engineer = FeatureEngineer(args.tracking_csv)
    features = feature_engineer.extract_features()
    
    # Save updated features
    updated_tracking_csv = os.path.join(os.path.dirname(args.tracking_csv), "updated_tracking.csv")
    features.to_csv(updated_tracking_csv, index=False)
    print(f"Saved updated tracking data to {updated_tracking_csv}")

    # 2. Classification
    print("\nRunning classification...")
    classifier = Classifier(args.output_dir)
    original_preds, updated_preds, best_preds = classifier.predict(args.tracking_csv, args.target_csv)

    # 3. Post-processing / Output Filtering
    print("\nApplying output filtering...")
    output_filter = OutputFilter()
    filtered_predictions = output_filter.filter(best_preds)

    # 4. Highlight Extraction
    print("\nCreating highlights...")
    highlight_maker = HighlightMaker(args.video_raw, filtered_predictions, args.output_dir)
    highlight_maker.create_highlights()

    print("\nPipeline complete! Check the output directory for results and visualizations.")

if __name__ == "__main__":
    main()
