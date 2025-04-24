# Prediction Comparison Report

## Basic Statistics

- Original predictions count: 28586
- Updated predictions count: 28601
- Original Class 1 count: 4579 (16.02%)
- Updated Class 1 count: 5771 (20.18%)

## Frame Overlap

- Frames in both datasets: 28586
- Frames only in original: 0
- Frames only in updated: 15

## Agreement Analysis

- Agreement: 24556 (85.90%)
- Disagreement: 4030 (14.10%)

## Detailed Comparison

- Both predict 0: 21396 (74.85%)
- Both predict 1: 3160 (11.05%)
- Original=0, Updated=1: 2611 (9.13%)
- Original=1, Updated=0: 1419 (4.96%)

## Key Findings

- The updated model predicts 4.16% more Class 1 events than the original model.
- When models disagree, the updated model is more likely to predict Class 1 (in 2611 cases vs 1419 cases).
- Overall agreement between the two models is 85.90%.

## Visualizations

The following visualizations are included in this directory:

1. **prediction_matrix.png**: A confusion-style matrix showing agreement/disagreement patterns
2. **agreement_pie.png**: A pie chart showing the overall agreement percentage
3. **class_distribution.png**: A bar chart comparing class distributions between models
4. **disagreement_by_frame.png**: A scatter plot showing where in the sequence the models disagree
