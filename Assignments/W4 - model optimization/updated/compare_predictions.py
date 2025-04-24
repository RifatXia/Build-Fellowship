import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for comparison results
os.makedirs('comparison_results', exist_ok=True)

# Load the prediction files
original_preds = pd.read_csv('../predictions.csv')
updated_preds = pd.read_csv('predictions.csv')

# Basic statistics
orig_count = len(original_preds)
updated_count = len(updated_preds)
orig_class1_count = original_preds['value'].sum()
updated_class1_count = updated_preds['value'].sum()
orig_class1_pct = orig_class1_count / orig_count * 100
updated_class1_pct = updated_class1_count / updated_count * 100

print(f"Original predictions count: {orig_count}")
print(f"Updated predictions count: {updated_count}")
print(f"Original Class 1 count: {orig_class1_count} ({orig_class1_pct:.2f}%)")
print(f"Updated Class 1 count: {updated_class1_count} ({updated_class1_pct:.2f}%)")

# Identify overlapping frames
original_frames = set(original_preds['frame'])
updated_frames = set(updated_preds['frame'])
common_frames = original_frames.intersection(updated_frames)
only_in_original = original_frames - updated_frames
only_in_updated = updated_frames - original_frames

print(f"Frames in both datasets: {len(common_frames)}")
print(f"Frames only in original: {len(only_in_original)}")
print(f"Frames only in updated: {len(only_in_updated)}")

# Merge datasets on frame to compare predictions
merged_df = pd.merge(
    original_preds, 
    updated_preds, 
    on='frame', 
    how='inner',
    suffixes=('_original', '_updated')
)

# Calculate agreement and disagreement
agreement = (merged_df['value_original'] == merged_df['value_updated']).sum()
disagreement = len(merged_df) - agreement
agreement_pct = agreement / len(merged_df) * 100
disagreement_pct = disagreement / len(merged_df) * 100

print(f"Agreement: {agreement} ({agreement_pct:.2f}%)")
print(f"Disagreement: {disagreement} ({disagreement_pct:.2f}%)")

# Identify specific types of disagreements
agreement_0_0 = ((merged_df['value_original'] == 0) & (merged_df['value_updated'] == 0)).sum()
agreement_1_1 = ((merged_df['value_original'] == 1) & (merged_df['value_updated'] == 1)).sum()
disagreement_0_1 = ((merged_df['value_original'] == 0) & (merged_df['value_updated'] == 1)).sum()
disagreement_1_0 = ((merged_df['value_original'] == 1) & (merged_df['value_updated'] == 0)).sum()

print(f"Both predict 0: {agreement_0_0} ({agreement_0_0/len(merged_df)*100:.2f}%)")
print(f"Both predict 1: {agreement_1_1} ({agreement_1_1/len(merged_df)*100:.2f}%)")
print(f"Original=0, Updated=1: {disagreement_0_1} ({disagreement_0_1/len(merged_df)*100:.2f}%)")
print(f"Original=1, Updated=0: {disagreement_1_0} ({disagreement_1_0/len(merged_df)*100:.2f}%)")

# Create confusion matrix
confusion_matrix = pd.DataFrame({
    'Original=0': [agreement_0_0, disagreement_0_1],
    'Original=1': [disagreement_1_0, agreement_1_1]
}, index=['Updated=0', 'Updated=1'])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Prediction Comparison Matrix')
plt.tight_layout()
plt.savefig('comparison_results/prediction_matrix.png')
plt.close()

# Create pie chart for agreement/disagreement
plt.figure(figsize=(10, 6))
plt.pie([agreement, disagreement], 
        labels=['Agreement', 'Disagreement'],
        autopct='%1.1f%%',
        colors=['#5cb85c', '#d9534f'],
        explode=(0, 0.1))
plt.title('Agreement between Original and Updated Predictions')
plt.tight_layout()
plt.savefig('comparison_results/agreement_pie.png')
plt.close()

# Create bar chart for class distribution comparison
plt.figure(figsize=(10, 6))
plt.bar(['Original Class 0', 'Original Class 1', 'Updated Class 0', 'Updated Class 1'],
        [orig_count - orig_class1_count, orig_class1_count, 
         updated_count - updated_class1_count, updated_class1_count])
plt.title('Class Distribution Comparison')
plt.grid(axis='y', alpha=0.3)
plt.savefig('comparison_results/class_distribution.png')
plt.close()

# Check distribution of disagreements by frame
disagreement_df = merged_df[merged_df['value_original'] != merged_df['value_updated']].copy()
disagreement_df = disagreement_df.sort_values('frame')

plt.figure(figsize=(15, 6))
plt.scatter(disagreement_df['frame'], 
            disagreement_df['value_updated'], 
            label='Updated', 
            alpha=0.7)
plt.scatter(disagreement_df['frame'], 
            disagreement_df['value_original'], 
            label='Original', 
            alpha=0.7)
plt.xlabel('Frame Number')
plt.ylabel('Predicted Class')
plt.title('Disagreement by Frame')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('comparison_results/disagreement_by_frame.png')
plt.close()

# Generate markdown report
with open('comparison_results/prediction_comparison_report.md', 'w') as f:
    f.write("# Prediction Comparison Report\n\n")
    
    f.write("## Basic Statistics\n\n")
    f.write(f"- Original predictions count: {orig_count}\n")
    f.write(f"- Updated predictions count: {updated_count}\n")
    f.write(f"- Original Class 1 count: {orig_class1_count} ({orig_class1_pct:.2f}%)\n")
    f.write(f"- Updated Class 1 count: {updated_class1_count} ({updated_class1_pct:.2f}%)\n\n")
    
    f.write("## Frame Overlap\n\n")
    f.write(f"- Frames in both datasets: {len(common_frames)}\n")
    f.write(f"- Frames only in original: {len(only_in_original)}\n")
    f.write(f"- Frames only in updated: {len(only_in_updated)}\n\n")
    
    f.write("## Agreement Analysis\n\n")
    f.write(f"- Agreement: {agreement} ({agreement_pct:.2f}%)\n")
    f.write(f"- Disagreement: {disagreement} ({disagreement_pct:.2f}%)\n\n")
    
    f.write("## Detailed Comparison\n\n")
    f.write(f"- Both predict 0: {agreement_0_0} ({agreement_0_0/len(merged_df)*100:.2f}%)\n")
    f.write(f"- Both predict 1: {agreement_1_1} ({agreement_1_1/len(merged_df)*100:.2f}%)\n")
    f.write(f"- Original=0, Updated=1: {disagreement_0_1} ({disagreement_0_1/len(merged_df)*100:.2f}%)\n")
    f.write(f"- Original=1, Updated=0: {disagreement_1_0} ({disagreement_1_0/len(merged_df)*100:.2f}%)\n\n")
    
    f.write("## Key Findings\n\n")
    
    # Calculate which dataset predicts more class 1 events
    if updated_class1_pct > orig_class1_pct:
        f.write(f"- The updated model predicts {updated_class1_pct - orig_class1_pct:.2f}% more Class 1 events than the original model.\n")
    else:
        f.write(f"- The original model predicts {orig_class1_pct - updated_class1_pct:.2f}% more Class 1 events than the updated model.\n")
    
    # Calculate net direction of disagreements
    if disagreement_0_1 > disagreement_1_0:
        f.write(f"- When models disagree, the updated model is more likely to predict Class 1 (in {disagreement_0_1} cases vs {disagreement_1_0} cases).\n")
    else:
        f.write(f"- When models disagree, the updated model is more likely to predict Class 0 (in {disagreement_1_0} cases vs {disagreement_0_1} cases).\n")
    
    f.write(f"- Overall agreement between the two models is {agreement_pct:.2f}%.\n\n")
    
    f.write("## Visualizations\n\n")
    f.write("The following visualizations are included in this directory:\n\n")
    f.write("1. **prediction_matrix.png**: A confusion-style matrix showing agreement/disagreement patterns\n")
    f.write("2. **agreement_pie.png**: A pie chart showing the overall agreement percentage\n")
    f.write("3. **class_distribution.png**: A bar chart comparing class distributions between models\n")
    f.write("4. **disagreement_by_frame.png**: A scatter plot showing where in the sequence the models disagree\n")

print("Comparison report and visualizations generated in 'comparison_results' directory.") 