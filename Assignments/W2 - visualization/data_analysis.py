import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('provided_data.csv')

# Display the first 5 rows
print(df.head())

# Display basic information about the dataset
print(df.info())

# Calculate and print summary statistics
print(df.describe())

# Get the number of columns
num_columns = df.shape[1]

# Plotting
for i in range(1, num_columns):  # Start from 1 to skip the first column (Frame Number)
    plt.figure(figsize=(10, 6))
    plt.plot(df.iloc[:, 0], df.iloc[:, i])
    plt.xlabel('Frame Number')
    plt.ylabel(f'Value (Column {i+1})')
    plt.title(f'Column {i+1} vs Frame Number')
    plt.grid(True)
    plt.savefig(f'plot_column_{i+1}.png')
    plt.close()  # Close the figure to free up memory

print(f"Created {num_columns - 1} plots.")
