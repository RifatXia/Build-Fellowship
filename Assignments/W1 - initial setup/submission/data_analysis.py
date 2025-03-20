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

# Plotting
plt.figure(figsize=(10, 6))
for i in range(len(df.columns)-1):
    plt.plot(df.iloc[:, 0], df.iloc[:, i+1])
    plt.xlabel('Frame Number')
    plt.ylabel(f'Column {i+1} Value')
    plt.title(f'Column {i+1} vs Frame Number')
    plt.grid(True)
    plt.savefig(f'plot{i+1}.png')
    plt.show()

# Calculate the mean of each column
means = df.mean()
