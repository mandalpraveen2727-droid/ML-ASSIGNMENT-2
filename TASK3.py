# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()

# Convert it into a DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)

plt.title("Correlation Heatmap of Wine Dataset")
plt.tight_layout()
plt.show()

# Remove diagonal values and duplicate feature pairs
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))

# Find the strongest positive correlation
strongest_pair = upper_triangle.stack().idxmax()
strongest_value = upper_triangle.stack().max()

print("\nStrongest Positive Correlation")
print("--------------------------------")
print("First Feature :", strongest_pair[0])
print("Second Feature:", strongest_pair[1])
print("Correlation   :", round(strongest_value, 4))