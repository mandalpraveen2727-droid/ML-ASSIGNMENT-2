import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

correlation_matrix = df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(14, 10))

sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)

plt.title("Correlation Heatmap of Wine Dataset")
plt.tight_layout()
plt.show()

upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))

strongest_pair = upper_triangle.stack().idxmax()
strongest_value = upper_triangle.stack().max()

print("\nStrongest Positive Correlation")
print("--------------------------------")
print("First Feature :", strongest_pair[0])
print("Second Feature:", strongest_pair[1])
print("Correlation   :", round(strongest_value, 4))
