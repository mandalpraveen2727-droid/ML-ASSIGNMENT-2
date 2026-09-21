# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()

# Convert it into a DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Create separate boxplots for every feature
number_of_columns = 3
number_of_rows = 5

plt.figure(figsize=(18, 20))

for index, column in enumerate(df.columns, start=1):
    plt.subplot(number_of_rows, number_of_columns, index) 
    sns.boxplot(y=df[column],color="skyblue")
    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)

plt.suptitle(
    "Boxplots of Numerical Features in the Wine Dataset",fontsize=18)
print("Potential Outliers Using the IQR Method")
print("-" * 50)
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_limit) |(df[column] > upper_limit)]
print(f"{column}: {len(outliers)} outlier(s)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()