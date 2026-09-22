import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df["class_name"] = df["target"].map({0: wine.target_names[0], 1: wine.target_names[1], 2: wine.target_names[2]})

print("First Five Rows:")
print(df.head())
print("\nLast Five Rows:")
print(df.tail())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())
print("\nClass Distribution:")
print(df["class_name"].value_counts())
plt.figure(figsize=(7, 5))
sns.countplot(data=df,x="class_name",hue="class_name",palette="viridis",legend=False)

plt.title("Distribution of Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

df[wine.feature_names].hist(figsize=(16, 12),bins=20,color="skyblue",edgecolor="black")

plt.suptitle("Distribution of Wine Dataset Features", fontsize=16)
plt.tight_layout()
plt.show()

correlation_matrix = df.drop(columns=["class_name"]).corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Correlation Heatmap of Wine Dataset")
plt.tight_layout()
plt.show()
