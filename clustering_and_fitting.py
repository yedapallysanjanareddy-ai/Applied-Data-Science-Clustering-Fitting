# ---------------------------------------------
# Data Science Assignment: Clustering and Fitting
# Student Code (Beginner Friendly)
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Prevent KMeans Windows warning
os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

df = pd.read_csv("world_top_restaurants_dataset.csv")
print("Dataset loaded successfully!")
print(df.head())

# -----------------------------------------------------
# 2. STATISTICAL MOMENTS (mean, variance, skewness, kurtosis)
# -----------------------------------------------------

numeric_cols = ["Average_Price_USD", "Michelin_Stars", "Rating", "Years_Operating"]

print("\n----- Statistical Moments -----")
for col in numeric_cols:
    print(f"\nColumn: {col}")
    print(" Mean:", df[col].mean())
    print(" Variance:", df[col].var())
    print(" Skewness:", df[col].skew())
    print(" Kurtosis:", df[col].kurt())

# -----------------------------------------------------
# 3. RELATIONAL PLOT (Scatter Plot)
# -----------------------------------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Average_Price_USD", y="Rating")
plt.title("Rating vs Average Price")
plt.xlabel("Average Price (USD)")
plt.ylabel("Rating")
plt.show()

# -----------------------------------------------------
# 4. CATEGORICAL PLOT (Bar Chart)
# -----------------------------------------------------

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Continent")
plt.title("Number of Restaurants per Continent")
plt.xticks(rotation=45)
plt.show()

# -----------------------------------------------------
# 5. STATISTICAL PLOT (Correlation Heatmap)
# -----------------------------------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------------------
# 6. K-MEANS CLUSTERING
# -----------------------------------------------------

X = df[["Average_Price_USD", "Rating"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Average_Price_USD", y="Rating", hue="Cluster", palette="Set2")
plt.title("K-Means Clustering (Price vs Rating)")
plt.show()

# -----------------------------------------------------
# 7. LINEAR REGRESSION (FITTING)
# -----------------------------------------------------

x = df["Years_Operating"].values.reshape(-1, 1)
y = df["Rating"].values

model = LinearRegression()
model.fit(x, y)

plt.figure(figsize=(8,5))
plt.scatter(df["Years_Operating"], df["Rating"])
plt.plot(df["Years_Operating"], model.predict(x))
plt.title("Linear Regression: Rating vs Years Operating")
plt.xlabel("Years Operating")
plt.ylabel("Rating")
plt.show()

print("\nAssignment Completed Successfully!")
