# -----------------------------------------------------------
# Applied Data Science - Clustering and Fitting Assignment
# Student Code - Simple, Beginner Friendly
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------

df = pd.read_csv("world_top_restaurants_dataset.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------------------------------------
# 2. Statistical Moments
# -----------------------------------------------------------

numeric_cols = ["Average_Price_USD", "Michelin_Stars", "Rating", "Years_Operating"]

print("\n--- Statistical Moments ---")
for col in numeric_cols:
    print(f"\nColumn: {col}")
    print(" Mean:", df[col].mean())
    print(" Variance:", df[col].var())
    print(" Skewness:", df[col].skew())
    print(" Kurtosis:", df[col].kurt())

# -----------------------------------------------------------
# 3. Relational Plot (Scatter Plot)
# ----------------------------------------------------
