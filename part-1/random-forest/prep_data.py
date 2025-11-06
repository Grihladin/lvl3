import pandas as pd
import numpy as np
import os

# Load the hourly bike sharing data
print("Loading data...")
df = pd.read_csv("../data/raw/hour.csv")  # Go up to bike directory, then into data
print(f"Original dataset shape: {df.shape}")

# Display first few rows and info
print("\nDataset info:")
print(df.info())

# Separate features and target
# Target variable: 'cnt' (total count of bike rentals)
# Remove columns that shouldn't be used for training
# Note: For Random Forest, we keep categorical variables as integers (no one-hot encoding needed)
# and we don't scale numerical features (tree-based models are scale-invariant)
columns_to_drop = ["instant", "dteday", "casual", "registered", "atemp", "cnt"]
features = df.drop(columns=columns_to_drop)
target = df["cnt"]

print(f"\nFeatures shape before processing: {features.shape}")
print(f"Target shape: {target.shape}")

# Define categorical and numerical columns
# For Random Forest, categorical variables can remain as integers
categorical_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
numerical_cols = ["temp", "hum", "windspeed"]

print(f"\nCategorical columns (kept as integers): {categorical_cols}")
print(f"Numerical columns (no scaling applied): {numerical_cols}")

# For Random Forest, minimal preprocessing is needed
# Categorical variables are already encoded as integers in the dataset
# Numerical features don't need scaling for tree-based models
print("\nPreprocessing data for Random Forest...")
print("Note: Random Forest doesn't require feature scaling or one-hot encoding")

# Create the processed dataframe with target variable added back
df_processed = features
df_processed["cnt"] = target.values

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(f"Feature columns: {[col for col in df_processed.columns if col != 'cnt']}")

# Create output directory if it doesn't exist
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Save processed data
output_path = os.path.join(output_dir, "hour_processed_rf.csv")
df_processed.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")

# Display summary statistics
print("\nProcessed data summary:")
print(df_processed.describe())
print(f"\nFirst few rows of processed data:")
print(df_processed.head())

# Print value ranges for verification
print("\nValue ranges for categorical variables:")
for col in categorical_cols:
    print(f"{col}: {df_processed[col].min()} to {df_processed[col].max()}")
print("\nValue ranges for numerical variables:")
for col in numerical_cols:
    print(f"{col}: {df_processed[col].min():.4f} to {df_processed[col].max():.4f}")
