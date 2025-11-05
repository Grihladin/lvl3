import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Load the hourly bike sharing data
print("Loading data...")
df = pd.read_csv(
    "../../data/raw/hour.csv"
)  # Go up to part-1, then up to bike, then into data
print(f"Original dataset shape: {df.shape}")

# Display first few rows and info
print("\nDataset info:")
print(df.info())

# Separate features and target
# Target variable: 'cnt' (total count of bike rentals)
# Remove columns that shouldn't be used for training
columns_to_drop = ["instant", "dteday", "casual", "registered", "cnt"]
target = df["cnt"]
features = df.drop(columns=columns_to_drop)

print(f"\nFeatures shape: {features.shape}")
print(f"Target shape: {target.shape}")

# Define categorical and numerical columns
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
numerical_cols = ["temp", "atemp", "hum", "windspeed"]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
    ],
    remainder="passthrough",
)

# Fit and transform the data
print("\nPreprocessing data...")
X_processed = preprocessor.fit_transform(features)

# Get feature names after preprocessing
num_feature_names = numerical_cols
cat_feature_names = []
for i, cat_col in enumerate(categorical_cols):
    categories = preprocessor.named_transformers_["cat"].categories_[i][
        1:
    ]  # skip first due to drop='first'
    cat_feature_names.extend([f"{cat_col}_{cat}" for cat in categories])

all_feature_names = num_feature_names + cat_feature_names

# Create processed DataFrame
df_processed = pd.DataFrame(X_processed, columns=all_feature_names)

# Add target variable back
df_processed["cnt"] = target.values

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(f"Feature names: {list(df_processed.columns)}")

# Create output directory if it doesn't exist
output_dir = "data/prepeared"
os.makedirs(output_dir, exist_ok=True)

# Save processed data
output_path = os.path.join(output_dir, "hour_processed.csv")
df_processed.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")

# Display summary statistics
print("\nProcessed data summary:")
print(df_processed.describe())
print(f"\nFirst few rows of processed data:")
print(df_processed.head())
