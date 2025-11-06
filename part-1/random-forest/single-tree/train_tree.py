import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv("../data/hour_processed_rf.csv")
print(f"Dataset shape: {df.shape}")

# Separate features and target
# Drop features with zero importance from previous runs
features_to_drop = [
    "mnth",
    "holiday",
    "weekday",
    "weathersit",
    "hum",
    "cnt",
    "windspeed",
]
X = df.drop(columns=features_to_drop)
y = df["cnt"]

print(
    f"\nFeatures dropped (zero importance): {[f for f in features_to_drop if f != 'cnt']}"
)
print(f"Features used: {list(X.columns)}")
print(f"Target: cnt (bike rental count)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train a single decision tree
print("\n" + "=" * 60)
print("TRAINING SINGLE DECISION TREE")
print("=" * 60)

# Create a tree with optimal parameters (found via hyperparameter tuning)
tree = DecisionTreeRegressor(
    max_depth=None,  # No limit - let tree grow naturally
    min_samples_split=20,  # Minimum samples required to split
    min_samples_leaf=10,  # Minimum samples in leaf node
    max_features=None,  # Consider all features at each split
    random_state=42,
)

print("\nTree Parameters:")
print(f"  max_depth: {tree.max_depth}")
print(f"  min_samples_split: {tree.min_samples_split}")
print(f"  min_samples_leaf: {tree.min_samples_leaf}")

# Fit the tree
print("\nTraining the model...")
tree.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make predictions
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

# Quick evaluation
print("\n" + "=" * 60)
print("QUICK MODEL EVALUATION")
print("=" * 60)

print("\nTraining Set:")
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"  RMSE: {train_rmse:.2f} bikes")
print(f"  MAE:  {train_mae:.2f} bikes")
print(f"  R²:   {train_r2:.4f}")

print("\nTest Set:")
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"  RMSE: {test_rmse:.2f} bikes")
print(f"  MAE:  {test_mae:.2f} bikes")
print(f"  R²:   {test_r2:.4f}")

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": tree.feature_importances_}
).sort_values("importance", ascending=False)

print("\nAll Features Importance:")
print(feature_importance.to_string(index=False))

# Save the model and data
print("\n" + "=" * 60)
print("SAVING MODEL AND DATA")
print("=" * 60)

# Create results directory
os.makedirs("results", exist_ok=True)

# Save the trained model
model_path = "results/decision_tree_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(tree, f)
print(f"✅ Model saved to: {model_path}")

# Save train/test splits
data_path = "results/train_test_data.pkl"
with open(data_path, "wb") as f:
    pickle.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
        },
        f,
    )
print(f"✅ Train/test data saved to: {data_path}")

# Save feature importance
feature_importance.to_csv("results/feature_importance.csv", index=False)
print(f"✅ Feature importance saved to: results/feature_importance.csv")

print("\n" + "=" * 60)
print("🎯 MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nRun 'evaluate_tree.py' to perform detailed evaluation and visualization.")
