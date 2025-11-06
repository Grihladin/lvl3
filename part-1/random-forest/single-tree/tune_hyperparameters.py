import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
import pickle
import time

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv("../data/hour_processed_rf.csv")
print(f"Dataset shape: {df.shape}")

# Separate features and target
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

print(f"\nFeatures used: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)

# Define parameter grid to search
param_grid = {
    "max_depth": [5, 10, 15, 20, 25, 30, None],  # Tree depth
    "min_samples_split": [2, 10, 20, 50],  # Min samples to split a node
    "min_samples_leaf": [1, 5, 10, 20],  # Min samples in a leaf
    "max_features": [None, "sqrt", "log2"],  # Features to consider at each split
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = (
    len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
    * len(param_grid["max_features"])
)
print(f"\nTotal combinations to test: {total_combinations}")

# Choose between Grid Search (thorough) or Random Search (faster)
print("\nChoose search method:")
print("1. Grid Search - Tests ALL combinations (slow but thorough)")
print("2. Random Search - Tests random sample (faster)")

# For this demo, we'll use Random Search to save time
use_random = True
n_random_iterations = 50

if use_random:
    print(f"\nUsing Random Search with {n_random_iterations} iterations")
    print("This will test a random sample of parameter combinations")
else:
    print("\nUsing Grid Search")
    print("This will test ALL parameter combinations")

# Create the base model
base_model = DecisionTreeRegressor(random_state=42)

# Set up the search
if use_random:
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_random_iterations,
        cv=5,  # 5-fold cross-validation
        scoring="neg_mean_squared_error",  # Minimize MSE
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
        random_state=42,
        return_train_score=True,
    )
else:
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

# Run the search
print("\n" + "=" * 80)
print("STARTING HYPERPARAMETER SEARCH...")
print("=" * 80)
print("This may take several minutes. Progress will be shown below.\n")

start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()

print("\n" + "=" * 80)
print("SEARCH COMPLETE!")
print("=" * 80)
print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")

# Get the best parameters
best_params = search.best_params_
best_score = -search.best_score_  # Convert back to positive MSE
best_rmse = np.sqrt(best_score)

print("\n" + "=" * 80)
print("BEST PARAMETERS FOUND")
print("=" * 80)
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation RMSE: {best_rmse:.2f} bikes")

# Train final model with best parameters
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("=" * 80)

best_model = search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Set:")
print(f"  RMSE: {train_rmse:.2f} bikes")
print(f"  MAE:  {train_mae:.2f} bikes")
print(f"  R²:   {train_r2:.4f}")

print("\nTest Set:")
print(f"  RMSE: {test_rmse:.2f} bikes")
print(f"  MAE:  {test_mae:.2f} bikes")
print(f"  R²:   {test_r2:.4f}")

print("\nOverfitting Check:")
print(f"  R² Gap: {train_r2 - test_r2:.4f}")
print(f"  RMSE Gap: {test_rmse - train_rmse:.2f} bikes")

# Show top 10 parameter combinations
print("\n" + "=" * 80)
print("TOP 10 PARAMETER COMBINATIONS")
print("=" * 80)

results_df = pd.DataFrame(search.cv_results_)
results_df["mean_test_rmse"] = np.sqrt(-results_df["mean_test_score"])
results_df["mean_train_rmse"] = np.sqrt(-results_df["mean_train_score"])
results_df["rmse_gap"] = results_df["mean_test_rmse"] - results_df["mean_train_rmse"]

top_results = results_df.nsmallest(10, "mean_test_rmse")[
    [
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
        "mean_test_rmse",
        "mean_train_rmse",
        "rmse_gap",
    ]
]

print("\n" + top_results.to_string(index=False))

# Save best model
print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

with open("results/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Best model saved to: results/best_model.pkl")

# Save search results
results_df.to_csv("results/hyperparameter_search_results.csv", index=False)
print("Full search results saved to: results/hyperparameter_search_results.csv")

# Save best parameters
with open("results/best_params.txt", "w") as f:
    f.write("BEST HYPERPARAMETERS\n")
    f.write("=" * 50 + "\n\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nCross-Validation RMSE: {best_rmse:.2f} bikes\n")
    f.write(f"Test RMSE: {test_rmse:.2f} bikes\n")
    f.write(f"Test R²: {test_r2:.4f}\n")
print("Best parameters saved to: results/best_params.txt")

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING COMPLETE!")
print("=" * 80)
print("\nUpdate train_tree.py with the best parameters shown above.")
