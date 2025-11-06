import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.model_selection import learning_curve, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
from scipy import stats
import pickle
import os

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("DECISION TREE EVALUATION")
print("=" * 80)

# Load the trained model and data
print("\nLoading model and data...")
with open("results/decision_tree_model.pkl", "rb") as f:
    tree = pickle.load(f)

with open("results/train_test_data.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

print("Model and data loaded successfully!")
print(f"\nModel: {type(tree).__name__}")
print(f"  max_depth: {tree.max_depth}")
print(f"  min_samples_split: {tree.min_samples_split}")
print(f"  min_samples_leaf: {tree.min_samples_leaf}")
print(f"  Actual tree depth: {tree.get_depth()}")
print(f"  Number of leaves: {tree.get_n_leaves()}")

# Make predictions
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred


# ============================================================================
# 1. METRICS EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("1. REGRESSION METRICS ANALYSIS")
print("=" * 80)


def calculate_metrics(y_true, y_pred, set_name=""):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    explained_var = explained_variance_score(y_true, y_pred)

    # Additional custom metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))

    # Mean Percentage Error (can be negative, shows bias)
    mpe = np.mean((y_true - y_pred) / y_true) * 100

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape,
        "MPE": mpe,
        "Explained Variance": explained_var,
        "Max Error": max_error,
        "Median AE": median_ae,
    }


train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# Create comparison table
metrics_df = pd.DataFrame(
    {"Training Set": train_metrics, "Test Set": test_metrics, "Difference": {}}
)

# Calculate differences
for metric in train_metrics.keys():
    if metric in ["R²", "Explained Variance"]:
        # For R² and explained variance, lower test score = worse
        metrics_df.loc[metric, "Difference"] = (
            train_metrics[metric] - test_metrics[metric]
        )
    else:
        # For error metrics, higher test score = worse
        metrics_df.loc[metric, "Difference"] = (
            test_metrics[metric] - train_metrics[metric]
        )

print("\n📊 Performance Metrics Comparison:")
print(metrics_df.to_string())

# Overfitting analysis
print("\n" + "-" * 80)
print("Overfitting Analysis:")
print("-" * 80)
r2_gap = train_metrics["R²"] - test_metrics["R²"]
rmse_gap = test_metrics["RMSE"] - train_metrics["RMSE"]

print(f"R² Gap (Train - Test): {r2_gap:.4f}")
print(f"RMSE Gap (Test - Train): {rmse_gap:.2f} bikes")

# Business interpretation
print("\n" + "-" * 80)
print("Business Interpretation:")
print("-" * 80)
print(f"Average prediction error: ±{test_metrics['MAE']:.1f} bikes")
print(f"Worst case error: {test_metrics['Max Error']:.0f} bikes")
print(f"Model explains {test_metrics['R²']*100:.1f}% of bike rental variance")
print(f"Median prediction error: {test_metrics['Median AE']:.1f} bikes")
print(f"Mean Percentage Error (MPE): {test_metrics['MPE']:.2f}%")

# ============================================================================
# 2. RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. RESIDUAL ANALYSIS")
print("=" * 80)

# Statistical tests on residuals
residual_mean = np.mean(test_residuals)
residual_std = np.std(test_residuals)
residual_skew = stats.skew(test_residuals)
residual_kurtosis = stats.kurtosis(test_residuals)

print(f"\nResidual Statistics (Test Set):")
print(f"  Mean: {residual_mean:.4f}")
print(f"  Std Dev: {residual_std:.2f}")
print(f"  Skewness: {residual_skew:.4f}")
print(f"  Kurtosis: {residual_kurtosis:.4f}")

# Normality test
_, p_value_normality = stats.shapiro(
    test_residuals[:5000] if len(test_residuals) > 5000 else test_residuals
)
print(f"\nShapiro-Wilk Normality Test:")
print(f"  p-value: {p_value_normality:.4f}")

# Homoscedasticity check (constant variance)
print(f"\nHeteroscedasticity Check:")
bins = pd.qcut(y_test_pred, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
variance_by_bin = pd.Series(test_residuals).groupby(bins).var()
print(f"  Residual variance by prediction quartile:")
for label, var in variance_by_bin.items():
    print(f"    {label}: {var:.2f}")

variance_ratio = variance_by_bin.max() / variance_by_bin.min()
print(f"  Variance ratio: {variance_ratio:.2f}")


# ============================================================================
# 3. CROSS-VALIDATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. CROSS-VALIDATION ANALYSIS")
print("=" * 80)

print("\nPerforming 5-fold cross-validation...")
cv_results = cross_validate(
    tree,
    X_train,
    y_train,
    cv=5,
    scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
    return_train_score=True,
    n_jobs=-1,
)

cv_summary = pd.DataFrame(
    {
        "RMSE": {
            "Train": np.sqrt(-cv_results["train_neg_mean_squared_error"]).mean(),
            "Val": np.sqrt(-cv_results["test_neg_mean_squared_error"]).mean(),
            "Std": np.sqrt(-cv_results["test_neg_mean_squared_error"]).std(),
        },
        "MAE": {
            "Train": -cv_results["train_neg_mean_absolute_error"].mean(),
            "Val": -cv_results["test_neg_mean_absolute_error"].mean(),
            "Std": -cv_results["test_neg_mean_absolute_error"].std(),
        },
        "R²": {
            "Train": cv_results["train_r2"].mean(),
            "Val": cv_results["test_r2"].mean(),
            "Std": cv_results["test_r2"].std(),
        },
    }
)

print("\nCross-Validation Results (5-fold):")
print(cv_summary.to_string())

print(f"\nModel stability: R² Std = {cv_summary.loc['Std', 'R²']:.4f}")


# ============================================================================
# 4. PREDICTION QUALITY BY RANGE
# ============================================================================
print("\n" + "=" * 80)
print("4. PREDICTION QUALITY BY DEMAND RANGE")
print("=" * 80)

# Categorize predictions by demand level
demand_bins = pd.qcut(
    y_test,
    q=4,
    labels=["Low Demand", "Medium-Low", "Medium-High", "High Demand"],
)

range_analysis = pd.DataFrame()
for label in demand_bins.unique():
    mask = demand_bins == label
    range_analysis[label] = {
        "Count": mask.sum(),
        "Actual Mean": y_test[mask].mean(),
        "Predicted Mean": y_test_pred[mask].mean(),
        "MAE": mean_absolute_error(y_test[mask], y_test_pred[mask]),
        "RMSE": np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask])),
        "R²": r2_score(y_test[mask], y_test_pred[mask]),
        "MAPE (%)": mean_absolute_percentage_error(y_test[mask], y_test_pred[mask])
        * 100,
    }

range_analysis = range_analysis.T
print("\nPerformance by Demand Range:")
print(range_analysis.to_string())

print("\nInsights:")
best_range = range_analysis["R²"].idxmax()
worst_range = range_analysis["R²"].idxmin()
print(
    f"  Best predictions: {best_range} (R² = {range_analysis.loc[best_range, 'R²']:.4f})"
)
print(
    f"  Worst predictions: {worst_range} (R² = {range_analysis.loc[worst_range, 'R²']:.4f})"
)


# Load feature importance
feature_importance = pd.read_csv("results/feature_importance.csv")

# Learning Curve Analysis
print("\n" + "=" * 60)
print("GENERATING LEARNING CURVES")
print("=" * 60)

# Calculate learning curve with MSE
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores_mse, test_scores_mse = learning_curve(
    tree,
    X_train,
    y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
)

# Convert negative MSE to positive
train_scores_mse_pos = -train_scores_mse
test_scores_mse_pos = -test_scores_mse

# Calculate mean and std for MSE
train_mean_mse = np.mean(train_scores_mse_pos, axis=1)
train_std_mse = np.std(train_scores_mse_pos, axis=1)
test_mean_mse = np.mean(test_scores_mse_pos, axis=1)
test_std_mse = np.std(test_scores_mse_pos, axis=1)

# Convert MSE to RMSE
train_scores_rmse = np.sqrt(train_scores_mse_pos)
test_scores_rmse = np.sqrt(test_scores_mse_pos)

# Calculate mean and std for RMSE
train_mean_rmse = np.mean(train_scores_rmse, axis=1)
train_std_rmse = np.std(train_scores_rmse, axis=1)
test_mean_rmse = np.mean(test_scores_rmse, axis=1)
test_std_rmse = np.std(test_scores_rmse, axis=1)

# Calculate learning curve with MAE
_, train_scores_mae, test_scores_mae = learning_curve(
    tree,
    X_train,
    y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,
)

# Convert negative MAE to positive
train_scores_mae_pos = -train_scores_mae
test_scores_mae_pos = -test_scores_mae

# Calculate mean and std for MAE
train_mean_mae = np.mean(train_scores_mae_pos, axis=1)
train_std_mae = np.std(train_scores_mae_pos, axis=1)
test_mean_mae = np.mean(test_scores_mae_pos, axis=1)
test_std_mae = np.std(test_scores_mae_pos, axis=1)

# Calculate learning curve with R²
_, train_scores_r2, test_scores_r2 = learning_curve(
    tree,
    X_train,
    y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
)

# Calculate mean and std for R²
train_mean_r2 = np.mean(train_scores_r2, axis=1)
train_std_r2 = np.std(train_scores_r2, axis=1)
test_mean_r2 = np.mean(test_scores_r2, axis=1)
test_std_r2 = np.std(test_scores_r2, axis=1)

print("Learning curves calculated (MSE, RMSE, MAE, and R²)")

# # Visualize the tree
# print("\n" + "=" * 60)
# print("VISUALIZING DECISION TREE")
# print("=" * 60)

# # Create a large figure for the tree
# fig, ax = plt.subplots(figsize=(25, 15))

# plot_tree(
#     tree,
#     feature_names=feature_names,
#     filled=True,
#     rounded=True,
#     fontsize=10,
#     ax=ax,
#     proportion=True,  # Show proportions instead of absolute values
#     precision=1,
# )

# plt.title(
#     f"Decision Tree for Bike Rental Prediction\n"
#     f"Max Depth: {tree.max_depth} | Test R²: {test_metrics['R²']:.4f} | Test MAE: {test_metrics['MAE']:.2f} bikes",
#     fontsize=16,
#     fontweight="bold",
#     pad=20,
# )

# plt.tight_layout()
# plt.savefig("results/single_decision_tree.png", dpi=300, bbox_inches="tight")
# print("Tree visualization saved to: results/single_decision_tree.png")

# Create a feature importance plot with seaborn
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_top = feature_importance.head(len(feature_importance))
sns.barplot(
    x="importance", y="feature", data=feature_importance_top, ax=ax, palette="viridis"
)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.set_title("Feature Importance in Decision Tree", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("results/tree_feature_importance.png", dpi=300, bbox_inches="tight")
print("Feature importance plot saved to: results/tree_feature_importance.png")

# Create learning curve plots (MSE, RMSE, MAE, and R²)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: MSE Learning Curve
sns.lineplot(
    x=train_sizes_abs,
    y=train_mean_mse,
    marker="o",
    label="Training MSE",
    color="blue",
    ax=axes[0, 0],
)
axes[0, 0].fill_between(
    train_sizes_abs,
    train_mean_mse - train_std_mse,
    train_mean_mse + train_std_mse,
    alpha=0.2,
    color="blue",
)

sns.lineplot(
    x=train_sizes_abs,
    y=test_mean_mse,
    marker="o",
    label="Cross-Validation MSE",
    color="red",
    ax=axes[0, 0],
)
axes[0, 0].fill_between(
    train_sizes_abs,
    test_mean_mse - test_std_mse,
    test_mean_mse + test_std_mse,
    alpha=0.2,
    color="red",
)

axes[0, 0].set_xlabel("Training Set Size", fontsize=12)
axes[0, 0].set_ylabel("MSE (bikes²)", fontsize=12)
axes[0, 0].set_title("Learning Curve - MSE", fontsize=14, fontweight="bold")
axes[0, 0].legend(loc="best", fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: RMSE Learning Curve
sns.lineplot(
    x=train_sizes_abs,
    y=train_mean_rmse,
    marker="o",
    label="Training RMSE",
    color="blue",
    ax=axes[0, 1],
)
axes[0, 1].fill_between(
    train_sizes_abs,
    train_mean_rmse - train_std_rmse,
    train_mean_rmse + train_std_rmse,
    alpha=0.2,
    color="blue",
)

sns.lineplot(
    x=train_sizes_abs,
    y=test_mean_rmse,
    marker="o",
    label="Cross-Validation RMSE",
    color="red",
    ax=axes[0, 1],
)
axes[0, 1].fill_between(
    train_sizes_abs,
    test_mean_rmse - test_std_rmse,
    test_mean_rmse + test_std_rmse,
    alpha=0.2,
    color="red",
)

axes[0, 1].set_xlabel("Training Set Size", fontsize=12)
axes[0, 1].set_ylabel("RMSE (bikes)", fontsize=12)
axes[0, 1].set_title("Learning Curve - RMSE", fontsize=14, fontweight="bold")
axes[0, 1].legend(loc="best", fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: MAE Learning Curve
sns.lineplot(
    x=train_sizes_abs,
    y=train_mean_mae,
    marker="o",
    label="Training MAE",
    color="blue",
    ax=axes[1, 0],
)
axes[1, 0].fill_between(
    train_sizes_abs,
    train_mean_mae - train_std_mae,
    train_mean_mae + train_std_mae,
    alpha=0.2,
    color="blue",
)

sns.lineplot(
    x=train_sizes_abs,
    y=test_mean_mae,
    marker="o",
    label="Cross-Validation MAE",
    color="red",
    ax=axes[1, 0],
)
axes[1, 0].fill_between(
    train_sizes_abs,
    test_mean_mae - test_std_mae,
    test_mean_mae + test_std_mae,
    alpha=0.2,
    color="red",
)

axes[1, 0].set_xlabel("Training Set Size", fontsize=12)
axes[1, 0].set_ylabel("MAE (bikes)", fontsize=12)
axes[1, 0].set_title("Learning Curve - MAE", fontsize=14, fontweight="bold")
axes[1, 0].legend(loc="best", fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: R² Learning Curve
sns.lineplot(
    x=train_sizes_abs,
    y=train_mean_r2,
    marker="o",
    label="Training R²",
    color="blue",
    ax=axes[1, 1],
)
axes[1, 1].fill_between(
    train_sizes_abs,
    train_mean_r2 - train_std_r2,
    train_mean_r2 + train_std_r2,
    alpha=0.2,
    color="blue",
)

sns.lineplot(
    x=train_sizes_abs,
    y=test_mean_r2,
    marker="o",
    label="Cross-Validation R²",
    color="red",
    ax=axes[1, 1],
)
axes[1, 1].fill_between(
    train_sizes_abs,
    test_mean_r2 - test_std_r2,
    test_mean_r2 + test_std_r2,
    alpha=0.2,
    color="red",
)

axes[1, 1].set_xlabel("Training Set Size", fontsize=12)
axes[1, 1].set_ylabel("R² Score", fontsize=12)
axes[1, 1].set_title("Learning Curve - R²", fontsize=14, fontweight="bold")
axes[1, 1].legend(loc="best", fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/learning_curves.png", dpi=300, bbox_inches="tight")
print("Learning curves saved to: results/learning_curves.png")

plt.show()
