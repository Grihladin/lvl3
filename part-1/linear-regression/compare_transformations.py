"""
Script to compare different transformations side by side.
Runs models with: original, sqrt, cbrt, and log transformations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import (
    load_and_prepare_data,
    compare_transformations,
)
from model_training import train_linear_regression, inverse_transform_predictions
from model_evaluation import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def run_single_transformation(
    X, y_transformed, continuous_cols, categorical_cols, transform_type
):
    """Run a single model with given transformation."""
    print(f"\n{'='*70}")
    print(f"Running model with {transform_type.upper()} transformation...")
    print(f"{'='*70}")

    results = train_linear_regression(
        X,
        y_transformed,
        continuous_cols,
        categorical_cols,
        transform_type=transform_type,
    )

    # Inverse transform predictions
    y_train_original, y_train_pred_original = inverse_transform_predictions(
        results["y_train"], results["y_train_pred_transformed"], transform_type
    )

    y_test_original, y_test_pred_original = inverse_transform_predictions(
        results["y_test"], results["y_test_pred_transformed"], transform_type
    )

    # Evaluate model
    metrics = evaluate_model(
        results["y_train"],
        results["y_test"],
        results["y_train_pred_transformed"],
        results["y_test_pred_transformed"],
        y_train_original,
        y_test_original,
        y_train_pred_original,
        y_test_pred_original,
        transform_type=transform_type,
    )

    return {
        "results": results,
        "metrics": metrics,
        "y_test_original": y_test_original,
        "y_test_pred_original": y_test_pred_original,
    }


def compare_all_transformations():
    """Compare all transformation types."""
    # Configuration
    DATA_PATH = (
        "../../data/raw/hour.csv"  # Go up to part-1, then up to bike, then into data
    )
    CONTINUOUS_COLS = ["temp", "hum", "windspeed"]
    CATEGORICAL_COLS = ["season", "hr", "weekday", "weathersit", "yr"]

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    X, y, df = load_and_prepare_data(DATA_PATH, CONTINUOUS_COLS, CATEGORICAL_COLS)

    # Get all transformations
    transformation_results = compare_transformations(y)

    # Transformations to test
    transformations = ["original", "sqrt", "cbrt", "log"]

    # Store results for each transformation
    all_results = {}

    for transform_type in transformations:
        y_transformed = transformation_results[transform_type]
        result = run_single_transformation(
            X, y_transformed, CONTINUOUS_COLS, CATEGORICAL_COLS, transform_type
        )
        all_results[transform_type] = result

    # Create comparison summary
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON OF ALL TRANSFORMATIONS")
    print("=" * 70)

    # Create comparison DataFrame
    comparison_data = []
    for transform_type in transformations:
        metrics = all_results[transform_type]["metrics"]
        comparison_data.append(
            {
                "Transformation": transform_type.upper(),
                "Test R² (Transformed)": metrics["test_r2_trans"],
                "Test MSE (Transformed)": metrics["test_mse_trans"],
                "Test RMSE (Transformed)": metrics["test_rmse_trans"],
                "Test MAE (Transformed)": metrics["test_mae_trans"],
                "Test MSLE (Transformed)": metrics.get("test_msle_trans", np.nan),
                "Test R² (Original)": metrics.get("test_r2", metrics["test_r2_trans"]),
                "Test MSE (Original)": metrics.get(
                    "test_mse", metrics["test_mse_trans"]
                ),
                "Test RMSE (Original)": metrics.get(
                    "test_rmse", metrics["test_rmse_trans"]
                ),
                "Test MAE (Original)": metrics.get(
                    "test_mae", metrics["test_mae_trans"]
                ),
                "Test MSLE (Original)": metrics.get("test_msle", np.nan),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    print("\n📊 TRANSFORMED SCALE METRICS (Model's prediction space)")
    print("-" * 70)
    transformed_cols = [
        "Transformation",
        "Test R² (Transformed)",
        "Test MSE (Transformed)",
        "Test RMSE (Transformed)",
        "Test MAE (Transformed)",
        "Test MSLE (Transformed)",
    ]
    print(comparison_df[transformed_cols].to_string(index=False))

    print("\n\n🚴 ORIGINAL SCALE METRICS (Actual bike count - back-transformed)")
    print("-" * 70)
    original_cols = [
        "Transformation",
        "Test R² (Original)",
        "Test MSE (Original)",
        "Test RMSE (Original)",
        "Test MAE (Original)",
        "Test MSLE (Original)",
    ]
    print(comparison_df[original_cols].to_string(index=False))

    # Find best performing transformation
    best_r2_trans = comparison_df.loc[
        comparison_df["Test R² (Transformed)"].idxmax(), "Transformation"
    ]
    best_r2_orig = comparison_df.loc[
        comparison_df["Test R² (Original)"].idxmax(), "Transformation"
    ]
    best_mse_orig = comparison_df.loc[
        comparison_df["Test MSE (Original)"].idxmin(), "Transformation"
    ]
    best_mae_orig = comparison_df.loc[
        comparison_df["Test MAE (Original)"].idxmin(), "Transformation"
    ]
    best_msle_orig = comparison_df.loc[
        comparison_df["Test MSLE (Original)"].idxmin(), "Transformation"
    ]

    print("\n\n🏆 BEST PERFORMERS")
    print("-" * 70)
    print(f"Best R² on Transformed Scale:  {best_r2_trans}")
    print(f"Best R² on Original Scale:     {best_r2_orig}")
    print(f"Lowest MSE on Original Scale:  {best_mse_orig}")
    print(f"Lowest MAE on Original Scale:  {best_mae_orig}")
    print(f"Lowest MSLE on Original Scale: {best_msle_orig}")

    # Create visualization
    create_comparison_plots(all_results, comparison_df, transformations)

    return all_results, comparison_df


def create_comparison_plots(all_results, comparison_df, transformations):
    """Create comprehensive comparison plots."""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    # Row 1: Actual vs Predicted for each transformation (Transformed Scale)
    for i, transform_type in enumerate(transformations):
        ax = fig.add_subplot(gs[0, i])
        results = all_results[transform_type]["results"]
        y_test = results["y_test"]
        y_test_pred = results["y_test_pred_transformed"]

        ax.scatter(y_test, y_test_pred, alpha=0.5, s=20, color=colors[i])
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        r2 = all_results[transform_type]["metrics"]["test_r2_trans"]
        ax.set_xlabel("Actual (Transformed)", fontsize=10)
        ax.set_ylabel("Predicted (Transformed)", fontsize=10)
        ax.set_title(
            f"{transform_type.upper()}\nR² = {r2:.4f}", fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Actual vs Predicted for each transformation (Original Scale)
    for i, transform_type in enumerate(transformations):
        ax = fig.add_subplot(gs[1, i])
        y_test_orig = all_results[transform_type]["y_test_original"]
        y_test_pred_orig = all_results[transform_type]["y_test_pred_original"]

        ax.scatter(y_test_orig, y_test_pred_orig, alpha=0.5, s=20, color=colors[i])
        ax.plot(
            [y_test_orig.min(), y_test_orig.max()],
            [y_test_orig.min(), y_test_orig.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        r2 = all_results[transform_type]["metrics"].get(
            "test_r2", all_results[transform_type]["metrics"]["test_r2_trans"]
        )
        ax.set_xlabel("Actual Bike Count", fontsize=10)
        ax.set_ylabel("Predicted Bike Count", fontsize=10)
        ax.set_title(
            f"{transform_type.upper()} (Original Scale)\nR² = {r2:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 3: Comparison bar charts
    # R² Comparison
    ax1 = fig.add_subplot(gs[2, 0:2])
    x = np.arange(len(transformations))
    width = 0.35
    r2_trans = [
        comparison_df.iloc[i]["Test R² (Transformed)"]
        for i in range(len(transformations))
    ]
    r2_orig = [
        comparison_df.iloc[i]["Test R² (Original)"] for i in range(len(transformations))
    ]

    ax1.bar(
        x - width / 2,
        r2_trans,
        width,
        label="Transformed Scale",
        color="steelblue",
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2, r2_orig, width, label="Original Scale", color="coral", alpha=0.8
    )
    ax1.set_xlabel("Transformation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("R² Score", fontsize=11, fontweight="bold")
    ax1.set_title(
        "R² Comparison: Transformed vs Original Scale", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.upper() for t in transformations])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # MSE Comparison (Original Scale)
    ax2 = fig.add_subplot(gs[2, 2:4])
    mse_orig = [
        comparison_df.iloc[i]["Test MSE (Original)"]
        for i in range(len(transformations))
    ]
    ax2.bar(transformations, mse_orig, color=colors, alpha=0.8)
    ax2.set_xlabel("Transformation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MSE (bikes²)", fontsize=11, fontweight="bold")
    ax2.set_title("MSE Comparison on Original Scale", fontsize=12, fontweight="bold")
    ax2.set_xticklabels([t.upper() for t in transformations])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add values on top of bars
    for i, v in enumerate(mse_orig):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")

    plt.suptitle(
        "Transformation Comparison: Original vs SQRT vs CBRT vs LOG",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save figure
    plt.savefig("results/transformation_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n📈 Comparison plots saved to: results/transformation_comparison.png")
    plt.show()


if __name__ == "__main__":
    all_results, comparison_df = compare_all_transformations()

    # Save comparison table
    comparison_df.to_csv("results/transformation_comparison.csv", index=False)
    print(f"\n💾 Comparison table saved to: results/transformation_comparison.csv")
