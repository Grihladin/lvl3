"""
Main script for bike sharing demand prediction.
This orchestrates data analysis, model training, and evaluation.
"""

import numpy as np
from data_analysis import (
    load_and_prepare_data,
    analyze_target_distribution,
    compare_transformations,
)
from model_training import train_linear_regression, inverse_transform_predictions
from model_evaluation import (
    evaluate_model,
    plot_model_results,
    print_prediction_summary,
)


def main():
    # Configuration
    DATA_PATH = (
        "../../data/raw/hour.csv"  # Go up to part-1, then up to bike, then into data
    )
    CONTINUOUS_COLS = ["temp", "hum", "windspeed"]
    CATEGORICAL_COLS = ["season", "hr", "weekday", "weathersit", "yr"]
    TRANSFORM_TYPE = "cbrt"  # 'cbrt', 'log', 'sqrt', or 'original'

    # Step 1: Load and analyze data
    print("=" * 70)
    print("STEP 1: DATA LOADING AND ANALYSIS")
    print("=" * 70)

    X, y, df = load_and_prepare_data(DATA_PATH, CONTINUOUS_COLS, CATEGORICAL_COLS)

    # Analyze target distribution
    analyze_target_distribution(y, "TARGET VARIABLE (cnt) ANALYSIS")

    # Compare transformations
    transformation_results = compare_transformations(y)

    # Select transformation
    if TRANSFORM_TYPE == "cbrt":
        y_transformed = transformation_results["cbrt"]
    elif TRANSFORM_TYPE == "log":
        y_transformed = transformation_results["log"]
    elif TRANSFORM_TYPE == "sqrt":
        y_transformed = transformation_results["sqrt"]
    else:
        y_transformed = transformation_results["original"]

    # Step 2: Train model
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)

    results = train_linear_regression(
        X,
        y_transformed,
        CONTINUOUS_COLS,
        CATEGORICAL_COLS,
        transform_type=TRANSFORM_TYPE,
    )

    # Step 3: Inverse transform predictions
    y_train_original, y_train_pred_original = inverse_transform_predictions(
        results["y_train"], results["y_train_pred_transformed"], TRANSFORM_TYPE
    )

    y_test_original, y_test_pred_original = inverse_transform_predictions(
        results["y_test"], results["y_test_pred_transformed"], TRANSFORM_TYPE
    )

    # Step 4: Evaluate model
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)

    metrics = evaluate_model(
        results["y_train"],
        results["y_test"],
        results["y_train_pred_transformed"],
        results["y_test_pred_transformed"],
        y_train_original,
        y_test_original,
        y_train_pred_original,
        y_test_pred_original,
        transform_type=TRANSFORM_TYPE,
    )

    # Step 5: Visualize results
    plot_model_results(
        results["model"],
        results["y_test"],
        results["y_test_pred_transformed"],
        y_test_original,
        y_test_pred_original,
        results["feature_names"],
        results["vif_data"],
        metrics,
    )

    # Step 6: Print summary
    print_prediction_summary(
        results["y_test"],
        results["y_test_pred_transformed"],
        y_test_original,
        y_test_pred_original,
        metrics,
    )

    return results, metrics


if __name__ == "__main__":
    results, metrics = main()
