import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import os

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def evaluate_model(
    y_train,
    y_test,
    y_train_pred,
    y_test_pred,
    y_train_original=None,
    y_test_original=None,
    y_train_pred_original=None,
    y_test_pred_original=None,
    transform_type="cbrt",
):
    """
    Evaluate model performance on both transformed and original scales.

    Parameters:
    -----------
    y_train, y_test : array-like
        Actual values (transformed)
    y_train_pred, y_test_pred : array-like
        Predicted values (transformed)
    y_train_original, y_test_original : array-like, optional
        Actual values (original scale)
    y_train_pred_original, y_test_pred_original : array-like, optional
        Predicted values (original scale)
    transform_type : str
        Type of transformation used

    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Metrics on transformed scale
    print(f"\n🎯 PRIMARY METRICS: Predicting cnt^(1/3) (Cube Root) - Scale [0, 10]")
    print("=" * 50)
    print("Training Set:")
    train_mse_trans = mean_squared_error(y_train, y_train_pred)
    train_rmse_trans = np.sqrt(train_mse_trans)
    train_mae_trans = mean_absolute_error(y_train, y_train_pred)
    train_r2_trans = r2_score(y_train, y_train_pred)

    print(f"  MSE:  {train_mse_trans:.4f}")
    print(f"  RMSE: {train_rmse_trans:.4f} (on 0-10 scale)")
    print(f"  MAE:  {train_mae_trans:.4f} (on 0-10 scale)")
    print(f"  R²:   {train_r2_trans:.4f}")

    print("\nTest Set:")
    test_mse_trans = mean_squared_error(y_test, y_test_pred)
    test_rmse_trans = np.sqrt(test_mse_trans)
    test_mae_trans = mean_absolute_error(y_test, y_test_pred)
    test_r2_trans = r2_score(y_test, y_test_pred)
    # MSLE requires non-negative values
    try:
        test_msle_trans = mean_squared_log_error(np.maximum(0, y_test), np.maximum(0, y_test_pred))
    except:
        test_msle_trans = np.nan

    print(f"  MSE:  {test_mse_trans:.4f}")
    print(f"  RMSE: {test_rmse_trans:.4f} (on 0-10 scale)")
    print(f"  MAE:  {test_mae_trans:.4f} (on 0-10 scale)")
    print(f"  MSLE: {test_msle_trans:.4f} (log scale)")
    print(f"  R²:   {test_r2_trans:.4f}")

    print(
        f"\nActual cnt^(1/3) range - Train: [{y_train.min():.2f}, {y_train.max():.2f}]"
    )
    print(
        f"Predicted cnt^(1/3) range - Train: [{y_train_pred.min():.2f}, {y_train_pred.max():.2f}]"
    )
    print(f"Actual cnt^(1/3) range - Test: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(
        f"Predicted cnt^(1/3) range - Test: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]"
    )

    metrics = {
        "train_mse_trans": train_mse_trans,
        "train_rmse_trans": train_rmse_trans,
        "train_mae_trans": train_mae_trans,
        "train_r2_trans": train_r2_trans,
        "test_mse_trans": test_mse_trans,
        "test_rmse_trans": test_rmse_trans,
        "test_mae_trans": test_mae_trans,
        "test_msle_trans": test_msle_trans,
        "test_r2_trans": test_r2_trans,
    }

    # Metrics on original scale (if provided)
    if y_train_original is not None and y_train_pred_original is not None:
        print("\n" + "=" * 50)
        print("� BACK-TRANSFORMED METRICS: Original Scale (Actual Bike Count)")
        print("=" * 50)
        print("📌 These are the predictions after converting back from cnt^(1/3) → cnt")
        print("   by cubing the predictions: (prediction)^3 = bike count")
        print("-" * 50)
        print("Training Set:")
        train_mse = mean_squared_error(y_train_original, y_train_pred_original)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_original, y_train_pred_original)
        train_r2 = r2_score(y_train_original, y_train_pred_original)

        print(f"  MSE:  {train_mse:.2f} bikes²")
        print(f"  RMSE: {train_rmse:.2f} bikes")
        print(f"  MAE:  {train_mae:.2f} bikes")
        print(f"  R²:   {train_r2:.4f}")

        print("\nTest Set:")
        test_mse = mean_squared_error(y_test_original, y_test_pred_original)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
        test_r2 = r2_score(y_test_original, y_test_pred_original)
        # MSLE on original scale
        try:
            test_msle = mean_squared_log_error(y_test_original, y_test_pred_original)
        except:
            test_msle = np.nan

        print(f"  MSE:  {test_mse:.2f} bikes²")
        print(f"  RMSE: {test_rmse:.2f} bikes")
        print(f"  MAE:  {test_mae:.2f} bikes")
        print(f"  MSLE: {test_msle:.4f} (log scale)")
        print(f"  R²:   {test_r2:.4f}")
        
        print("\n💡 Interpretation:")
        print(f"   On average, our predictions are off by ±{test_mae:.1f} bikes")
        print(f"   The model explains {test_r2*100:.1f}% of the variance in bike rentals")
        print(f"   MSLE = {test_msle:.4f} (lower is better, penalizes under-predictions more)")

        metrics.update(
            {
                "train_mse": train_mse,
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_msle": test_msle,
                "test_r2": test_r2,
            }
        )

    return metrics


def plot_model_results(
    model,
    y_test,
    y_test_pred,
    y_test_original,
    y_test_pred_original,
    feature_names,
    vif_data,
    metrics,
    save_path="results/linear_regression_results.png",
):
    """
    Create comprehensive visualization of model results.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    y_test : array-like
        Actual test values (transformed)
    y_test_pred : array-like
        Predicted test values (transformed)
    y_test_original : array-like
        Actual test values (original scale)
    y_test_pred_original : array-like
        Predicted test values (original scale)
    feature_names : list
        List of feature names
    vif_data : DataFrame
        VIF analysis results
    metrics : dict
        Dictionary of evaluation metrics
    save_path : str
        Path to save the plot
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get coefficients
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    print("\n" + "=" * 50)
    print("MODEL COEFFICIENTS")
    print("=" * 50)
    print(f"Intercept: {model.intercept_:.2f}")
    print("\nTop 10 Most Important Features:")
    print(coef_df.head(10).to_string(index=False))

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Plot 1: Actual vs Predicted (Transformed Scale)
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, s=30, ax=axes[0, 0])
    axes[0, 0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0, 0].set_xlabel("Actual cnt^(1/3)", fontsize=12)
    axes[0, 0].set_ylabel("Predicted cnt^(1/3)", fontsize=12)
    axes[0, 0].set_title(
        f"Actual vs Predicted cnt^(1/3) [0-10 scale]\nR² = {metrics['test_r2_trans']:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 0].legend()

    # Plot 2: Residuals Distribution (Transformed Scale)
    residuals_trans = y_test - y_test_pred
    sns.histplot(residuals_trans, kde=True, bins=50, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Residuals (cnt^(1/3) scale)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency", fontsize=12)
    axes[0, 1].set_title(
        "Residuals Distribution [0-10 scale]", fontsize=14, fontweight="bold"
    )

    # Plot 3: Residuals vs Predicted (Transformed Scale)
    sns.scatterplot(x=y_test_pred, y=residuals_trans, alpha=0.6, s=30, ax=axes[1, 0])
    axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Predicted cnt^(1/3)", fontsize=12)
    axes[1, 0].set_ylabel("Residuals (cnt^(1/3) scale)", fontsize=12)
    axes[1, 0].set_title("Residual Plot [0-10 scale]", fontsize=14, fontweight="bold")

    # Plot 4: Top 10 Feature Coefficients
    top_features = coef_df.head(10).copy()
    sns.barplot(x="Coefficient", y="Feature", data=top_features, ax=axes[1, 1])
    axes[1, 1].set_xlabel("Coefficient Value", fontsize=12)
    axes[1, 1].set_ylabel("Feature", fontsize=12)
    axes[1, 1].set_title(
        "Top 10 Most Important Features", fontsize=14, fontweight="bold"
    )
    axes[1, 1].axvline(x=0, color="black", linestyle="-", lw=0.5)

    # Plot 5: VIF Heatmap (Top 20 features)
    top_20_vif = vif_data.head(20).copy()
    colors = [
        "red" if x > 10 else "orange" if x > 5 else "green" for x in top_20_vif["VIF"]
    ]
    sns.barplot(x="VIF", y="Feature", data=top_20_vif, palette=colors, ax=axes[0, 2])
    axes[0, 2].set_xlabel("VIF Value", fontsize=12)
    axes[0, 2].set_ylabel("Feature", fontsize=12)
    axes[0, 2].set_title(
        "Top 20 Features by VIF\n(Red: VIF>10, Orange: VIF>5, Green: VIF<5)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 2].axvline(
        x=10, color="red", linestyle="--", lw=1, alpha=0.7, label="VIF=10"
    )
    axes[0, 2].axvline(
        x=5, color="orange", linestyle="--", lw=1, alpha=0.7, label="VIF=5"
    )
    axes[0, 2].legend()

    # Plot 6: Q-Q plot for residuals normality check
    stats.probplot(residuals_trans, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title(
        "Q-Q Plot - Residuals Normality\n[cnt^(1/3) scale]",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlots saved to: {save_path}")
    plt.show()


def print_prediction_summary(
    y_test, y_test_pred, y_test_original, y_test_pred_original, metrics
):
    """
    Print summary of predictions.

    Parameters:
    -----------
    y_test : array-like
        Actual test values (transformed)
    y_test_pred : array-like
        Predicted test values (transformed)
    y_test_original : array-like
        Actual test values (original scale)
    y_test_pred_original : array-like
        Predicted test values (original scale)
    metrics : dict
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 50)
    print("🎯 PREDICTION SUMMARY - Transformed Scale cnt^(1/3) [0-10]")
    print("=" * 50)
    print(f"Mean actual cnt^(1/3):      {y_test.mean():.4f}")
    print(f"Mean predicted cnt^(1/3):   {y_test_pred.mean():.4f}")
    print(f"Min actual:    {y_test.min():.4f}")
    print(f"Max actual:    {y_test.max():.4f}")
    print(f"Min predicted: {y_test_pred.min():.4f}")
    print(f"Max predicted: {y_test_pred.max():.4f}")
    print(f"\nMean Absolute Error: {metrics['test_mae_trans']:.4f} (on 0-10 scale)")
    print(f"Root Mean Squared Error: {metrics['test_rmse_trans']:.4f} (on 0-10 scale)")

    if y_test_original is not None and y_test_pred_original is not None:
        print("\n" + "=" * 50)
        print("� BACK-TRANSFORMED - Original Scale [0-1000 bikes]")
        print("=" * 50)
        print("📌 After cubing predictions: (cnt^(1/3))^3 = cnt")
        print(f"Mean actual bike count:     {y_test_original.mean():.2f}")
        print(f"Mean predicted bike count:  {y_test_pred_original.mean():.2f}")
        print(f"\nMin actual:    {y_test_original.min():.0f} bikes")
        print(f"Max actual:    {y_test_original.max():.0f} bikes")
        print(f"Min predicted: {y_test_pred_original.min():.0f} bikes")
        print(f"Max predicted: {y_test_pred_original.max():.0f} bikes")
        print(f"\n🎯 Performance on Original Scale:")
        print(f"   MSE:  {metrics.get('test_mse', 0):.2f} bikes²")
        print(f"   RMSE: {metrics.get('test_rmse', 0):.2f} bikes")
        print(f"   MAE:  {metrics.get('test_mae', 0):.2f} bikes")
        print(f"   MSLE: {metrics.get('test_msle', 0):.4f} (log scale)")
        print(f"   R²:   {metrics.get('test_r2', 0):.4f}")

    print("\n" + "=" * 50)
    print("Model evaluation completed successfully!")
    print("=" * 50)
