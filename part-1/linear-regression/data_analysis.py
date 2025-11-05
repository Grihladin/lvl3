import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def analyze_target_distribution(y, title="Target Variable Analysis"):
    """
    Analyze and visualize the distribution of the target variable.

    Parameters:
    -----------
    y : array-like
        Target variable
    title : str
        Title for the analysis
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"Mean: {y.mean():.2f}")
    print(f"Median: {y.median():.2f}")
    print(f"Std Dev: {y.std():.2f}")
    print(f"Min: {y.min()}")
    print(f"Max: {y.max()}")
    print(f"Skewness: {y.skew():.4f}")
    print(f"Kurtosis: {y.kurtosis():.4f}")


def compare_transformations(y, save_path="results/cnt_transformations.png"):
    """
    Compare different transformations of the target variable.

    Parameters:
    -----------
    y : array-like
        Original target variable
    save_path : str
        Path to save the comparison plot

    Returns:
    --------
    dict : Dictionary with transformation results and statistics
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "=" * 50)
    print("TRANSFORMING TARGET VARIABLE")
    print("=" * 50)

    # Test different transformations
    y_log = np.log1p(y)  # log(1 + y) to handle zeros
    y_sqrt = np.sqrt(y)
    y_cbrt = np.cbrt(y)  # cube root transformation

    print("\nOriginal cnt:")
    print(f"  Skewness: {y.skew():.4f}")
    print(f"  Kurtosis: {y.kurtosis():.4f}")

    print("\nLog-transformed cnt:")
    print(f"  Skewness: {y_log.skew():.4f}")
    print(f"  Kurtosis: {y_log.kurtosis():.4f}")

    print("\nSquare-root transformed cnt:")
    print(f"  Skewness: {y_sqrt.skew():.4f}")
    print(f"  Kurtosis: {y_sqrt.kurtosis():.4f}")

    print("\nCube-root transformed cnt:")
    print(f"  Skewness: {y_cbrt.skew():.4f}")
    print(f"  Kurtosis: {y_cbrt.kurtosis():.4f}")

    # Compare transformations visually
    fig = plt.figure(figsize=(20, 10))

    # Original distribution
    plt.subplot(2, 4, 1)
    sns.histplot(y, bins=50, kde=True)
    plt.xlabel("Bike Count (cnt)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title(
        f"Original Distribution\nSkewness: {y.skew():.3f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.axvline(
        y.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {y.mean():.0f}",
    )
    plt.axvline(
        y.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {y.median():.0f}",
    )
    plt.legend(fontsize=9)

    plt.subplot(2, 4, 5)
    stats.probplot(y, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Original)", fontsize=13, fontweight="bold")

    # Log transformation
    plt.subplot(2, 4, 2)
    sns.histplot(y_log, bins=50, kde=True, color="orange")
    plt.xlabel("Log(1 + cnt)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title(
        f"Log-Transformed\nSkewness: {y_log.skew():.3f}", fontsize=13, fontweight="bold"
    )
    plt.axvline(
        y_log.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {y_log.mean():.2f}",
    )
    plt.axvline(
        y_log.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {y_log.median():.2f}",
    )
    plt.legend(fontsize=9)

    plt.subplot(2, 4, 6)
    stats.probplot(y_log, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Log)", fontsize=13, fontweight="bold")

    # Square root transformation
    plt.subplot(2, 4, 3)
    sns.histplot(y_sqrt, bins=50, kde=True, color="green")
    plt.xlabel("sqrt(cnt)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title(
        f"Square-Root\nSkewness: {y_sqrt.skew():.3f}", fontsize=13, fontweight="bold"
    )
    plt.axvline(
        y_sqrt.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {y_sqrt.mean():.2f}",
    )
    plt.axvline(
        y_sqrt.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {y_sqrt.median():.2f}",
    )
    plt.legend(fontsize=9)

    plt.subplot(2, 4, 7)
    stats.probplot(y_sqrt, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Square-Root)", fontsize=13, fontweight="bold")

    # Cube root transformation
    plt.subplot(2, 4, 4)
    sns.histplot(y_cbrt, bins=50, kde=True, color="purple")
    plt.xlabel("cbrt(cnt)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title(
        f"Cube-Root\nSkewness: {y_cbrt.skew():.3f}", fontsize=13, fontweight="bold"
    )
    plt.axvline(
        y_cbrt.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {y_cbrt.mean():.2f}",
    )
    plt.axvline(
        y_cbrt.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {y_cbrt.median():.2f}",
    )
    plt.legend(fontsize=9)

    plt.subplot(2, 4, 8)
    stats.probplot(y_cbrt, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Cube-Root)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nTransformation comparison plot saved to: {save_path}")
    plt.show()

    # Select best transformation based on skewness (closest to 0)
    transformations = {
        "original": (y, abs(y.skew())),
        "log": (y_log, abs(y_log.skew())),
        "sqrt": (y_sqrt, abs(y_sqrt.skew())),
        "cbrt": (y_cbrt, abs(y_cbrt.skew())),
    }

    best_transform = min(transformations.items(), key=lambda x: x[1][1])
    print(
        f"\n✓ Best transformation: {best_transform[0]} (Skewness: {best_transform[1][0].skew():.4f})"
    )

    return {
        "original": y,
        "log": y_log,
        "sqrt": y_sqrt,
        "cbrt": y_cbrt,
        "best": best_transform[0],
        "transformations": transformations,
    }


def load_and_prepare_data(
    filepath, continuous_cols, categorical_cols, target_col="cnt"
):
    """
    Load and prepare the dataset.

    Parameters:
    -----------
    filepath : str
        Path to the data file
    continuous_cols : list
        List of continuous feature names
    categorical_cols : list
        List of categorical feature names
    target_col : str
        Name of the target column

    Returns:
    --------
    tuple : (X, y, df) where X is features, y is target, df is full dataframe
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")

    all_features = continuous_cols + categorical_cols
    X = df[all_features]
    y = df[target_col]

    print(f"\nFeatures used:")
    print(f"Continuous: {continuous_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, df
