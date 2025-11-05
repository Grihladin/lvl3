import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def create_preprocessor(continuous_cols, categorical_cols):
    """
    Create a preprocessing pipeline for features.

    Parameters:
    -----------
    continuous_cols : list
        List of continuous feature names
    categorical_cols : list
        List of categorical feature names

    Returns:
    --------
    ColumnTransformer : Preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor, continuous_cols, categorical_cols):
    """
    Get feature names after preprocessing.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    continuous_cols : list
        List of continuous feature names
    categorical_cols : list
        List of categorical feature names

    Returns:
    --------
    list : All feature names after preprocessing
    """
    num_feature_names = continuous_cols
    cat_feature_names = []
    for i, cat_col in enumerate(categorical_cols):
        categories = preprocessor.named_transformers_["cat"].categories_[i][1:]
        cat_feature_names.extend([f"{cat_col}_{cat}" for cat in categories])

    all_feature_names = num_feature_names + cat_feature_names
    return all_feature_names


def calculate_vif(X_processed, feature_names):
    """
    Calculate Variance Inflation Factor for multicollinearity detection.

    Parameters:
    -----------
    X_processed : array-like
        Processed feature matrix
    feature_names : list
        List of feature names

    Returns:
    --------
    DataFrame : VIF values for each feature
    """
    print("\n" + "=" * 50)
    print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("=" * 50)
    print("Checking for multicollinearity...")
    print("VIF > 10: High multicollinearity")
    print("VIF > 5: Moderate multicollinearity")
    print("VIF < 5: Low multicollinearity\n")

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [
        variance_inflation_factor(X_processed, i) for i in range(X_processed.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print(vif_data.to_string(index=False))

    # Identify problematic features
    high_vif = vif_data[vif_data["VIF"] > 10]
    if len(high_vif) > 0:
        print(
            f"\n⚠️  Warning: {len(high_vif)} feature(s) with high multicollinearity (VIF > 10):"
        )
        print(high_vif.to_string(index=False))
    else:
        print("\n✓ No features with high multicollinearity detected.")

    moderate_vif = vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)]
    if len(moderate_vif) > 0:
        print(
            f"\n⚠️  {len(moderate_vif)} feature(s) with moderate multicollinearity (VIF > 5):"
        )
        print(moderate_vif.to_string(index=False))

    return vif_data


def train_linear_regression(
    X,
    y,
    continuous_cols,
    categorical_cols,
    transform_type="cbrt",
    test_size=0.2,
    random_state=42,
):
    """
    Train a linear regression model with preprocessing.

    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    continuous_cols : list
        List of continuous feature names
    categorical_cols : list
        List of categorical feature names
    transform_type : str
        Type of transformation applied to target ('cbrt', 'log', 'sqrt', 'original')
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Dictionary containing model, preprocessor, data splits, and metadata
    """
    print(f"\nUsing {transform_type} transformation for modeling...")
    if transform_type == "cbrt":
        print(f"Model will predict: cnt^(1/3) (cube root of bike count)")
        print(
            f"Then we'll cube the predictions to get actual bike count: (cnt^(1/3))^3 = cnt"
        )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create and fit preprocessor
    preprocessor = create_preprocessor(continuous_cols, categorical_cols)

    print("\nPreprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")

    # Get feature names
    all_feature_names = get_feature_names(
        preprocessor, continuous_cols, categorical_cols
    )

    # VIF Analysis
    vif_data = calculate_vif(X_train_processed, all_feature_names)

    # Train model
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    print(
        f"Training Linear Regression model to predict cnt^(1/3)..."
        if transform_type == "cbrt"
        else "Training Linear Regression model..."
    )

    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    print("✓ Model trained successfully!")

    # Make predictions
    print("\nMaking predictions...")
    y_train_pred_transformed = model.predict(X_train_processed)
    y_test_pred_transformed = model.predict(X_test_processed)
    print(f"✓ Predictions complete (in {transform_type} space)")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "y_train_pred_transformed": y_train_pred_transformed,
        "y_test_pred_transformed": y_test_pred_transformed,
        "feature_names": all_feature_names,
        "vif_data": vif_data,
        "transform_type": transform_type,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
    }


def inverse_transform_predictions(
    y_transformed, predictions_transformed, transform_type
):
    """
    Inverse transform predictions back to original scale.

    Parameters:
    -----------
    y_transformed : array-like
        Transformed target values
    predictions_transformed : array-like
        Predictions in transformed space
    transform_type : str
        Type of transformation ('cbrt', 'log', 'sqrt', 'original')

    Returns:
    --------
    tuple : (y_original, predictions_original)
    """
    if transform_type == "log":
        y_original = np.expm1(y_transformed)
        predictions_original = np.expm1(predictions_transformed)
    elif transform_type == "cbrt":
        y_original = y_transformed**3
        predictions_original = predictions_transformed**3
    elif transform_type == "sqrt":
        y_original = y_transformed**2
        predictions_original = predictions_transformed**2
    else:  # original
        y_original = y_transformed
        predictions_original = predictions_transformed

    return y_original, predictions_original
