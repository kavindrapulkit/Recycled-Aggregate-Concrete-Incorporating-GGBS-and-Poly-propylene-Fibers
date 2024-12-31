# Import required libraries
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import pandas as pd
import numpy as np

# Initialize H2O
h2o.init()

# Load the new dataset
split_data = pd.read_csv("split_str.csv")
split_data.columns = split_data.columns.str.strip()  # Clean column names

# Preprocessing: Handle missing values
for col in split_data.columns:
    if split_data[col].isnull().sum() > 0:
        if split_data[col].dtype in ['float64', 'int64']:
            # Numerical columns: Impute with median
            median_val = split_data[col].median()
            split_data[col] = split_data[col].fillna(median_val)
        else:
            # Categorical columns: Impute with mode
            mode_val = split_data[col].mode()[0]
            split_data[col] = split_data[col].fillna(mode_val)

# Convert MIX to a categorical feature
if "MIX" in split_data.columns:
    split_data["MIX"] = split_data["MIX"].astype("category")

# Convert to H2OFrame
h2o_split_data = h2o.H2OFrame(split_data)

# Define predictors and target
split_target = "Split_Tensile_Strength"
if split_target not in h2o_split_data.columns:
    raise ValueError(f"Target column '{split_target}' not found in dataset!")

split_predictors = [col for col in h2o_split_data.columns if col != split_target]

# Split data into train and test sets
split_train, split_test = h2o_split_data.split_frame(ratios=[0.8], seed=42)

# DRF hyperparameter tuning
split_drf_params = {
    "ntrees": [50, 100, 200],
    "max_depth": [10, 20, 30],
    "min_rows": [5, 10],
    "sample_rate": [0.8, 1.0],
}

split_drf_grid = H2OGridSearch(
    model=H2ORandomForestEstimator(
        seed=42,
        nfolds=3,
        keep_cross_validation_predictions=True
    ),
    hyper_params=split_drf_params,
    grid_id="split_drf_grid"
)
split_drf_grid.train(x=split_predictors, y=split_target, training_frame=split_train)

# Get the best DRF model
split_best_drf = split_drf_grid.get_grid(sort_by="mse", decreasing=False).models[0]

# GBM hyperparameter tuning
split_gbm_params = {
    "ntrees": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "learn_rate": [0.01, 0.1],
    "sample_rate": [0.8, 1.0],
}

split_gbm_grid = H2OGridSearch(
    model=H2OGradientBoostingEstimator(
        seed=42,
        nfolds=3,
        keep_cross_validation_predictions=True
    ),
    hyper_params=split_gbm_params,
    grid_id="split_gbm_grid"
)
split_gbm_grid.train(x=split_predictors, y=split_target, training_frame=split_train)

# Get the best GBM model
split_best_gbm = split_gbm_grid.get_grid(sort_by="mse", decreasing=False).models[0]

# Stacked Ensemble using DRF and GBM models
split_ensemble = H2OStackedEnsembleEstimator(
    base_models=[split_best_drf.model_id, split_best_gbm.model_id],
    seed=42
)
split_ensemble.train(x=split_predictors, y=split_target, training_frame=split_train)

# Function to calculate and return metrics, including MAPE and MRE
def calculate_metrics(model, data_frame, target_col):
    # Generate predictions
    predictions = model.predict(data_frame).as_data_frame()["predict"].values
    actuals = data_frame.as_data_frame()[target_col].values

    # Calculate metrics
    mse = model.model_performance(data_frame).mse()
    rmse = model.model_performance(data_frame).rmse()
    r2 = model.model_performance(data_frame).r2()
    mae = model.model_performance(data_frame).mae()

    # MAPE calculation (Mean Absolute Percentage Error)
    mape = (abs((actuals - predictions) / actuals).mean()) * 100

    # MRE calculation (Mean Relative Error)
    mre = np.mean(np.abs((actuals - predictions) / actuals))  # Mean Relative Error

    return {
        "MSE": mse,
        "MAPE": mape,
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse
    }

# Function to collect and return all metrics for both training and testing datasets
def get_all_metrics(model, model_name, train_frame, test_frame, target_col):
    # Calculate metrics for training data
    train_metrics = calculate_metrics(model, train_frame, target_col)
    # Calculate metrics for test data
    test_metrics = calculate_metrics(model, test_frame, target_col)

    return {
        "Approach": model_name,
        "Train MSE": train_metrics["MSE"],
        "Train MAPE (%)": train_metrics["MAPE"],
        "Train R²": train_metrics["R²"],
        "Train MAE": train_metrics["MAE"],
        "Train RMSE": train_metrics["RMSE"],
        "Test MSE": test_metrics["MSE"],
        "Test MAPE (%)": test_metrics["MAPE"],
        "Test R²": test_metrics["R²"],
        "Test MAE": test_metrics["MAE"],
        "Test RMSE": test_metrics["RMSE"]
    }

# Collect metrics for all models
metrics_drf = get_all_metrics(split_best_drf, "Split DRF", split_train, split_test, split_target)
metrics_gbm = get_all_metrics(split_best_gbm, "Split GBM", split_train, split_test, split_target)
metrics_ensemble = get_all_metrics(split_ensemble, "Split Stacked Ensemble", split_train, split_test, split_target)

# Convert collected metrics into a DataFrame for easy printing
metrics_df = pd.DataFrame([metrics_drf, metrics_gbm, metrics_ensemble])

# Print the results in tabular format
print(metrics_df)

# Shutdown H2O (if needed)
# h2o.shutdown(prompt=False)
