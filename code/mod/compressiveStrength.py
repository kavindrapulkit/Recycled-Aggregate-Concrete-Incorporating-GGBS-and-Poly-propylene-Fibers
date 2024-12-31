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

# Load dataset
data = pd.read_csv("FinalData.csv")
data.columns = data.columns.str.strip()  # Clean column names

# Preprocessing: Handle missing values
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype in ['float64', 'int64']:
            # Numerical columns: Impute with median
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
        else:
            # Categorical columns: Impute with mode
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)

# Convert to H2OFrame
h2o_data = h2o.H2OFrame(data)

# Define predictors and target
target = "Strength (output)"  # Ensure this matches your dataset
if target not in h2o_data.columns:
    raise ValueError(f"Target column '{target}' not found in dataset!")

predictors = [col for col in h2o_data.columns if col != target]

# Split data into train and test sets
train, test = h2o_data.split_frame(ratios=[0.8], seed=42)

# DRF hyperparameter tuning
drf_params = {
    "ntrees": [50, 100, 200],
    "max_depth": [10, 20, 30],
    "min_rows": [5, 10],
    "sample_rate": [0.8, 1.0],
}

drf_grid = H2OGridSearch(
    model=H2ORandomForestEstimator(
        seed=42,
        nfolds=3,
        keep_cross_validation_predictions=True
    ),
    hyper_params=drf_params,
    grid_id="drf_grid"
)
drf_grid.train(x=predictors, y=target, training_frame=train)

# Get the best DRF model
best_drf = drf_grid.get_grid(sort_by="mse", decreasing=False).models[0]

# GBM hyperparameter tuning
gbm_params = {
    "ntrees": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "learn_rate": [0.01, 0.1],
    "sample_rate": [0.8, 1.0],
}

gbm_grid = H2OGridSearch(
    model=H2OGradientBoostingEstimator(
        seed=42,
        nfolds=3,
        keep_cross_validation_predictions=True
    ),
    hyper_params=gbm_params,
    grid_id="gbm_grid"
)
gbm_grid.train(x=predictors, y=target, training_frame=train)

# Get the best GBM model
best_gbm = gbm_grid.get_grid(sort_by="mse", decreasing=False).models[0]

# Stacked Ensemble using DRF and GBM models
ensemble = H2OStackedEnsembleEstimator(
    base_models=[best_drf.model_id, best_gbm.model_id],
    seed=42
)
ensemble.train(x=predictors, y=target, training_frame=train)

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

    # Return all metrics as a dictionary
    return {
        "MSE": mse,
        "MAPE": mape,
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse
    }

# Evaluate models on both training and testing datasets
results = []

for model, name in [(best_drf, "Distributed Random Forest"),
                    (best_gbm, "Gradient Boosting Machine"),
                    (ensemble, "Stacked Ensemble")]:
    train_metrics = calculate_metrics(model, train, target)
    test_metrics = calculate_metrics(model, test, target)

    # Append results in the required format
    results.append({
        "Approach": name,
        "Train MSE": train_metrics["MSE"],
        "Train MAPE": train_metrics["MAPE"],
        "Train R²": train_metrics["R²"],
        "Train MAE": train_metrics["MAE"],
        "Train RMSE": train_metrics["RMSE"],
        "Test MSE": test_metrics["MSE"],
        "Test MAPE": test_metrics["MAPE"],
        "Test R²": test_metrics["R²"],
        "Test MAE": test_metrics["MAE"],
        "Test RMSE": test_metrics["RMSE"]
    })

# Convert results into a DataFrame for display
results_df = pd.DataFrame(results, columns=[
    "Approach",
    "Train MSE", "Train MAPE", "Train R²", "Train MAE", "Train RMSE",
    "Test MSE", "Test MAPE", "Test R²", "Test MAE", "Test RMSE"
])

# Print the results in tabular format
print("\n--- Model Performance Metrics ---")
print(results_df.to_string(index=False))

# Shutdown H2O (uncomment the following line if needed)
# h2o.shutdown(prompt=False)
