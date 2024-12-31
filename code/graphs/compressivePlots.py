import matplotlib.pyplot as plt
import numpy as np

# Restart H2O
h2o.init()

# Reload the dataset (if needed, ensure you use the same preprocessing steps as before)
h2o_data = h2o.H2OFrame(data)

# Split data again
train, test = h2o_data.split_frame(ratios=[0.8], seed=42)

# Generate predictions for plotting
train_drf_pred = best_drf.predict(train).as_data_frame()["predict"].values
train_gbm_pred = best_gbm.predict(train).as_data_frame()["predict"].values
train_ensemble_pred = ensemble.predict(train).as_data_frame()["predict"].values

test_drf_pred = best_drf.predict(test).as_data_frame()["predict"].values
test_gbm_pred = best_gbm.predict(test).as_data_frame()["predict"].values
test_ensemble_pred = ensemble.predict(test).as_data_frame()["predict"].values

# Extract actual values
train_actual = train.as_data_frame()[target].values
test_actual = test.as_data_frame()[target].values

# Function to plot actual vs predicted
def plot_actual_vs_predicted(actual, predicted, model_name, title_suffix="Training"):
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color="red", linestyle="--", label="Ideal Fit")
    plt.title(f"{model_name} - Actual vs Predicted ({title_suffix})")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# Training plots
plot_actual_vs_predicted(train_actual, train_drf_pred, "Distributed Random Forest", title_suffix="Training")
plot_actual_vs_predicted(train_actual, train_gbm_pred, "Gradient Boosting Machine", title_suffix="Training")
plot_actual_vs_predicted(train_actual, train_ensemble_pred, "Stacked Ensemble", title_suffix="Training")

# Testing plots
plot_actual_vs_predicted(test_actual, test_drf_pred, "Distributed Random Forest", title_suffix="Testing")
plot_actual_vs_predicted(test_actual, test_gbm_pred, "Gradient Boosting Machine", title_suffix="Testing")
plot_actual_vs_predicted(test_actual, test_ensemble_pred, "Stacked Ensemble", title_suffix="Testing")
