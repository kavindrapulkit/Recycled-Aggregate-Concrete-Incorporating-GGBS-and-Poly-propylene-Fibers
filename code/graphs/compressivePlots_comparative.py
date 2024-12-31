# Combine predictions and actual values into a single dictionary
combined_data = {
    "Measured": np.concatenate([test_actual] * 3),  # Repeat actual values for each model
    "Predicted": np.concatenate([
        test_drf_pred,
        test_gbm_pred,
        test_ensemble_pred
    ]),
    "Model": (["Distributed Random Forest"] * len(test_drf_pred) +
              ["Gradient Boosting Machine"] * len(test_gbm_pred) +
              ["Stacked Ensemble"] * len(test_ensemble_pred))
}

# Create the plot
plt.figure(figsize=(10, 8))

# Plot each model's data with different styles
models = ["Distributed Random Forest", "Gradient Boosting Machine", "Stacked Ensemble"]
colors = ["blue", "green", "orange"]
markers = ["o", "s", "d"]  # Different markers for each model

for model, color, marker in zip(models, colors, markers):
    model_data = [(combined_data["Measured"][i], combined_data["Predicted"][i])
                  for i in range(len(combined_data["Model"])) if combined_data["Model"][i] == model]
    measured = [point[0] for point in model_data]
    predicted = [point[1] for point in model_data]
    plt.scatter(measured, predicted, label=model, color=color, marker=marker, alpha=0.6, edgecolors="k", linewidth=0.5)

# Add ideal fit line
plt.plot([min(test_actual), max(test_actual)],
         [min(test_actual), max(test_actual)],
         color="red", linestyle="--", label="Ideal Fit")

# Add labels, title, and legend
plt.xlabel("Measured Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("Predicted vs Measured Compressive Strength (All Models)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
# Combine predictions and actual values into a single dictionary
combined_data = {
    "Measured": np.concatenate([test_actual] * 3),  # Repeat actual values for each model
    "Predicted": np.concatenate([
        test_drf_pred,
        test_gbm_pred,
        test_ensemble_pred
    ]),
    "Model": (["Distributed Random Forest"] * len(test_drf_pred) +
              ["Gradient Boosting Machine"] * len(test_gbm_pred) +
              ["Stacked Ensemble"] * len(test_ensemble_pred))
}

# Create the plot
plt.figure(figsize=(10, 8))

# Plot each model's data with different styles
models = ["Distributed Random Forest", "Gradient Boosting Machine", "Stacked Ensemble"]
colors = ["blue", "green", "orange"]
markers = ["o", "s", "d"]  # Different markers for each model

for model, color, marker in zip(models, colors, markers):
    model_data = [(combined_data["Measured"][i], combined_data["Predicted"][i])
                  for i in range(len(combined_data["Model"])) if combined_data["Model"][i] == model]
    measured = [point[0] for point in model_data]
    predicted = [point[1] for point in model_data]
    plt.scatter(measured, predicted, label=model, color=color, marker=marker, alpha=0.6, edgecolors="k", linewidth=0.5)

# Add ideal fit line
plt.plot([min(test_actual), max(test_actual)],
         [min(test_actual), max(test_actual)],
         color="red", linestyle="--", label="Ideal Fit")

# Add labels, title, and legend
plt.xlabel("Measured Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("Predicted vs Measured Compressive Strength (All Models)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
