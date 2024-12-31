# Combine predictions and actual values into a single dictionary
combined_data = {
    "Measured": np.concatenate([split_test_actual] * 3),  # Repeat actuals for each model
    "Predicted": np.concatenate([
        split_test_drf_pred,
        split_test_gbm_pred,
        split_test_ensemble_pred
    ]),
    "Model": (["Split DRF"] * len(split_test_drf_pred) +
              ["Split GBM"] * len(split_test_gbm_pred) +
              ["Split Stacked Ensemble"] * len(split_test_ensemble_pred))
}

# Create the plot
plt.figure(figsize=(10, 8))

# Plot each model's data with different styles
models = ["Split DRF", "Split GBM", "Split Stacked Ensemble"]
colors = ["blue", "green", "orange"]
markers = ["o", "s", "d"]  # Different markers for each model

for model, color, marker in zip(models, colors, markers):
    model_data = [(combined_data["Measured"][i], combined_data["Predicted"][i])
                  for i in range(len(combined_data["Model"])) if combined_data["Model"][i] == model]
    measured = [point[0] for point in model_data]
    predicted = [point[1] for point in model_data]
    plt.scatter(measured, predicted, label=model, color=color, marker=marker, alpha=0.6, edgecolors="k", linewidth=0.5)

# Add ideal fit line
plt.plot([min(split_test_actual), max(split_test_actual)],
         [min(split_test_actual), max(split_test_actual)],
         color="red", linestyle="--", label="Ideal Fit")

# Add labels, title, and legend
plt.xlabel("Measured Split Tensile Strength")
plt.ylabel("Predicted Split Tensile Strength")
plt.title("Predicted vs Measured Split Tensile Strength (All Models)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
