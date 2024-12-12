import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import shap
from pathlib import Path


OUTPUTS_PATH = Path('outputs/model_report/')
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

# Load your trained model
# Ensure you have the CatBoost model already trained
# Replace 'model_path' with your model file path if loading from file
model = CatBoostRegressor()
model.load_model("models/catboost.cbm")

# Load test data (adjust file paths and column selection as needed)
test_data = pd.read_csv("outputs/predictions.csv")  # Replace with your test dataset file path
features = [col for col in test_data.columns if col != "deal_probability"]  # Adjust for your target column
# test_pool = Pool(test_data[features])

# 1. Compute Feature Importance (Model-based)
feature_importances = model.get_feature_importance(prettified=True)
feature_importance_df = pd.DataFrame(feature_importances)

# Save feature importance to CSV
feature_importance_df.to_csv(OUTPUTS_PATH / "feature_importance_model_based.csv", index=False)

# Plot Feature Importance (Model-based)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature Id"], feature_importance_df["Importances"], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Model-based)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUTS_PATH / "feature_importance_model_based.png")

# 2. Compute Loss-Function Change Importance
loss_change_importances = model.get_feature_importance(type='LossFunctionChange')
loss_importance_df = pd.DataFrame({
    "Feature": features,
    "LossFunctionChangeImportance": loss_change_importances
})

# Save loss-function change importance to CSV
loss_importance_df.to_csv(OUTPUTS_PATH / "feature_importance_loss_change.csv", index=False)

# Plot Loss-Function Change Importance
plt.figure(figsize=(10, 6))
plt.barh(loss_importance_df["Feature"], loss_importance_df["LossFunctionChangeImportance"], color='salmon')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Loss Function Change)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUTS_PATH / "feature_importance_loss_change.png")

# 3. Compute SHAP Values
shap_values = model.get_feature_importance(data=test_pool, type="ShapValues")[:, :-1]  # Exclude last column (base value)

# # Create SHAP Summary Plot
# shap.summary_plot(shap_values, test_data[features], show=False)
# plt.tight_layout()
# plt.savefig("shap_summary_plot.png")

# # Create SHAP Dependence Plot for the most important feature (e.g., 'price')
# most_important_feature = feature_importance_df.loc[0, "Feature Id"]  # Adjust based on ranking
# shap.dependence_plot(most_important_feature, shap_values, test_data[features], show=False)
# plt.tight_layout()
# plt.savefig(f"shap_dependence_plot_{most_important_feature}.png")

# # Save SHAP values to CSV for further analysis
# shap_df = pd.DataFrame(shap_values, columns=features)
# shap_df.to_csv("shap_values.csv", index=False)

print("Feature importance analysis completed. Results saved as CSV and PNG.")
