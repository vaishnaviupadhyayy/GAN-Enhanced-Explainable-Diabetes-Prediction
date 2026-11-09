import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. LOAD MODEL AND DATA
# -----------------------------------------

print("\nLoading Random Forest model...")

model = joblib.load("random_forest_model.pkl")

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data.drop("Outcome", axis=1)
X_test = test_data.drop("Outcome", axis=1)

# Keep the same feature order used during model training
feature_names = list(X_train.columns)
X_test = X_test[feature_names]

print("Model loaded successfully!")
print("Training records:", len(X_train))
print("Test records:", len(X_test))


# -----------------------------------------
# 2. CREATE SHAP EXPLAINER
# -----------------------------------------

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_result = explainer(X_test)

# For binary classification, select class 1 (Diabetes)
if shap_result.values.ndim == 3:

    shap_values = shap_result.values[:, :, 1]

    base_values = np.asarray(shap_result.base_values)

    if base_values.ndim == 2:
        base_values = base_values[:, 1]

    elif base_values.ndim == 1 and len(base_values) == 2:
        base_values = base_values[1]

else:

    shap_values = shap_result.values
    base_values = shap_result.base_values

# Create a SHAP Explanation for Diabetes class
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=base_values,
    data=X_test.values,
    feature_names=feature_names
)

print("SHAP values calculated successfully!")


# -----------------------------------------
# 3. GLOBAL FEATURE IMPORTANCE
# -----------------------------------------

print("\nCalculating global feature importance...")

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute SHAP Value": importance
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP Value",
    ascending=False
)

importance_df.to_csv(
    "shap_global_importance.csv",
    index=False
)

print("\nGLOBAL FEATURE IMPORTANCE")
print(importance_df.to_string(index=False))


# -----------------------------------------
# 4. GLOBAL SHAP BAR PLOT
# -----------------------------------------

print("\nGenerating global SHAP bar plot...")

shap.plots.bar(
    shap_explanation,
    max_display=8,
    show=False
)

plt.title("Global SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    "shap_global_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# 5. SHAP BEESWARM PLOT
# -----------------------------------------

print("Generating SHAP beeswarm plot...")

shap.plots.beeswarm(
    shap_explanation,
    max_display=8,
    show=False
)

plt.title("SHAP Summary: Feature Impact on Diabetes Risk Estimate")
plt.tight_layout()

plt.savefig(
    "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# 6. INDIVIDUAL PATIENT EXPLANATION
# -----------------------------------------

print("\nGenerating individual SHAP explanation...")

# Select first record from the test dataset
patient_index = 0

patient_data = X_test.iloc[patient_index]

patient_shap_values = shap_values[patient_index]

patient_base_value = (
    base_values[patient_index]
    if np.ndim(base_values) > 0
    else base_values
)

patient_explanation = shap.Explanation(
    values=patient_shap_values,
    base_values=patient_base_value,
    data=patient_data.values,
    feature_names=feature_names
)

# Generate waterfall plot
shap.plots.waterfall(
    patient_explanation,
    max_display=8,
    show=False
)

plt.title("Individual SHAP Explanation - Test Record 1")
plt.tight_layout()

plt.savefig(
    "shap_patient_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# 7. INDIVIDUAL CONTRIBUTION TABLE
# -----------------------------------------

contribution_df = pd.DataFrame({
    "Feature": feature_names,
    "Feature Value": patient_data.values,
    "SHAP Value": patient_shap_values
})

contribution_df["Direction"] = np.where(
    contribution_df["SHAP Value"] > 0,
    "Increases model output for Diabetes",
    "Decreases model output for Diabetes"
)

contribution_df["Absolute SHAP Value"] = np.abs(
    contribution_df["SHAP Value"]
)

contribution_df = contribution_df.sort_values(
    by="Absolute SHAP Value",
    ascending=False
)

contribution_df.to_csv(
    "shap_patient_contributions.csv",
    index=False
)


# -----------------------------------------
# 8. DISPLAY MODEL PREDICTION
# -----------------------------------------

prediction = model.predict(
    X_test.iloc[[patient_index]]
)[0]

probability = model.predict_proba(
    X_test.iloc[[patient_index]]
)[0][1]

print("\n====================================")
print("INDIVIDUAL MODEL EXPLANATION")
print("====================================")

print("Test Record:", patient_index + 1)

print(
    "Model Prediction:",
    "Diabetes" if prediction == 1 else "No Diabetes"
)

print(
    "Model-Estimated Diabetes Probability:",
    round(probability * 100, 2), "%"
)

print("\nFEATURE CONTRIBUTIONS")

print(
    contribution_df[
        ["Feature", "Feature Value", "SHAP Value", "Direction"]
    ].to_string(index=False)
)


# -----------------------------------------
# 9. COMPLETION
# -----------------------------------------

print("\n====================================")
print("SHAP ANALYSIS COMPLETED")
print("====================================")

print("Generated files:")

print("1. shap_global_importance.csv")
print("2. shap_global_bar.png")
print("3. shap_beeswarm.png")
print("4. shap_patient_waterfall.png")
print("5. shap_patient_contributions.csv")

print("\nAll SHAP explanations saved successfully!")