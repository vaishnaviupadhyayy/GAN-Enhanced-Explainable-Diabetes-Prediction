import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


# -----------------------------------------
# 1. LOAD MODEL AND DATA
# -----------------------------------------

print("\nLoading Random Forest model...")

model = joblib.load("random_forest_model.pkl")

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data.drop("Outcome", axis=1)
X_test = test_data.drop("Outcome", axis=1)

feature_names = list(X_train.columns)

X_train = X_train[feature_names]
X_test = X_test[feature_names]

print("Model and datasets loaded successfully!")


# -----------------------------------------
# 2. CREATE LIME EXPLAINER
# -----------------------------------------

print("\nCreating LIME explainer...")

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["No Diabetes", "Diabetes"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


# -----------------------------------------
# 3. SELECT TEST RECORD
# -----------------------------------------

patient_index = 0

patient_data = X_test.iloc[patient_index]

print("\nSelected Test Record:", patient_index + 1)


# -----------------------------------------
# 4. GENERATE LIME EXPLANATION
# -----------------------------------------

print("\nGenerating LIME explanation...")


def predict_probability(data):

    data_frame = pd.DataFrame(
        data,
        columns=feature_names
    )

    return model.predict_proba(data_frame)


lime_explanation = explainer.explain_instance(
    data_row=patient_data.values,
    predict_fn=predict_probability,
    num_features=8,
    num_samples=5000,
    labels=(1,)
)


# -----------------------------------------
# 5. EXTRACT FEATURE CONTRIBUTIONS
# -----------------------------------------

lime_values = lime_explanation.as_list(label=1)

lime_df = pd.DataFrame(
    lime_values,
    columns=["Feature Condition", "LIME Contribution"]
)

lime_df["Direction"] = np.where(
    lime_df["LIME Contribution"] > 0,
    "Supports Diabetes prediction",
    "Supports No Diabetes prediction"
)

lime_df["Absolute Contribution"] = np.abs(
    lime_df["LIME Contribution"]
)

lime_df = lime_df.sort_values(
    by="Absolute Contribution",
    ascending=False
)

lime_df.to_csv(
    "lime_patient_contributions.csv",
    index=False
)


# -----------------------------------------
# 6. GENERATE LIME CHART
# -----------------------------------------

print("\nGenerating LIME contribution chart...")

lime_explanation.as_pyplot_figure(label=1)

plt.title(
    "LIME Explanation - Individual Diabetes Prediction"
)

plt.tight_layout()

plt.savefig(
    "lime_patient_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# 7. SAVE HTML EXPLANATION
# -----------------------------------------

lime_explanation.save_to_file(
    "lime_patient_explanation.html"
)


# -----------------------------------------
# 8. DISPLAY PREDICTION
# -----------------------------------------

prediction = model.predict(
    X_test.iloc[[patient_index]]
)[0]

probability = model.predict_proba(
    X_test.iloc[[patient_index]]
)[0][1]

print("\n====================================")
print("LIME INDIVIDUAL EXPLANATION")
print("====================================")

print(
    "Model Prediction:",
    "Diabetes" if prediction == 1 else "No Diabetes"
)

print(
    "Model-Estimated Diabetes Probability:",
    round(probability * 100, 2),
    "%"
)

print("\nLIME FEATURE CONTRIBUTIONS")

print(lime_df.to_string(index=False))


# -----------------------------------------
# 9. COMPLETION
# -----------------------------------------

print("\n====================================")
print("LIME ANALYSIS COMPLETED")
print("====================================")

print("Generated files:")

print("1. lime_patient_contributions.csv")
print("2. lime_patient_explanation.png")
print("3. lime_patient_explanation.html")

print("\nAll LIME explanations saved successfully!")