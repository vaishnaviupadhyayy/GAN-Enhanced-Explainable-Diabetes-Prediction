
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import shap
import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


# =========================================
# 1. INITIALIZE FLASK APPLICATION
# =========================================

app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

os.makedirs("static", exist_ok=True)


# =========================================
# 2. LOAD MODEL AND TRAINING DATA
# =========================================

print("\nLoading Random Forest model...")

model = joblib.load("random_forest_model.pkl")

train_data = pd.read_csv("train_data.csv")

X_train = train_data.drop("Outcome", axis=1)

feature_names = list(X_train.columns)

print("Random Forest model loaded successfully!")


# =========================================
# 3. INITIALIZE SHAP EXPLAINER
# =========================================

print("\nInitializing SHAP...")

shap_explainer = shap.TreeExplainer(model)

print("SHAP initialized successfully!")


# =========================================
# 4. INITIALIZE LIME EXPLAINER
# =========================================

print("\nInitializing LIME...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["No Diabetes", "Diabetes"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

print("LIME initialized successfully!")


# =========================================
# 5. HOME ROUTE
# =========================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================
# 6. HEALTH CHECK ROUTE
# =========================================

@app.route("/health")
def health():

    return jsonify({
        "status": "success",
        "message": "Diabetes Risk Assessment System is running",
        "model": "Random Forest",
        "explainability": ["SHAP", "LIME"]
    })


# =========================================
# 7. GENERATE INDIVIDUAL SHAP WATERFALL
# =========================================

def generate_shap_waterfall(shap_values, base_value, input_values):

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=input_values,
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        explanation,
        max_display=8,
        show=False
    )

    plt.savefig(
        "static/shap_patient_waterfall.png",
        bbox_inches="tight",
        dpi=180
    )

    plt.close("all")


# =========================================
# 8. GENERATE INDIVIDUAL LIME PLOT
# =========================================

def generate_lime_plot(lime_result):

    figure = lime_result.as_pyplot_figure(label=1)

    figure.set_size_inches(10, 6)

    figure.savefig(
        "static/lime_patient_explanation.png",
        bbox_inches="tight",
        dpi=180
    )

    plt.close(figure)


# =========================================
# 9. PREDICTION ROUTE
# =========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive input measurements
        input_data = request.get_json(silent=True)

        if not isinstance(input_data, dict):

            return jsonify({
                "error": "Please provide measurements as JSON."
            }), 400

        # Check all eight features
        missing_features = [
            feature for feature in feature_names
            if feature not in input_data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing input measurements.",
                "missing_features": missing_features
            }), 400

        # Convert inputs into numerical values
        values = []

        for feature in feature_names:

            try:

                value = float(input_data[feature])

            except (TypeError, ValueError):

                return jsonify({
                    "error": f"{feature} must be a valid number."
                }), 400

            if not np.isfinite(value) or value < 0:

                return jsonify({
                    "error": f"{feature} must be a finite, non-negative number."
                }), 400

            values.append(value)

        if values[6] > 2:

            return jsonify({
                "error": "DiabetesPedigreeFunction is outside the supported input range."
            }), 400

        if values[7] <= 0:

            return jsonify({
                "error": "Age must be greater than zero."
            }), 400

        if values[0] != int(values[0]) or values[7] != int(values[7]):

            return jsonify({
                "error": "Pregnancies and Age must be whole numbers."
            }), 400

        # Create input DataFrame
        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )


        # =================================
        # 10. RANDOM FOREST PREDICTION
        # =================================

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]

        diabetes_probability = float(probabilities[1])

        prediction_label = (
            "Diabetes class"
            if prediction == 1
            else "No Diabetes class"
        )


        # =================================
        # 11. GENERATE SHAP EXPLANATION
        # =================================

        shap_result = shap_explainer(input_df)

        shap_array = shap_result.values

        if shap_array.ndim == 3:

            patient_shap = shap_array[0, :, 1]

        else:

            patient_shap = shap_array[0]

        # Get base value for Diabetes class
        base_values = np.asarray(shap_result.base_values)

        if base_values.ndim >= 2:

            if base_values.shape[-1] > 1:
                base_value = float(base_values[0, 1])
            else:
                base_value = float(base_values[0, 0])

        elif base_values.ndim == 1:

            if base_values.size > 1:
                base_value = float(base_values[1])
            else:
                base_value = float(base_values[0])

        else:

            base_value = float(base_values)

        shap_contributions = []

        for i, feature in enumerate(feature_names):

            contribution = float(patient_shap[i])

            shap_contributions.append({

                "feature": feature,

                "value": float(values[i]),

                "contribution": contribution,

                "direction": (
                    "Increases model output for Diabetes"
                    if contribution > 0
                    else "Decreases model output for Diabetes"
                )

            })

        shap_contributions.sort(
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )

        # Generate updated waterfall plot
        generate_shap_waterfall(
            patient_shap,
            base_value,
            values
        )


        # =================================
        # 12. GENERATE LIME EXPLANATION
        # =================================

        def predict_probability(data):

            data_frame = pd.DataFrame(
                data,
                columns=feature_names
            )

            return model.predict_proba(data_frame)

        lime_result = lime_explainer.explain_instance(

            data_row=input_df.iloc[0].values,

            predict_fn=predict_probability,

            num_features=8,

            num_samples=5000,

            labels=(1,)

        )

        lime_values = lime_result.as_list(label=1)

        lime_contributions = []

        for condition, contribution in lime_values:

            contribution = float(contribution)

            lime_contributions.append({

                "condition": condition,

                "contribution": contribution,

                "direction": (
                    "Supports Diabetes prediction"
                    if contribution > 0
                    else "Supports No Diabetes prediction"
                )

            })

        # Generate updated LIME plot
        generate_lime_plot(lime_result)


        # =================================
        # 13. CREATE IMAGE VERSION
        # =================================

        plot_version = str(time.time_ns())

        shap_waterfall_url = (
            "/static/shap_patient_waterfall.png?v=" + plot_version
        )

        lime_plot_url = (
            "/static/lime_patient_explanation.png?v=" + plot_version
        )


        # =================================
        # 14. RETURN COMPLETE RESULT
        # =================================

        result = {

            "status": "success",

            "prediction": prediction_label,

            "prediction_class": prediction,

            "diabetes_probability": round(
                diabetes_probability * 100, 2
            ),

            "input_measurements": {

                feature: float(values[i])

                for i, feature in enumerate(feature_names)

            },

            "shap_explanation": shap_contributions,

            "lime_explanation": lime_contributions,

            "shap_waterfall_url": shap_waterfall_url,

            "lime_plot_url": lime_plot_url,

            "disclaimer": (
                "This is a research decision-support prototype, "
                "not a clinically validated diagnostic system. "
                "The Pima Indians Diabetes dataset contains female "
                "participants only. Results should not replace "
                "professional medical assessment or clinical testing."
            )

        }

        return jsonify(result)

    except Exception as error:

        print("\nPrediction error:", str(error))

        import traceback
        traceback.print_exc()

        return jsonify({

            "error": (
                "An error occurred while generating the prediction. "
                "Check the Flask Terminal for details."
            )

        }), 500


# =========================================
# 15. RUN APPLICATION
# =========================================

if __name__ == "__main__":

    print("\n====================================")
    print("DIABETES RISK ASSESSMENT SYSTEM")
    print("====================================")

    print("Starting Flask server...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
