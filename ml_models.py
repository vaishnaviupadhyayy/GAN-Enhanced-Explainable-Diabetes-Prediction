
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# Load datasets
train_data = pd.read_csv("train_data.csv")
augmented_data = pd.read_csv("augmented_diabetes.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and target
X_test = test_data.drop("Outcome", axis=1)
y_test = test_data["Outcome"]

# Define training datasets
experiments = {
    "Original Data": train_data,
    "GAN Augmented Data": augmented_data
}

# Store results
results = []
reports = []

# Train and evaluate models
for experiment_name, dataset in experiments.items():

    print("\n===================================")
    print("Experiment:", experiment_name)
    print("Training Records:", len(dataset))
    print("===================================")

    X_train = dataset.drop("Outcome", axis=1)
    y_train = dataset["Outcome"]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        # Train model
        model.fit(X_train, y_train)

        # Predict test results
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) else 0

        # Save metrics
        results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1 Score": f1
        })

        # Generate classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Diabetes", "Diabetes"],
            zero_division=0
        )

        reports.append(
            "\n" + "=" * 55 +
            "\nExperiment: " + experiment_name +
            "\nModel: " + model_name +
            "\n" + "=" * 55 +
            "\n" + report
        )

        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Precision:", round(precision * 100, 2), "%")
        print("Recall:", round(recall * 100, 2), "%")
        print("Specificity:", round(specificity * 100, 2), "%")
        print("F1 Score:", round(f1 * 100, 2), "%")

        print("\nClassification Report:")
        print(report)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"]
        )

        plt.title(experiment_name + " - " + model_name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        filename = (
            experiment_name.replace(" ", "_") +
            "_" + model_name.replace(" ", "_") +
            "_confusion_matrix.png"
        )

        plt.savefig(filename)
        plt.close()

        # Save augmented Random Forest model
        if model_name == "Random Forest":

            if experiment_name == "Original Data":
                joblib.dump(model, "random_forest_model.pkl")
                print("\nOriginal Random Forest model saved!")

            else:
                joblib.dump(model, "random_forest_augmented_model.pkl")
                print("GAN Augmented Random Forest model saved!")

# Save comparison results
comparison = pd.DataFrame(results)

comparison.to_csv("model_comparison.csv", index=False)

# Save all classification reports
with open("classification_reports.txt", "w") as file:
    file.write("\n".join(reports))

# Display comparison
print("\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

print(comparison.round(4).to_string(index=False))

print("\nModel comparison saved as model_comparison.csv")
print("Classification reports saved as classification_reports.txt")
print("Confusion matrices saved successfully.")
print("All model training and evaluation completed!")
