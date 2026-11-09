
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

data = pd.read_csv("diabetes.csv", header=None, names=columns)

# Display basic information
print("Dataset Information:")
print(data.info())

print("\nStatistical Description:")
print(data.describe())

print("\nDuplicate Records:")
print(data.duplicated().sum())

print("\nDiabetes Outcome Distribution:")
print(data["Outcome"].value_counts())

# Plot outcome distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="Outcome", data=data)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])

plt.savefig("outcome_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
