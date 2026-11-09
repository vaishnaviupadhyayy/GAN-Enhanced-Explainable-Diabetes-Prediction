
import pandas as pd

# Load dataset
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

print("Original Dataset Shape:", data.shape)

# Columns where zero represents a missing measurement
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Count zero values before replacement
print("\nZero Values Before Preprocessing:")
for column in zero_columns:
    print(column, ":", (data[column] == 0).sum())

# Replace zero values with NaN
for column in zero_columns:
    data[column] = data[column].replace(0, float("nan"))

# Replace missing values with median
for column in zero_columns:
    median_value = data[column].median()
    data[column] = data[column].fillna(median_value)

# Check missing values after preprocessing
print("\nMissing Values After Preprocessing:")
print(data.isnull().sum())

# Save cleaned dataset
data.to_csv("clean_diabetes.csv", index=False)

print("\nCleaned Dataset Shape:", data.shape)
print("Cleaned dataset saved successfully as clean_diabetes.csv")
