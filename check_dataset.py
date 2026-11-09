
import pandas as pd

# Load the diabetes dataset
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

# Display dataset information
print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nOutcome Distribution:")
print(data["Outcome"].value_counts())

print("\nZero Values in Medical Columns:")
for column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    print(column, ":", (data[column] == 0).sum())
