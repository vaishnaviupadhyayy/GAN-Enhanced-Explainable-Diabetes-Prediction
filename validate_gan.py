
import pandas as pd

# Load original and augmented datasets
train_data = pd.read_csv("train_data.csv")
augmented_data = pd.read_csv("augmented_diabetes.csv")

# Separate real and synthetic diabetic records
real_diabetic = train_data[train_data["Outcome"] == 1]

synthetic_data = augmented_data.iloc[len(train_data):].copy()
synthetic_data = synthetic_data.drop("Outcome", axis=1)

# Display synthetic data information
print("Real Diabetic Records:", len(real_diabetic))
print("Synthetic Records:", len(synthetic_data))

# Compare feature ranges
print("\nFeature Range Comparison:")

for column in synthetic_data.columns:

    real_min = real_diabetic[column].min()
    real_max = real_diabetic[column].max()

    synthetic_min = synthetic_data[column].min()
    synthetic_max = synthetic_data[column].max()

    outside_range = (
        (synthetic_data[column] < real_min) |
        (synthetic_data[column] > real_max)
    ).sum()

    print("\n", column)
    print("Real Range:", real_min, "to", real_max)
    print("Synthetic Range:", round(synthetic_min, 2),
          "to", round(synthetic_max, 2))
    print("Values Outside Real Range:", outside_range)

# Compare average values
print("\nReal Diabetic Feature Means:")
print(real_diabetic.drop("Outcome", axis=1).mean().round(2))

print("\nSynthetic Feature Means:")
print(synthetic_data.mean().round(2))

print("\nGAN Data Validation Completed!")
