
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
data = pd.read_csv("clean_diabetes.csv")

# Separate input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and target
train_data = X_train.copy()
train_data["Outcome"] = y_train

test_data = X_test.copy()
test_data["Outcome"] = y_test

# Save training and testing datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Display results
print("Original Dataset Shape:", data.shape)

print("\nTraining Dataset Shape:", train_data.shape)
print("Testing Dataset Shape:", test_data.shape)

print("\nTraining Outcome Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Outcome Distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining and Testing datasets saved successfully.")
