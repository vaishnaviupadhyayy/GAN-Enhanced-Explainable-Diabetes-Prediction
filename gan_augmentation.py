
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# Load training dataset
train_data = pd.read_csv("train_data.csv")

# Select only diabetic records
diabetic_data = train_data[train_data["Outcome"] == 1]

X_diabetic = diabetic_data.drop("Outcome", axis=1)

print("Original Training Records:", len(train_data))
print("Diabetic Training Records:", len(X_diabetic))

# Normalize diabetic records
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_diabetic)

# GAN parameters
noise_dim = 100
feature_count = X_scaled.shape[1]
epochs = 1000
batch_size = 32
synthetic_samples = 200

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(noise_dim,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(feature_count, activation="sigmoid")
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(feature_count,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

# GAN model
discriminator.trainable = False

gan_input = tf.keras.layers.Input(shape=(noise_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)

gan = tf.keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

# Training GAN
for epoch in range(epochs):

    # Select real samples
    indices = np.random.randint(0, X_scaled.shape[0], batch_size)
    real_data = X_scaled[indices]

    # Generate fake samples
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_data = generator.predict(noise, verbose=0)

    # Train discriminator
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    discriminator.trainable = True

    discriminator.train_on_batch(real_data, real_labels)
    discriminator.train_on_batch(fake_data, fake_labels)

    # Train generator
    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    generator_labels = np.ones((batch_size, 1))

    gan.train_on_batch(noise, generator_labels)

    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch + 1, "completed")

# Generate synthetic diabetic records
noise = np.random.normal(0, 1, (synthetic_samples, noise_dim))

synthetic_scaled = generator.predict(noise, verbose=0)

synthetic_data = scaler.inverse_transform(synthetic_scaled)

synthetic_df = pd.DataFrame(
    synthetic_data,
    columns=X_diabetic.columns
)

# Convert integer-valued features
synthetic_df["Pregnancies"] = synthetic_df["Pregnancies"].round().astype(int)
synthetic_df["Age"] = synthetic_df["Age"].round().astype(int)

# Prevent negative values
for column in X_diabetic.columns:
    synthetic_df[column] = synthetic_df[column].clip(lower=0)

# Assign diabetic outcome
synthetic_df["Outcome"] = 1

# Combine original training data and synthetic data
augmented_data = pd.concat(
    [train_data, synthetic_df],
    ignore_index=True
)

# Save augmented dataset
augmented_data.to_csv("augmented_diabetes.csv", index=False)

print("\nGAN Training Completed!")
print("Synthetic Records Generated:", len(synthetic_df))
print("Augmented Dataset Shape:", augmented_data.shape)

print("\nFinal Outcome Distribution:")
print(augmented_data["Outcome"].value_counts())
