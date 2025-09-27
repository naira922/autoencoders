from data import load_fashion_mnist
from model import build_encoder

# Load data
x_train, x_test = load_fashion_mnist()

# Build encoder only
encoder = build_encoder(latent_dim=64)

# Compile encoder (optional for feature extraction)
encoder.compile(optimizer="adam", loss="mse")

# Summary
encoder.summary()

# Example: encode first 5 test images
latent_vectors = encoder.predict(x_test[:5])
print("Latent space representation shape:", latent_vectors.shape)
