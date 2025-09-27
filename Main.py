import matplotlib.pyplot as plt
from data import load_fashion_mnist
from model import build_encoder, build_decoder, build_autoencoder

# -----------------------------
# Load data
# -----------------------------
x_train, x_test = load_fashion_mnist()

# -----------------------------
# Build models
# -----------------------------
encoder, shape_before_flattening = build_encoder()
decoder = build_decoder(shape_before_flattening)
autoencoder = build_autoencoder(encoder, decoder)

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# -----------------------------
# Train
# -----------------------------
history = autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

# -----------------------------
# Save models
# -----------------------------
autoencoder.save("autoencoder.keras")
encoder.save("encoder.keras")
decoder.save("decoder.keras")
print("✅ Models saved successfully!")

# -----------------------------
# Visualization: Loss Curve
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
print("📊 Loss curve saved as loss_curve.png")

# -----------------------------
# Visualization: Reconstructions
# -----------------------------
def save_reconstructions(model, data, n=10, filename="reconstructions.png"):
    sample_imgs = data[:n]
    reconstructed = model.predict(sample_imgs)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_imgs[i].squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.savefig(filename)
    print(f"Reconstructions saved as {filename}")

save_reconstructions(autoencoder, x_test)