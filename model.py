from tensorflow.keras import layers, models

# Encoder
def build_encoder(latent_dim=64):
    encoder_input = layers.Input(shape=(28, 28, 1))  # Fashion-MNIST input shape
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = models.Model(encoder_input, latent, name="encoder")
    return encoder
