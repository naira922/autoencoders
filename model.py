import numpy as np
from tensorflow.keras import layers, models, backend as K

# -----------------------------
# Encoder
# -----------------------------
def build_encoder(image_size=28, channels=1, embedding_dim=64):
    encoder_input = layers.Input(shape=(image_size, image_size, channels), name="encoder_input")

    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)

    shape_before_flattening = K.int_shape(x)[1:]  # needed for decoder

    x = layers.Flatten()(x)
    encoder_output = layers.Dense(embedding_dim, name="encoder_output")(x)

    encoder = models.Model(encoder_input, encoder_output, name="encoder")
    return encoder, shape_before_flattening


# -----------------------------
# Decoder
# -----------------------------
def build_decoder(shape_before_flattening, embedding_dim=64, channels=1):
    decoder_input = layers.Input(shape=(embedding_dim,), name="decoder_input")

    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)

    x = layers.Conv2D(channels, (3, 3), strides=1, activation="sigmoid", padding="same")(x)

    # crop 32x32 → 28x28
    decoder_output = layers.Cropping2D(((2, 2), (2, 2)))(x)

    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    return decoder


# -----------------------------
# Autoencoder
# -----------------------------
def build_autoencoder(encoder, decoder):
    autoencoder_input = encoder.input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    return autoencoder