def build_decoder(shape_before_flattening, embedding_dim=64, channels=1):
    decoder_input = layers.Input(shape=(embedding_dim,), name="decoder_input")

    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)

    x = layers.Conv2D(channels, (3, 3), strides=1, activation="sigmoid", padding="same")(x)

    # Crop from (32,32,1) → (28,28,1)
    decoder_output = layers.Cropping2D(((2, 2), (2, 2)))(x)

    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    return decoder

def build_autoencoder(encoder, decoder):
    autoencoder_input = encoder.input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    return autoencoder

