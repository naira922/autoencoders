from tensorflow.keras import layers, models, backend as K

def build_encoder(image_size=28, channels=1, embedding_dim=64):
  
    encoder_input = layers.Input(
        shape=(image_size, image_size, channels), name="encoder_input"
    )

    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)

    shape_before_flattening = K.int_shape(x)[1:]  # for decoder

    x = layers.Flatten()(x)
    encoder_output = layers.Dense(embedding_dim, name="encoder_output")(x)

    encoder = models.Model(encoder_input, encoder_output, name="encoder")

    return encoder, shape_before_flattening

encoder, shape_before_flattening = build_encoder(image_size=28, channels=1, embedding_dim=64)
encoder.summary()