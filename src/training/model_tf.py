import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(n_mels: int, num_classes: int, dropout: float = 0.3, lr: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_mels, None, 1), name="logmel_input")

    def conv_block(x, filters, pool=True):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    x = conv_block(inputs, 32, pool=True)
    x = conv_block(x, 64, pool=True)
    x = conv_block(x, 128, pool=True)
    x = layers.Dropout(dropout)(x)

    x = conv_block(x, 256, pool=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="heart_sound_cnn")

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
    # Remove AUC here; we'll compute ROC/AUC after training
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model