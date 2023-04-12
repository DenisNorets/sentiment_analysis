import tensorflow as tf
from metrics import f1_score


def bilstm_model():
    inputs = tf.keras.layers.Input((None, 768), dtype=tf.float32)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM((256), dropout=0.3, recurrent_dropout=0.3))(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(7, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    return model
