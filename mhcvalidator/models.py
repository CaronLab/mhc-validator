import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import backend as K

def get_model_with_lstm_peptide_encoding(ms_feature_length: int,
                                         max_pep_length: int = 15,
                                         dense_layer_node_factor: int = 2):
    pep_input = keras.Input(shape=(max_pep_length, 21), dtype="float32")
    pep_x = layers.Bidirectional(layers.LSTM(max_pep_length, return_sequences=False))(pep_input)
    pep_x = layers.Dropout(0.5)(pep_x)
    pep_x = layers.Dense(6, activation="relu")(pep_x)
    pep_x = layers.Dropout(0.5)(pep_x)
    pep_out_flat = layers.Flatten()(pep_x)

    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    ms_plus_pep = layers.concatenate([ms_feature_input, pep_out_flat])
    n_nodes = int(int(ms_plus_pep.shape[1]) * dense_layer_node_factor)

    x = layers.Dense(n_nodes, activation='relu')(ms_plus_pep)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(n_nodes, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(n_nodes, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


def get_model_with_peptide_encoding(ms_feature_length: int, max_pep_length: int = 15, dropout: float = 0.6,
                                    hidden_layers: int = 3):
    pep_input = keras.Input(shape=(max_pep_length, 21))
    p = layers.BatchNormalization(input_shape=(max_pep_length, 21))(pep_input)
    p = layers.Conv1D(12, 4, padding="valid", activation=tf.nn.tanh)(p)
    p = layers.MaxPool1D()(p)
    p = layers.Dropout(dropout)(p)
    p = layers.Flatten()(p)
    pep_out_flat = layers.Dense(4, activation=tf.nn.relu)(p)

    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(ms_feature_input)
    x = layers.concatenate([x, pep_out_flat])
    n_nodes = int(int(x.shape[1]) * 3)
    for i in range(hidden_layers):
        x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


def get_model_without_peptide_encoding(ms_feature_length: int, max_pep_length: int, dropout: float = 0.6,
                                       hidden_layers: int = 3):
    n_nodes = ms_feature_length * 3
    input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(input)
    for i in range(hidden_layers):
        x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=input, outputs=output)
    return model
