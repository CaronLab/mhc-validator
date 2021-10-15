import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def get_model_with_peptide_encoding(ms_feature_length: int, max_pep_length: int = 15):
    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    x = layers.Dense(10, activation=tf.nn.relu)(ms_feature_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(20, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    ms_out_flat = layers.Flatten()(x)

    pep_input = keras.Input(shape=(max_pep_length, 21))
    p = layers.Conv1D(4, 5, padding="same", activation=tf.nn.relu)(pep_input)
    p = layers.MaxPool1D()(p)
    p = layers.Dense(4, activation=tf.nn.relu)(p)
    p = layers.Dropout(0.2)(p)
    pep_out_flat = layers.Flatten()(p)

    merge = layers.concatenate([ms_out_flat, pep_out_flat])
    merged_hidden1 = layers.Dense(7, activation=tf.nn.relu)(merge)
    m = layers.Dropout(0.2)(merged_hidden1)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(m)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


def get_bigger_model_with_peptide_encoding(ms_feature_length: int, pep_length: int = 15):
    pep_input = keras.Input(shape=(pep_length, 21))
    p = layers.Conv1D(4, 4, padding="same", activation=tf.nn.relu)(pep_input)
    p = layers.MaxPool1D()(p)
    p = layers.Dense(6, activation=tf.nn.relu)(p)
    #p = layers.Dropout(0.2)(p)
    #p = layers.Dense(1, activation=tf.nn.relu)(p)
    pep_out_flat = layers.Flatten()(p)

    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    ms_plus_pep = layers.concatenate([ms_feature_input, pep_out_flat])
    n_nodes = ms_plus_pep.shape[1] * 1
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(ms_plus_pep)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    #ms_out_flat = layers.Flatten()(x)

    #merge = layers.concatenate([ms_out_flat, pep_out_flat])
    #merged_hidden1 = layers.Dense(10, activation=tf.nn.relu)(merge)
    #m = layers.Dropout(0.2)(merge)
    #output = layers.Dense(1, activation=tf.nn.sigmoid)(m)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


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


def get_bigger_model_with_peptide_encoding2(ms_feature_length: int, max_pep_length: int = 15, dropout: float = 0.6):
    pep_input = keras.Input(shape=(max_pep_length, 21))
    p = layers.BatchNormalization(input_shape=(max_pep_length, 21))(pep_input)
    p = layers.Conv1D(12, 4, padding="valid", activation=tf.nn.tanh)(p)
    #p = layers.Conv1D(12, 4, padding="valid", activation=tf.nn.tanh)(p)
    #p = layers.Dropout(0.3)(p)
    p = layers.MaxPool1D()(p)
    p = layers.Dropout(dropout)(p)
    p = layers.Flatten()(p)
    pep_out_flat = layers.Dense(4, activation=tf.nn.relu)(p)
    #pep_out_flat = layers.Dropout(0.2)(pep_out_flat)

    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(ms_feature_input)
    ms_plus_pep = layers.concatenate([x, pep_out_flat])
    n_nodes = int(int(ms_plus_pep.shape[1]) * 3)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(ms_plus_pep)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    #ms_out_flat = layers.Flatten()(x)

    #merge = layers.concatenate([ms_out_flat, pep_out_flat])
    #merged_hidden1 = layers.Dense(10, activation=tf.nn.relu)(merge)
    #m = layers.Dropout(0.2)(merge)
    #output = layers.Dense(1, activation=tf.nn.sigmoid)(m)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


def get_model_without_peptide_encoding(ms_feature_length: int, max_pep_length: int, dropout: float = 0.6):
    n_nodes = ms_feature_length * 3
    input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(input)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=input, outputs=output)
    return model
