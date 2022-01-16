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


def peptide_sequence_encoding():
    pep_input = keras.Input(shape=(15, 21))
    p = layers.BatchNormalization(input_shape=(15, 21))(pep_input)
    p = layers.Conv1D(12, 4, padding="valid", activation=tf.nn.tanh)(p) # this should perhaps be 18, not 12. there are up to three anchor sites and 6 alleles
    p = layers.MaxPool1D()(p)
    p = layers.Flatten()(p)
    p = layers.Dropout(0.5)(p)
    out = layers.Dense(6)(p)
    model = keras.Model(inputs=pep_input, outputs=out)
    return model


def peptide_sequence_label_prediction():
    input = keras.Input(shape=(6,))
    p = K.relu(input)
    out = layers.Dense(1, activation=tf.nn.sigmoid)(p)
    model = keras.Model(inputs=input, outputs=out)
    return model


def peptide_sequence_encoder(dropout: float = 0.6, max_pep_length: int = 15, encoding_size: int = 3,
                             dense_layers: int = 2):
    pep_input = keras.Input(shape=(max_pep_length, 21))
    p = layers.BatchNormalization(input_shape=(max_pep_length, 21))(pep_input)
    p = layers.Conv1D(12, 4, padding="valid", activation=tf.nn.tanh)(p)
    #p = layers.MaxPool1D()(p)
    p = layers.Dropout(dropout)(p)
    p = layers.Flatten()(p)
    p = layers.Dense(max_pep_length*2, activation='relu')(p)
    p = layers.Dropout(dropout)(p)
    p = layers.Dense(encoding_size, name='encoded_peptides', activation='relu')(p)
    #p = layers.Dropout(dropout)(p)
    out = layers.Dense(1, activation=tf.nn.sigmoid)(p)
    model = keras.Model(inputs=pep_input, outputs=out)
    return model


def peptide_sequence_autoencoder(dropout: float = 0.6, max_pep_length: int = 15, encoding_size: int = 3):
    pep_input = keras.Input(shape=(max_pep_length, 21))
    #p = layers.BatchNormalization(input_shape=(max_pep_length, 21))(pep_input)
    p = layers.Conv1D(4, 4, padding="valid", activation='relu')(pep_input)
    encoded = layers.MaxPool1D(3, name='encoded_peptides')(p)

    x = layers.UpSampling1D(3)(encoded)
    x = layers.Conv1DTranspose(4, 4, activation='relu', padding='valid')(x)
    decoded = layers.Conv1DTranspose(21, 1, activation='sigmoid', padding='valid')(x)

    autoencoder = keras.Model(pep_input, decoded)

    return autoencoder


def get_model_with_peptide_encoding(ms_feature_length: int, max_pep_length: int = 15, dropout: float = 0.6,
                                    hidden_layers_after_convolutions: int = 2, convolutional_layers: int = 1,
                                    filter_size: int = 4, n_filters: int = 12,
                                    filter_stride: int = 4, n_encoded_sequence_features: int = 6,
                                    after_convolutions_width_ratio: float = 5) -> keras.Model:
    pep_input = keras.Input(shape=(max_pep_length, 21))
    p = layers.BatchNormalization(input_shape=(max_pep_length, 21))(pep_input)
    for i in range(convolutional_layers):
        p = layers.Conv1D(n_filters, filter_size, filter_stride, padding="valid", activation=tf.nn.tanh)(p)
        p = layers.MaxPool1D()(p)
    p = layers.Dropout(dropout)(p)
    p = layers.Flatten()(p)
    p = layers.Dense(n_encoded_sequence_features*3, activation=tf.nn.relu)(p)
    pep_out_flat = layers.Dense(n_encoded_sequence_features, activation=tf.nn.relu)(p)

    ms_feature_input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(ms_feature_input)
    x = layers.concatenate([x, pep_out_flat])
    n_nodes = int(round(int(x.shape[1]) * after_convolutions_width_ratio))
    for i in range(hidden_layers_after_convolutions):
        x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=[ms_feature_input, pep_input], outputs=output)

    return model


def get_model_without_peptide_encoding(ms_feature_length: int, max_pep_length: int, dropout: float = 0.6,
                                       hidden_layers: int = 2, width_ratio: float = 5) -> keras.Model:
    n_nodes = int(round(ms_feature_length * width_ratio))
    input = keras.Input(shape=(ms_feature_length,))
    x = layers.BatchNormalization(input_shape=(ms_feature_length,))(input)
    for i in range(hidden_layers):
        x = layers.Dense(n_nodes, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=input, outputs=output)
    return model


def reset_weights(model):
  for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))
