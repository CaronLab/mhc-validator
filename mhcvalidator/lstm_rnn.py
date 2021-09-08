"""
I will try a bidirectional LSTM for the peptide sequence encoding. This code is adapted from
https://keras.io/examples/nlp/bidirectional_lstm_imdb/

I will start by simply encoding the amino acids as integers
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from mhcvalidator.predictors import MhcValidator
import matplotlib.pyplot as plt
from mhcvalidator.encoding import pad_and_encode_multiple_aa_seq

max_features = 20  # There are 20 common amino acids we will deal with
maxlen = 15  # Our max peptide length is 15. For class II it is 30, but we will just start with 15 for now.

COMMON_AA = "ARNDCQEGHILKMFPSTWYV"
COMMON_AA_LIST = list(COMMON_AA)

v = MhcValidator()
v.set_mhc_params(['A0201', 'A0217', 'B4002', 'B4101', 'C0202', 'C1701'], 'I')
v.load_data('/Data/Data/SupremaReplicates/JY_Human/20211512_analysis/JY_301120_S3.pin',
            filetype='pin')
_, unique_idx = np.unique(v.peptides, return_index=True)
X = v.peptides[unique_idx]
y = np.array(v.labels[unique_idx])

X = pad_and_encode_multiple_aa_seq(X, max_length=maxlen, padding='post').astype(np.float32)

random_idx = np.random.permutation(len(y))
X = X[random_idx]
y = y[random_idx]

split = int(len(y) * 0.5)

x_train, x_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]


# Input for variable-length sequences of integers
inputs = keras.Input(shape=(maxlen, 21), dtype="float32")
# Embed each integer in a 128-dimensional vector
#x = layers.Embedding(max_features, 20)(inputs)  # this is where we would use the BLOSUM encoding, I think
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(maxlen, return_sequences=False))(inputs)
x = layers.Dropout(0.5)(x)
#x = layers.Bidirectional(layers.LSTM(int(maxlen/2)))(x)
#x = layers.Dropout(0.5)(x)
# Add a classifier
x = layers.Dense(maxlen, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(maxlen, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_val, y_val))

preds = model.predict(x_train)
plt.hist(x=preds[y_train==1], label='Targets', bins=40, range=(0, 1), alpha=0.8)
plt.hist(x=preds[y_train==0], label='Decoys', bins=40, range=(0, 1), alpha=0.8)
plt.xlim((0, 1))
plt.legend()
plt.title('Training set')
plt.show()

preds = model.predict(x_val)
plt.hist(x=preds[y_val==1], label='Targets', bins=40, range=(0, 1), alpha=0.8)
plt.hist(x=preds[y_val==0], label='Decoys', bins=40, range=(0, 1), alpha=0.8)
plt.xlim((0, 1))
plt.legend()
plt.title('Validation set')
plt.show()

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss (BCE)')
plt.legend()
plt.show()
