import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K


def Chi2Loss(y_true, y_pred):
    h_true = tf.histogram_fixed_width(y_true, value_range=(0, 1.), nbins=20)
    h_pred = tf.histogram_fixed_width(y_pred, value_range=(0, 1.), nbins=20)
    h_true = tf.cast(h_true, dtype=tf.dtypes.float32)
    h_pred = tf.cast(h_pred, dtype=tf.dtypes.float32)
    return K.mean(K.square(h_true - h_pred) + y_pred * 0)


def weighted_bce(weight_on_FP: float = 10, weight_on_uncertain: float = 1, bce_weight: float = 0.5) -> callable:
    def loss(y_true, y_pred):
        # y = y_true * 0.9 + 0.05
        # y_pred_labels = tf.cast(y_pred > 0.5, dtype=tf.dtypes.float32)
        # y_shift = tf.clip_by_value(y_pred - 0.4, clip_value_min=0.0, clip_value_max=1.0) * (1/0.6)
        # clipped_y_pred = tf.clip_by_value(y_pred - 0.5, 0, 1) + 0.5
        FP_weight = tf.math.abs(y_true - 1.) * y_pred * weight_on_FP
        #center_weight = tf.math.square(y_pred - 0.5) * weight_on_uncertain
        center_weight = tf.math.exp(-1 * tf.math.abs(y_pred - 0.5)) * weight_on_uncertain
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.mean(bce * (FP_weight + center_weight + bce_weight))
    return loss


def total_fdr(y_true, y_pred):
    y_pred_labels = tf.round(y_pred)
    false_positives = tf.reduce_sum(tf.math.abs(y_true - 1.) * y_pred_labels)
    true_positives = tf.reduce_sum(y_true * y_pred_labels)
    return false_positives / (false_positives + true_positives)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_pred_labels = tf.round(y_pred)
    false_positives = tf.reduce_sum(tf.math.abs(y_true - 1.) * y_pred_labels)
    true_positives = tf.reduce_sum(y_true * y_pred_labels)
    return true_positives / (false_positives + true_positives)


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))