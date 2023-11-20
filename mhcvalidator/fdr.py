import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


'''def calculate_tensor_qs(metrics: tf.Tensor, labels: tf.Tensor, higher_better: bool = True):

    ordered_idx = tf.argsort(metrics)
    sorted_metrics = tf.gather(metrics, ordered_idx)
    sorted_labels = tf.gather(labels, ordered_idx)
    if not higher_better:
        sorted_metrics = tf.reverse(sorted_metrics)
        sorted_labels = tf.reverse(sorted_labels)
        ordered_idx = tf.reverse(ordered_idx)

    target_counts = Counter(metrics[labels == 1])
    decoy_counts = Counter(metrics[labels == 0])
    for key, value in decoy_counts.items():
        if key not in target_counts:
            target_counts[key] = 0
    for key, value in target_counts.items():
        if key not in decoy_counts:
            decoy_counts[key] = 0

    qs = np.ones_like(metrics, dtype=float)
    N_targets = np.sum(labels == 1)
    N_decoys = np.sum(labels == 0)

    for i in range(len(sorted_metrics)):
        if i == len(sorted_metrics) - 1:
            if sorted_labels[i] == 1:
                qs[ordered_idx[i]] = 0
            else:
                qs[ordered_idx[i]] = 1
            continue
        qs[ordered_idx[i]] = N_decoys / N_targets

        if sorted_metrics[i+1] != sorted_metrics[i]:
            N_targets -= target_counts[sorted_metrics[i]]
            N_decoys -= decoy_counts[sorted_metrics[i]]

    #if not higher_better:
    #    return np.flip(qs)
    #else:
    #    return qs
    return qs'''


def calculate_tensor_qs(metrics: tf.Tensor, labels: tf.Tensor, higher_better: bool = True):
    with tf.compat.v1.Session().as_default():
        metrics = metrics.numpy()
        labels = labels.numpy()
    ordered_idx = np.argsort(metrics)
    sorted_metrics = metrics[ordered_idx]
    sorted_labels = labels[ordered_idx]
    if not higher_better:
        sorted_metrics = np.flip(sorted_metrics)
        sorted_labels = np.flip(sorted_labels)
        ordered_idx = np.flip(ordered_idx)

    target_counts = Counter(metrics[labels == 1])
    decoy_counts = Counter(metrics[labels == 0])
    for key, value in decoy_counts.items():
        if key not in target_counts:
            target_counts[key] = 0
    for key, value in target_counts.items():
        if key not in decoy_counts:
            decoy_counts[key] = 0

    qs = np.ones_like(metrics, dtype=float)
    N_targets = np.sum(labels == 1)
    N_decoys = np.sum(labels == 0)

    for i in range(len(sorted_metrics)):
        if i == len(sorted_metrics) - 1:
            if sorted_labels[i] == 1:
                qs[ordered_idx[i]] = 0
            else:
                qs[ordered_idx[i]] = 1
            continue
        qs[ordered_idx[i]] = N_decoys / N_targets

        if sorted_metrics[i+1] != sorted_metrics[i]:
            N_targets -= target_counts[sorted_metrics[i]]
            N_decoys -= decoy_counts[sorted_metrics[i]]

    #if not higher_better:
    #    return np.flip(qs)
    #else:
    #    return qs
    return qs


def calculate_qs(metrics, labels, higher_better: bool = True):
    metrics = np.array(metrics)
    labels = np.array(labels)
    ordered_idx = np.argsort(metrics)
    sorted_metrics = metrics[ordered_idx]
    sorted_labels = labels[ordered_idx]
    if not higher_better:
        sorted_metrics = np.flip(sorted_metrics)
        sorted_labels = np.flip(sorted_labels)
        ordered_idx = np.flip(ordered_idx)

    target_counts = Counter(metrics[labels == 1])
    decoy_counts = Counter(metrics[labels == 0])
    for key, value in decoy_counts.items():
        if key not in target_counts:
            target_counts[key] = 0
    for key, value in target_counts.items():
        if key not in decoy_counts:
            decoy_counts[key] = 0

    qs = np.ones_like(metrics, dtype=float)
    N_targets = np.sum(labels == 1)
    N_decoys = np.sum(labels == 0)

    for i in range(len(sorted_metrics)):
        if i == len(sorted_metrics) - 1:
            if sorted_labels[i] == 1:
                qs[ordered_idx[i]] = 0
            else:
                qs[ordered_idx[i]] = 1
            continue
        qs[ordered_idx[i]] = N_decoys / N_targets

        if sorted_metrics[i+1] != sorted_metrics[i]:
            N_targets -= target_counts[sorted_metrics[i]]
            N_decoys -= decoy_counts[sorted_metrics[i]]

    #if not higher_better:
    #    return np.flip(qs)
    #else:
    #    return qs
    return qs


def calculate_roc(qs,
                  labels,
                  qvalue_cutoff: float = 0.05):
    qs = np.array(qs, dtype=float)
    labels = np.array(labels, dtype=int)

    qs = qs[(labels == 1) & (qs <= qvalue_cutoff)]
    sort_idx = np.argsort(qs)
    qs = qs[sort_idx]
    qs, counts = np.unique(qs, return_counts=True) # Counter(qs)

    N = 0
    n_psms = np.empty_like(qs, dtype=float)

    for i in range(len(qs)):
        N += counts[i]
        n_psms[i] = N

    return [qs, n_psms]


def calculate_peptide_level_qs(metrics, labels, peptides, higher_better=True):
    max_len = np.max(np.vectorize(len)(peptides))
    n_peps = len(np.unique(peptides))
    best_x = np.empty(n_peps, dtype=float)
    best_y = np.empty(n_peps, dtype=int)
    best_peps = np.empty(n_peps, dtype=f'U{max_len}')

    peptide_counter = Counter(peptides)

    ordered_idx = np.argsort(peptides)
    x = np.asarray([metrics[i] for i in ordered_idx])
    y = np.asarray([labels[i] for i in ordered_idx])
    peps = np.asarray([peptides[i] for i in ordered_idx], dtype=f'U{max_len}')

    current_peptide = peps[0]
    current_metric = x[0]
    pep_idx = 0
    max_i = len(x)

    assert int(max_i) == len(x) == len(y)

    for i in range(1, max_i):
        if peps[i] == current_peptide:
            if higher_better:
                if x[i] > current_metric:
                    current_metric = x[i]
            else:
                if x[i] < current_metric:
                    current_metric = x[i]
            if i == max_i - 1:  # ensure the last entry gets set correctly
                best_x[pep_idx] = current_metric
                pre_i = i - 1
                best_y[pep_idx] = y[pre_i]
                best_peps[pep_idx] = current_peptide
        else:
            best_x[pep_idx] = current_metric
            pre_i = i - 1
            best_y[pep_idx] = y[pre_i]
            best_peps[pep_idx] = current_peptide

            pep_idx += 1
            current_peptide = peps[i]
            current_metric = x[i]

            if i == max_i - 1:  # ensure the last entry gets set correctly
                best_x[pep_idx] = current_metric
                pre_i = i - 1
                best_y[pep_idx] = y[pre_i]
                best_peps[pep_idx] = current_peptide

    qs = calculate_qs(metrics=best_x, labels=best_y, higher_better=higher_better)

    peptide_counts = [peptide_counter[p] for p in best_peps]

    return np.asarray(qs), np.asarray(best_x), np.asarray(best_y), np.asarray(best_peps), np.array(peptide_counts)


def get_confident_data(X, y, metrics, fdr: float = 0.01, higher_better: bool = True):
    qs = calculate_qs(metrics, y, higher_better)
    return X[((y == 1) & (qs <= fdr)) | (y == 0)], y[((y == 1) & (qs <= fdr)) | (y == 0)]