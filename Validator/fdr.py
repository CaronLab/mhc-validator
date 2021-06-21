import numpy as np
import pandas as pd
from numba import njit, jit
from tqdm import tqdm
from typing import List, Union, Iterable
from copy import deepcopy


def calculate_qs(metrics, labels, higher_better: bool = True):
    x = metrics.flatten()
    y = labels.flatten()

    targets = x[y == 1]
    decoys = x[y == 0]
    if higher_better:
        targets = np.sum(x[:, np.newaxis] <= targets, axis=1)
        decoys = np.sum(x[:, np.newaxis] <= decoys, axis=1)
    else:
        targets = np.sum(x[:, np.newaxis] >= targets, axis=1)
        decoys = np.sum(x[:, np.newaxis] >= decoys, axis=1)
    qs = decoys / targets

    return qs


@jit(nopython=True)
def calculate_qs_fast(metrics: np.ndarray, labels: np.ndarray, higher_better: bool = True):
    x = metrics
    y = labels

    qs = np.empty_like(x)

    for i in range(x.shape[0]):
        n_targets = 0
        n_decoys = 0
        for j in range(x.shape[0]):
            if x[j] >= x[i] if higher_better else x[j] <= x[i]:
                if y[j] == 1:
                    n_targets += 1
                else:
                    n_decoys += 1
        qs[i] = n_decoys / n_targets

    return qs

def calculate_qs_2(metrics: np.ndarray, labels: np.ndarray, higher_better: bool = True):
    x = deepcopy(metrics.flatten())
    y = deepcopy(labels.flatten())

    sorted = x.argsort()
    x = x[sorted]
    y = y[sorted]

    qs = np.empty_like(x)

    for i in tqdm(range(x.shape[0])):
        better = x[i:] >= x[i] if higher_better else x <= x[i:]
        n_targets = np.sum(better & (y[i:] == 1))
        n_decoys = np.sum(better & (y[i:] == 0))
        qs[i] = n_decoys / n_targets

    return qs

@jit(nopython=True)
def calculate_qs_fast2(metrics: np.ndarray, labels: np.ndarray, higher_better: bool = True):
    x = metrics.flatten()
    y = labels.flatten()

    qs = np.empty_like(x)

    for i in range(len(x)):
        better = x >= x[i] if higher_better else x <= x[i]
        n_targets = np.sum(better & (y == 1))
        n_decoys = np.sum(better & (y == 0))
        qs[i] = n_decoys / n_targets

    return qs


def calculate_peptide_level_qs(metrics, labels, peptides, higher_better: bool = True):
    df = pd.DataFrame()
    df['label'] = list(labels)
    df['metric'] = list(metrics)
    df['peptide'] = list(peptides)

    grouped_df = df.groupby('peptide')
    if higher_better:
        x = grouped_df['metric'].max().to_numpy(np.float32)
    else:
        x = grouped_df['metric'].min().to_numpy(np.float32)
    y = grouped_df['label'].prod().to_numpy(np.float32)
    peptides = grouped_df['peptide'].first().to_numpy()

    targets = x[y == 1]
    decoys = x[y == 0]
    if higher_better:
        targets = np.sum(x[:, np.newaxis] <= targets, axis=1)
        decoys = np.sum(x[:, np.newaxis] <= decoys, axis=1)
    else:
        targets = np.sum(x[:, np.newaxis] >= targets, axis=1)
        decoys = np.sum(x[:, np.newaxis] >= decoys, axis=1)
    qs = decoys / targets

    return qs, y, peptides


def get_confident_data(X, y, metrics, fdr: float = 0.01, higher_better: bool = True):
    qs = calculate_qs(metrics, y, higher_better)
    return X[((y == 1) & (qs <= fdr)) | (y == 0)], y[((y == 1) & (qs <= fdr)) | (y == 0)]