import numpy as np
import pandas as pd
from numba import njit, jit
from tqdm import tqdm
from typing import List, Union, Iterable
from copy import deepcopy
import cython


cdef double[] calculate_qs(np.array[double, dim=1] metrics, np.array[int, dim=1] labels, bool higher_better = True):
    cdef np.array[double, dim=1] x = metrics
    cdef np.array[int, dim=1] y = labels
    cdef double[double, dim=1] qs = np.one(x.shape[0])
    cdef int i = 0

    for i in range(x.shape[0]):
        cdef int n_targets = 0
        cdef int n_decoys = 0
        cdef int j = 0
        for j in range(x.shape[0]):
            if (x[j] >= x[i] if higher_better else x[j] <= x[i]):
                if y[j] == 1:
                    n_targets += 1
                else:
                    n_decoys += 1
        qs[i] = n_decoys / n_targets

    return qs


def calculate_peptide_level_qs(np.array[double, dim=1] metrics,
                               np.array[int, dim=1] labels,
                               np.array[str, dim=1] peptides,
                               bool higher_better = True):
    cdef np.array[double, dim=1] x
    cdef np.array[int, dim=1] y
    cdef double[double, dim=1] qs = np.one(x.shape[0])
    cdef np.array[str, dim=1] peps

    cdef float best_metric
    cdef str current_peptide

    ordered_idx = np.argsort(peptides)
    x = metrics[ordered_idx]
    y = labels[ordered_idx]
    peps = peptides[ordered_idx]
    ##### CONTINUE FROM HERE

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