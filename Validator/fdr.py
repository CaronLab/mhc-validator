import numpy as np
import pandas as pd


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