import numpy as np
from typing import Iterable, Tuple, List, Union
from collections import Counter
from itertools import cycle


def k_fold_split(peptides: Iterable[str],
                 k_folds: int = 3,
                 random_state: Union[int, np.random.RandomState] = 0):

    if isinstance(random_state, int):
        random_state: np.random.RandomState = np.random.RandomState(random_state)
    else:
        random_state: np.random.RandomState = random_state
    # get the number of counts of each peptide sequence
    peptide_counts = Counter(peptides)

    unique_peptide_indices = []
    nonunique_peptide_indices = {}

    for i, pep in enumerate(peptides):
        if peptide_counts[i] == 1:
            unique_peptide_indices.append(i)
        else:
            if pep not in nonunique_peptide_indices:
                nonunique_peptide_indices[pep] = []
            nonunique_peptide_indices[pep].append(i)

    # shuffle the unique_peptide_indices
    rand_idx = random_state.choice(len(unique_peptide_indices), len(unique_peptide_indices), False)
    unique_peptide_indices = [unique_peptide_indices[i] for i in rand_idx]

    # get initial splits, include only unique peptide indices
    split_size = int(len(rand_idx) / k_folds)
    train_indices = []
    val_indices = []
    for k in range(k_folds):
        val_indices.append(unique_peptide_indices[k*split_size: k*split_size + split_size])
        train_indices.append((unique_peptide_indices[0: k*split_size] +
                              unique_peptide_indices[k*split_size + split_size:]))

    # now we need to add the peptides with more than one PSM. We make sure all the PSMs of a given peptide end up in
    # a single validation set, and distributed among the training sets of other splits
    k = list(range(k_folds))  # randomly get one of the splits to start
    random_state.shuffle(k)
    k = cycle(k)
    for indices in nonunique_peptide_indices.values():
        val_k = next(k)
        val_indices[val_k] += indices  # add the indices to a single validation set

        # and now add them to the training sets in all the other splits
        train_ks = list(range(k_folds))
        train_ks.remove(val_k)
        for i in train_ks:
            train_indices[i] += indices

    for k in range(k_folds):
        random_state.shuffle(train_indices[k])
        random_state.shuffle(val_indices[k])
        train_indices[k] = np.array(train_indices[k])
        val_indices[k] = np.array(val_indices[k])

    train_test_splits = list(zip(train_indices, val_indices))

    return train_test_splits

