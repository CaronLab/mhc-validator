import re
from typing import Union, List, Type
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from collections import Counter


def remove_modifications(peptides: Union[List[str], str, Type[np.array]]):
    if isinstance(peptides, str):
        peptide = ''.join(re.findall('[a-zA-Z]+', peptides))
        if peptide.startswith('n'):
            return peptide[1:]
        elif peptide.endswith('c'):
            return peptide[:-1]
        else:
            return peptide
    unmodified_peps = []
    for pep in peptides:
        pep = ''.join(re.findall('[a-zA-Z]+', pep))
        unmodified_peps.append(pep)
    for i, peptide in enumerate(unmodified_peps):
        if peptide.startswith('n'):
            unmodified_peps[i] = peptide[1:]
        elif peptide.endswith('c'):
            unmodified_peps[i] = peptide[:-1]
    return unmodified_peps


def remove_previous_and_next_aa(peptides: Union[List[str], str, Type[np.array]]):
    peptides = deepcopy(peptides)
    return_one = False
    if isinstance(peptides, str):
        peptides = [peptides]
        return_one = True
    for i in range(len(peptides)):
        if peptides[i][1] == '.':
            peptides[i] = peptides[i][2:]
        if peptides[i][-2] == '.':
            peptides[i] = peptides[i][:-2]
    if return_one:
        return peptides[0]
    return peptides


def encode_peptide_modifications(peptides: Union[List[str], Type[np.array]], modification_encoding: dict = None,
                                 return_encoding_dictionary: bool = False):
    peptides = deepcopy(peptides)
    peptides = np.array(peptides)
    peptides = remove_previous_and_next_aa(peptides)

    if modification_encoding is not None:  # a dictionary containing modification encoding was passed
        mod_dict = modification_encoding
    else:  # we need to build the modification encoding dictionary
        modifications = []
        for pep in peptides:
            mods = re.findall('([a-zA-z][[({][0-9.a-zA-Z]+[](}])', pep)
            modifications += mods
        modifications = set(modifications)

        mod_dict = {}
        for i, mod in enumerate(modifications):
            mod_dict[mod] = str(i + 1)

    # replace the modifications in the peptide strings with the number encodings
    for i, pep in enumerate(peptides):
        for mod in mod_dict.keys():
            while mod in pep:
                pep = pep.replace(mod, mod_dict[mod])
        peptides[i] = pep

    if return_encoding_dictionary:
        return peptides, mod_dict
    else:
        return peptides


def clean_peptide_sequences(peptides: List[str]) -> List[str]:
    return remove_modifications(remove_previous_and_next_aa(peptides))


def resolve_duplicates_between_splits(index1, index2, peptides, k: int, random_seed=0):

    #### PROBLEM: SOME OF THE PSMS END UP NEVER BEING IN THE PREDICTION SPLIT
    #### I HAVE MESSED UP THE K_FOLDS HERE, NEED TO FIX!!!!!!!

    rs = np.random.RandomState(random_seed)
    new_index1 = []
    new_index2 = []

    set1_peps = set([peptides[i] for i in index1])
    set2_peps = set([peptides[i] for i in index2])

    duplicates = set1_peps & set2_peps
    set1_unique_peps = set1_peps - set2_peps
    set2_unique_peps = set2_peps - set1_peps

    newly_added_set1_peps = []
    newly_added_set2_peps = []

    i = rs.randint(1e6)
    for idx in index1:
        p = peptides[idx]
        if p in duplicates:
            if p in newly_added_set1_peps:
                new_index1.append(idx)
            elif p in newly_added_set2_peps:
                new_index2.append(idx)
            else:
                new_index1.append(idx)
                newly_added_set1_peps.append(p)
                '''if i % k == 0:
                    new_index1.append(idx)
                    newly_added_set1_peps.append(p)
                else:
                    new_index2.append(idx)
                    newly_added_set2_peps.append(p)
                i += 1'''
        else:
            new_index1.append(idx)

    for idx in index2:
        p = peptides[idx]
        if p in duplicates:
            if p in newly_added_set1_peps:
                new_index1.append(idx)
            elif p in newly_added_set2_peps:
                new_index2.append(idx)
        else:
            new_index2.append(idx)

    assert set(new_index1) | set(new_index2) == set(index1) | set(index2)

    new_index1 = np.array(new_index1, dtype=int)
    new_index2 = np.array(new_index2, dtype=int)

    return new_index1, new_index2


def resolve_missing_peptides_in_k_folds(k_folds_splits, peptides):
    # GOAL: find in which sets each duplicate peptide sequence is present.
    psm_presence = {i: set() for i in range(len(peptides))}

    for index1, index2 in k_folds_splits:
        for i, idx in enumerate([set(index1), set(index2)]):
            for scan in range(len(peptides)):
                if scan in idx:
                    psm_presence[scan].add(i)

