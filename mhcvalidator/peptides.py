import re
from typing import Union, List
from copy import deepcopy


def remove_modifications(peptides: Union[List[str], str]):
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


def remove_previous_and_next_aa(peptides: Union[List[str], str]):
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


def clean_peptide_sequences(peptides: List[str]) -> List[str]:
    return remove_modifications(remove_previous_and_next_aa(peptides))
