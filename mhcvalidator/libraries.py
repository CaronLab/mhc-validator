import pandas as pd
from pathlib import Path
from typing import List, Union, Iterable
from os import PathLike


def load_biognosys_library(filepath: Union[str, PathLike]):
    lib = pd.read_table(filepath, index_col=False, low_memory=False)
    return lib


def load_library(filepath: Union[str, PathLike], lib_format: str = 'biognosys'):
    accepted_formats = ['biognosys']
    if lib_format not in accepted_formats:
        raise ValueError(f'lib_format must be one of {accepted_formats}')
    if lib_format == 'biognosys':
        return load_biognosys_library(filepath)


def filter_library(library: pd.DataFrame,
                   peptide_list: Iterable[str],
                   lib_format: str = 'biognosys',
                   pep_column: str = None) -> pd.DataFrame:
    if pep_column is None:
        if lib_format == 'biognosys':
            pep_column = 'StrippedPeptide'
    peps = set(peptide_list)

    to_keep = list(library[pep_column].apply(lambda x: x in peps))

    return library.loc[to_keep, :]
