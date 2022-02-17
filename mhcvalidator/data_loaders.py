# define a function to read files into pandas dataframes. We need a custom function to do this because the data
# produced by some tools is ragged (e.g. proteins just run willy-nilly off to right. Not cool.)

from pathlib import Path
from typing import Union, List
import pandas as pd
import numpy as np
from pyteomics import pepxml, mzid, tandem
from os import PathLike
from mhcvalidator.constants import COMMON_AA
from mhcvalidator.peptides import clean_peptide_sequences

sep_tag = '@@'

columns_for_training = {
    'PIN': ['ExpMass', 'abs_ppm', 'log10_evalue', 'hyperscore', 'delta_hyperscore',
            'matched_ion_num', 'matched_ion_fraction', 'peptide_length', 'charge_1',
            'charge_2', 'charge_3', 'charge_4', 'charge_5',
            'charge_6', 'charge_7'],

    'POUT': ['score', 'posterior_error_prob'],

    'MSFRAGGER': ['precursor_neutral_mass', 'retention_time', 'charge',
                  'num_matched_ions', 'tot_num_ions', 'calc_neutral_pep_mass', 'massdiff',
                  'hyperscore', 'nextscore', 'expectscore']
}


def load_tabular_data(filepath: Union[str, Path],
                      id_column=None,
                      protein_column: str = None,
                      decoy_prefix: str = 'rev_',
                      sep='\t'):
    """
    Loads a tabular file into a Pandas DataFrame. If there are ragged entries (i.e. not all row
    lengths are the same), then it is assumed the last column is proteins any ragged entries are joined
    with the special separator '@@' (chosen because it is extremely unlikely that
    we will ever encounter such a string in a FASTA file by chance.)

    :param filepath: Path to the tabular file
    :param id_column: A column to use for the dataframe index. Optional.
    :param decoy_prefix: The decoy prefix from the database search. Optional
    :param protein_column: The name of the column containing protein IDs. Optional.
    :param sep: The column delimiter. Must be the literal separator, not a word. E.g. use "\t" for TSV or "," fot CSV
    :return: A Pandas Dataframe containing the loaded data
    """
    if id_column and not isinstance(id_column, str):
        raise TypeError('id_column must be a string or None')
    if not isinstance(sep, str):
        raise TypeError('sep must be a string')

    with open(filepath, 'r') as f:
        header = f.readline().rstrip().split(sep)
        contents = [x.rstrip().split(sep) for x in f.readlines()]
    n_columns = len(header)
    new_contents = []
    for line in contents:
        new_contents.append(line[:n_columns - 1] + ['@@'.join(line[n_columns - 1:])])
    df = pd.DataFrame(data=new_contents, columns=header)
    if id_column is not None:
        df.index = list(df[id_column])
    if protein_column:
        add_label_feature_from_protein_id(df,
                                          prot_column=protein_column,
                                          decoy_prefix=decoy_prefix,
                                          in_place=True)
    return df


def load_spectromine_data(filepath: Union[str, Path],
                          sep='\t'):
    """
    NOT CURRENTLY USED. Loads a tabular file into a Pandas DataFrame. If there are ragged entries (i.e. not all row
    lengths are the same), then it is assumed the last column is proteins any ragged entries are joined
    with the special separator '@@' (chosen because it is extremely unlikely that
    we will ever encounter such a string in a FASTA file by chance.)

    :param filepath: Path to the tabular file
    :param id_column: A column to use for the dataframe index. Optional.
    :param decoy_prefix: The decoy prefix from the database search. Optional
    :param protein_column: The name of the column containing protein IDs. Optional.
    :param sep: The column delimiter. Must be the literal separator, not a word. E.g. use "\t" for TSV or "," fot CSV
    :return: A Pandas Dataframe containing the loaded data
    """
    if not isinstance(sep, str):
        raise TypeError('sep must be a string')

    with open(filepath, 'r') as f:
        header = f.readline().rstrip().split(sep)
        contents = [x.rstrip().split(sep) for x in f.readlines()]
    n_columns = len(header)
    new_contents = []
    for line in contents:
        new_contents.append(line[:n_columns - 1] + ['@@'.join(line[n_columns - 1:])])
    df = pd.DataFrame(data=new_contents, columns=header)
    apply_target_decoy_label_function(df=df,
                                      func=lambda x: 0 if x == 'True' else 1,
                                      column='PSM.IsDecoy',
                                      in_place=True)
    return df


def load_pin_data(filepath: Union[str, Path],
                  decoy_tag: str = 'rev_',
                  tag_is_prefix: bool = True,
                  protein_column: str = 'Proteins') -> pd.DataFrame:
    """
    Load a Percolator PIN file into a Pandas DataFrame.

    NOTE: WE NEED TO STRIP MODIFICATIONS FROM THE PEPTIDES. AS IT IS THEY ARE MODIFIED

    :param filepath: Path to the PIN file
    :param decoy_tag: The decoy tag used in the database search. Can be ignored if target-decoy competition was not
    performed.
    :param tag_is_prefix: True if the tag is a prefix, False if the tag is a suffix.
    :param protein_column: The column containing protein IDs. Needed for extracting target-decoy labels if the
    Label column is incorrect.
    :return: A Pandas DataFrame containing the loaded data
    """

    df = load_tabular_data(filepath, id_column='SpecId')
    if len(df['Label'].unique()) == 2:
        apply_target_decoy_label_function(df,
                                          column='Label',
                                          func=lambda x: 0 if int(x) == -1 else 1,
                                          in_place=True)
    else:
        print('INFO: The Label column in the PIN file is not properly formatted. Attempting to extract target/decoy '
              'labels from the protein IDs. Please ensure the following information is correct. If not, specify in '
              'the "load_data" function call:\n'
              f'  Protein ID column: {protein_column}\n'
              f'  Decoy tag: {decoy_tag}\n'
              f'  Decoy tag is a prefix: {tag_is_prefix}')

        if tag_is_prefix:
            def is_decoy(x):
                return x.startswith(decoy_tag)
        else:
            def is_decoy(x):
                return x.endswith(decoy_tag)

        if protein_column is None or decoy_tag is None or not isinstance(tag_is_prefix, bool):
            raise ValueError('One or more required arguments are missing to extract target/decoy labels from the '
                             'protein ID column. "protein_column", "decoy_tag" and "tag_is_prefix" must all be '
                             'defined in the function call.')
        apply_target_decoy_label_function(df,
                                          column=protein_column,
                                          func=lambda x: 0 if is_decoy(x) else 1,
                                          in_place=True)
    return df


def load_pout_data(target_filepath: Union[str, Path, None],
                   decoy_filepath: Union[str, Path, None],
                   min_len: int = 8,
                   max_len: int = 15) -> pd.DataFrame:
    """
    Load Percolator POUT file(s) into a Pandas DataFrame. Can specify a target POUT file, decoy POUT file, or
    both.

    :param target_filepath: Path to a target POUT file
    :param decoy_filepath: Path to a decoy POUT file
    :return: A Pandas DataFrame containing the loaded data
    """
    if not target_filepath and not decoy_filepath:
        raise ValueError("You must indicate a target filepath, decoy filepath, or both.")
    df = pd.DataFrame()
    if target_filepath:
        targets = load_tabular_data(target_filepath, id_column='PSMId')
        add_target_decoy_label(targets,
                               label=1,
                               in_place=True)
        df = pd.concat([df, targets])
    if decoy_filepath:
        decoys = load_tabular_data(decoy_filepath, id_column='PSMId')
        add_target_decoy_label(decoys,
                               label=0,
                               in_place=True)
        df = pd.concat([df, decoys])
    peps = np.array(clean_peptide_sequences(list(df['peptide'])))
    lengths = np.vectorize(len)(peps)
    only_common_aas = np.vectorize(sequence_contains_only_common_aas)(peps)
    df = df.loc[(min_len <= lengths) & (lengths <= max_len) & (only_common_aas), :]
    df.reset_index(inplace=True, drop=True)
    return df


# define a function to add a label column with a single value to a dataframe. this is pure convenience for when
# we need to load separate files for targets and decoys

def add_target_decoy_label(df: pd.DataFrame,
                           label: Union[int, str],
                           replace_existing: bool = False,
                           in_place: bool = True) -> Union[pd.DataFrame, None]:
    """
    Add a "Label" column to a DataFrame, populated with a single value.

    :param df: The DataFrame
    :param label: The label to add to a "Label" column
    :param replace_existing: Replace an existing Label column
    :param in_place: Perform the operation in place
    :return: The labeled DataFrame
    """
    if not replace_existing and 'Label' in df.columns:
        raise IndexError('The DataFrame already contains a "Label" column.')
    if not in_place:
        df_out = df.copy(deep=True)
        df_out['Label'] = label
        return df_out
    else:
        df['Label'] = label


def apply_target_decoy_label_function(df: pd.DataFrame,
                                      func: callable = lambda x: 1 if 1 else 0,
                                      column: str = 'Label',
                                      in_place: bool = True) -> Union[pd.DataFrame, None]:
    """
    Create or modify and existing Label column in a dataframe with a callable function.

    :param df: The DataFrame
    :param func: The function to apply to the column indicated by the value of the "column" argument. The return
    value of this function is used for the label.
    :param column: The column to which the function is applied. For example, to detect target/decoy labels based on
    protein ID, you would use something like "protein" or "proteinIDs" here.
    :param in_place: Perform the operation in place
    :return: The labeled DataFrame
    """
    if not in_place:
        df_out = df.copy(deep=True)
        df_out['Label'] = df[column].apply(func)
        return df_out
    else:
        df['Label'] = df[column].apply(func)


def add_label_feature_from_protein_id(df: pd.DataFrame,
                                      func: callable = None,
                                      prot_column: str = 'protein',
                                      decoy_prefix: str = 'rev_',
                                      in_place: bool = True) -> Union[pd.DataFrame, None]:
    """
    Add a label column tpo a dataframe based on the prefix of the protein column indicated. If the decoy is indicated
    using a suffix or something more complicated, you can use apply_target_decoy_label_function for more flexibility.

    :param df:
    :param func:
    :param prot_column:
    :param decoy_prefix:
    :param in_place:
    :return:
    """
    try:
        if not func:
            def func(x):
                if x[0].startswith(decoy_prefix):
                    return 0
                else:
                    return 1

        if not in_place:
            df_out = df.copy(deep=True)
            df_out['Label'] = df_out[prot_column].apply(func)
            return df_out
        else:
            df['Label'] = df[prot_column].apply(func)
    except Exception as e:
        print(prot_column)
        print(df[prot_column])
        raise e


# define a function to load a pepXML file into a DataFrame

def load_pepxml_data(filepath: Union[str, Path],
                     decoy_prefix: str = 'rev_',
                     prot_column: str = 'protein',
                     extract_label_func: callable = None):
    df = pepxml.DataFrame(filepath)
    before = len(df)
    df.dropna(axis=0, subset=['peptide', prot_column], inplace=True)
    df.reset_index(inplace=True, drop=True)
    after = len(df)
    if after != before:
        print(f"INFO: {before-after} spectra had no identifications and were dropped.")
    df[prot_column] = df[prot_column].apply(lambda x: '@@'.join(x))
    add_label_feature_from_protein_id(df=df,
                                      func=extract_label_func if extract_label_func is not None
                                      else lambda x: 0 if x.startswith(decoy_prefix) else 1,
                                      prot_column=prot_column,
                                      decoy_prefix=decoy_prefix)
    return df


# define a function to load a mzIdentML file into a DataFrame

def load_mzid_data(filepath: Union[str, Path],
                   decoy_prefix: str = 'rev_',
                   prot_column: str = 'protein description',
                   extract_label_func: callable = None):
    """
    Not currently used.
    :param filepath:
    :param decoy_prefix:
    :param prot_column:
    :param extract_label_func:
    :return:
    """
    df = mzid.DataFrame(filepath)
    before = len(df)
    df.dropna(axis=0, subset=['PeptideSequence', prot_column], inplace=True)
    df.reset_index(inplace=True, drop=True)
    after = len(df)
    if after != before:
        print(f"INFO: {before-after} spectra had no identifications and were dropped.")
    add_label_feature_from_protein_id(df=df,
                                      func=extract_label_func,
                                      prot_column=prot_column,
                                      decoy_prefix=decoy_prefix)
    return df


# define a function to load a mzIdentML file into a DataFrame

def load_tandem_data(filepath: Union[str, Path],
                     decoy_prefix: str = 'rev_',
                     prot_column: str = 'protein',
                     extract_label_func: callable = None):
    df = tandem.DataFrame(filepath)
    add_label_feature_from_protein_id(df=df,
                                      func=extract_label_func,
                                      prot_column=prot_column,
                                      decoy_prefix=decoy_prefix)
    return df


def combine_protein_and_alternative_protein_columns(df: pd.DataFrame,
                                                    protein: str = 'protein',
                                                    alt_prots: str = 'alternative_proteins',
                                                    in_place: bool = True):
    """
    Only useful for Percolator files, I think.
    """
    if not in_place:
        df = df.copy(deep=True)
    proteins_out = list(df.apply(lambda x: x[protein] + '@@' + x[alt_prots]
    if x[alt_prots] != None else x[protein], axis=1))
    df[protein] = proteins_out
    df.drop(columns=[alt_prots], inplace=in_place)
    df.reset_index(inplace=True, drop=True)
    if not in_place:
        return df


def sequence_contains_only_common_aas(sequence: str):
    for aa in sequence:
        if aa not in COMMON_AA:
            return False
    return True


def load_file(filename: Union[str, PathLike],
              filetype: str,
              decoy_tag: str = 'rev_',
              protein_column: str = None,
              tag_is_prefix: bool = True,
              file_sep: str = '\t',
              min_len: int = 8,
              max_len: int = 15):
    """

    :param max_len:
    :param min_len:
    :param tag_is_prefix:
    :param filename:
    :param filetype:
    :param decoy_tag:
    :param protein_column:
    :param file_sep:
    :return:
    """
    if filetype not in ['pin', 'pepxml', 'tabular', 'tandem', 'mhcv']:
        raise ValueError("filetype must be one of "
                         "{'auto', 'pin', 'pepxml', 'tabular', 'tandem', 'mhcv}")

    if filetype == 'pin' or filetype == 'mhcv':
        df = load_pin_data(filename, decoy_tag=decoy_tag, protein_column=protein_column, tag_is_prefix=tag_is_prefix)
        pep_col = 'peptide' if 'peptide' in df else 'Peptide'
    elif filetype == 'pepxml':
        df = load_pepxml_data(filename, decoy_prefix=decoy_tag)
        pep_col = 'peptide' if 'peptide' in df else 'Peptide'
    elif filetype == 'tandem':
        df = load_tandem_data(filename, decoy_prefix=decoy_tag)
        pep_col = 'peptide' if 'peptide' in df else 'Peptide'
    else:
        if not (protein_column and decoy_tag):
            raise ValueError("the protein_column and decoy_tag arguments must be specified for arbitrary tabular data.")
        df = load_tabular_data(filename, protein_column=protein_column, decoy_prefix=decoy_tag, sep=file_sep)
        pep_col = 'peptide' if 'peptide' in df else 'Peptide'

    peps = np.array(clean_peptide_sequences(list(df[pep_col])))
    lengths = np.vectorize(len)(peps)
    only_common_aas = np.vectorize(sequence_contains_only_common_aas)(peps)
    before = len(df)
    df = df.loc[(min_len <= lengths) & (lengths <= max_len) & (only_common_aas), :]
    after = len(df)
    if before != after:
        print(f"INFO: {before-after} PSMs were outside tolerable peptide lengths or contained uncommon amino acids "
              f"and were dropped.")
    df.reset_index(inplace=True, drop=True)
    return df
