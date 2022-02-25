import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy.random
import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Union, List, Tuple
from os import PathLike
from pathlib import Path
from mhcvalidator.data_loaders import load_file, load_pout_data
from mhcvalidator.features import prepare_features, eliminate_common_peptides_between_sets
from mhcvalidator.predictions_parsers import add_mhcflurry_to_feature_matrix, add_netmhcpan_to_feature_matrix
from mhcvalidator.predictions_parsers import format_mhcflurry_predictions_dataframe, format_netmhcpan_prediction_dataframe
from mhcvalidator.netmhcpan_helper import NetMHCpanHelper, format_class_II_allele
from mhcvalidator.losses_and_metrics import global_accuracy
from mhcvalidator.fdr import calculate_qs, calculate_peptide_level_qs, calculate_roc
import matplotlib.pyplot as plt
from mhcflurry.encodable_sequences import EncodableSequences
from mhcvalidator.models import get_model_without_peptide_encoding, get_model_with_peptide_encoding
from mhcvalidator.peptides import clean_peptide_sequences, resolve_duplicates_between_splits
from mhcvalidator.nd_standard_scalar import NDStandardScaler
from mhcnames import normalize_allele_name
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from mhcvalidator.libraries import load_library, filter_library
import tempfile
from collections import Counter
import tensorflow.python.util.deprecation as deprecation
from matplotlib.gridspec import GridSpec
import subprocess
import matplotlib.backends.backend_pdf as plt_pdf
from sklearn.mixture import GaussianMixture
# from deeplc import DeepLC
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.pyll.base import scope
from inspect import signature
from matplotlib.cm import get_cmap
from mhcvalidator.rt_prediction import train_predict as train_predict_rt
from mhcvalidator.rt_prediction import extract_rt
from mhcvalidator.pepxml_parser import pepxml_to_mhcv
from mhcvalidator.datasets import k_fold_split

deprecation._PRINT_DEPRECATION_WARNINGS = False

# This can be uncommented to prevent the GPU from getting used.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DEFAULT_TEMP_MODEL_DIR = str(Path(tempfile.gettempdir()) / 'validator_models')


class MhcValidator:
    def __init__(self,
                 random_seed: int = 0,
                 model_dir: Union[str, PathLike] = DEFAULT_TEMP_MODEL_DIR,
                 max_threads: int = -1):
        self.filename: str = None
        self.filepath: Path = None
        self.model: keras.Model = None
        self.raw_data: Union[pd.DataFrame, None] = None
        self.feature_matrix: Union[pd.DataFrame, None] = None
        self.labels: Union[List[int], None] = None
        self.peptides: Union[List[str], None] = None
        self.encoded_peptides = None
        self.loaded_filetype: Union[str, None] = None
        self.random_seed: int = random_seed
        self.predictions: np.array = None
        self.qs: np.array = None
        self.roc = None
        self.percolator_qs = None
        self.obs_rts: np.array = None
        self.pred_rts: np.array = None
        self.mhc_class: str = None
        self.alleles: List[str] = None
        self.min_len: int = 5
        self.max_len: int = 100
        self.model_dir = Path(model_dir)
        self._mhcflurry_predictions: bool = False
        self._netmhcpan_predictions: bool = False
        self._retention_time_features: bool = False
        self.mhcflurry_predictions: pd.DataFrame = None
        self.netmhcpan_predictions: pd.DataFrame = None
        self.annotated_data: pd.DataFrame = None
        self.modificationss = {'15.9949': 'Oxidation',
                               '0.9840': 'Deamidation'}
        if max_threads < 1:
            self.max_threads: int = os.cpu_count()
        else:
            self.max_threads: int = max_threads

    def set_mhc_params(self,
                       alleles: Union[str, List[str]] = None,
                       mhc_class: str = None,
                       max_pep_len: int = None,
                       min_pep_len: int = None) -> None:
        """
        Set the MHC-specific parameters.

        :param alleles: The alleles to be used by MhcFlurry or NetMHCpan.
        :param mhc_class: The MHC class of the peptides. Must be one of {'I', 'II'}
        :param min_pep_len: Maximum length of peptides allowed. Will default to 16 for class I and 30 for class II. Note
        that MhcFlurry does not accept peptide lengths greater than 16. There is no length restriction for NetMHCpan.
        :param max_pep_len: Minimum length of peptides allowed. Will default to 8 for class I and 9 for class II. Note
        that NetMHC(II)pan does not accept peptide lengths less than 8 for class I or 9 for class I. NetMHCpan predictions
        take much longer for longer peptides.
        :return: None
        """
        if alleles is None and mhc_class is None:
            return None

        if isinstance(alleles, str):
            alleles = [alleles]

        if mhc_class is not None:
            if mhc_class not in ['I', 'II']:
                raise ValueError("mhc_class must be one of {'I', 'II'}")

        if alleles:
            self.alleles = [normalize_allele_name(a).replace('*', '').replace(':', '') for a in alleles]
        if mhc_class:
            self.mhc_class = mhc_class

        if max_pep_len is not None:
            self.max_len = max_pep_len
        else:
            if self.mhc_class == 'I':
                self.max_len = 15
            else:
                self.max_len = 30

        if min_pep_len is not None:
            self.min_len = min_pep_len
        else:
            if self.mhc_class == 'I':
                self.min_len = 8
            else:
                self.min_len = 9

    def _check_peptide_lengths(self):
        max_len = self.max_len
        longest_peptide = np.max(np.vectorize(len)(self.peptides))
        if max_len > longest_peptide:
            print(f'Longest peptide ({longest_peptide} mer) is shorter than set maximum length ({max_len} mer). '
                  f'Changing max_len to {longest_peptide}.')
            self.max_len = longest_peptide

    def load_data(self,
                  filepath: Union[str, PathLike],
                  filetype='auto',
                  decoy_tag='rev_',
                  peptide_column: str = None,
                  protein_column: str = None,
                  tag_is_prefix: bool = True,
                  file_delimiter: str = '\t',
                  use_features: Union[List[str], None] = None):
        """
        Load the results of an upstream search or validation tool. PIN, pepXML, mzID, X! Tandem, Spectromine and
        generic tabular formats are accepted. To load POUT files, use the separate 'load_pout_data' function. You can
        load both PIN and POUT files from a single experiment using the separate 'load_percolator_data' function.
        Generic tabular files must contain a column titled 'Peptide' or 'peptide' which contains the peptide sequences.

        :param filepath: The path to the file you want to load. Can be absolute or relative.
        :param filetype: The type of file. Must be one of {'auto', 'pin', 'pepxml', 'tabular', 'mzid', 'tandem',
            'spectromine'}. If you choose 'auto', the file type will be inferred from the file extension. Be
            cautious when loading pepXML and X! Tandem files, as the extensions are similar. It is best to be explicit
            in these cases.
        :param decoy_tag: The decoy tag used in the upstream FASTA to differentiate decoy proteins from target proteins.
        :param protein_column: The header of the column containing the protein IDs. Required for tabular data of an
            unspecified format.
        :param tag_is_prefix: Whether or not the decoy tag is a prefix. If False, it is assumed the tag is a suffix.
        :param file_delimiter: The delimiter used if the file is tabular.
        :param use_features: A list of column headers to be used as training features. Not required  If your tabular data
            contains a column indicating the target/decoy label of each PSM, DO NOT INCLUDE THIS COLUMN! The label will
            be determined from the protein IDs.
        :return: None
        """

        if filetype == 'auto':
            if str(filepath).lower().endswith('pin') or str(filepath).lower().endswith('mhcv'):
                filetype = 'pin'
            elif str(filepath).lower().endswith('pepxml'):
                filetype = 'tandem'
            elif str(filepath).lower().endswith('pep.xml'):
                filetype = 'pepxml'
            else:
                raise ValueError('File type could not be inferred from filename. You must explicitly specify the '
                                 'filetype.')
        else:
            if filetype not in ['auto', 'pin', 'pepxml', 'tabular', 'mhcv']:
                raise ValueError("filetype must be one of "
                                 "{'auto', 'pin', 'pepxml', 'tabular', 'mhcv'}")

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file...')
        self.raw_data = load_file(filename=filepath, filetype=filetype, decoy_tag=decoy_tag,
                                  protein_column=protein_column, file_sep=file_delimiter,
                                  tag_is_prefix=tag_is_prefix, min_len=self.min_len, max_len=self.max_len)
        self.labels = self.raw_data['Label'].to_numpy()

        if peptide_column is not None:
            self.peptides = list(self.raw_data[peptide_column])
        elif filetype == 'pin':
            self.peptides = list(self.raw_data['Peptide'])
        #elif filetype == 'mzid':
        #    self.peptides = list(self.raw_data['PeptideSequence'])
        #elif filetype == 'spectromine':
        #    self.peptides = list(self.raw_data['PEP.StrippedSequence'])
        else:
            if 'peptide' in self.raw_data.columns:
                self.peptides = list(self.raw_data['peptide'])
            elif 'Peptide' in self.raw_data.columns:
                self.peptides = list(self.raw_data['Peptide'])
            else:
                raise IndexError('Peptide field could not be automatically found. Please indicate the column '
                                 'containing the peptide sequences')

        self.peptides = np.array(clean_peptide_sequences(self.peptides))
        self._check_peptide_lengths()

        print(f'Loaded {len(self.peptides)} PSMs')

        self.loaded_filetype = filetype
        self.filename = Path(filepath).name
        self.filepath = Path(filepath).expanduser().resolve()

        print('Preparaing training features')
        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_features)
        self._mhcflurry_predictions = False
        self._netmhcpan_predictions = False
        self._retention_time_features = False

    def load_pout_data(self,
                       targets_pout: Union[str, PathLike],
                       decoys_pout: Union[str, PathLike],
                       use_features: Union[List[str], None] = None) -> None:
        """
        Load POUT files generated by Percolator. You must have created both target and decoy POUT files from Percolator.

        :param targets_pout: The path to the targets POUT file.
        :param decoys_pout: The path to the decoys POUT file.
        :param use_features: (Optional) A list of features (i.e. columns) to load.
        :return: None
        """

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file')
        self.raw_data = load_pout_data(targets_pout, decoys_pout, self.min_len, self.max_len)
        self.labels = self.raw_data['Label'].values
        self.peptides = list(self.raw_data['peptide'])
        self.peptides = np.array(clean_peptide_sequences(self.peptides))
        self._check_peptide_lengths()

        # self.raw_data.drop(columns=['Label'], inplace=True)
        self.loaded_filetype = 'pout'
        self.filename = (Path(targets_pout).name, Path(decoys_pout).name)
        self.filepath = (Path(targets_pout).expanduser().resolve(), Path(decoys_pout).expanduser().resolve())

        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_features)
        self._mhcflurry_predictions = False
        self._netmhcpan_predictions = False
        self._retention_time_features = False

    def load_percolator_data(self,
                             pin_file: Union[str, PathLike],
                             target_pout_file: Union[str, PathLike],
                             decoy_pout_file: Union[str, PathLike],
                             use_features: Union[List[str], None] = None,
                             decoy_tag='rev_',
                             tag_is_prefix: bool = True
                             ) -> None:
        """
        Load PIN and POUT files from a single experiment.You must have created both target and decoy POUT files from
        Percolator.

        :param pin_file: Path to the PIN file.
        :param target_pout_file: The path to the targets POUT file.
        :param decoy_pout_file: The path to the decoys POUT file.
        :param use_features: (Optional) A list of features (i.e. columns) to load.
        :param decoy_tag: The decoy tag used to indicate decoys in the upstream FASTA file.
        :param tag_is_prefix: Whether or not the decoy tag is a prefix. If False, it is assumed the tag is a suffix.
        :return: None
        """
        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')
        print('Loading PSM file')

        pout_data = load_pout_data(target_pout_file, decoy_pout_file, self.min_len, self.max_len)

        pin_data = load_file(filename=pin_file, filetype='pin', decoy_tag=decoy_tag,
                             protein_column='Proteins', file_sep='\t',
                             tag_is_prefix=tag_is_prefix, min_len=self.min_len, max_len=self.max_len)
        pout_data.drop(columns=['peptide', 'proteinIds'], inplace=True)
        pin_data.drop(columns=['Label'], inplace=True)

        self.raw_data = pin_data.join(pout_data.set_index('PSMId'), on='SpecId')
        self.percolator_qs = self.raw_data['q-value'].to_numpy(np.float32)
        self.raw_data.drop(columns=['q-value'], inplace=True)

        self.labels = self.raw_data['Label'].to_numpy(np.float32)
        self.peptides = list(self.raw_data['Peptide'])
        self.peptides = np.array(clean_peptide_sequences(self.peptides))
        self._check_peptide_lengths()

        self.loaded_filetype = 'PIN_POUT'
        self.filename = (Path(pin_file).name,
                         Path(target_pout_file).name,
                         Path(target_pout_file).name)
        self.filepath = (Path(pin_file).expanduser().resolve(),
                         Path(target_pout_file).expanduser().resolve(),
                         Path(target_pout_file).expanduser().resolve())
        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype='pin',  # PIN file processing works for this
                                               use_features=use_features)
        self._mhcflurry_predictions = False
        self._netmhcpan_predictions = False
        self._retention_time_features = False

    def encode_peptide_sequences(self):
        """
        Use a BLOSUM62 substitution matrix to numerically encode each peptide sequence. Uses the EncodableSequences
        class from MhcFlurry. Encoded peptides are saved in self.encoded_peptides.
        :return:
        """

        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        self.encoded_peptides = deepcopy(encoded_peps)

    def add_mhcflurry_predictions(self, force: bool = False):
        """
        Run MhcFlurry and add presentation predictions to the training feature matrix.

        :return: None
        """
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using mhcvalidator.set_mhc_params')
        if self.mhc_class == 'II':
            raise RuntimeError('MhcFlurry is only compatible with MHC class I')
        if self.feature_matrix is None:
            raise RuntimeError('It looks like you haven\'t loaded any data. Load some data!')
        if np.max(np.vectorize(len)(self.peptides)) > 16:
            raise RuntimeError('MhcFlurry cannot make predictions on peptides over length 16.')
        if self._mhcflurry_predictions:
            if not force:
                raise RuntimeError('MhcFlurry predictions have already been added to this instance. If you want to '
                                   'run anyway, set the `force` argument to True.')
            else:
                print('MhcFlurry predictions have already been added to this instance. Forcing run.')
        print('Running MhcFlurry')

        # we will run MhcFlurry in a separate process so the Tensorflow space doesn't get messed up. I don't know why it
        # happens, but it does, and then either MhcFlurry or TensorFlow Decision Forests stops working.
        # I think perhaps because MhcFlurry uses some legacy code from TFv1 (I think), though this is only
        # a suspicion.

        with tempfile.NamedTemporaryFile('w', delete=False) as pep_file:
            pep_file.write('allele,peptide\n')
            for pep in self.peptides:
                for allele in self.alleles:
                    pep_file.write(f'{allele},{pep}\n')
            pep_file_path = pep_file.name
        with tempfile.NamedTemporaryFile('w') as results:
            results_file = results.name

        command = f'mhcflurry-predict --out {results_file} {pep_file_path}'.split()
        p = subprocess.Popen(command)
        _ = p.communicate()

        preds = pd.read_csv(results_file, index_col=False)

        self.mhcflurry_predictions = format_mhcflurry_predictions_dataframe(preds,
                                                                            self.peptides,
                                                                            True)

        self.feature_matrix = add_mhcflurry_to_feature_matrix(self.feature_matrix,
                                                              mhcflurry_predictions=preds,
                                                              from_file=True,
                                                              peptide_list=self.peptides)
        self._mhcflurry_predictions = True

    def add_netmhcpan_predictions(self, n_processes: int = 0, force: bool = False):
        """
        Run NetMHCpan and add presentation predictions to the training feature matrix.

        :return: None
        """
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using mhcvalidator.set_mhc_params')
        if self.mhc_class == 'II':
            alleles = [format_class_II_allele(x) for x in self.alleles]
        else:
            alleles = self.alleles

        if self.feature_matrix is None:
            raise RuntimeError('It looks like you haven\'t loaded any data. Load some data!')
        if self._netmhcpan_predictions:
            if not force:
                raise RuntimeError('NetMHCpan predictions have already been added to this instance. If you want to '
                                   'run anyway, set the `force` argument to True.')
            else:
                print('NetMHCpan predictions have already been added to this instance. Forcing run.')
        print(f'Running NetMHC{"II" if self.mhc_class == "II" else ""}pan')
        netmhcpan = NetMHCpanHelper(peptides=self.peptides,
                                    alleles=alleles,
                                    mhc_class=self.mhc_class,
                                    n_threads=n_processes)
        preds = netmhcpan.predict_df()
        self.netmhcpan_predictions = format_netmhcpan_prediction_dataframe(preds, peptide_list=list(self.peptides))
        to_drop = [x for x in preds.columns if 'rank' in x.lower()]
        preds.drop(columns=to_drop, inplace=True)
        self.feature_matrix = add_netmhcpan_to_feature_matrix(self.feature_matrix, preds)
        self._netmhcpan_predictions = True

    '''
    MhcNuggets does not seem to be working with more recent versions of tensorflow
    def add_mhcnuggets_predictions(self):
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using mhcvalidator.set_mhc_params')
        if self.mhc_class == 'II':
            alleles = [format_class_II_allele(x) for x in self.alleles]
        else:
            alleles = self.alleles

        preds = get_mhcnuggets_preds(self.mhc_class, alleles, self.peptides)
        self.feature_matrix = self.feature_matrix.join(preds)
    '''

    def add_all_available_predictions(self, verbose_errors: bool = False):
        """
        Try to run both MhcFlurry and NetMHCpan and add their predictions to the training features.

        :param verbose_errors:
        :return:
        """
        try:
            self.add_netmhcpan_predictions()
        except Exception as e:
            if verbose_errors:
                print(e)
            print(f'Unable to run NetMHC{"II" if self.mhc_class == "II" else ""}pan.'
                  f'{" See exception information above." if verbose_errors else ""}')
        try:
            self.add_mhcflurry_predictions()
        except Exception as e:
            if verbose_errors:
                print(e)
            print(f'Unable to run MhcFlurry.'
                  f'{" See exception information above." if verbose_errors else ""}')

    def make_autort_predictions(self,
                                mzml_file: Union[str, PathLike],
                                scan_list: List[int] = None,
                                epochs_for_prerun: int = 5,
                                qvalue_for_training: float = 0.01,
                                add_to_feature_matrix: bool = True,
                                force: bool = False):
        if (self._mhcflurry_predictions or self._netmhcpan_predictions) and not force:
            raise ValueError('MhcFlurry or NetMHCpan predictions have been added to the feature matrix. It is '
                             'recommended that you add AutoRT features prior to MHC predictions. To run anyway, '
                             'set the `force` argument to True.')

        if scan_list is None:
            scan_list = self.raw_data['ScanNr']
        observed_rts = extract_rt(scan_list, mzml_file)

        print('Running initial validation to get training set.')
        self.run(model=self.get_nn_model(), epochs=epochs_for_prerun, verbose=0, visualize=False)
        train_peptides = self.raw_data.loc[(self.qs <= qvalue_for_training) & (self.labels == 1), 'Peptide'].values
        train_rts = observed_rts[(self.qs <= qvalue_for_training) & (self.labels == 1)]
        max_rt = int(np.ceil(max(observed_rts)))

        predicted_rts = train_predict_rt(self.raw_data['Peptide'].values, train_peptides, train_rts,
                                         max_retention_time=max_rt, encode_modifications=True)

        self.obs_rts = observed_rts
        self.pred_rts = predicted_rts

        if add_to_feature_matrix:
            self.raw_data['mhcv_observed_rt'] = observed_rts
            self.raw_data['mhcv_predicted_rt'] = predicted_rts
            self.raw_data['rel_rt_error'] = (predicted_rts - observed_rts) / observed_rts
            self.raw_data['rt_error'] = predicted_rts - observed_rts
            self.feature_matrix['rel_rt_error'] = (predicted_rts - observed_rts) / observed_rts
            self.feature_matrix['rt_error'] = predicted_rts - observed_rts

        self._retention_time_features = True

    @staticmethod
    def _string_contains(string: str, pattern: Union[List[str], str]):
        if isinstance(pattern, str):
            pattern = [pattern]
        for x in pattern:
            if x in string:
                return True
        return False

    def get_qvalue_mask_from_features(self,
                                      X=None,
                                      y=None,
                                      cutoff: float = 0.05,
                                      n: int = 1,
                                      features_to_use: Union[List[str], str] = 'all',
                                      verbosity: int = 1):

        if isinstance(features_to_use, str):
            if features_to_use.lower() == 'mhc_only' and not (self._mhcflurry_predictions | self._netmhcpan_predictions):
                raise RuntimeError("mhc_only has been specified for creating a qvalue mask, but MHC predictions have not "
                                   "been added to the feature matrix.")

            if features_to_use == 'all':
                columns = list(self.feature_matrix.columns)
            elif features_to_use == 'mhc' or features_to_use == 'mhc_only':
                columns = [x for x in self.feature_matrix.columns if
                           self._string_contains(x.lower(), ['netmhcpan', 'mhcflurry', 'netmhciipan'])]
            else:
                columns = [features_to_use]
        else:
            columns = features_to_use

        if X is None:
            X = self.feature_matrix.copy(deep=True).to_numpy()
        else:
            X = np.array(X)
        if y is None:
            y = deepcopy(self.labels)

        col_idx = [list(self.feature_matrix.columns).index(x) for x in columns]
        passes = np.zeros(X.shape, dtype=bool)
        if verbosity == 1:
            print('Calculating masks based on feature q-values.')
        for i, col in enumerate(columns):
            ii = col_idx[i]
            col = col.lower()
            if len(np.unique(X[:, ii])) < 100:  # the data is too quantized
                continue
            if 'mass' in col or 'length' in col or 'mz' in col:  # we won't rank based on mass or length
                continue
            if verbosity > 1:
                print(f'Calculating q-values: {columns[i]}')
            higher_better = np.median(X[y == 1, ii]) > np.median(X[y == 0, ii])
            qs = calculate_qs(X[:, ii], y, higher_better)
            passes[:, ii] = (y == 1) & (qs <= cutoff)
            passes[y == 0, ii] = True  # keep all the decoys in there
            if verbosity > 1:
                print(f'  Target PSMs at {cutoff} FDR: {np.sum((y == 1) & (qs <= cutoff))}')
        mask = np.sum(passes, axis=1) >= n
        if verbosity > 1:
            print(f'Total target PSMs with {n} or more features passing {cutoff} q-value cutoff: {np.sum((mask) & (y == 1))}')
        return mask

    def _set_seed(self, random_seed: int = None):
        if random_seed is None:
            random_seed = self.random_seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    def _get_model_func(self, encode_peptide_sequences: bool = False):
        if encode_peptide_sequences:
            get_model = get_model_with_peptide_encoding
        else:
            get_model = get_model_without_peptide_encoding
        return get_model

    def _initialize_model(self, encode_peptide_sequences: bool = False,
                          hidden_layers: int = 3,
                          dropout: float = 0.5,
                          learning_rate: float = 0.001,
                          early_stopping_patience: int = 15,
                          loss_fn=tf.losses.BinaryCrossentropy()) -> [keras.Model, str, List[int]]:
        get_model_function = self._get_model_func(encode_peptide_sequences)

        # CREATE CALLBACKS
        now = str(datetime.now()).replace(' ', '_').replace(':', '-')
        model_name = str(self.model_dir / f'mhcvalidator_{now}.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(model_name,
                                                     monitor='val_loss', verbose=0,
                                                     save_best_only=True, mode='min')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

        callbacks_list = [checkpoint, early_stopping]
        # create optimizer, model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        if self.mhc_class == 'I':
            encoded_pep_length = self.max_len
        else:
            encoded_pep_length = 2 * self.max_len
        model = get_model_function(self.feature_matrix.shape[1],
                                   dropout=dropout,
                                   hidden_layers=hidden_layers,
                                   max_pep_length=encoded_pep_length)
        model.compile(loss=loss_fn,
                      optimizer=optimizer,
                      metrics=[global_accuracy])
        return model, model_name, callbacks_list

    @staticmethod
    def _simple_peptide_encoding(peptide: str):
        """
        Return the first four and last four amino acids of the peptide sequence.

        :param peptide: The peptide sequence.
        :return:
        """
        return list(peptide[:4] + peptide[-4:])

    def get_gradient_boosted_tree_model(self,
                                        num_trees: int = 2000,
                                        max_depth: int = 1,
                                        shrinkage: float = 0.05,
                                        tfdf_hyperparameter_template: str = 'benchmark_rank1',
                                        **kwargs):
        """
        Return a Tensorflow Decision Forest GradientBoostedTreesModel.

        :param num_trees: The maximum number of trees in the forest.
        :param max_depth: The depth of the forest.
        :param shrinkage: Shrinkage. Sort of analogous to learning rate.
        :param tfdf_hyperparameter_template: The name of a hyperparameter template to use. Default: benchmark_rank1
        :param kwargs: Additional keyword arguments to pass to tfdf.keras.GradientBoostedTreesModel
        :return: An instance of tfdf.keras.GradientBoostedTreesModel
        """

        import tensorflow_decision_forests as tfdf
        model = tfdf.keras.GradientBoostedTreesModel(num_trees=num_trees,
                                                     max_depth=max_depth,
                                                     shrinkage=shrinkage,
                                                     hyperparameter_template=tfdf_hyperparameter_template,
                                                     **kwargs)
        model.compile(metrics=['accuracy'])

        return model

    def get_nn_model(self,
                     learning_rate: float = 0.001,
                     dropout: float = 0.5,
                     hidden_layers: int = 2,
                     width_ratio: float = 5.0,
                     loss_fn=tf.losses.BinaryCrossentropy()
                     ):
        """
        Return a compiled multilayer perceptron neural network with the indicated architecture.

        :param learning_rate: Learning rate used by the optimizer (adam).
        :param dropout: Dropout between each layer.
        :param hidden_layers: Number of hidden layers.
        :param width_ratio: Ratio of width of hidden layers to width of input layer.
        :param loss_fn: The loss function to use.
        :return: A compiled keras.Model
        """

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = get_model_without_peptide_encoding(self.feature_matrix.shape[1],
                                                   dropout=dropout,
                                                   hidden_layers=hidden_layers,
                                                   max_pep_length=self.max_len,
                                                   width_ratio=width_ratio)
        model.compile(loss=loss_fn, optimizer=optimizer)

        return model

    def get_nn_model_with_sequence_encoding(self,
                                            learning_rate: float = 0.001,
                                            dropout: float = 0.5,
                                            hidden_layers: int = 2,
                                            width_ratio: float = 5.0,
                                            convolutional_layers: int = 1,
                                            filter_size: int = 4,
                                            n_filters: int = 12,
                                            filter_stride: int = 3,
                                            n_encoded_sequence_features: int = 6,
                                            loss_fn=tf.losses.BinaryCrossentropy()):
        """
        Return a compiled neural network, similar to get_nn_model but also includes a convolutional network for
        encoding peptide sequences which feeds into the multilayer perceptron.

        :param learning_rate: Learning rate used by the optimizer (adam).
        :param dropout:  Dropout between each layer.
        :param hidden_layers: Number of hidden layers.
        :param width_ratio: Ratio of width of hidden layers to width of input layer.
        :param convolutional_layers: Number of convolutional layers.
        :param filter_size: Convolution filter size.
        :param n_filters: Number of filters.
        :param filter_stride: Filter stride.
        :param n_encoded_sequence_features: Number of nodes in the output of the convolutional network.
        :param loss_fn: The loss function to use.
        :return: A compiled keras.Model
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = get_model_with_peptide_encoding(ms_feature_length=self.feature_matrix.shape[1],
                                                dropout=dropout,
                                                hidden_layers_after_convolutions=hidden_layers,
                                                after_convolutions_width_ratio=width_ratio,
                                                convolutional_layers=convolutional_layers,
                                                filter_size=filter_size,
                                                n_filters=n_filters,
                                                filter_stride=filter_stride,
                                                n_encoded_sequence_features=n_encoded_sequence_features,
                                                max_pep_length=self.max_len
                                                )
        model.compile(optimizer=optimizer, loss=loss_fn)

        return model

    @staticmethod
    def _visualize_splits(k_fold_splits, split='train', ax=None):
        """
        Visualize the distribution of examples between the k-fold splits.

        :param k_fold_splits: The splits, as returned by scikit-learn StratifiedKFold.
        :param split: Which split to visualize. If 'train', the training set will be visualized. Any other string
        will visualize the validation set.
        :param ax: Optional, a matplotlib axis object on which to plot.
        :return: None
        """

        if ax is None:
            fig, ax = plt.subplots()
        colors = get_cmap('tab10')
        n_splits = len(k_fold_splits)
        x = 0 if split == 'train' else 1

        if x == 0:
            alpha = 0.1
        else:
            alpha = min(1.0, 0.1 * (n_splits - 1))

        for i, split in enumerate(k_fold_splits):
            ax.vlines(split[x], i, i + 1, label=f'split {i + 1}',
                      lw=0.1, colors=colors(i), alpha=alpha)
        ax.set_yticks([])

    def run(self,
            model='BASIC',
            model_fit_function='fit',
            model_predict_function='predict',
            post_prediction_fn=lambda x: x,
            additional_training_data=None,
            return_prediction_data_and_model: bool = False,
            n_splits: int = 3,
            early_stopping_patience: int = 10,
            #q_value_subset: float = 1.0,
            #features_for_subset: Union[List[str], str] = 'all',
            #subset_threshold: int = 1,
            weight_by_inverse_peptide_counts: bool = True,
            visualize: bool = True,
            random_seed: int = None,
            clear_session: bool = True,
            alternate_labels=None,
            initial_model_weights: str = None,
            fit_model: bool = True,
            fig_pdf: Union[str, PathLike] = None,
            report_directory: Union[str, PathLike] = None,
            **kwargs):

        """
        Run the validation algorithm.

        :param model: The model to train on the target/decoy data. Can be a Python object with a fit and predict
        function, or string in {'BASIC', 'SEQUENCE_ENCODING'}. BASIC will load the basic fully-connected neural
        network supported by MhcValidator, while SEQUENCE_ENCODING will load the neural network which also performs
        convolutional peptide sequence encoding. Default is 'BASIC'.
        :param model_fit_function: The function which fits the model. Default is 'fit'.
        :param model_predict_function: The function used to make predictions with the fitted model. Default is 'predict'.
        :param post_prediction_fn: A function applied to the output of the predict function. Useful if the output is
        multidimensional, as all downstream processes expect a probability between 0 and 1.
        :param additional_training_data: Additional data for training the model. Only used if the model
        expects two inputs. If you are using the provided neural network which encodes peptide sequences, then you must
        pass self.X_encoded_peps.
        :param return_prediction_data_and_model: Whether to return predictions, q-values, etc in a dictionary. This
        data is available from the attributes of this MhcValidator instance after running, but it can be useful to
        return the data if you will be manipulating it downstream.
        :param n_splits: Number of splits used for training and validation/predicting (ala k-fold cross-validation).
        :param weight_by_inverse_peptide_counts: Whether to weight training by inverse peptide counts (i.e. number of
        times a sequence is identified in the data).
        :param visualize: Visualize the results.
        :param random_seed: Random seed used.
        :param clear_session: Clear the Tensorflow session before running.
        :param alternate_labels: Alternate labels to use for training. Possibly useful in an iterative variation of the
        algorithm.
        :param initial_model_weights: A file containing model weights to load before training using the models "load"
        function, if it has one.
        :param fit_model: Whether or not to fit the model. You would only set this to false if you were loading weights
        from an already-fitted model.
        :param fig_pdf: Filepath to save a PDF version of the training report.
        :param report_directory: Save all run information to a specified location. Includes: annotated input data,
        feature matrix, NetMHCpan and MHCFlurry predictions (if applicable), model weights, training report PDF.
        :param kwargs: Additional keyword arguments passed to model fit function.
        :return:
        """

        if clear_session:
            K.clear_session()

        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed(random_seed)

        if model == 'BASIC':
            model_args = {key: arg for key, arg in kwargs.items() if key in signature(self.get_nn_model).parameters}
            kwargs = {key: arg for key, arg in kwargs.items() if key not in model_args}
            model = self.get_nn_model(**model_args)
        elif model == 'SEQUENCE_ENCODING':
            model_args = {key: arg for key, arg in kwargs.items() if key in
                          signature(self.get_nn_model_with_sequence_encoding()).parameters}
            kwargs = {key: arg for key, arg in kwargs.items() if key not in model_args}
            model = self.get_nn_model_with_sequence_encoding(**model_args)
            if self.encoded_peptides is None:
                self.encode_peptide_sequences()
            additional_training_data = self.encoded_peptides

        if initial_model_weights is not None:
            model.load_weights(initial_model_weights)

        # check if the model is a Keras model, and if so check if number of epochs and batch size have been specified.
        # If they haven't set them to default values of 30 and 512, respectively. Otherwise things will go poorly.
        if isinstance(model, keras.Model):
            if 'epochs' not in kwargs.keys():
                print('`epochs` was not passed as a keyword argument. Setting it to default value of 30')
                kwargs['epochs'] = 30
            if 'batch_size' not in kwargs.keys():
                print('`batch_size` was not passed as a keyword argument. Setting it to default value of 512')
                kwargs['batch_size'] = 512

        # prepare data for training
        all_data = self.feature_matrix.copy(deep=True)

        if alternate_labels is None:
            labels = deepcopy(self.labels)
        else:
            labels = alternate_labels

        all_data = all_data.values
        peptides = self.peptides

        # we might make the splits better if our stratification takes feature q-values into account.
        # e.g. we calculate q-values for expect value and MHC predictions, and make sure we include good examples
        # from each allele.
        # stratification_labels = self.get_stratification_labels()
        '''skf = list(StratifiedKFold(n_splits=n_splits,
                                   random_state=random_seed,
                                   shuffle=True).split(all_data, labels))'''
        skf = k_fold_split(peptides=peptides, k_folds=n_splits, random_state=random_seed)

        predictions = np.zeros_like(labels, dtype=float)
        k_splits = np.zeros_like(labels, dtype=int)

        output = []
        history = []

        if isinstance(model, keras.Model):
            now = str(datetime.now()).replace(' ', '_').replace(':', '-')
            initial_model_weights = str(self.model_dir / f'mhcvalidator_initial_weights_{now}.h5')
            model.save(initial_model_weights)
        else:
            initial_model_weights = ''

        for k_fold, (train_index, predict_index) in enumerate(skf):
            print('-----------------------------------')
            print(f'Training on split {k_fold+1}')
            self._set_seed(random_seed)

            # make sure peptide sequences aren't duplicated bewtween train and test sets
            '''train_index, predict_index = resolve_duplicates_between_splits(index1=train_index,
                                                                           index2=predict_index,
                                                                           peptides=peptides,
                                                                           k=n_splits,
                                                                           random_seed=random_seed)'''

            if isinstance(model, keras.Model):
                model.load_weights(initial_model_weights)
            feature_matrix = deepcopy(all_data)

            '''if q_value_subset < 1.:
                mask = self.get_qvalue_mask_from_features(X=feature_matrix[train_index],
                                                          y=labels[train_index],
                                                          cutoff=q_value_subset,
                                                          n=subset_threshold,
                                                          features_to_use=features_for_subset,
                                                          verbosity=1)
            else:
                mask = np.ones_like(labels[train_index], dtype=bool)'''
            mask = np.ones_like(labels[train_index], dtype=bool)  # just in case we implement the q-value subset again

            x_train = deepcopy(feature_matrix[train_index, :][mask])
            rnd_idx = RandomState(random_seed).choice(len(x_train), len(x_train), replace=False)
            x_train = x_train[rnd_idx]
            x_predict = deepcopy(feature_matrix[predict_index, :])
            input_scalar = NDStandardScaler()
            input_scalar = input_scalar.fit(x_train)
            x_train = input_scalar.transform(x_train)
            x_predict = input_scalar.transform(x_predict)
            feature_matrix = input_scalar.transform(feature_matrix)

            x = deepcopy(feature_matrix)
            x_train = deepcopy(x_train)
            x_predict = deepcopy(x_predict)
            train_labels = labels[train_index][mask][rnd_idx]
            predict_labels = labels[predict_index]
            print(f' Training split - {np.sum(train_labels == 1)} targets | {np.sum(train_labels == 0)} decoys')
            print(f' Prediction split - {np.sum(predict_labels == 1)} targets | {np.sum(predict_labels == 0)} decoys')

            if weight_by_inverse_peptide_counts:
                pep_counts = Counter(peptides[train_index][mask])
                weights = np.array([np.sqrt(1 / pep_counts[p]) for p in peptides[train_index][mask][rnd_idx]])
            else:
                weights = np.ones_like(labels[train_index][mask][rnd_idx])

            if additional_training_data is not None:
                additional_training_data = deepcopy(additional_training_data)
                x2_train = additional_training_data[train_index][mask][rnd_idx]
                x2_test = additional_training_data[predict_index]
                input_scalar2 = NDStandardScaler()
                input_scalar2 = input_scalar2.fit(x2_train)

                x2_train = input_scalar2.transform(x2_train)
                x2_test = input_scalar2.transform(x2_test)
                additional_training_data = input_scalar2.transform(additional_training_data)

                x_train = (x_train, x2_train)
                x_predict = (x_predict, x2_test)
                x = (x, additional_training_data)

            model_fit_parameters = eval(f'signature(model.{model_fit_function})').parameters
            if 'validation_data' in model_fit_parameters.keys():
                val_str = 'validation_data=(x_predict, predict_labels),'
            else:
                val_str = ''

            if 'sample_weight' in model_fit_parameters.keys():
                weight_str = 'sample_weight=weights,'
            else:
                weight_str = ''

            if isinstance(model, keras.Model):
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    verbose=1,
                    mode="auto",
                    restore_best_weights=False)
                now = str(datetime.now()).replace(' ', '_').replace(':', '-')
                model_name = str(self.model_dir / f'mhcvalidator_k={k_fold+1}_{now}.h5')
                checkpoint = keras.callbacks.ModelCheckpoint(model_name,
                                                             monitor='val_loss', verbose=0,
                                                             save_best_only=True, mode='min')
                callbacks_str = 'callbacks=[early_stopping, checkpoint],'
            else:
                callbacks_str = ''
                model_name = ''

            # Train the model
            if fit_model:
                fit_history = eval(f"model.{model_fit_function}(x_train, train_labels, "
                                   f"{val_str} {weight_str} {callbacks_str} **kwargs)")
                if model_name != '':
                    model.load_weights(model_name)
                    if report_directory is not None:
                        model.save(Path(report_directory) / f'{Path(self.filename).stem}'
                                                            f'.mhcvalidator_model_k={k_fold+1}.h5')
            else:
                fit_history = None

            if fit_history is not None and hasattr(fit_history, 'history'):
                history.append(fit_history)

            predict_preds = post_prediction_fn(eval(
                f"model.{model_predict_function}(x_predict)")).flatten()  # all these predictions are assumed to be arrays. we flatten them because sometimes the have an extra dimension of size 1
            train_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_train)")).flatten()
            predict_qs = calculate_qs(predict_preds.flatten(), predict_labels)
            train_qs = calculate_qs(train_preds.flatten(), train_labels)
            preds = post_prediction_fn(eval(f"model.{model_predict_function}(x)")).flatten()
            qs = calculate_qs(preds.flatten(), labels)
            predictions[predict_index] = predict_preds
            k_splits[predict_index] = k_fold + 1
            assert np.all(predict_labels == self.labels[predict_index])

            train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.05)
            val_roc = calculate_roc(predict_qs, predict_labels, qvalue_cutoff=0.05)
            roc = calculate_roc(qs, labels, qvalue_cutoff=0.05)

            pep_level_qs, _, pep_level_labels, peps, pep_counts = calculate_peptide_level_qs(predict_preds,
                                                                                             predict_labels,
                                                                                             self.peptides[predict_index])

            print(f' | PSMs in this split validated at 1% FDR: {np.sum((predict_qs <= 0.01) & (predict_labels == 1))}')
            print(f' | Extrapolated to whole dataset: {np.sum((predict_qs <= 0.01) & (predict_labels == 1)) * n_splits}')
            print(f' | Peptides in this split validated at 1% FDR (peptide-level): '
                  f'{np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))}')
            print('-----------------------------------')

            results = {'train_preds': train_preds, 'train_labels': train_labels, 'train_qs': train_qs,
                           'train_roc': train_roc, 'predict_preds': predict_preds, 'predict_labels': predict_labels,
                           'predict_qs': predict_qs,
                           'predict_roc': val_roc, 'preds': preds, 'labels': labels, 'qs': qs, 'roc': roc, 'model': model,
                       'train_index': train_index, 'predict_index': predict_index}
            output.append(results)

        self.predictions = np.empty(len(labels), dtype=float)
        self.qs = np.empty(len(labels), dtype=float)

        self.predictions = predictions
        self.qs = calculate_qs(predictions, labels)
        self.roc = calculate_roc(self.qs, self.labels)

        pep_level_qs, _, pep_level_labels, pep_level_peps, pep_counts = calculate_peptide_level_qs(self.predictions,
                                                                                                   self.labels,
                                                                                                   self.peptides)

        print('===================================')
        print('Validation results')
        print(f' | PSMs validated at 1% FDR: {np.sum((self.qs <= 0.01) & (self.labels == 1))}')
        print(f' | Peptides validated at 1% FDR (peptide-level): '
              f'{np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))}')
        print('===================================')

        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        fig.suptitle(self.filename, fontsize=16)
        gs = GridSpec(2, 2, figure=fig)

        # train = fig.add_subplot(gs[:4, 0])
        # val = fig.add_subplot(gs[4:8, 0])
        final: plt.Axes = fig.add_subplot(gs[0, 0])

        if len(history) > 0:  # the model returned a fit history we can use here
            dist: plt.Axes = fig.add_subplot(gs[0, 1])
            loss: plt.Axes = fig.add_subplot(gs[1, 1])
            #train_split = fig.add_subplot(gs[8:10, 1])
            #val_split = fig.add_subplot(gs[10:, 1])
        else:
            loss: plt.Axes = None
            dist: plt.Axes = fig.add_subplot(gs[0, 1])
            #train_split = fig.add_subplot(gs[6:9, 1])
            #val_split = fig.add_subplot(gs[9:, 1])

        if self._retention_time_features:
            rt_corr: plt.Axes = fig.add_subplot(gs[1, 0])
            self.plot_retention_time_correlation(target_fdr=0.05, ax=rt_corr, visualize=False)

        colormap = get_cmap("tab10")

        # self._visualize_splits(skf, split='train', ax=train_split)
        # self._visualize_splits(skf, split='val', ax=val_split)
        # train_split.set_title('K-fold splits')
        # train_split.set_ylabel('Training')
        # val_split.set_ylabel('Validation')
        # val_split.set_xlabel('Scan number')

        if loss:
            min_x = []
            min_y = []
            for i, h in enumerate(history):
                loss.plot(range(1, len(h.history['val_loss']) + 1),
                          h.history['val_loss'], c=colormap(i), marker=None, label=f'split {i+1}')
                min_y.append(np.min(h.history['val_loss']))
                min_x.append(np.argmin(h.history['val_loss']) + 1)
            loss.plot(min_x, min_y, ls='none', marker='x', ms='12', c='k', label='best models')
            loss.set_title('Validation loss')
            loss.set_xlabel('Epoch')
            loss.set_ylabel('Loss')
            loss.legend()

        # for i, r in enumerate(output):
        #     train.plot(*r['train_roc'], c=colormap(i), ms='3', ls='none', marker='.', label=f'split {i+1}', alpha=0.6)
        #     val.plot(*r['predict_roc'], c=colormap(i), ms='3', ls='none', marker='.', label=f'split {i+1}', alpha=0.6)
        final.plot(*self.roc, c=colormap(0), ms='3', ls='none', marker='.', alpha=0.6)
        n_psms_at_1percent = np.sum((self.qs <= 0.01) & (self.labels == 1))
        final.vlines(0.01, 0, n_psms_at_1percent, ls='--', lw=1, color='k', alpha=0.7)
        final.hlines(n_psms_at_1percent, 0, 0.01, ls='--', lw=1, color='k', alpha=0.7)

        _, bins, _ = dist.hist(self.predictions[self.labels == 1], label='Target', bins=30, alpha=0.5, color='g')
        dist.hist(self.predictions[self.labels == 0], label='Decoy', bins=bins, alpha=0.5, zorder=100, color='r')

        # train.set_xlim((0, 0.05))
        # val.set_xlim((0, 0.05))
        final.set_xlim((0, 0.05))

        # train.set_title('Training data')
        # train.set_xlabel('q-value')
        # train.set_ylabel('PSMs')
        # train.set_ylim((0, train.get_ylim()[1]))

        # val.set_title('Validation data')
        # val.set_xlabel('q-value')
        # val.set_ylabel('PSMs')
        # val.set_ylim((0, val.get_ylim()[1]))

        final.set_title('Final q-values')
        final.set_xlabel('q-value')
        final.set_ylabel('PSMs')
        final.set_ylim((0, final.get_ylim()[1]))

        dist.set_title('Prediction distributions')
        dist.set_xlabel('Target probability')
        dist.set_ylabel('PSMs')

        # train.legend(markerscale=3)
        # val.legend(markerscale=3)
        dist.legend()

        plt.tight_layout()
        if visualize:
            fig.show()
        if fig_pdf is not None:
            pdf = plt_pdf.PdfPages(str(fig_pdf), keep_empty=False)
            pdf.savefig(fig)
            pdf.close()
        if report_directory is not None:
            pdf_file = Path(report_directory) / f'{Path(self.filename).stem}.MhcValidator_training_report.pdf'
            pdf = plt_pdf.PdfPages(str(pdf_file), keep_empty=False)
            pdf.savefig(fig)
            pdf.close()
        plt.close(fig)

        # make peptide-level q-value lookup
        pep_q_lookup = {pep: q for pep, q in zip(pep_level_peps, pep_level_qs)}

        self.raw_data['mhcv_peptide'] = self.peptides
        self.raw_data['mhcv_prob'] = self.predictions
        self.raw_data['mhcv_label'] = self.labels
        self.raw_data['mhcv_q-value'] = self.qs
        self.raw_data['mhcv_pep-level_q-value'] = np.array([pep_q_lookup[p] for p in self.peptides])
        self.raw_data['mhcv_k-fold_split'] = k_splits

        self.annotated_data = self.raw_data.copy(deep=True)

        if report_directory is not None:
            self.annotated_data.to_csv(Path(report_directory) /
                                       f'{Path(self.filename).stem}.MhcValidator_annotated.tsv',
                                       index=False, sep='\t')
            if self._mhcflurry_predictions:
                self.mhcflurry_predictions.to_csv(Path(report_directory) /
                                                  f'{Path(self.filename).stem}.MhcFlurry_Predictions.tsv',
                                                  index=False, sep='\t')
            if self._netmhcpan_predictions:
                self.netmhcpan_predictions.to_csv(Path(report_directory) /
                                                  f'{Path(self.filename).stem}.NetMHCpan_Predictions.tsv',
                                                  index=False, sep='\t')

        if return_prediction_data_and_model:
            return output, {'predictions': deepcopy(self.predictions),
                            'qs': deepcopy(self.qs),
                            'roc': deepcopy(self.roc)}

    def proph_run(self,
                  prophet_file,
                  model = 'BASIC',
                  decoy_prefix: str = 'rev_',
                  model_fit_function='fit',
                  model_predict_function='predict',
                  post_prediction_fn=lambda x: x,
                  additional_training_data_for_model=None,
                  return_prediction_data_and_model: bool = False,
                  n_splits: int = 3,
                  early_stopping_patience: int = 10,
                  weight_by_inverse_peptide_counts: bool = True,
                  visualize: bool = True,
                  random_seed: int = None,
                  clear_session: bool = True,
                  alternate_labels=None,
                  initial_model_weights: str = None,
                  fit_model: bool = True,
                  fig_pdf: Union[str, PathLike] = None,
                  report_directory: Union[str, PathLike] = None,
                  **kwargs):

        """Experimental! Loads PeptideProphet or iProphet files, splits them into individual experiments (keeping
        all respective search engine scores and Prophet scores), processes each with MhcValidator, keeps best scoring
        PSMs for each MS scan in the dataset, calculates q-values on the combined results. We have done no validation
        of this function whatsoever, but feel free to play around with it so long as you don't actually use the
        results at this point."""

        now = str(datetime.now()).replace(' ', '_').replace(':', '-')
        prophet_file = Path(prophet_file)
        if report_directory is None:
            report_directory = prophet_file.parent / f'{prophet_file.stem.split(".")[0]}_MhcValidator_{now}'

        mhcv_files = pepxml_to_mhcv(prophet_file,
                                    report_directory,
                                    decoy_prefix=decoy_prefix,
                                    split_output=True)

        run_results = {}

        results = pd.DataFrame()

        for mhcv_file in mhcv_files:
            run_report_dir = report_directory / mhcv_file.stem
            self.load_data(filepath=mhcv_file,
                           filetype='mhcv')
            self.encode_peptide_sequences()

            self.run(model,
                     model_fit_function,
                     model_predict_function,
                     post_prediction_fn,
                     additional_training_data_for_model,
                     return_prediction_data_and_model,
                     n_splits,
                     early_stopping_patience,
                     weight_by_inverse_peptide_counts,
                     visualize,
                     random_seed,
                     clear_session,
                     alternate_labels,
                     initial_model_weights,
                     fit_model,
                     fig_pdf,
                     report_directory=run_report_dir,
                     **kwargs)

            run_results[mhcv_file.name] = self.raw_data.copy(deep=True)
            results = results.append(self.raw_data, ignore_index=True)

        idx = results.groupby(['SpecId'])['mhcv_prob'].transform(max) == results['mhcv_prob']
        results = results[idx]
        results.reset_index(inplace=True)

        results['mhcv_q-value'] = calculate_qs(results['mhcv_prob'], results['mhcv_label'], higher_better=True)
        qs, _, pep_labels, peps, _ = calculate_peptide_level_qs(results['mhcv_prob'], results['mhcv_label'],
                                                                results['mhcv_peptide'], higher_better=True)
        peptide_level_qs = {p: q for p, q in zip(peps, qs)}
        results['mhcv_pep-level_q-value'] = results['mhcv_peptide'].apply(lambda x: peptide_level_qs[x])

        self.raw_data = results
        self.qs = results['mhcv_q-value'].values
        self.labels = results['mhcv_label'].values
        self.predictions = results['mhcv_prob'].values

        n_psms = np.sum((self.qs <= 0.01) & (self.labels == 1))
        n_peps = np.sum((qs <= 0.01) & (pep_labels == 1))

        report = ("=============== MhcValidator Prophet Report ===============\n"
                  f" | PSMs validated at 1% FDR: {n_psms}\n"
                  f" | Peptides validated at 1% FDR (peptide-level): {n_peps}\n"
                  f"===========================================================")

        return run_results

    def get_highest_scoring_PSMs(self):
        """
        Placeholder for now. It might be used for getting a single PSMs per spectrum if I implement running iProphet
        files.
        :return:
        """
        # here's something that will be handy
        idx = self.raw_data.groupby(['ScanNr'])['mhcv_prob'].transform(max) == self.raw_data['mhcv_prob']
        df = self.raw_data[idx]  # this is the highest scoring PSM per spectrum

    def plot_retention_time_correlation(self,
                                        target_fdr: float = 0.01,
                                        plot_decoys: bool = True,
                                        rt_units: str = 'min',
                                        ax: plt.Axes = None,
                                        visualize: bool = False):
        if not self._retention_time_features:
            raise Exception('You must first add AutoRT predictions.')
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot([10, 60], [10, 60], ls='--', c='k', alpha=0.6)
        ax.fill_between([10, 60], [5, 55], [15, 65], color='red', alpha=0.2)

        if plot_decoys:
            ax.plot(self.obs_rts[(self.labels == 0)], self.pred_rts[(self.labels == 0)],
                    ls='none', marker='.', ms='3', label='Decoys', alpha=0.1, c='k')

        ax.plot(self.obs_rts[(self.qs <= target_fdr) & (self.labels == 1)],
                self.pred_rts[(self.qs <= target_fdr) & (self.labels == 1)],
                c='green', ls='none', marker='.', ms='3', label=f'Targets at {int(target_fdr*100)}% FDR')
        ax.legend(markerscale=2)
        ax.set_xlabel(f'Observed retention time ({rt_units})')
        ax.set_ylabel(f'Predicted retention time ({rt_units})')
        if visualize:
            plt.show()
            plt.close()

    def optimize_arbitrary_model(self,
                                 model,
                                 space,
                                 instantiate_model: bool = False,
                                 model_fit_function='fit',
                                 model_predict_function='predict',
                                 post_prediction_fn=lambda x: x,
                                 objective_fn=None,
                                 additional_training_data_for_model=None,
                                 return_best_results: bool = False,
                                 fdr_of_interest: float = 0.01,
                                 algo=tpe.suggest,
                                 n_evals: int = 100,
                                 visualize_trials: bool = False,
                                 visualize_best: bool = True,
                                 report_dir: Union[str, PathLike] = None,
                                 random_seed: int = None,
                                 fig_pdf: Union[str, PathLike] = None,
                                 additional_model_kwargs=None,
                                 additional_fit_kwargs=None):

        K.clear_session()
        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed()
        fmin_rstate = np.random.default_rng(random_seed)
        if additional_fit_kwargs is None:
            additional_fit_kwargs = {}
        if additional_model_kwargs is None:
            additional_model_kwargs = {}

        if objective_fn is None:
            def objective_fn(params):
                print(f'Parameters: {params}')
                if instantiate_model:
                    model_params = params['model_params']
                    model_to_fit = model(**model_params, **additional_model_kwargs)
                else:
                    model_to_fit = model
                fit_params = params['fit_params']
                run_params = params['run_params']
                results, compiled = self.run(model=model_to_fit,
                                             model_fit_function=model_fit_function,
                                             model_predict_function=model_predict_function,
                                             post_prediction_fn=post_prediction_fn,
                                             additional_training_data=additional_training_data_for_model,
                                             return_prediction_data_and_model=True,
                                             visualize=visualize_trials,
                                             **run_params,
                                             **fit_params,
                                             **additional_fit_kwargs)

                n_good_targets = np.sum((self.qs <= fdr_of_interest) & (self.labels == 1))
                n_decoys = np.sum(self.labels == 0)
                n_targets = np.sum(self.labels == 1)
                n_possible = n_targets - n_decoys
                #n_psms_score = (np.abs(n_possible - n_good_targets) + 1) / n_possible
                n_psms_score = np.abs(n_possible - n_good_targets)

                # calculate the variance of the splits at psm_of_interest
                n_psms = [np.sum((r['predict_qs'] <= fdr_of_interest) & (r['predict_labels'] == 1)) for r in results]
                n_psms_variance = np.var(n_psms)
                n_psms_mean = np.mean(n_psms)
                if n_psms_variance == 0 or n_psms_mean == 0:
                    n_psms_rel_variance = 1
                else:
                    n_psms_rel_variance = n_psms_variance / n_psms_mean

                combined_loss = n_psms_score * n_psms_rel_variance

                print('Trial results:')
                print(f'\tTotal PSMs at {fdr_of_interest} FDR: {n_good_targets}')
                print(f'\tTotal PSMs at {fdr_of_interest} FDR loss: {n_psms_score}')
                print(f'\tVariance at {fdr_of_interest} FDR: {n_psms_variance}')
                print(f'\tRelative variance at {fdr_of_interest} FDR: {n_psms_rel_variance}')
                print(f'\tCombined loss: {combined_loss}\n')

                return combined_loss

        best = fmin(fn=objective_fn,
                    space=space,
                    algo=algo,
                    max_evals=n_evals,
                    rstate=fmin_rstate)

        best_args = space_eval(space, best)
        print(f'Best parameters found: {best_args}')
        print(f'Running best model')
        if instantiate_model:
            model_params = best_args['model_params']
            model_to_fit = model(**model_params, **additional_model_kwargs)
        else:
            model_to_fit = model
        # and here get only the params we need for this part
        fit_params = best_args['fit_params']
        run_params = best_args['run_params']
        results = self.run(model=model_to_fit,
                           model_fit_function=model_fit_function,
                           model_predict_function=model_predict_function,
                           post_prediction_fn=post_prediction_fn,
                           additional_training_data=additional_training_data_for_model,
                           return_prediction_data_and_model=True,
                           visualize=visualize_best,
                           **run_params,
                           **fit_params, **additional_fit_kwargs)
        if return_best_results:
            return results

    def test_optimize_arbitrary_model(self):
        model = self.get_nn_model_with_sequence_encoding
        space = {'model_params': {'hidden_layers': hp.choice('hidden_layers', (1, 2, 3)),
                                  'convolutional_layers': hp.choice('convolutional_layers', (1, 2))},
                 'fit_params': {'epochs': hp.uniformint('epochs', 10, 60)}}
        additional_model_kwargs = {}
        additional_fit_kwargs = {'verbose': 0, 'batch_size': 256}
        self.prepare_data()
        extra_data = self.X_encoded_peps

        self.optimize_arbitrary_model(model, space, True, additional_training_data_for_model=extra_data,
                                      n_evals=100, additional_model_kwargs=additional_model_kwargs,
                                      additional_fit_kwargs=additional_fit_kwargs)

    def optimize_nn_w_sequence_encoding(self, return_results: bool = False):
        model = self.get_nn_model_with_sequence_encoding
        space = {'model_params': {'hidden_layers': hp.choice('hidden_layers', (0, 1, 2, 3)),
                                  'convolutional_layers': hp.choice('convolutional_layers', (1, 2)),
                                  'dropout': hp.quniform('dropout', 0.4, 0.7, 0.05),
                                  'width_ratio': hp.uniform('width_ratio', 1, 3),
                                  'convolution_filter_size': scope.int(hp.quniform('convolution_filter_size', 4, 12, 2)),
                                  'convolution_filter_stride': hp.choice('convolution_filter_stride', (1, 2, 3)),
                                  'n_encoded_sequence_features': hp.choice('n_encoded_sequence_features', (3, 4, 5, 6, 7, 8, 9))},
                 'fit_params': {'epochs': hp.uniformint('epochs', 10, 90),
                                'batch_size': hp.choice('batch_size', 64, 128, 256, 512)}}
        additional_model_kwargs = {}
        additional_fit_kwargs = {'verbose': 0}
        self.prepare_data()
        extra_data = self.X_encoded_peps

        results = self.optimize_arbitrary_model(model, space, True, additional_training_data_for_model=extra_data,
                                                n_evals=100, additional_model_kwargs=additional_model_kwargs,
                                                additional_fit_kwargs=additional_fit_kwargs,
                                                return_best_results=True)
        if return_results:
            return results

    def optimize_nn(self, return_results: bool = False):
        model = self.get_nn_model
        space = {'model_params': {'hidden_layers': hp.choice('hidden_layers', (1, 2, 3)),
                                  'dropout': hp.quniform('dropout', 0.4, 0.7, 0.05),
                                  'width_ratio': hp.uniform('width_ratio', 1, 3)},
                 'fit_params': {'epochs': hp.uniformint('epochs', 10, 90),
                                'batch_size': hp.choice('batch_size', 64, 128, 256, 512)}}
        additional_model_kwargs = {}
        additional_fit_kwargs = {'verbose': 0}
        self.prepare_data()
        extra_data = self.X_encoded_peps

        results = self.optimize_arbitrary_model(model, space, True,
                                                n_evals=100, additional_model_kwargs=additional_model_kwargs,
                                                additional_fit_kwargs=additional_fit_kwargs,
                                                return_best_results=True)
        if return_results:
            return results

    def optimize_forest(self, return_results: bool = False):
        model = self.get_gradient_boosted_tree_model
        space = {'model_params': {'num_trees': scope.int(hp.quniform('num_trees', 50, 2000, 25)),
                                  'max_depth': hp.choice('max_depth', (1, 2, 3, 4)),
                                  'shrinkage': hp.uniform('shrinkage', 0.02, 0.2),
                                  #'growing_strategy': hp.choice('growing_strategy', ('LOCAL', 'BEST_FIRST_GLOBAL')),
                                  #'l1_regularization': hp.uniform('l1_regularization', 0, 1),
                                  #'l2_regularization': hp.uniform('l2_regularization', 0, 1),
                                  'min_examples': hp.uniformint('min_examples', 5, 20),
                                  },#'sampling_method': hp.choice('sampling_method', ('NONE', 'RANDOM', 'GOSS'))},
                 'fit_params': {'verbose': False},
                 'run_params': {'n_splits': hp.choice('n_splits', (2, 3, 4))}}#'q_value_subset': hp.uniform('q_value_subset', 0.01, 1)}}
        additional_model_kwargs = {}
        additional_fit_kwargs = {'verbose': True}
        self.prepare_data()
        extra_data = self.X_encoded_peps

        results = self.optimize_arbitrary_model3(model, space, True,
                                                n_evals=50, additional_model_kwargs=additional_model_kwargs,
                                                additional_fit_kwargs=additional_fit_kwargs,
                                                return_best_results=True)
        if return_results:
            return results

    def find_best_sequence_encoding_w_hyperopt(self,
                                               target_fdr: float = 0.01,
                                               epoch_range: tuple = (6, 120),
                                               batch_size_space: tuple = (64, 128, 256, 512),
                                               learning_rate_range: tuple = (0.0001, 0.01),
                                               dropout_space: tuple = (0.1, 0.7),
                                               latent_space_range: tuple = (3, 9),
                                               q_value_subset_range: tuple = (0.005, 0.2),
                                               q_value_subset_threshold_space: tuple = (1, 2),
                                               n_evals: int = 50,
                                               random_seed_choices: tuple = tuple(range(1)),
                                               visualize: bool = False,
                                               random_seed: int = None):

        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed()
        fmin_rstate = np.random.default_rng(random_seed)

        space = [hp.uniformint('sequence_encoding_epochs', *epoch_range),
                 hp.choice('sequence_encoding_batch_size', batch_size_space),
                 hp.uniform('sequence_encoding_learning_rate', *learning_rate_range),
                 hp.uniform('sequence_encoding_dropout', *dropout_space),
                 hp.uniformint('sequence_encoding_latent_space_size', *latent_space_range),
                 hp.choice('q_value_subset',
                           (hp.choice('no_q_cutoff', [1]), hp.uniform('q_cutoff', *q_value_subset_range))),
                 hp.choice('q_value_subset_threshold', q_value_subset_threshold_space),
                 hp.choice('random_seed', random_seed_choices)]

        def objective_fn(params):
            sequence_encoding_epochs = int(params[0])
            sequence_encoding_batch_size = int(params[1])
            sequence_encoding_learning_rate = params[2]
            sequence_encoding_dropout = params[3]
            latent_space_size = int(params[4])
            q_value_subset = params[5]
            q_value_subset_threshold = params[6]
            random_seed = int(params[5])

            train_qs, val_qs = self.add_peptide_encodings(epochs=sequence_encoding_epochs,
                                                          batch_size=sequence_encoding_batch_size,
                                                          learning_rate=sequence_encoding_learning_rate,
                                                          dropout=sequence_encoding_dropout,
                                                          n_encoded_features=latent_space_size,
                                                          random_seed=random_seed,
                                                          q_value_subset=q_value_subset,
                                                          subset_threshold=q_value_subset_threshold,
                                                          visualize=visualize,
                                                          return_train_validation_qs=True)

            encoding_train_roc = np.array([np.sum((self.pep_encoding_train_qs <= x) & (self.y_train == 1))
                                           for x in np.linspace(0, 0.05, 100)])
            encoding_test_roc = np.array([np.sum((self.pep_encoding_test_qs <= x) & (self.y_val == 1))
                                          for x in np.linspace(0, 0.05, 100)])
            encoding_diff_score = np.sum((np.abs(encoding_train_roc - encoding_test_roc) /
                                          np.max((encoding_train_roc, encoding_test_roc), axis=0)))

            max_targets = np.sum(self.labels == 1) - np.sum(self.labels == 0)
            num_targets_score = np.abs(max_targets - np.sum((self.pep_encoding_qs <= target_fdr) & (self.labels == 1))) / max_targets

            return np.mean((encoding_diff_score, num_targets_score))

        best = fmin(fn=objective_fn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n_evals,
                    rstate=fmin_rstate)
        return best

    def find_best_w_hyperopt(self,
                             n_trees_range: tuple = (100, 2000),
                             max_depth_space: tuple = (1, 2),
                             q_value_subset_range: tuple = (0.01, 0.2),
                             q_value_subset_threshold_space: tuple = (1, 2),
                             #shrinkage_range: tuple = (0.005, 0.1),
                             n_evals: int = 50,
                             target_fdr: float = 0.01,
                             random_seed_choices: tuple = tuple(range(1)),
                             visualize: bool = False,
                             random_seed: int = None):

        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed()
        fmin_rstate = np.random.default_rng(random_seed)

        space = [hp.quniform('num_trees', *n_trees_range, 10),
                 hp.choice('max_depth', max_depth_space),
                 hp.choice('q_value_subset',
                           (hp.choice('no_q_cutoff', [1]), hp.uniform('q_cutoff', *q_value_subset_range))),
                 hp.choice('q_value_subset_threshold', q_value_subset_threshold_space),
                 #hp.uniform('shrinkage', *shrinkage_range),
                 hp.choice('random_seed', random_seed_choices)]

        def objective_fn(params):
            num_trees = int(params[0])
            max_depth = int(params[1])
            q_value_subset = params[2]
            q_value_subset_threshold = params[3]
            #shrinkage = params[4]
            random_seed = int(params[4])

            self.run(num_trees=num_trees,
                     max_depth=max_depth,
                     q_value_subset=q_value_subset,
                     subset_threshold=q_value_subset_threshold,
                     #shrinkage=shrinkage,
                     visualize=visualize)

            train_roc = np.array([np.sum((self.train_qs <= x) & (self.train_labels == 1))
                                  for x in np.linspace(0, 0.05, 100)])
            test_roc = np.array([np.sum((self.test_qs <= x) & (self.test_labels == 1))
                                 for x in np.linspace(0, 0.05, 100)])
            #diff_score = np.sum(np.abs(train_roc - test_roc) / ((train_roc + test_roc) / 2))
            diff_score = np.sum((np.abs(train_roc - test_roc) / np.max((train_roc, test_roc), axis=0))**2)

            max_targets = np.sum(self.labels == 1) - np.sum(self.labels == 0)
            num_targets_score = np.abs(max_targets - np.sum((self.qs <= target_fdr) & (self.labels == 1))) / max_targets

            return np.mean((diff_score, num_targets_score))

        best = fmin(fn=objective_fn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n_evals,
                    rstate=fmin_rstate)
        return best



    def add_peptide_clustering(self,
                               expected_grouping: int = -1,
                               random_seed: int = None):
        if expected_grouping == -1:
            expected_grouping = len(self.alleles) + 1
        self._set_seed(random_seed)
        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        encoded_peps = keras.utils.normalize(encoded_peps, 0)
        shape = np.shape(encoded_peps)
        encoded_peps = encoded_peps.reshape((shape[0], shape[1]*shape[2]))

        bgm = GaussianMixture(expected_grouping)
        bgm.fit(encoded_peps)
        preds = bgm.predict_proba(encoded_peps)

        df = pd.DataFrame(columns=[f'gmmc@@{x}' for x in range(np.shape(preds)[1])], data=preds)
        self.feature_matrix.drop(columns=[x for x in self.feature_matrix.columns if 'gmmc@@' in x], inplace=True)
        self.feature_matrix = self.feature_matrix.join(df)

    def filter_library(self,
                       filepath: Union[str, PathLike],
                       lib_format: str = 'biognosys',
                       fdr: float = 0.01,
                       suffix: str = 'VFiltered',
                       pep_list: List[str] = None):
        if pep_list is None:
            peps_to_use = self.peptides[self.qs <= fdr]
        else:
            peps_to_use = list(pep_list)
        lib = load_library(filepath, lib_format)
        filtered_lib = filter_library(lib, peptide_list=peps_to_use, lib_format=lib_format)
        pout = Path(filepath).parent / (Path(filepath).stem + f'_{suffix}{Path(filepath).suffix}')
        filtered_lib.to_csv(pout, sep='\t')

    @staticmethod
    def plot_histogram(predictions,
                       labels,
                       title: str = None,
                       log_xscale: bool = False,
                       log_yscale: bool = False,
                       outdir: Union[str, PathLike] = None,
                       visualize: bool = True,
                       return_fig: bool = False):
        predictions = np.log10(predictions) if log_xscale else predictions
        fig, ax = plt.subplots()
        D, bins, _ = ax.hist(x=np.array(predictions[labels == 0]).flatten(), label='Decoy', bins=100, alpha=0.6,
                             zorder=999)
        T, bins, _ = ax.hist(x=np.array(predictions[labels == 1]).flatten(), label='Target', bins=100, alpha=0.6,
                             range=(bins[0], bins[-1]))
        plt.title(title)
        if log_yscale:
            ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel("log10(target probability)") if log_xscale else ax.set_xlabel("target probability")
        ax.set_ylabel("PSM count")
        if outdir is not None:
            plt.savefig(str(Path(outdir, f'{title.replace(" ", "_")}.svg')))
        if visualize:
            plt.show()
        if return_fig:
            return fig, ax
        else:
            plt.clf()
            plt.close()
            return None, None

    @staticmethod
    def plot_roc(roc,
                 outdir: Union[str, PathLike] = None,
                 visualize: bool = True,
                 title: str = 'ROC',
                 return_fig: bool = True):
        qs = roc[0][roc[0] <= 0.05]
        response = roc[1][roc[0] <= 0.05]
        fig, ax = plt.subplots()
        ax.plot(qs, response, ls='none', marker='.', ms=1)
        ax.set_xlim((0, 0.05))
        ax.set_xlabel('FDR')
        ax.set_ylabel('Number of PSMs')
        plt.title(title)
        if outdir is not None:
            fig.savefig(str(Path(outdir, f'{title}.svg')))
        if visualize:
            fig.show()
        if return_fig:
            return fig, ax
        else:
            plt.clf()
            plt.close()
            return None, None

    def get_peptide_list_at_fdr(self, fdr: float, label: int = 1, peptide_level: bool = False):
        if peptide_level:
            qs, _, labels, peps, _ = calculate_peptide_level_qs(self.predictions, self.labels,
                                                                self.peptides)
            return peps[(qs <= fdr) & (labels == label)]
        else:
            return self.peptides[(self.qs <= fdr) & (self.labels == label)]
