import os
import logging
import sys

import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy.random
import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Union, List
from os import PathLike, devnull
from pathlib import Path
from mhcvalidator.data_loaders import load_file, load_pout_data
from mhcvalidator.features import prepare_features, eliminate_common_peptides_between_sets
from mhcvalidator.predictions_parsers import add_mhcflurry_to_feature_matrix, add_netmhcpan_to_feature_matrix
from mhcvalidator.predictions_parsers import format_mhcflurry_predictions_dataframe, format_netmhcpan_prediction_dataframe
from mhcvalidator.netmhcpan_helper import NetMHCpanHelper, format_class_II_allele
from mhcvalidator.losses_and_metrics import i_dunno_bce, global_accuracy, pickTopPredictions, sliding_bce
from mhcvalidator.fdr import calculate_qs, calculate_peptide_level_qs, calculate_roc
import matplotlib.pyplot as plt
from mhcflurry.encodable_sequences import EncodableSequences
from mhcvalidator.models import get_model_without_peptide_encoding, get_model_with_peptide_encoding, peptide_sequence_encoder, peptide_sequence_autoencoder
from mhcvalidator.peptides import clean_peptide_sequences, remove_previous_and_next_aa
from mhcnames import normalize_allele_name
from copy import deepcopy
from scipy.stats import percentileofscore
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from mhcvalidator.libraries import load_library, filter_library
import tempfile
from collections import Counter
import tensorflow.python.util.deprecation as deprecation
from multiprocessing import Process, Queue
import tensorflow_decision_forests as tfdf
from matplotlib.gridspec import GridSpec
import subprocess
import matplotlib.backends.backend_pdf as plt_pdf
from contextlib import nullcontext
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from deeplc import DeepLC
from pyteomics.mzml import MzML
from sklearn.base import TransformerMixin
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.pyll.base import scope
from scipy.stats import pearsonr
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from matplotlib.cm import get_cmap

deprecation._PRINT_DEPRECATION_WARNINGS = False


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# This can be uncommented to prevent the GPU from getting used.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from scipy.stats import gmean as geom_mean

# tf.config.threading.set_inter_op_parallelism_threads(0)
# tf.config.threading.set_intra_op_parallelism_threads(0)
# tf.config.set_soft_device_placement(enabled=True)

DEFAULT_TEMP_MODEL_DIR = str(Path(tempfile.gettempdir()) / 'validator_models')


class NDStandardScaler(TransformerMixin):
    def __init__(self):
        self._scaler = MinMaxScaler(copy=True)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


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
        self.previous_aa: Union[List[str], None] = None
        self.next_aa: Union[List[str], None] = None
        self.mzml = None
        self.loaded_filetype: Union[str, None] = None
        self.random_seed = random_seed
        self.fit_history = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.X_train_peps = None
        self.X_val_peps = None
        self.X_train_encoded_peps = None
        self.X_val_encoded_peps = None
        self.X_encoded_peps = None
        self.y_val = None
        self.training_weights = None
        self.search_score_names: List[str] = []
        self.predictions = None
        self.training_predictions = None
        self.testing_predictions = None
        self.validation_predictions = None
        self.qs = None
        self.roc = None
        self.percolator_qs = None
        self.mhc_class: str = None
        self.alleles: List[str] = None
        self.min_len: int = 5
        self.max_len: int = 100
        self.model_dir = Path(model_dir)
        self.mhcflurry_graph = tf.Graph()
        self.mhcflurry_session = tf.compat.v1.Session(graph=self.mhcflurry_graph)
        self.tfdf_graph = tf.Graph()
        self.tfdf_session = tf.compat.v1.Session(graph=self.tfdf_graph)
        self._mhcflurry_predictions: bool = False
        self._netmhcpan_predictions: bool = False
        self.mhcflurry_predictions: pd.DataFrame = None
        self.netmhcpan_predictions: pd.DataFrame = None
        self.modificationss = {'15.9949': 'Oxidation',
                               '0.9840': 'Deamidation'}
        if max_threads < 1:
            self.max_threads: int = os.cpu_count()
        else:
            self.max_threads: int = max_threads


    def set_mhc_params(self,
                       alleles: Union[str, List[str]] = ('HLA-A0201', 'HLA-B0702', 'HLA-C0702'),
                       mhc_class: str = 'I') -> None:
        """
        Set the MHC-specific parameters.

        :param alleles: The alleles to be used by MhcFlurry or NetMHCpan.
        :param mhc_class: The MHC class of the peptides. Must be one of {'I', 'II'}
        :return: None
        """
        if isinstance(alleles, str):
            alleles = alleles
        assert mhc_class in ['I', 'II']
        assert len(alleles) >= 1
        self.alleles = [normalize_allele_name(a).replace('*', '').replace(':', '') for a in alleles]
        self.mhc_class = mhc_class
        if self.mhc_class == 'I':
            self.min_len = 8
            self.max_len = 15
        else:
            self.min_len = 9
            self.max_len = 30

    def load_data(self,
                  filepath: Union[str, PathLike],
                  filetype='auto',
                  decoy_tag='rev_',
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
            if str(filepath).lower().endswith('pin'):
                filetype = 'pin'
            elif str(filepath).lower().endswith('pepxml'):
                filetype = 'tandem'
            elif str(filepath).lower().endswith('pep.xml'):
                filetype = 'pepxml'
            elif str(filepath).lower().endswith('mzid'):
                filetype = 'mzid'
            else:
                raise ValueError('File type could not be inferred from filename. You must explicitly specify the '
                                 'filetype.')
        else:
            if filetype not in ['auto', 'pin', 'pepxml', 'tabular', 'mzid', 'tandem', 'spectromine']:
                raise ValueError("filetype must be one of "
                                 "{'auto', 'pin', 'pepxml', 'tabular', 'mzid', 'tandem', 'spectromine'}")

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file...')
        self.raw_data = load_file(filename=filepath, filetype=filetype, decoy_tag=decoy_tag,
                                  protein_column=protein_column, file_sep=file_delimiter,
                                  tag_is_prefix=tag_is_prefix, min_len=self.min_len, max_len=self.max_len)
        self.labels = self.raw_data['Label'].to_numpy()

        if filetype == 'pin':
            self.peptides = list(self.raw_data['Peptide'])
        elif filetype == 'mzid':
            self.peptides = list(self.raw_data['PeptideSequence'])
        elif filetype == 'spectromine':
            self.peptides = list(self.raw_data['PEP.StrippedSequence'])
        else:
            self.peptides = list(self.raw_data['peptide'])
        self.peptides = np.array(clean_peptide_sequences(self.peptides))

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
        # self.raw_data.drop(columns=['Label'], inplace=True)
        self.loaded_filetype = 'pout'
        self.filename = (Path(targets_pout).name, Path(decoys_pout).name)
        self.filepath = (Path(targets_pout).expanduser().resolve(), Path(decoys_pout).expanduser().resolve())

        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_features)
        self._mhcflurry_predictions = False
        self._netmhcpan_predictions = False

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

    def prepare_data(self, validation_split: float = 0.5,
                     random_seed: int = None,
                     stratification_dimensions: int = 2,
                     subset_by_feature_qvalue: bool = False,
                     subset_by_mhc_features_only: bool = False,
                     subset_by_prior_qvalues: bool = False,
                     qvalue_cutoff: float = 0.05,
                     n_features_to_meet_cutoff: int = 1,
                     verbosity: int = 0):
        """
        Encode peptide sequences and split examples into training, validation, and testing sets. Optionally, ensure
        there are no common peptides sequences between the sets (this might result in imbalanced sets). Note that the
        training set is implicitly what does not go into the validation and testing sets.
        Be sure to leave a reasonable amount of data for it.

        :param validation_split: float between 0 and 1. Portion of the dataset to go into the validation set.
        :param holdout_split: float between 0 and 1. Portion of the dataset to go into the testing set.
        :param random_seed:
        :param stratification_dimensions: An integer in {1, 2}. If 1, only the target-decoy label is used for
        stratification. If 2, new classes are made for MhcFlurry and/or NetMHCpan which indicates if the respective
        peptide is predicted to be presented by any alleles or not (i.e. %rank  > 2% for any alleles is class 0, else
        is class 1). This ensures there are predicted binders and non-binders in the training, validation, and testing
        sets. For high-quality data this is not really necessary, but for low quality data with few identifiable
        spectra or few predicted binders it can make a difference.
        :return:
        """
        if self.raw_data is None:
            raise AttributeError("Data has not yet been loaded.")
        if stratification_dimensions not in [1, 2]:
            raise ValueError("stratification_dimensions must be in {1, 2}.")

        if random_seed is None:
            random_seed = self.random_seed

        X = self.feature_matrix.to_numpy(dtype=np.float32)
        y = np.array(self.labels, dtype=np.float32)
        peptides = np.array(self.peptides, dtype='U100')

        assert X.shape[0] == y.shape[0]

        # save X and y before we do any shuffling. We will need this in the original order for predictions later
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        # encode all peptide sequences
        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        self.X_encoded_peps = deepcopy(encoded_peps)

        if subset_by_feature_qvalue:
            mask = self.get_qvalue_mask_from_features(qvalue_cutoff,
                                                      n_features_to_meet_cutoff,
                                                      'mhc' if subset_by_mhc_features_only else 'all',
                                                      verbosity=verbosity)
        elif subset_by_prior_qvalues:
            mask = (self.qs <= qvalue_cutoff) | (self.labels == 0)
        elif subset_by_prior_qvalues and subset_by_feature_qvalue:
            mask1 = self.get_qvalue_mask_from_features(qvalue_cutoff,
                                                       n_features_to_meet_cutoff,
                                                       'mhc' if subset_by_mhc_features_only else 'all',
                                                       verbosity=verbosity)
            mask2 = (self.qs <= qvalue_cutoff) | (self.labels == 0)
            mask = mask1 & mask2
        else:
            mask = np.full_like(y, fill_value=True, dtype=bool)

        X = X[mask]
        y = y[mask]
        encoded_peps = encoded_peps[mask]
        peptides = peptides[mask]

        if stratification_dimensions == 1:
            stratification = y
        else:
            mhcflurry = []
            netmhcpan = []
            for c in self.feature_matrix.columns:
                if 'MhcFlurry_PresentationScore' in c:
                    mhcflurry.append(c)
                elif 'NetMHCpan_EL_score' in c:
                    netmhcpan.append(c)
            if not mhcflurry and not netmhcpan:
                print("No MHC binding predictions found. Using target-decoy label only for stratification.")
                stratification = y
            else:
                decoy_medians = self.feature_matrix.loc[y == 0, mhcflurry + netmhcpan].median(axis=0)
                # are any predictions greater than twice the decoy median?
                y2 = np.max(self.feature_matrix > 2 * decoy_medians, axis=1).astype(int)
                stratification = np.concatenate((y[:, np.newaxis], y2[:, np.newaxis]), axis=1)

        # get training and testing sets
        (X_train, X_val, y_train, y_val, X_train_peps,
         X_val_peps, X_train_encoded_peps, X_val_encoded_peps, _, _) = train_test_split(
            X, y, peptides, encoded_peps, stratification,
            random_state=random_seed,
            stratify=stratification,
            test_size=validation_split
        )

        assert X_train.shape[0] == y_train.shape[0] == X_train_peps.shape[0]
        assert X_train.shape[1] == X_val.shape[1]

        # scale the Xs
        input_scalar = NDStandardScaler()
        input_scalar = input_scalar.fit(X_train)
        X_train = input_scalar.transform(X_train)
        X_val = input_scalar.transform(X_val)
        self.X = input_scalar.transform(self.X)

        # scale the encoded peptides
        input_scalar = input_scalar.fit(X_train_encoded_peps)
        X_train_encoded_peps = input_scalar.transform(X_train_encoded_peps)
        X_val_encoded_peps = input_scalar.transform(X_val_encoded_peps)
        self.X_encoded_peps = input_scalar.transform(self.X_encoded_peps)

        self.X_train = X_train
        self.X_val = X_val
        self.X_train_peps = X_train_peps
        self.X_val_peps = X_val_peps
        self.X_train_encoded_peps = X_train_encoded_peps
        self.X_val_encoded_peps = X_val_encoded_peps
        self.y_train = y_train
        self.y_val = y_val

    def add_mhcflurry_predictions(self):
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
        if self._mhcflurry_predictions:
            raise RuntimeError('MhcFlurry predictions have already been added to this instance.')
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

    def add_netmhcpan_predictions(self, n_processes: int = 0):
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
            raise RuntimeError('NetMHCpan predictions have already been added to this instance.')
        print(f'Running NetMHC{"II" if self.mhc_class == "II" else ""}pan')
        netmhcpan = NetMHCpanHelper(peptides=self.peptides,
                                    alleles=alleles,
                                    mhc_class=self.mhc_class,
                                    n_threads=n_processes)
        preds = netmhcpan.predict_df()
        self.netmhcpan_predictions = format_netmhcpan_prediction_dataframe(preds)
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

    def add_deeplc_modifications(self, mods = 'default'):
        if mods == 'default':
            mods = {'15.9949': 'Oxidation',
                    '0.9840': 'Deamidation'}
        for key, value in mods.items():
            self.modificationss[key] = value

    def add_deeplc_features(self,
                            rt_file: str,
                            rt_header: str = 'retention_time_sec',
                            calibration_qvalue: float = 0.01,
                            calibration_features: Union[str, List[str]] = 'lnExpect',
                            n_features_must_pass: int = 1):
        peptides = remove_previous_and_next_aa(self.raw_data['Peptide'])
        mask = self.get_qvalue_mask_from_features(cutoff=calibration_qvalue,
                                                  n=n_features_must_pass,
                                                  features_to_use=calibration_features)
        #mzml = MzML(mzml_file)



    @staticmethod
    def _string_contains(string: str, pattern: Union[List[str], str]):
        if isinstance(pattern, str):
            pattern = [pattern]
        for x in pattern:
            if x in string:
                return True
        return False

    def get_qvalue_mask_from_features(self,
                                      X = None,
                                      y = None,
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

    '''def _calculate_peptides_similarity(self):
        n_targets = np.sum(self.y == 1)
        n_decoys = np.sum(self.y == 0)

        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)

        encoded_decoys = encoded_peps[self.y == 0]
        encoded_targets = encoded_peps[self.y == 1]

        summed_null_encoding = np.sum(encoded_decoys, axis=0)
        summed_target_encoding = np.sum(encoded_targets, axis=0)

        desired_target_encoding = (summed_target_encoding - summed_null_encoding) / (n_targets - n_decoys)
        desired_target_encoding = (desired_target_encoding - np.min(desired_target_encoding)) / (np.max(desired_target_encoding) - np.min(desired_target_encoding))
        def normalize(x):
            x = x.flatten()
            return (x-np.amin(x))/(np.amax(x)-np.amin(x)).flatten().reshape(1, -1)
        similarity = np.array([euclidean(normalize(x), desired_target_encoding.reshape(1, -1)) for x in tqdm(encoded_peps)])'''

    @staticmethod
    def _simple_peptide_encoding(peptide: str):
        odd = len(peptide) % 2
        if len(peptide) == 9:
            return list(peptide) + [odd]
        elif len(peptide) == 8:
            return list(peptide[:3] + ' ' + peptide[3:]) + [odd]
        else:
            middle = int(np.floor((len(peptide)) / 2))
            return list(peptide[:3] + peptide[middle-1:middle+2] + peptide[-3:]) + [odd]

    def get_gradient_boosted_tree_model(self,
                                        num_trees: int = 2000,
                                        max_depth: int = 1,
                                        shrinkage: float = 0.05,
                                        tfdf_hyperparameter_template: str = 'benchmark_rank1',
                                        **kwargs):

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
                     hidden_layers: int = 3,
                     width_ratio: float = 3.0,
                     loss_fn=tf.losses.BinaryCrossentropy()
                     ):

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
                                            hidden_layers: int = 3,
                                            width_ratio: float = 3.0,
                                            convolutional_layers: int = 1,
                                            convolution_filter_size: int = 4,
                                            convolution_filter_stride: int = 3,
                                            n_encoded_sequence_features: int = 4,
                                            loss_fn=tf.losses.BinaryCrossentropy()):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = get_model_with_peptide_encoding(ms_feature_length=self.feature_matrix.shape[1],
                                                dropout=dropout,
                                                hidden_layers_after_convolutions=hidden_layers,
                                                after_convolutions_width_ratio=width_ratio,
                                                convolutional_layers=convolutional_layers,
                                                conv_filter_size=convolution_filter_size,
                                                conv_filter_stride=convolution_filter_stride,
                                                n_encoded_sequence_features=n_encoded_sequence_features
                                                )
        model.compile(optimizer=optimizer, loss=loss_fn)

        return model

    def run2(self,
            model,
            model_fit_function='fit',
            model_predict_function='predict',
            post_prediction_fn=lambda x: x,
            additional_training_data_for_model=None,
            return_prediction_data_and_model: bool = False,
            q_value_subset: float = 1.0,
            features_for_subset: Union[List[str], str] = 'all',
            subset_threshold: int = 1,
            encode_peptide_sequences: bool = False,
            weight_by_inverse_peptide_counts: bool = True,
            visualize: bool = True,
            report_dir: Union[str, PathLike] = None,
            random_seed: int = None,
            report_vebosity: int = 1,
            clear_session: bool = True,
            alternate_labels=None,
            initial_model_weights: str = None,
            keep_best_loss: bool = True,
            fig_pdf: Union[str, PathLike] = None,
            **kwargs):

        if clear_session:
            K.clear_session()

        self._set_seed(random_seed)

        if initial_model_weights is not None:
            model.load(initial_model_weights)

        # prepare data for training
        all_data = self.feature_matrix.copy(deep=True)

        if alternate_labels is None:
            labels = deepcopy(self.labels)
        else:
            labels = alternate_labels

        if q_value_subset < 1.:
            mask = self.get_qvalue_mask_from_features(cutoff=q_value_subset,
                                                      n=subset_threshold,
                                                      features_to_use=features_for_subset,
                                                      verbosity=1)
        else:
            mask = np.ones_like(labels, dtype=bool)

        if encode_peptide_sequences:
            all_data[['AA1', 'AA2', 'AA3', 'AAm-1', 'AAm', 'AAm+1', 'AA-3', 'AA-2', 'AA-1', 'odd_length']] = ''
            peps = pd.Series(data=self.peptides[mask])
            all_data.iloc[:, -10:] = pd.DataFrame(peps.apply(self._simple_peptide_encoding).to_list())

        idx = np.random.RandomState(self.random_seed).choice(np.sum(mask), np.sum(mask), False)
        n = int(len(idx) * 0.5)

        set1_indices = idx[:n]
        set2_indices = idx[n:]

        model1 = deepcopy(model)
        model2 = deepcopy(model)

        output = []

        for train_index, predict_index, model in [(set1_indices, set2_indices, model1), (set2_indices, set1_indices, model2)]:
            self._set_seed(random_seed=random_seed)
            feature_matrix = all_data.copy(deep=True)
            x_train = feature_matrix[mask].iloc[train_index, :].copy(deep=True)
            x_test = feature_matrix[mask].iloc[predict_index, :].copy(deep=True)
            input_scalar = NDStandardScaler()
            input_scalar = input_scalar.fit(x_train.values)
            x_train.loc[:, :] = input_scalar.transform(x_train.values)
            x_test.loc[:, :] = input_scalar.transform(x_test.values)
            feature_matrix.loc[:, :] = input_scalar.transform(feature_matrix.values)

            #x_train = tfdf.keras.pd_dataframe_to_tf_dataset(x_train, label="Label", weight=weight)
            #x_test = tfdf.keras.pd_dataframe_to_tf_dataset(x_test, label="Label")
            #x = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix, label='Label')
            x = deepcopy(feature_matrix.values)
            x_train = deepcopy(x_train.values)
            x_test = deepcopy(x_test.values)
            train_labels = labels[mask][train_index]
            test_labels = labels[mask][predict_index]

            if weight_by_inverse_peptide_counts:
                pep_counts = Counter(self.peptides)
                weights = np.array([np.sqrt(1 / pep_counts[p]) for p in self.peptides[train_index]])
            else:
                weights = np.ones_like(labels[train_index])

            if additional_training_data_for_model is not None:
                additional_training_data_for_model = deepcopy(additional_training_data_for_model)
                x2_train = additional_training_data_for_model[mask][train_index]
                x2_test = additional_training_data_for_model[mask][predict_index]
                input_scalar2 = NDStandardScaler()
                input_scalar2 = input_scalar2.fit(x2_train)

                x2_train = input_scalar2.transform(x2_train)
                x2_test = input_scalar2.transform(x2_test)
                additional_training_data_for_model = input_scalar2.transform(additional_training_data_for_model)

                #x2_train = tfdf.keras.pd_dataframe_to_tf_dataset(x2_train, weight=weights)
                #x2_test = tfdf.keras.pd_dataframe_to_tf_dataset(x2_test)
                #additional_training_data_for_model = tfdf.keras.pd_dataframe_to_tf_dataset(additional_training_data_for_model)

                x_train = (x_train, x2_train)
                x_test = (x_test, x2_test)
                x = (x, additional_training_data_for_model)

            exec(f"model.{model_fit_function}(x_train, train_labels, sample_weight=weights, **kwargs)")

            test_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_test)")).flatten()  # all these predictions are assumed to be arrays. we flatten them because sometimes the have an extra dimension of size 1
            train_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_train)")).flatten()
            test_qs = calculate_qs(test_preds.flatten(), test_labels)
            train_qs = calculate_qs(train_preds.flatten(), train_labels)
            preds = post_prediction_fn(eval(f"model.{model_predict_function}(x)")).flatten()
            qs = calculate_qs(preds.flatten(), labels)

            self.test_preds = test_preds
            self.train_preds = train_preds
            self.test_qs = test_qs
            self.train_qs = train_qs
            self.test_labels = test_labels
            self.train_labels = train_labels
            self.predictions = preds
            self.qs = qs

            if fig_pdf is not None:
                pdf = plt_pdf.PdfPages(str(fig_pdf), keep_empty=False)

            fig = plt.figure(constrained_layout=True, figsize=(10, 10))
            gs = GridSpec(2, 2, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])

            ax1.hist(train_preds[train_labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax1.hist(train_preds[train_labels == 1], label='Target', bins=30, alpha=0.5)
            ax1.legend()
            ax1.set_title("Training data")
            ax1.set_xlim((0, 1))
            ax1.set_xlabel('Target probability')
            # plt.yscale('log')

            ax2.hist(test_preds[test_labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax2.hist(test_preds[test_labels == 1], label='Target', bins=30, alpha=0.5)
            ax2.legend()
            ax2.set_title("Testing data")
            ax2.set_xlim((0, 1))
            ax2.set_xlabel('Target probability')
            # plt.yscale('log')

            train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.05)
            val_roc = calculate_roc(test_qs, test_labels, qvalue_cutoff=0.05)

            if len(train_roc[1]) > 0:
                train_at_01_fdr = train_roc[1][train_roc[0] <= 0.01][-1]
            else:
                train_at_01_fdr = 0
            if len(val_roc[1]) > 0:
                val_at_01_fdr = val_roc[1][val_roc[0] <= 0.01][-1]
            else:
                val_at_01_fdr = 0

            ax3.plot(train_roc[0], train_roc[1], ls='-', lw='0.5', marker='.', label='Training predictions', c='#1F77B4', alpha=1)
            ax3.plot(val_roc[0], val_roc[1], ls='-', lw='0.5', marker='.', label='Validation predictions', c='#FF7F0F', alpha=1)
            ax3.vlines(0.01, 0, max(train_at_01_fdr, val_at_01_fdr), colors='k', ls='--')
            ax3.plot((0, 0.01), (train_at_01_fdr, train_at_01_fdr), c='k', ls='--', ms=None)
            ax3.plot((0, 0.01), (val_at_01_fdr, val_at_01_fdr), c='k', ls='--', marker=None)
            ax3.legend()
            ax3.set_title("ROC curve")
            ax3.set_xlabel("FDR")
            ax3.set_ylabel("Number of PSMs")
            ax3.set_ylim((0, ax3.get_ylim()[1]))
            ax3.set_xlim((0, 0.05))

            if q_value_subset < 1:
                fig.suptitle("Training and validation\n"
                             f"Subset: {subset_threshold} or more features with q-value <= {q_value_subset}", fontsize=14)
            else:
                fig.suptitle("Training and validation", fontsize=14)
            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(constrained_layout=True, figsize=(8, 8))
            gs = GridSpec(2, 2, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0, :])
            ax3 = fig.add_subplot(gs[1, :])

            ax1.hist(preds[labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax1.hist(preds[labels == 1], label='Target', bins=30, alpha=0.5)
            ax1.legend()
            ax1.set_title("Prediction distribution")
            ax1.set_xlim((0, 1))
            # plt.yscale('log')

            roc = calculate_roc(qs, labels, qvalue_cutoff=0.05)

            self.predictions = preds
            self.qs = qs
            self.roc = roc

            ax3.plot(roc[0], roc[1], ls='-', lw='0.5', marker='.', c='#1F77B4', alpha=1)
            ax3.axvline(0.01, c='k', ls='--')
            ax3.axvline(0.05, c='k', ls='--')
            ax3.set_title("ROC curve")
            ax3.set_xlabel("FDR")
            ax3.set_ylabel("Number of PSMs")

            fig.suptitle("Predictions for all data", fontsize=14)
            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

            '''logs = model.make_inspector().training_logs()
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
            ax1.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
            ax1.set_xlabel("Number of trees")
            ax1.set_ylabel("Accuracy (out-of-bag)")
    
            ax2.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
            ax2.set_xlabel("Number of trees")
            ax2.set_ylabel("Logloss (out-of-bag)")
    
            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close()'''
            if fig_pdf is not None:
                pdf.close()

            n_targets = np.sum(self.labels == 1)
            n_decoys = np.sum(self.labels == 0)
            psms = self.peptides[(qs <= 0.01) & (self.labels == 1)]
            n_psm_targets = len(psms)
            n_unique_psms = len(set(psms))
            pep_level_qs, _, pep_level_labels, peps, pep_counts = calculate_peptide_level_qs(preds,
                                                                                             self.labels,
                                                                                             self.peptides)
            n_unique_peps = np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))

            ratio = len(train_labels) / len(test_labels)
            n_training_targets = np.sum((train_qs <= 0.01) & (train_labels == 1))
            n_testing_targets = np.sum((test_qs <= 0.01) & (test_labels == 1))

            report = '\n========== REPORT ==========\n\n' \
                     f'Total PSMs: {len(self.labels)}\n' \
                     f'Labeled as targets: {n_targets}\n' \
                     f'Labeled as decoys: {n_decoys}\n' \
                     f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                     f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                     '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                     f'Training PSMs at 1% FDR: {n_training_targets}\n' \
                     f'Testing PSMs at 1% FDR: {n_testing_targets}\n' \
                     f'Ratio of training set size to testing set size: {ratio}\n' \
                     f'Ratio of training PSMs to testing PSMs at 1% FDR: {n_training_targets / n_testing_targets}\n\n' \
                     f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                     f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                     f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n'

            if report_vebosity > 0:
                print(report)
            self.raw_data['mhcv_prob'] = list(self.predictions)
            self.raw_data['mhcv_q-value'] = list(self.qs)
            self.raw_data['mhcv_label'] = list(self.labels)
            self.raw_data['mhcv_peptide'] = list(self.peptides)

            output.append({'train_preds': train_preds, 'train_labels': train_labels, 'train_qs': train_qs,
                      'train_roc': train_roc, 'test_preds': test_preds, 'test_labels': test_labels, 'test_qs': test_qs,
                      'test_roc': val_roc, 'preds': preds, 'labels': labels, 'qs': qs, 'roc': roc, 'model': model})

        if return_prediction_data_and_model:
            return output

    def run3(self,
             model,
             model_fit_function='fit',
             model_predict_function='predict',
             post_prediction_fn=lambda x: x,
             additional_training_data_for_model=None,
             return_prediction_data_and_model: bool = False,
             n_splits: int = 3,
             q_value_subset: float = 1.0,
             features_for_subset: Union[List[str], str] = 'all',
             subset_threshold: int = 1,
             encode_peptide_sequences: bool = False,
             weight_by_inverse_peptide_counts: bool = True,
             visualize: bool = True,
             report_dir: Union[str, PathLike] = None,
             random_seed: int = None,
             report_vebosity: int = 1,
             clear_session: bool = True,
             alternate_labels=None,
             initial_model_weights: str = None,
             keep_best_loss: bool = True,
             fig_pdf: Union[str, PathLike] = None,
             disable_weights: bool = False,
             **kwargs):

        if clear_session:
            K.clear_session()

        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed(random_seed)

        if initial_model_weights is not None:
            model.load(initial_model_weights)

        # prepare data for training
        all_data = self.feature_matrix.copy(deep=True)

        if alternate_labels is None:
            labels = deepcopy(self.labels)
        else:
            labels = alternate_labels

        if encode_peptide_sequences:
            all_data[['AA1', 'AA2', 'AA3', 'AAm-1', 'AAm', 'AAm+1', 'AA-3', 'AA-2', 'AA-1', 'odd_length']] = ''
            peps = pd.Series(data=self.peptides)
            all_data.iloc[:, -10:] = pd.DataFrame(peps.apply(self._simple_peptide_encoding).to_list())

        all_data = all_data.values
        peptides = self.peptides

        shuffle_idx = np.random.RandomState(random_seed).choice(len(peptides), len(peptides), replace=False)
        all_data = all_data[shuffle_idx]
        peptides = peptides[shuffle_idx]
        labels = labels[shuffle_idx]

        skf = list(StratifiedKFold(n_splits=n_splits,
                                   random_state=random_seed,
                                   shuffle=True).split(all_data, labels))
        models = [deepcopy(model) for m in range(n_splits)]

        splits = list(zip(*zip(*skf), models))

        predictions = np.zeros_like(labels, dtype=float)

        output = []

        for train_index, predict_index, model in splits:
            self._set_seed(random_seed=random_seed)
            feature_matrix = deepcopy(all_data)

            if q_value_subset < 1.:
                mask = self.get_qvalue_mask_from_features(X=feature_matrix[train_index],
                                                          y=labels[train_index],
                                                          cutoff=q_value_subset,
                                                          n=subset_threshold,
                                                          features_to_use=features_for_subset,
                                                          verbosity=1)
            else:
                mask = np.ones_like(labels[train_index], dtype=bool)

            x_train = deepcopy(feature_matrix[train_index, :][mask])
            rnd_idx = RandomState(random_seed).choice(len(x_train), len(x_train), replace=False)
            x_train = x_train[rnd_idx]
            x_predict = deepcopy(feature_matrix[predict_index, :])
            input_scalar = NDStandardScaler()
            input_scalar = input_scalar.fit(x_train)
            x_train = input_scalar.transform(x_train)
            x_predict = input_scalar.transform(x_predict)
            feature_matrix = input_scalar.transform(feature_matrix)

            # x_train = tfdf.keras.pd_dataframe_to_tf_dataset(x_train, label="Label", weight=weight)
            # x_test = tfdf.keras.pd_dataframe_to_tf_dataset(x_test, label="Label")
            # x = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix, label='Label')
            x = deepcopy(feature_matrix)
            x_train = deepcopy(x_train)
            x_predict = deepcopy(x_predict)
            train_labels = labels[train_index][mask][rnd_idx]
            predict_labels = labels[predict_index]

            if weight_by_inverse_peptide_counts:
                pep_counts = Counter(peptides)
                weights = np.array([np.sqrt(1 / pep_counts[p]) for p in peptides[train_index][mask][rnd_idx]])
            else:
                weights = np.ones_like(labels[train_index][mask][rnd_idx])

            if additional_training_data_for_model is not None:
                additional_training_data_for_model = deepcopy(additional_training_data_for_model)[shuffle_idx]
                x2_train = additional_training_data_for_model[train_index][mask][rnd_idx]
                x2_test = additional_training_data_for_model[predict_index]
                input_scalar2 = NDStandardScaler()
                input_scalar2 = input_scalar2.fit(x2_train)

                x2_train = input_scalar2.transform(x2_train)
                x2_test = input_scalar2.transform(x2_test)
                additional_training_data_for_model = input_scalar2.transform(additional_training_data_for_model)

                x_train = (x_train, x2_train)
                x_predict = (x_predict, x2_test)
                x = (x, additional_training_data_for_model)

            if disable_weights:
                exec(f"model.{model_fit_function}(x_train, train_labels, **kwargs)")
            else:
                exec(f"model.{model_fit_function}(x_train, train_labels, sample_weight=weights, **kwargs)")

            predict_preds = post_prediction_fn(eval(
                f"model.{model_predict_function}(x_predict)")).flatten()  # all these predictions are assumed to be arrays. we flatten them because sometimes the have an extra dimension of size 1
            train_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_train)")).flatten()
            predict_qs = calculate_qs(predict_preds.flatten(), predict_labels)
            train_qs = calculate_qs(train_preds.flatten(), train_labels)
            preds = post_prediction_fn(eval(f"model.{model_predict_function}(x)")).flatten()
            qs = calculate_qs(preds.flatten(), labels)
            predictions[predict_index] = predict_preds
            assert np.all(predict_labels == self.labels[shuffle_idx][predict_index])

            if fig_pdf is not None:
                pdf = plt_pdf.PdfPages(str(fig_pdf), keep_empty=False)

            fig = plt.figure(constrained_layout=True, figsize=(10, 10))
            gs = GridSpec(2, 2, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])

            ax1.hist(train_preds[train_labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax1.hist(train_preds[train_labels == 1], label='Target', bins=30, alpha=0.5)
            ax1.legend()
            ax1.set_title("Training data")
            ax1.set_xlim((0, 1))
            ax1.set_xlabel('Target probability')
            # plt.yscale('log')

            ax2.hist(predict_preds[predict_labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax2.hist(predict_preds[predict_labels == 1], label='Target', bins=30, alpha=0.5)
            ax2.legend()
            ax2.set_title("Testing data")
            ax2.set_xlim((0, 1))
            ax2.set_xlabel('Target probability')
            # plt.yscale('log')

            train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.05)
            val_roc = calculate_roc(predict_qs, predict_labels, qvalue_cutoff=0.05)

            if len(train_roc[1][train_roc[0] <= 0.01]) > 0:
                train_at_01_fdr = train_roc[1][train_roc[0] <= 0.01][-1]
            else:
                train_at_01_fdr = 0
            if len(val_roc[1][val_roc[0] <= 0.01]) > 0:
                val_at_01_fdr = val_roc[1][val_roc[0] <= 0.01][-1]
            else:
                val_at_01_fdr = 0

            ax3.plot(train_roc[0], train_roc[1], ls='-', lw='0.5', marker='.', label='Training predictions',
                     c='#1F77B4', alpha=1)
            ax3.plot(val_roc[0], val_roc[1], ls='-', lw='0.5', marker='.', label='Validation predictions', c='#FF7F0F',
                     alpha=1)
            ax3.vlines(0.01, 0, max(train_at_01_fdr, val_at_01_fdr), colors='k', ls='--')
            ax3.plot((0, 0.01), (train_at_01_fdr, train_at_01_fdr), c='k', ls='--', ms=None)
            ax3.plot((0, 0.01), (val_at_01_fdr, val_at_01_fdr), c='k', ls='--', marker=None)
            ax3.legend()
            ax3.set_title("ROC curve")
            ax3.set_xlabel("FDR")
            ax3.set_ylabel("Number of PSMs")
            ax3.set_ylim((0, ax3.get_ylim()[1]))
            ax3.set_xlim((0, 0.05))

            if q_value_subset < 1:
                fig.suptitle("Training and validation\n"
                             f"Subset: {subset_threshold} or more features with q-value <= {q_value_subset}",
                             fontsize=14)
            else:
                fig.suptitle("Training and validation", fontsize=14)
            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(constrained_layout=True, figsize=(8, 8))
            gs = GridSpec(2, 2, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0, :])
            ax3 = fig.add_subplot(gs[1, :])

            ax1.hist(preds[labels == 0], label='Decoy', bins=30, alpha=0.5)
            ax1.hist(preds[labels == 1], label='Target', bins=30, alpha=0.5)
            ax1.legend()
            ax1.set_title("Prediction distribution")
            ax1.set_xlim((0, 1))
            # plt.yscale('log')

            roc = calculate_roc(qs, labels, qvalue_cutoff=0.05)

            ax3.plot(roc[0], roc[1], ls='-', lw='0.5', marker='.', c='#1F77B4', alpha=1)
            ax3.axvline(0.01, c='k', ls='--')
            ax3.axvline(0.05, c='k', ls='--')
            ax3.set_title("ROC curve")
            ax3.set_xlabel("FDR")
            ax3.set_ylabel("Number of PSMs")

            fig.suptitle("Predictions for all data", fontsize=14)
            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

            '''logs = model.make_inspector().training_logs()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
            ax1.set_xlabel("Number of trees")
            ax1.set_ylabel("Accuracy (out-of-bag)")

            ax2.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
            ax2.set_xlabel("Number of trees")
            ax2.set_ylabel("Logloss (out-of-bag)")

            if visualize:
                fig.show()
            if fig_pdf is not None:
                pdf.savefig(fig)
            plt.close()'''
            if fig_pdf is not None:
                pdf.close()

            n_targets = np.sum(labels == 1)
            n_decoys = np.sum(labels == 0)
            psms = peptides[(qs <= 0.01) & (labels == 1)]
            n_psm_targets = len(psms)
            n_unique_psms = len(set(psms))
            pep_level_qs, _, pep_level_labels, peps, pep_counts = calculate_peptide_level_qs(preds, labels, peptides)
            n_unique_peps = np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))

            ratio = len(train_labels) / len(predict_labels)
            n_training_targets = np.sum((train_qs <= 0.01) & (train_labels == 1))
            n_testing_targets = np.sum((predict_qs <= 0.01) & (predict_labels == 1))

            report = '\n========== REPORT ==========\n\n' \
                     f'Total PSMs: {len(labels)}\n' \
                     f'Labeled as targets: {n_targets}\n' \
                     f'Labeled as decoys: {n_decoys}\n' \
                     f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                     f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                     '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                     f'Training PSMs at 1% FDR: {n_training_targets}\n' \
                     f'Testing PSMs at 1% FDR: {n_testing_targets}\n' \
                     f'Ratio of training set size to testing set size: {ratio}\n' \
                     f'Ratio of training PSMs to testing PSMs at 1% FDR: {n_training_targets / n_testing_targets}\n\n' \
                     f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                     f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                     f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n'

            if report_vebosity > 0:
                print(report)
            #self.raw_data['mhcv_prob'] = list(self.predictions)
            #self.raw_data['mhcv_q-value'] = list(self.qs)
            #self.raw_data['mhcv_label'] = list(self.labels)
            #self.raw_data['mhcv_peptide'] = list(self.peptides)

            results = {'train_preds': train_preds, 'train_labels': train_labels, 'train_qs': train_qs,
                           'train_roc': train_roc, 'predict_preds': predict_preds, 'predict_labels': predict_labels,
                           'predict_qs': predict_qs,
                           'predict_roc': val_roc, 'preds': preds, 'labels': labels, 'qs': qs, 'roc': roc, 'model': model,
                       'train_index': train_index, 'predict_index': predict_index}
            output.append(results)

        self.predictions = np.empty(len(labels), dtype=float)
        self.qs = np.empty(len(labels), dtype=float)

        self.predictions[shuffle_idx] = predictions
        self.qs[shuffle_idx] = calculate_qs(predictions, labels)
        self.roc = calculate_roc(self.qs, self.labels)

        if visualize:
            fig = plt.figure(constrained_layout=True, figsize=(6, 10))
            gs = GridSpec(12, 2, figure=fig)
            fig.suptitle('ROC curves for each cross-validation split')
            # create sub plots as grid
            train = fig.add_subplot(gs[:4, 0])
            val = fig.add_subplot(gs[4:8, 0])
            all_pred = fig.add_subplot(gs[8:, 0])
            dist = fig.add_subplot(gs[:6, 1])
            final = fig.add_subplot(gs[6:, 1])

            colormap = get_cmap("tab10")
            for i, r in enumerate(output):
                train.plot(*r['train_roc'], c=colormap(i), ms='3', ls='none', marker='.', label=f'split {i+1}', alpha=0.6)
                val.plot(*r['predict_roc'], c=colormap(i), ms='3', ls='none', marker='.', label=f'split {i+1}', alpha=0.6)
                all_pred.plot(*r['roc'], c=colormap(i), ms='3', ls='none', marker='.', label=f'split {i+1}', alpha=0.6)
            final.plot(*self.roc, c=colormap(0), ms='3', ls='none', marker='.', alpha=0.6)

            train.set_xlim((0, 0.05))
            val.set_xlim((0, 0.05))
            all_pred.set_xlim((0, 0.05))
            final.set_xlim((0, 0.05))

            train.set_title('Training data')
            val.set_title('Validation data')
            all_pred.set_title('All data')
            final.set_title('Final q-values')

            train.legend(markerscale=3)
            val.legend(markerscale=3)
            all_pred.legend(markerscale=3)

            plt.show()
            plt.close()

        if return_prediction_data_and_model:
            return output, {'predictions': deepcopy(self.predictions),
                            'qs': deepcopy(self.qs),
                            'roc': deepcopy(self.roc),
                            'shuffle_index': shuffle_idx}

    def run(self,
            model,
            model_fit_function='fit',
            model_predict_function='predict',
            post_prediction_fn=lambda x: x,
            additional_training_data_for_model=None,
            return_prediction_data_and_model: bool = False,
            q_value_subset: float = 1.0,
            features_for_subset: Union[List[str], str] = 'all',
            subset_threshold: int = 1,
            encode_peptide_sequences: bool = False,
            validation_split: float = 0.5,
            weight_by_inverse_peptide_counts: bool = True,
            visualize: bool = True,
            report_dir: Union[str, PathLike] = None,
            random_seed: int = None,
            report_vebosity: int = 1,
            clear_session: bool = True,
            alternate_labels=None,
            initial_model_weights: str = None,
            keep_best_loss: bool = True,
            fig_pdf: Union[str, PathLike] = None,
            **kwargs):

        if clear_session:
            K.clear_session()

        self._set_seed(random_seed)

        if initial_model_weights is not None:
            model.load(initial_model_weights)

        # prepare data for training
        feature_matrix = self.feature_matrix.copy(deep=True)

        if alternate_labels is None:
            labels = deepcopy(self.labels)
        else:
            labels = alternate_labels

        if q_value_subset < 1.:
            mask = self.get_qvalue_mask_from_features(cutoff=q_value_subset,
                                                      n=subset_threshold,
                                                      features_to_use=features_for_subset,
                                                      verbosity=1)
        else:
            mask = np.ones_like(labels, dtype=bool)

        if encode_peptide_sequences:
            feature_matrix[['AA1', 'AA2', 'AA3', 'AAm-1', 'AAm', 'AAm+1', 'AA-3', 'AA-2', 'AA-1', 'odd_length']] = ''
            peps = pd.Series(data=self.peptides[mask])
            feature_matrix.iloc[:, -10:] = pd.DataFrame(peps.apply(self._simple_peptide_encoding).to_list())

        idx = np.random.RandomState(self.random_seed).choice(np.sum(mask), np.sum(mask), False)
        n = int(len(idx) * (1 - validation_split))

        x_train = feature_matrix[mask].iloc[idx[:n], :].copy(deep=True)
        x_test = feature_matrix[mask].iloc[idx[n:], :].copy(deep=True)
        input_scalar = NDStandardScaler()
        input_scalar = input_scalar.fit(x_train.values)
        x_train.loc[:, :] = input_scalar.transform(x_train.values)
        x_test.loc[:, :] = input_scalar.transform(x_test.values)
        feature_matrix.loc[:, :] = input_scalar.transform(feature_matrix.values)

        #x_train = tfdf.keras.pd_dataframe_to_tf_dataset(x_train, label="Label", weight=weight)
        #x_test = tfdf.keras.pd_dataframe_to_tf_dataset(x_test, label="Label")
        #x = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix, label='Label')
        x = feature_matrix.values
        x_train = x_train.values
        x_test = x_test.values
        train_labels = labels[mask][idx[:n]]
        test_labels = labels[mask][idx[n:]]

        if weight_by_inverse_peptide_counts:
            pep_counts = Counter(self.peptides)
            weights = np.array([np.sqrt(1 / pep_counts[p]) for p in self.peptides[idx[:n]]])
        else:
            weights = np.ones_like(labels[idx[:n]])

        if additional_training_data_for_model is not None:
            additional_training_data_for_model = deepcopy(additional_training_data_for_model)
            x2_train = additional_training_data_for_model[mask][idx[:n]]
            x2_test = additional_training_data_for_model[mask][idx[n:]]
            input_scalar2 = NDStandardScaler()
            input_scalar2 = input_scalar2.fit(x2_train)

            x2_train = input_scalar2.transform(x2_train)
            x2_test = input_scalar2.transform(x2_test)
            additional_training_data_for_model = input_scalar2.transform(additional_training_data_for_model)

            #x2_train = tfdf.keras.pd_dataframe_to_tf_dataset(x2_train, weight=weights)
            #x2_test = tfdf.keras.pd_dataframe_to_tf_dataset(x2_test)
            #additional_training_data_for_model = tfdf.keras.pd_dataframe_to_tf_dataset(additional_training_data_for_model)

            x_train = (x_train, x2_train)
            x_test = (x_test, x2_test)
            x = (x, additional_training_data_for_model)

        exec(f"model.{model_fit_function}(x_train, train_labels, sample_weight=weights, **kwargs)")

        test_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_test)")).flatten()  # all these predictions are assumed to be arrays. we flatten them because sometimes the have an extra dimension of size 1
        train_preds = post_prediction_fn(eval(f"model.{model_predict_function}(x_train)")).flatten()
        test_qs = calculate_qs(test_preds.flatten(), test_labels)
        train_qs = calculate_qs(train_preds.flatten(), train_labels)
        preds = post_prediction_fn(eval(f"model.{model_predict_function}(x)")).flatten()
        qs = calculate_qs(preds.flatten(), labels)

        self.test_preds = test_preds
        self.train_preds = train_preds
        self.test_qs = test_qs
        self.train_qs = train_qs
        self.test_labels = test_labels
        self.train_labels = train_labels
        self.predictions = preds
        self.qs = qs

        if fig_pdf is not None:
            pdf = plt_pdf.PdfPages(str(fig_pdf), keep_empty=False)

        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        gs = GridSpec(2, 2, figure=fig)

        # create sub plots as grid
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        ax1.hist(train_preds[train_labels == 0], label='Decoy', bins=30, alpha=0.5)
        ax1.hist(train_preds[train_labels == 1], label='Target', bins=30, alpha=0.5)
        ax1.legend()
        ax1.set_title("Training data")
        ax1.set_xlim((0, 1))
        ax1.set_xlabel('Target probability')
        # plt.yscale('log')

        ax2.hist(test_preds[test_labels == 0], label='Decoy', bins=30, alpha=0.5)
        ax2.hist(test_preds[test_labels == 1], label='Target', bins=30, alpha=0.5)
        ax2.legend()
        ax2.set_title("Testing data")
        ax2.set_xlim((0, 1))
        ax2.set_xlabel('Target probability')
        # plt.yscale('log')

        train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.05)
        val_roc = calculate_roc(test_qs, test_labels, qvalue_cutoff=0.05)

        if len(train_roc[1]) > 0:
            train_at_01_fdr = train_roc[1][train_roc[0] <= 0.01][-1]
        else:
            train_at_01_fdr = 0
        if len(val_roc[1]) > 0:
            val_at_01_fdr = val_roc[1][val_roc[0] <= 0.01][-1]
        else:
            val_at_01_fdr = 0

        ax3.plot(train_roc[0], train_roc[1], ls='-', lw='0.5', marker='.', label='Training predictions', c='#1F77B4', alpha=1)
        ax3.plot(val_roc[0], val_roc[1], ls='-', lw='0.5', marker='.', label='Validation predictions', c='#FF7F0F', alpha=1)
        ax3.vlines(0.01, 0, max(train_at_01_fdr, val_at_01_fdr), colors='k', ls='--')
        ax3.plot((0, 0.01), (train_at_01_fdr, train_at_01_fdr), c='k', ls='--', ms=None)
        ax3.plot((0, 0.01), (val_at_01_fdr, val_at_01_fdr), c='k', ls='--', marker=None)
        ax3.legend()
        ax3.set_title("ROC curve")
        ax3.set_xlabel("FDR")
        ax3.set_ylabel("Number of PSMs")
        ax3.set_ylim((0, ax3.get_ylim()[1]))
        ax3.set_xlim((0, 0.05))

        if q_value_subset < 1:
            fig.suptitle("Training and validation\n"
                         f"Subset: {subset_threshold} or more features with q-value <= {q_value_subset}", fontsize=14)
        else:
            fig.suptitle("Training and validation", fontsize=14)
        if visualize:
            fig.show()
        if fig_pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        gs = GridSpec(2, 2, figure=fig)

        # create sub plots as grid
        ax1 = fig.add_subplot(gs[0, :])
        ax3 = fig.add_subplot(gs[1, :])

        ax1.hist(preds[labels == 0], label='Decoy', bins=30, alpha=0.5)
        ax1.hist(preds[labels == 1], label='Target', bins=30, alpha=0.5)
        ax1.legend()
        ax1.set_title("Prediction distribution")
        ax1.set_xlim((0, 1))
        # plt.yscale('log')

        roc = calculate_roc(qs, labels, qvalue_cutoff=0.05)

        self.predictions = preds
        self.qs = qs
        self.roc = roc

        ax3.plot(roc[0], roc[1], ls='-', lw='0.5', marker='.', c='#1F77B4', alpha=1)
        ax3.axvline(0.01, c='k', ls='--')
        ax3.axvline(0.05, c='k', ls='--')
        ax3.set_title("ROC curve")
        ax3.set_xlabel("FDR")
        ax3.set_ylabel("Number of PSMs")

        fig.suptitle("Predictions for all data", fontsize=14)
        if visualize:
            fig.show()
        if fig_pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

        '''logs = model.make_inspector().training_logs()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
        ax1.set_xlabel("Number of trees")
        ax1.set_ylabel("Accuracy (out-of-bag)")

        ax2.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
        ax2.set_xlabel("Number of trees")
        ax2.set_ylabel("Logloss (out-of-bag)")

        if visualize:
            fig.show()
        if fig_pdf is not None:
            pdf.savefig(fig)
        plt.close()'''
        if fig_pdf is not None:
            pdf.close()

        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)
        psms = self.peptides[(qs <= 0.01) & (self.labels == 1)]
        n_psm_targets = len(psms)
        n_unique_psms = len(set(psms))
        pep_level_qs, _, pep_level_labels, peps, pep_counts = calculate_peptide_level_qs(preds,
                                                                                         self.labels,
                                                                                         self.peptides)
        n_unique_peps = np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))

        ratio = len(train_labels) / len(test_labels)
        n_training_targets = np.sum((train_qs <= 0.01) & (train_labels == 1))
        n_testing_targets = np.sum((test_qs <= 0.01) & (test_labels == 1))

        report = '\n========== REPORT ==========\n\n' \
                 f'Total PSMs: {len(self.labels)}\n' \
                 f'Labeled as targets: {n_targets}\n' \
                 f'Labeled as decoys: {n_decoys}\n' \
                 f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                 f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                 '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                 f'Training PSMs at 1% FDR: {n_training_targets}\n' \
                 f'Testing PSMs at 1% FDR: {n_testing_targets}\n' \
                 f'Ratio of training set size to testing set size: {ratio}\n' \
                 f'Ratio of training PSMs to testing PSMs at 1% FDR: {n_training_targets / n_testing_targets}\n\n' \
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                 f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n'

        if report_vebosity > 0:
            print(report)
        self.raw_data['mhcv_prob'] = list(self.predictions)
        self.raw_data['mhcv_q-value'] = list(self.qs)
        self.raw_data['mhcv_label'] = list(self.labels)
        self.raw_data['mhcv_peptide'] = list(self.peptides)

        if return_prediction_data_and_model:
            output = {'train_preds': train_preds, 'train_labels': train_labels, 'train_qs': train_qs,
                      'train_roc': train_roc, 'test_preds': test_preds, 'test_labels': test_labels, 'test_qs': test_qs,
                      'test_roc': val_roc, 'preds': preds, 'labels': labels, 'qs': qs, 'roc': roc, 'model': model}

            return output

        # self.run(initial_model_weights=model_name, learning_rate=second_fit_learning_rate)

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
        fmin_rstate = numpy.random.default_rng(random_seed)
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
                results = self.run(model=model_to_fit,
                                   model_fit_function=model_fit_function,
                                   model_predict_function=model_predict_function,
                                   post_prediction_fn=post_prediction_fn,
                                   additional_training_data_for_model=additional_training_data_for_model,
                                   return_prediction_data_and_model=True,
                                   visualize=visualize_trials,
                                   **run_params,
                                   **fit_params,
                                   **additional_fit_kwargs)

                upper_q = fdr_of_interest + 0.01
                lower_q = fdr_of_interest - 0.01
                train_q_dist, _ = np.histogram(results['train_qs'][(results['train_qs'] >= lower_q) &
                                                                   (results['train_qs'] <= upper_q) &
                                                                   (results['train_labels'] == 1)])
                val_q_dist, _ = np.histogram(results['test_qs'][(results['test_qs'] >= lower_q) &
                                                                (results['test_qs'] <= upper_q) &
                                                                (results['test_labels'] == 1)])
                dist_correlation_score = 1 - pearsonr(train_q_dist, val_q_dist)[0]

                cum_train_q_dist = [np.sum(train_q_dist[:i]) for i in range(len(train_q_dist))]
                cum_val_q_dist = [np.sum(val_q_dist[:i]) for i in range(len(val_q_dist))]
                cum_dist_correlation_score = 1 - pearsonr(cum_train_q_dist, cum_val_q_dist)[0]

                n_train = np.sum(train_q_dist)
                n_val = np.sum(val_q_dist)
                n_diff_psm_score = np.sum(np.abs(train_q_dist - val_q_dist) / np.max((train_q_dist, val_q_dist), axis=0))

                n_possible = np.sum(self.labels == 1) - np.sum(self.labels == 0)
                n_targets = np.sum((results['qs'] <= 0.01) & (results['labels'] == 1))
                n_psm_score = (n_possible - n_targets) / n_possible

                combined_loss = np.mean((n_diff_psm_score * 2, n_psm_score))

                print('Trial results:')
                print(f'\tDistribution correlation loss: {dist_correlation_score}')
                print(f'\tCumulative distribution correlation loss: {cum_dist_correlation_score}')
                print(f'\tDifference between train and val at {fdr_of_interest} FDR loss: {n_diff_psm_score}')
                print(f'\tPSMs at {fdr_of_interest} loss: {n_psm_score}')
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
                           additional_training_data_for_model=additional_training_data_for_model,
                           return_prediction_data_and_model=True,
                           visualize=visualize_best,
                           **run_params,
                           **fit_params, **additional_fit_kwargs)
        if return_best_results:
            return results

    def optimize_arbitrary_model3(self,
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
        fmin_rstate = numpy.random.default_rng(random_seed)
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
                results, compiled = self.run3(model=model_to_fit,
                                   model_fit_function=model_fit_function,
                                   model_predict_function=model_predict_function,
                                   post_prediction_fn=post_prediction_fn,
                                   additional_training_data_for_model=additional_training_data_for_model,
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
        results = self.run3(model=model_to_fit,
                            model_fit_function=model_fit_function,
                            model_predict_function=model_predict_function,
                            post_prediction_fn=post_prediction_fn,
                            additional_training_data_for_model=additional_training_data_for_model,
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
        fmin_rstate = numpy.random.default_rng(random_seed)

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
        fmin_rstate = numpy.random.default_rng(random_seed)

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

    def train_peptide_encoder(self,
                              epochs=20,
                              validation_split=0.5,
                              batch_size=64,
                              n_encoded_features=3,
                              learning_rate=0.001,
                              dropout: float = 0.5,
                              q_value_subset: float = 1.0,
                              features_for_subset: Union[List[str], str] = 'all',
                              subset_threshold: int = 1,
                              random_seed: int = None,
                              weight_by_peptide_counts: bool = True,
                              label_smoothing: float = 0.0,
                              visualize: bool = True,
                              pdf_out: Union[str, PathLike] = None):

        self._set_seed(random_seed)
        encoder: keras.Model = peptide_sequence_encoder(max_pep_length=self.max_len,
                                                        dropout=dropout,
                                                        encoding_size=n_encoded_features)
        #top_n_picker = pickTopPredictions(np.sum(self.labels == 1), np.sum(self.labels == 0), epochs)
        #callbacks = [top_n_picker]
        #loss_fn = sliding_bce(top_n_picker.top_n)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = tf.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.prepare_data(validation_split=validation_split, stratification_dimensions=1,
                          subset_by_feature_qvalue=(q_value_subset < 1),
                          qvalue_cutoff=q_value_subset,
                          n_features_to_meet_cutoff=subset_threshold
                          )

        if weight_by_peptide_counts:
            peptide_counts = Counter(self.X_train_peps)
            weights = np.array([1 / np.sqrt(peptide_counts[x]) for x in self.X_train_peps])
        else:
            weights = np.ones_like(self.y_train)

        self.fit_history = encoder.fit(self.X_train_encoded_peps,
                                       self.y_train,
                                       validation_data=(self.X_val_encoded_peps, self.y_val),
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       use_multiprocessing=self.max_threads > 1,
                                       workers=self.max_threads,
                                       verbose=2,
                                       sample_weight=weights)

        if pdf_out is not None:
            pdf = plt_pdf.PdfPages(pdf_out, False)

        plt.plot(self.fit_history.history['loss'], label='Training')
        plt.plot(self.fit_history.history['val_loss'], label='Validation')
        plt.legend()
        if visualize:
            plt.show()
        if pdf_out is not None:
            pdf.savefig()
        plt.close()

        preds = encoder.predict(self.X_encoded_peps).flatten()
        qs = calculate_qs(preds, self.labels)
        train_preds = encoder.predict(self.X_train_encoded_peps).flatten()
        val_preds = encoder.predict(self.X_val_encoded_peps).flatten()
        train_qs = calculate_qs(train_preds, self.y_train)
        val_qs = calculate_qs(val_preds, self.y_val)
        self.pep_encoding_train_qs = train_qs
        self.pep_encoding_test_qs = val_qs
        self.pep_encoding_qs = qs

        # These need to optionally not show the plot and save it to a PDF
        hist1, _ = self.plot_histogram(train_preds,
                            self.y_train,
                            title='Training predictions',
                            visualize=visualize,
                            return_fig=True)
        hist2, _ = self.plot_histogram(val_preds,
                            self.y_val,
                            title='Validation predictions',
                            visualize=visualize,
                            return_fig=True)
        hist3, _ = self.plot_histogram(preds,
                            self.y,
                            title='All predictions',
                            visualize=visualize,
                            return_fig=True)

        roc, _ = self.plot_roc(calculate_roc(qs, self.labels), visualize=visualize,
                               return_fig=True)

        train_roc = calculate_roc(train_qs, self.y_train,
                                  qvalue_cutoff=0.05)
        val_roc = calculate_roc(val_qs, self.y_val, qvalue_cutoff=0.05)
        rocs, _ = self.compare_rocs(train_roc, val_roc, visualize=visualize,
                                    return_fig=True)

        if pdf_out is not None:
            for fig in [hist1, hist2, hist3, roc, rocs]:
                pdf.savefig(fig)
            pdf.close()
        plt.close('all')

        print(f"Training target PSMs at 1% FDR: {np.sum((train_qs <= 0.01) & (self.y_train == 1))}")
        print(f"Validation target PSMs at 1% FDR: {np.sum((val_qs <= 0.01) & (self.y_val == 1))}")
        print(f"Target PSMs at 1% using peptide sequences alone: {np.sum((qs <= 0.01) & (self.y == 1))}")

        model = keras.Model(inputs=encoder.input, outputs=encoder.get_layer('encoded_peptides').output)
        encoded_peps = model.predict(self.X_encoded_peps)
        shape = np.shape(encoded_peps)

        return encoded_peps, train_qs, val_qs  # .reshape(shape[0], shape[1]*shape[2])

    def add_peptide_encodings(self,
                              epochs=20,
                              validation_split=0.5,
                              batch_size=64,
                              n_encoded_features: int = 3,
                              learning_rate=0.001,
                              q_value_subset: float = 1.0,
                              features_for_subset: Union[List[str], str] = 'all',
                              subset_threshold: int = 1,
                              dropout: float = 0.5,
                              random_seed: int = None,
                              weight_by_peptide_counts: bool = False,
                              label_smoothing: float = 0.0,
                              visualize: bool = True,
                              pdf_out: Union[str, PathLike] = None,
                              return_train_validation_qs: bool = False):

        encoded_peps, train_qs, val_qs = self.train_peptide_encoder(epochs=epochs,
                                                                    validation_split=validation_split,
                                                                    batch_size=batch_size,
                                                                    learning_rate=learning_rate,
                                                                    dropout=dropout,
                                                                    q_value_subset=q_value_subset,
                                                                    features_for_subset=features_for_subset,
                                                                    subset_threshold=subset_threshold,
                                                                    random_seed=random_seed,
                                                                    weight_by_peptide_counts=weight_by_peptide_counts,
                                                                    n_encoded_features=n_encoded_features,
                                                                    label_smoothing=label_smoothing,
                                                                    visualize=visualize,
                                                                    pdf_out=pdf_out)
        df = pd.DataFrame(data=encoded_peps, columns=[f'mhcv_seq_encoding@@{x}' for x in range(np.shape(encoded_peps)[1])])
        self.feature_matrix.drop(columns=[x for x in self.feature_matrix.columns
                                          if 'mhcv_seq_encoding@@' in x], inplace=True)
        self.feature_matrix = self.feature_matrix.join(df)
        if return_train_validation_qs:
            return train_qs, val_qs

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

    def grid_search(self,
                    batch_sizes: List[int] = (64, 128, 256, 512, 1028, 2056, 4112),
                    epochs: List[int] = (15, 30, 60, 120),
                    early_stopping_patiences: List[int] = (15,),
                    learning_rates: List[float] = (0.001,),
                    holdout_splits: List[float] = (0.25,),
                    validation_splits: List[float] = (0.25,),
                    hidden_layers=(3,),
                    dropouts=(0.6,),
                    output_dir: str = None,
                    encode_peptide_sequences: bool = False,
                    test_stratify_based_on_MHC_presentation: bool = True,
                    visualize: bool = False,
                    title: str = None):
        """
        Run a simple grid search of the following hyperparameters: batch_size, epochs, learning_rate, early
        stopping patience. The training curves (loss and accuracy) are plotted for each, along with the number
        of PSMs and unique peptides identified at 1% FDR. Note that using many values for each parameter will
        result in a very long run time and generate many many plots. It is best to start with a coarse grid and
        refine later.

        :param dropouts:
        :param title:
        :param visualize:
        :param test_stratify_based_on_MHC_presentation:
        :param hidden_layers:
        :param batch_sizes:
        :param epochs:
        :param early_stopping_patiences:
        :param learning_rates:
        :param holdout_splits:
        :param validation_splits:
        :param output_dir:
        :param encode_peptide_sequences:
        :return:
        """

        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)
        max_accuracy = round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)
        theoretical_possible_targets = n_targets - n_decoys

        import matplotlib.backends.backend_pdf as plt_pdf
        if output_dir is None:
            time = str(datetime.now()).replace(' ', '_').replace(':', '-')
            pdf_file = f'/tmp/mhcvalidator_gridsearch_{time}.pdf'
        else:
            if not Path(output_dir).exists():
                Path(output_dir).mkdir()
            pdf_file = str(Path(output_dir) / 'mhcvalidator_gridsearch.pdf')

        if test_stratify_based_on_MHC_presentation:
            stratify = [True, False]
        else:
            stratify = [False]

        total = len(batch_sizes) * len(epochs) * len(stratify) * len(early_stopping_patiences) * len(learning_rates) * \
                len(validation_splits) * len(holdout_splits) * len(dropouts) * len(hidden_layers)
        i = 1

        print(f'Saving plots to {pdf_file}')

        with plt_pdf.PdfPages(pdf_file) as pdf:
            for learning_rate in learning_rates:
                for max_epochs in epochs:
                    for batch_size in batch_sizes:
                        for validation_split in validation_splits:
                            for holdout_split in holdout_splits:
                                for early_stopping_patience in early_stopping_patiences:
                                    for layers in hidden_layers:
                                        for dropout in dropouts:
                                            for test_stratify in stratify:
                                                print(f'\nStep {i}/{total}'
                                                      f' - epochs: {max_epochs}'
                                                      f' - batch size: {batch_size}'
                                                      f' - early stopping patience: {early_stopping_patience}'
                                                      f' - holdout split: {holdout_split}'
                                                      f' - validation split: {validation_split}'
                                                      f' - learning rate: {learning_rate}')
                                                print('Fitting model...')

                                                i += 1
                                                self.run(batch_size=batch_size,
                                                         epochs=max_epochs,
                                                         learning_rate=learning_rate,
                                                         early_stopping_patience=early_stopping_patience,
                                                         fit_verbosity=0,
                                                         report_vebosity=0,
                                                         visualize=False,
                                                         encode_peptide_sequences=encode_peptide_sequences,
                                                         validation_split=validation_split,
                                                         holdout_split=holdout_split,
                                                         dropout=dropout,
                                                         hidden_layers=layers,
                                                         stratify_based_on_MHC_presentation=test_stratify)

                                                xs = range(1, len(self.fit_history.history['val_loss']) + 1)

                                                val_loss = np.min(self.fit_history.history['val_loss'])
                                                stopping_idx = self.fit_history.history['val_loss'].index(val_loss)

                                                n_psms_01 = np.sum((self.qs <= 0.01) & (self.labels == 1))
                                                n_uniqe_peps_01 = len(
                                                    np.unique(self.peptides[(self.qs <= 0.01) & (self.labels == 1)]))

                                                n_psms_05 = np.sum((self.qs <= 0.05) & (self.labels == 1))
                                                n_uniqe_peps_05 = len(
                                                    np.unique(self.peptides[(self.qs <= 0.05) & (self.labels == 1)]))

                                                text = f'Estimated max possible target PSMs: {theoretical_possible_targets}\n' \
                                                       f'  Target PSMs at 1% FDR: {n_psms_01}\n' \
                                                       f'  Target peptides at 1% FDR: {n_uniqe_peps_01}\n' \
                                                       f'  Target PSMs at 5% FDR: {n_psms_05}\n' \
                                                       f'  Target peptides at 5% FDR: {n_uniqe_peps_05}\n\n' \
                                                       f'Title: {title}\n' \
                                                       f'  stratification based on MHC preds: {test_stratify}\n' \
                                                       f'  epochs: {max_epochs}\n' \
                                                       f'  batch size: {batch_size}\n' \
                                                       f'  hidden layers: {layers}\n' \
                                                       f'  dropout: {dropout}\n' \
                                                       f'  early stopping patience: {early_stopping_patience}\n' \
                                                       f'  holdout split: {holdout_split}\n' \
                                                       f'  validation split: {validation_split}\n' \
                                                       f'  learning rate: {learning_rate}'

                                                fig, (ax, text_ax) = plt.subplots(2, 1, figsize=(8, 10))
                                                text_ax.axis('off')

                                                tl = ax.plot(xs, self.fit_history.history['loss'], c='#3987bc',
                                                             label='Training loss')
                                                vl = ax.plot(xs, self.fit_history.history['val_loss'], c='#ff851a',
                                                             label='Validation loss')
                                                ax.set_ylabel('Loss')
                                                ax2 = ax.twinx()
                                                ta = ax2.plot(xs, self.fit_history.history['global_accuracy'],
                                                              c='#3987bc', label='Training accuracy', ls='--')
                                                va = ax2.plot(xs, self.fit_history.history['val_global_accuracy'],
                                                              c='#ff851a', label='Validation accuracy', ls='--')
                                                ax2.set_ylabel('Accuracy (targets and decoys)')
                                                ax.plot(range(1, max_epochs + 1), [val_loss] * max_epochs, ls=':',
                                                        c='gray')
                                                ma = ax2.plot(range(1, max_epochs + 1), [max_accuracy] * max_epochs,
                                                              ls='-.', c='k', zorder=0,
                                                              label='Predicted max accuracy')
                                                bm = ax.plot(self.fit_history.history['val_loss'].index(val_loss) + 1,
                                                             val_loss,
                                                             marker='o', mec='red', mfc='none', ms='12', ls='none',
                                                             label='best model')

                                                lines = tl + vl + bm + ta + va + ma
                                                labels = [l.get_label() for l in lines]
                                                plt.legend(lines, labels, bbox_to_anchor=(0, -.12, 1, 0),
                                                           loc='upper center',
                                                           mode='expand', ncol=2)

                                                ax.set_xlabel('Epoch')
                                                ylim = ax.get_ylim()

                                                ax.plot([stopping_idx + 1, stopping_idx + 1], [0, 1], ls=':', c='gray')
                                                ax.set_ylim(ylim)
                                                ax.set_xlim((1, max_epochs))
                                                ax2.set_xlim((1, max_epochs))

                                                text_ax.text(0, 0.1, text, transform=text_ax.transAxes, size=12)

                                                plt.tight_layout()
                                                if visualize:
                                                    plt.show()

                                                pdf.savefig(fig)
                                                plt.close('all')

                                                # if output_dir:
                                                #    self.raw_data.to_csv(str(Path(output_dir) / f'{self.filename}_MhcV.txt'),
                                                #                         index=False)

    @staticmethod
    def compare_rocs(train_roc,
                     val_roc,
                     title: str = "Comparison of training and validation ROCs",
                     visualize: bool = True,
                     return_fig: bool = True):

        fig, ax = plt.subplots()
        ax.plot(train_roc[0], train_roc[1], ls='-', lw=0.5, marker='.', label='Training predictions', c='#1F77B4')
        ax.plot(val_roc[0], val_roc[1], ls='-', lw=0.5, marker='.', label='Validation predictions', c='#FF7F0F')
        ax.legend()
        plt.title(title)
        ax.set_xlabel("FDR")
        ax.set_ylabel("Number of PSMs")
        fig.tight_layout()
        if visualize:
            fig.show()
            plt.close(fig)
            return None, None
        if return_fig:
            return fig, ax

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

    def visualize_training(self,
                           outdir: Union[str, PathLike] = None,
                           log_yscale: bool = False,
                           log_xscale: bool = True,
                           save_only: bool = False,
                           stopping_idx: int = None,
                           plot_accuracy: bool = False):
        if self.fit_history is None or self.X_test is None or self.y_test is None:
            raise AttributeError("Model has not yet been trained. Use run to train.")
        if outdir is not None:
            if not Path(outdir).exists():
                Path(outdir).mkdir(parents=True)

        self.plot_histogram(self.training_predictions, self.y_train, 'Predictions for training data', log_xscale, log_yscale)
        self.plot_histogram(self.validation_predictions, self.y_val, 'Predictions for validation data', log_xscale, log_yscale)
        self.plot_histogram(self.testing_predictions, self.y_test, 'Predictions for testing data', log_xscale, log_yscale)
        self.plot_histogram(self.predictions, self.y, 'Predictions for all data', log_xscale, log_yscale)

        qs = self.roc[0][self.roc[0] <= 0.05]
        response = self.roc[1][self.roc[0] <= 0.05]
        plt.plot(qs, response, ls='none', marker='.', ms=1)
        plt.xlim((0, 0.05))
        plt.xlabel('FDR')
        plt.ylabel('Number of PSMs')
        plt.title('ROC')
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'ROC.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)
        max_accuracy = round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)

        n_epochs = len(self.fit_history.epoch)

        xs = range(1, len(self.fit_history.history['val_loss']) + 1)

        val_loss = np.min(self.fit_history.history['val_loss'])
        if stopping_idx is None:
            stopping_idx = self.fit_history.history['val_loss'].index(val_loss)
        n_psms = np.sum((self.qs <= 0.01) & (self.labels == 1))
        n_uniqe_peps = len(np.unique(self.peptides[(self.qs <= 0.01) & (self.labels == 1)]))

        fig, ax = plt.subplots()

        tl = ax.plot(xs, self.fit_history.history['loss'], c='#3987bc', label='Training loss')
        vl = ax.plot(xs, self.fit_history.history['val_loss'], c='#ff851a', label='Validation loss')
        ax.set_ylabel('Loss')
        if plot_accuracy:
            ax2 = ax.twinx()
            ta = ax2.plot(xs, self.fit_history.history['global_accuracy'], c='#3987bc', label='Training accuracy',
                          ls='--')
            va = ax2.plot(xs, self.fit_history.history['val_global_accuracy'], c='#ff851a', label='Validation accuracy',
                          ls='--')
            ax2.set_ylabel('Global accuracy (targets and decoys)')
            ax.plot(xs, [val_loss] * n_epochs, ls=':', c='gray')
            ma = ax2.plot(xs, [max_accuracy] * n_epochs, ls='-.', c='k', zorder=0,
                          label='Predicted max accuracy')
            ax2.set_xlim((1, n_epochs))

        bm = ax.plot(stopping_idx + 1, val_loss,
                     marker='o', mec='red', mfc='none', ms='12', ls='none', label='best model')
        if plot_accuracy:
            lines = tl + vl + bm + ta + va + ma
        else:
            lines = tl + vl + bm
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, bbox_to_anchor=(0, -.12, 1, 0), loc='upper center',
                   mode='expand', ncol=2)

        ax.set_xlabel('Epoch')
        ylim = ax.get_ylim()

        ax.plot([stopping_idx + 1, stopping_idx + 1], [0, 1], ls=':', c='gray')
        ax.set_ylim(ylim)
        ax.set_xlim((1, n_epochs))
        ax.set_title(f'Training curves\n#PSMs={n_psms} - #Peptides={n_uniqe_peps}')
        plt.tight_layout()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'training_history.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        plt.close('all')

    def write_table(self, filename: Union[str, PathLike] = None):
        if self.predictions is None or self.model is None:
            raise RuntimeError("The model must be trained first.")

    '''def plot_roc(self, q_cutoff=0.05):
        qs = self.qs[self.y == 1]  # get q-values of targets
        qs = qs[qs <= q_cutoff]
        roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
        plt.plot(qs, roc, ls='none', marker='.', ms=1)
        plt.xlabel('FDR')
        plt.ylabel('Number of PSMs')
        plt.title('ROC')
        plt.xlim((0, q_cutoff))
        plt.show()'''

    def get_peptide_list(self, fdr: float, label: int = 1):
        return self.peptides[(self.qs <= fdr) & (self.labels == label)]
