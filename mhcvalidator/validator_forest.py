import os

import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Union, List
from os import PathLike
from pathlib import Path
from mhcvalidator.data_loaders import load_file, load_pout_data
from mhcvalidator.features import prepare_features, eliminate_common_peptides_between_sets
from mhcvalidator.predictions_parsers import add_mhcflurry_to_feature_matrix, add_netmhcpan_to_feature_matrix
from mhcvalidator.netmhcpan_helper import NetMHCpanHelper, format_class_II_allele
from mhcvalidator.losses_and_metrics import i_dunno_bce, global_accuracy, total_fdr, loss_coteaching
from mhcvalidator.fdr import calculate_qs, calculate_peptide_level_qs, calculate_roc
import matplotlib.pyplot as plt
from mhcflurry.encodable_sequences import EncodableSequences
from mhcvalidator.models import get_model_without_peptide_encoding, get_model_with_peptide_encoding
from mhcvalidator.peptides import clean_peptide_sequences
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

deprecation._PRINT_DEPRECATION_WARNINGS = False

# This can be uncommented to prevent the GPU from getting used.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from scipy.stats import gmean as geom_mean

# tf.config.threading.set_inter_op_parallelism_threads(0)
# tf.config.threading.set_intra_op_parallelism_threads(0)
# tf.config.set_soft_device_placement(enabled=True)


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
        self.previous_aa: Union[List[str], None] = None
        self.next_aa: Union[List[str], None] = None
        self.mzml = None
        self.loaded_filetype: Union[str, None] = None
        self.random_seed = random_seed
        self.fit_history = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.X_train_peps = None
        self.X_test_peps = None
        self.X_val_peps = None
        self.X_train_encoded_peps = None
        self.X_test_encoded_peps = None
        self.X_val_encoded_peps = None
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

    def prepare_data(self, validation_split: float = 0.33, holdout_split: float = 0.33, random_seed: int = None,
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
        if validation_split + holdout_split > 1:
            raise ValueError("The validation and holdout splits cannot sum to greater than one.")
        if validation_split + holdout_split > 0.75:
            print(f'WARNING: The validation and holdout splits sum to {validation_split + holdout_split}. '
                  f'Only {1 - validation_split - holdout_split} of the data is going into the training set.')

        if random_seed is None:
            random_seed = self.random_seed

        X = self.feature_matrix.to_numpy(dtype=np.float32)
        y = np.array(self.labels, dtype=np.float32)
        peptides = np.array(self.peptides, dtype='U100')

        assert X.shape[0] == y.shape[0]

        input_scalar = MinMaxScaler()
        # save X and y before we do any shuffling. We will need this in the original order for predictions later
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        input_scalar = input_scalar.fit(self.X)
        self.X = input_scalar.transform(self.X)

        # encode all peptide sequences
        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        encoded_peps = keras.utils.normalize(encoded_peps, 0)
        self.X_encoded_peps = encoded_peps

        if subset_by_feature_qvalue:
            mask = self.get_qvalue_mask_from_features(X,
                                                      y,
                                                      qvalue_cutoff,
                                                      n_features_to_meet_cutoff,
                                                      subset_by_mhc_features_only,
                                                      verbosity=verbosity)
        elif subset_by_prior_qvalues:
            mask = (self.qs <= qvalue_cutoff) | (self.labels == 0)
        elif subset_by_prior_qvalues and subset_by_feature_qvalue:
            mask1 = self.get_qvalue_mask_from_features(X,
                                                      y,
                                                      qvalue_cutoff,
                                                      n_features_to_meet_cutoff,
                                                      subset_by_mhc_features_only,
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

        # first get training and testing sets
        (X_train, X_the_rest, y_train, y_the_rest, X_train_peps,
         X_the_rest_peps, X_train_encoded_peps, X_the_rest_encoded_peps, _, stratification) = train_test_split(
            X, y, peptides, encoded_peps, stratification,
            random_state=random_seed,
            stratify=stratification,
            test_size=validation_split + holdout_split
        )

        '''# resolve common peptide sequences
        if prevent_common_peptides:
            X_the_rest, X_train, y_the_rest, y_train, X_the_rest_peps, X_train_peps, X_the_rest_encoded_peps, X_train_encoded_peps = \
                eliminate_common_peptides_between_sets(X_the_rest,
                                                       X_train,
                                                       y_the_rest,
                                                       y_train,
                                                       X_the_rest_peps,
                                                       X_train_peps,
                                                       X_the_rest_encoded_peps,
                                                       X_train_encoded_peps,
                                                       random_state=np.random.RandomState(random_seed))'''

        # now split the testing and validation sets out of "the_rest"
        (X_test, X_val, y_test, y_val, X_test_peps,
         X_val_peps, X_test_encoded_peps, X_val_encoded_peps) = train_test_split(
            X_the_rest, y_the_rest, X_the_rest_peps, X_the_rest_encoded_peps,
            random_state=random_seed + 1,
            stratify=stratification,
            test_size=validation_split / (validation_split + holdout_split)
        )

        # if encode_peptide_sequences:
        # resolve common peptide sequences
        '''if prevent_common_peptides:
            X_test, X_val, y_test, y_val, X_test_peps, X_val_peps, X_test_encoded_peps, X_val_encoded_peps = \
                eliminate_common_peptides_between_sets(X_test,
                                                       X_val,
                                                       y_test,
                                                       y_val,
                                                       X_test_peps,
                                                       X_val_peps,
                                                       X_test_encoded_peps,
                                                       X_val_encoded_peps,
                                                       random_state=np.random.RandomState(random_seed))'''

        assert X_train.shape[0] == y_train.shape[0] == X_train_peps.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

        # normalize the Xs
        input_scalar = input_scalar.fit(X_train)
        X_train = input_scalar.transform(X_train)  # keras.utils.normalize(X_train, 0)
        input_scalar = input_scalar.fit(X_test)
        X_test = input_scalar.transform(X_test)  # keras.utils.normalize(X_test, 0)
        input_scalar = input_scalar.fit(X_val)
        X_val = input_scalar.transform(X_val)  # keras.utils.normalize(X_test, 0)

        X_train_encoded_peps = keras.utils.normalize(X_train_encoded_peps, 0)
        X_test_encoded_peps = keras.utils.normalize(X_test_encoded_peps, 0)
        X_val_encoded_peps = keras.utils.normalize(X_val_encoded_peps, 0)

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.X_train_peps = X_train_peps
        self.X_test_peps = X_test_peps
        self.X_val_peps = X_val_peps
        self.X_train_encoded_peps = X_train_encoded_peps
        self.X_test_encoded_peps = X_test_encoded_peps
        self.X_val_encoded_peps = X_val_encoded_peps
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

    def train_validate_split(self, random_seed: int = None):
        if self.raw_data is None:
            raise AttributeError("Data has not yet been loaded.")

        if random_seed is None:
            random_seed = self.random_seed

        X = self.feature_matrix.to_numpy(dtype=np.float32)
        y = np.array(self.labels, dtype=np.float32)
        peptides = np.array(self.peptides, dtype='U100')

        assert X.shape[0] == y.shape[0]

        input_scalar = MinMaxScaler()
        # save X and y before we do any shuffling. We will need this in the original order for predictions later
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        input_scalar = input_scalar.fit(self.X)
        self.X = input_scalar.transform(self.X)

        # encode all peptide sequences
        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        encoded_peps = keras.utils.normalize(encoded_peps, 0)
        self.X_encoded_peps = encoded_peps

        stratification = y
        random_idx = np.random.RandomState(random_seed).choice(len(y), size=len(y), replace=False)
        X = X[random_idx]
        y = y[random_idx]
        peptides = peptides[random_idx]
        encoded_peps = encoded_peps[random_idx]
        stratification = stratification[random_idx]

        # first get training and testing sets
        (X1, X2, y1, y2, peps1, peps2, encoded_peps1, encoded_peps2, _, stratification) = train_test_split(
            X, y, peptides, encoded_peps, stratification, train_size=0.5,
            random_state=random_seed,
            stratify=stratification)

        # normalize
        input_scalar = input_scalar.fit(X1)
        X1 = input_scalar.transform(X1)  # keras.utils.normalize(X_train, 0)
        input_scalar = input_scalar.fit(X2)
        X2 = input_scalar.transform(X2)  # keras.utils.normalize(X_test, 0)

        encoded_peps1 = keras.utils.normalize(encoded_peps1, 0)
        encoded_peps2 = keras.utils.normalize(encoded_peps2, 0)

        return X1, X2, y1, y2, peps1, peps2, encoded_peps1, encoded_peps2

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

    @staticmethod
    def _string_contains(string: str, pattern: Union[List[str], str]):
        if isinstance(pattern, str):
            pattern = [pattern]
        for x in pattern:
            if x in string:
                return True
        return False

    def get_qvalue_mask_from_features(self,
                                      cutoff: float = 0.05,
                                      n: int = 1,
                                      features_to_use: Union[List[str], str] = 'all',
                                      verbosity: int = 1):

        if features_to_use.lower() == 'mhc_only' and not (self._mhcflurry_predictions | self._netmhcpan_predictions):
            raise RuntimeError("mhc_only has been specified for creating a qvalue mask, but MHC predictions have not "
                               "been added to the feature matrix.")

        if features_to_use == 'all':
            columns = list(self.feature_matrix.columns)
        elif features_to_use == 'mhc' or features_to_use == 'mhc_only':
            columns = [x for x in self.feature_matrix.columns if
                       self._string_contains(x.lower, ['netmhcpan', 'mhcflurry', 'netmhciipan'])]
        else:
            columns = features_to_use

        X = self.feature_matrix.copy(deep=True).to_numpy()
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
        odd = len(peptide) % 2
        if len(peptide) == 9:
            return list(peptide) + [odd]
        elif len(peptide) == 8:
            return list(peptide[:3] + ' ' + peptide[3:]) + [odd]
        else:
            middle = int(np.floor((len(peptide)) / 2))
            return list(peptide[:3] + peptide[middle-1:middle+2] + peptide[-3:]) + [odd]

    #@staticmethod
    #def _simple_encode_peptide_list(peptides: List[str]):


    def run(self,
            q_value_subset: float = 1.0,
            features_for_subset: Union[List[str], str] = 'all',
            subset_threshold: int = 1,
            encode_peptide_sequences: bool = False,
            validation_split: float = 0.5,
            num_trees: int = 2000,
            max_depth: int = 1,
            shrinkage: float = 0.05,
            tfdf_hyperparameter_template: str = 'benchmark_rank1',
            early_stopping_patience: int = 15,
            weight_by_inverse_peptide_counts: bool = True,
            visualize: bool = True,
            report_dir: Union[str, PathLike] = None,
            random_seed: int = None,
            report_vebosity: int = 1,
            clear_session: bool = True,
            kwargs=None):

        if clear_session:
            K.clear_session()

        if kwargs is None:
            kwargs = {}

        # with self.graph.as_default():
        # tf.compat.v1.enable_eager_execution()

        if kwargs is None:
            kwargs = dict()

        self._set_seed(random_seed)

        # prepare data for training
        feature_matrix = self.feature_matrix.copy(deep=True)
        feature_matrix['Label'] = deepcopy(self.labels)

        if q_value_subset < 1.:
            mask = self.get_qvalue_mask_from_features(cutoff=q_value_subset,
                                                      n=subset_threshold,
                                                      features_to_use=features_for_subset,
                                                      verbosity=1)
        else:
            mask = np.ones_like(self.labels, dtype=bool)

        feature_matrix = self.feature_matrix[mask].copy(deep=True)
        labels = deepcopy(self.labels[mask])
        feature_matrix['Label'] = labels
        if encode_peptide_sequences:
            feature_matrix[['AA1', 'AA2', 'AA3', 'AAm-1', 'AAm', 'AAm+1', 'AA-3', 'AA-2', 'AA-1', 'odd_length']] = ''
            peps = pd.Series(data=self.peptides[mask])
            feature_matrix.iloc[:, -10:] = pd.DataFrame(peps.apply(self._simple_peptide_encoding).to_list())

        idx = np.random.RandomState(self.random_seed).choice(len(feature_matrix), len(feature_matrix), False)
        n = int(len(feature_matrix) * (1 - validation_split))
        x_train = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix.iloc[idx[:n], :], label="Label")
        x_test = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix.iloc[idx[n:], :], label="Label")
        train_labels = labels[idx[:n]]
        test_labels = labels[idx[n:]]

        self.model = tfdf.keras.GradientBoostedTreesModel(num_trees=num_trees, max_depth=max_depth,
                                                     hyperparameter_template=tfdf_hyperparameter_template,
                                                     shrinkage=shrinkage,
                                                     **kwargs)
        # growing_strategy='LOCAL', l1_regularization=0.9)
        # model = tfdf.keras.CartModel()
        self.model.fit(x_train, verbose=0)
        self.model.compile(metrics=['accuracy'])

        test_preds = self.model.predict(x_test)
        train_preds = self.model.predict(x_train)
        test_qs = calculate_qs(test_preds.flatten(), test_labels)
        train_qs = calculate_qs(train_preds.flatten(), train_labels)

        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        gs = GridSpec(2, 2, figure=fig)

        # create sub plots as grid
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        ax1.hist(train_preds[train_labels == 0], label='Decoy', bins=30, alpha=0.5)
        ax1.hist(train_preds[train_labels == 1], label='Target', bins=30, alpha=0.5)
        ax1.legend()
        ax1.set_title("Training data")
        ax1.set_xlim((0, 1))
        # plt.yscale('log')

        ax2.hist(test_preds[test_labels == 0], label='Decoy', bins=30, alpha=0.5)
        ax2.hist(test_preds[test_labels == 1], label='Target', bins=30, alpha=0.5)
        ax2.legend()
        ax2.set_title("Testing data")
        ax2.set_xlim((0, 1))
        # plt.yscale('log')

        train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.05)
        val_roc = calculate_roc(test_qs, test_labels, qvalue_cutoff=0.05)

        ax3.plot(train_roc[0], train_roc[1], ls='-', lw='0.5', marker='.', label='Training predictions', c='#1F77B4', alpha=1)
        ax3.plot(val_roc[0], val_roc[1], ls='-', lw='0.5', marker='.', label='Validation predictions', c='#FF7F0F', alpha=1)
        ax3.axvline(0.01, c='k', ls='--')
        ax3.axvline(0.05, c='k', ls='--')
        ax3.legend()
        ax3.set_title("ROC curve")
        ax3.set_xlabel("FDR")
        ax3.set_ylabel("Number of PSMs")
        ax3.set_ylim((0, ax3.get_ylim()[1]))

        train_roc = calculate_roc(train_qs, train_labels, qvalue_cutoff=0.5)
        val_roc = calculate_roc(test_qs, test_labels, qvalue_cutoff=0.5)

        ax4.plot(train_roc[0], train_roc[1], ls='-', lw='0.5', marker='.', label='Training predictions', c='#1F77B4', alpha=1)
        ax4.plot(val_roc[0], val_roc[1], ls='-', lw='0.5', marker='.', label='Validation predictions', c='#FF7F0F', alpha=1)
        ax4.axvline(0.01, c='k', ls='--')
        ax4.axvline(0.05, c='k', ls='--')
        ax4.legend()
        ax4.set_title("Expanded ROC curve")
        ax4.set_xlabel("FDR")
        ax4.set_ylabel("Number of PSMs")
        ax4.set_ylim((0, ax4.get_ylim()[1]))

        if q_value_subset < 1:
            fig.suptitle("Subset used for training and validation", fontsize=14)
        else:
            fig.suptitle("Training and validation", fontsize=14)
        fig.show()
        plt.close(fig)

        ratio = len(train_labels) / len(test_labels)
        n_training_targets = np.sum((train_qs <= 0.01) & (train_labels == 1))
        n_testing_targets = np.sum((test_qs <= 0.01) & (test_labels == 1))
        print(f'Training PSMs at 1% FDR: {n_training_targets}')
        print(f'Testing PSMs at 1% FDR: {n_testing_targets}')
        print(f'Ratio of training set size to testing set size: {ratio}')
        print(f'Ratio of training PSMs to testing PSMs at 1% FDR: {n_training_targets / n_testing_targets}')

        # Now use all the examples
        feature_matrix = self.feature_matrix.copy(deep=True)
        labels = deepcopy(self.labels)
        if encode_peptide_sequences:
            feature_matrix[['AA1', 'AA2', 'AA3', 'AAm-1', 'AAm', 'AAm+1', 'AA-3', 'AA-2', 'AA-1', 'odd_length']] = ''
            peps = pd.Series(data=self.peptides)
            feature_matrix.iloc[:, -10:] = pd.DataFrame(peps.apply(self._simple_peptide_encoding).to_list())
        x = tfdf.keras.pd_dataframe_to_tf_dataset(feature_matrix)

        preds = self.model.predict(x)
        qs = calculate_qs(preds.flatten(), labels)

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
        roc = calculate_roc(qs, labels, qvalue_cutoff=0.05)

        ax3.plot(roc[0], roc[1], ls='-', lw='0.5', marker='.', c='#1F77B4', alpha=1)
        ax3.axvline(0.01, c='k', ls='--')
        ax3.axvline(0.05, c='k', ls='--')
        ax3.set_title("ROC curve")
        ax3.set_xlabel("FDR")
        ax3.set_ylabel("Number of PSMs")

        fig.suptitle("Predictions for all data", fontsize=14)
        fig.show()
        plt.close(fig)

        logs = self.model.make_inspector().training_logs()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Accuracy (out-of-bag)")

        plt.subplot(1, 2, 2)
        plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Logloss (out-of-bag)")

        plt.show()

        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)
        psms = self.peptides[(qs <= 0.01) & (self.labels == 1)]
        n_psm_targets = len(psms)
        n_unique_psms = len(set(psms))
        pep_level_qs, _, pep_level_labels, peps = calculate_peptide_level_qs(preds, self.labels, self.peptides)
        n_unique_peps = np.sum((pep_level_qs <= 0.01) & (pep_level_labels == 1))


        report = '\n========== REPORT ==========\n\n' \
                 f'Total PSMs: {len(self.labels)}\n' \
                 f'Labeled as targets: {n_targets}\n' \
                 f'Labeled as decoys: {n_decoys}\n' \
                 f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                 f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                 '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                 f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n'

        if report_vebosity > 0:
            print(report)
        self.raw_data['mhcv_prob'] = list(preds)
        self.raw_data['mhcv_q-value'] = list(qs)
        self.raw_data['mhcv_label'] = list(labels)
        self.raw_data['mhcv_peptide'] = list(self.peptides)

    def test_initial_subset_by_mhc_predictions(self,
                                               encode_peptide_sequences: bool = False,
                                               epochs: int = 30,
                                               batch_size: int = 256,
                                               loss_fn=tf.losses.BinaryCrossentropy(),
                                               holdout_split: float = 0.25,
                                               validation_split: float = 0.25,
                                               initial_learning_rate: float = 0.001,
                                               subsequent_iterations_learning_rate: float = 0.0001,
                                               iterations: int = 1,
                                               early_stopping_patience: int = 15,
                                               dropout: float = 0.5,
                                               hidden_layers: int = 3,
                                               subset_by_feature_qvalue: bool = True,
                                               subset_by_mhc_prediction_only: bool = True,
                                               n_features_to_meet_cutoff: int = 1,
                                               qvalue_cutoff: float = 0.1,
                                               stratify_based_on_MHC_presentation: bool = False,
                                               weight_by_inverse_peptide_counts: bool = True,
                                               visualize: bool = True,
                                               log_yscale: bool = False,
                                               log_xscale: bool = True,
                                               report_dir: Union[str, PathLike] = None,
                                               random_seed: int = None,
                                               return_model: bool = False,
                                               fit_verbosity: int = 2,
                                               report_vebosity: int = 2,
                                               clear_session: bool = True,
                                               initial_model_weights: str = None,
                                               keep_best_loss: bool = True,
                                               plot_accuracy: bool = False):

        if clear_session:
            K.clear_session()

        self._set_seed(random_seed)

        # prepare data for training
        self.prepare_data(validation_split=validation_split,
                          holdout_split=holdout_split,
                          random_seed=random_seed,
                          stratification_dimensions=2 if stratify_based_on_MHC_presentation else 1,
                          subset_by_feature_qvalue=subset_by_feature_qvalue,
                          subset_by_mhc_features_only=subset_by_mhc_prediction_only,
                          n_features_to_meet_cutoff=n_features_to_meet_cutoff,
                          qvalue_cutoff=qvalue_cutoff)

        model_name, callbacks_list = self._initialize_model(encode_peptide_sequences,
                                                            hidden_layers,
                                                            dropout,
                                                            initial_learning_rate,
                                                            early_stopping_patience,
                                                            loss_fn)

        # if we are encoding peptide sequences, add them to the input
        if encode_peptide_sequences:
            X_train = [self.X_train, self.X_train_encoded_peps]
            X_val = [self.X_val, self.X_val_encoded_peps]
            X_test = [self.X_test, self.X_test_encoded_peps]
            X = [self.X, self.X_encoded_peps]
        else:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
            X = self.X

        #
        peptide_counts = Counter(self.X_train_peps)
        if weight_by_inverse_peptide_counts:
            weights = np.array([1 / np.sqrt(peptide_counts[x]) for x in self.X_train_peps])
        else:
            weights = np.ones_like(self.y_train)
        self.fit_history = self.model.fit(X_train,
                                          self.y_train,
                                          validation_data=(X_val, self.y_val),
                                          sample_weight=weights,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          verbose=fit_verbosity,
                                          callbacks=callbacks_list)

        # load the best model
        if keep_best_loss:
            self.model.load_weights(model_name)

        self.predictions = self.model.predict(X).flatten()
        self.training_predictions = self.model.predict(X_train).flatten()
        self.testing_predictions = self.model.predict(X_test).flatten()
        self.validation_predictions = self.model.predict(X_val).flatten()
        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        self.qs = np.asarray(self.qs, dtype=float)
        self.roc = calculate_roc(self.qs, self.labels)
        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)

        min_val_loss = np.min(self.fit_history.history['val_loss'])
        stopping_idx = self.fit_history.history['val_loss'].index(min_val_loss) if keep_best_loss else \
            len(self.fit_history.history['val_loss']) - 1
        pep_qs, pep_xs, pep_ys, peps = calculate_peptide_level_qs(self.predictions, self.y, self.peptides,
                                                                  higher_better=True)
        psm_target_mask = (self.qs <= 0.01) & (self.y == 1)
        n_psm_targets = np.sum(psm_target_mask)
        n_unique_psms = len(np.unique(self.peptides[psm_target_mask]))
        n_unique_peps = np.sum((pep_qs <= 0.01) & (pep_ys == 1))
        evaluate = self.model.evaluate(X_test, self.y_test, batch_size=batch_size, verbose=0)

        report = '\n========== REPORT ==========\n\n' \
                 f'Total PSMs: {len(self.labels)}\n' \
                 f'Labeled as targets: {n_targets}\n' \
                 f'Labeled as decoys: {n_decoys}\n' \
                 f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                 f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                 f'Theoretical maximum global accuracy: {round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)}\n' \
                 f'  --Be wary if the accuracies below are higher than this value.\n' \
                 '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                 f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n' \
                 f'\n' \
                 f'----- MODEL TRAINING, VALIDATION, TESTING -----\n' \
                 f'Training loss: {round(self.fit_history.history["loss"][stopping_idx], 3)} - ' \
                 f'Validation loss: {round(self.fit_history.history["val_loss"][stopping_idx], 3)} - ' \
                 f'Testing loss: {round(evaluate[0], 3)}\n' \
                 f'Training accuracy: {round(self.fit_history.history["global_accuracy"][stopping_idx], 3)} - ' \
                 f'Validation accuracy: {round(self.fit_history.history["val_global_accuracy"][stopping_idx], 3)} - ' \
                 f'Testing accuracy: {round(evaluate[1], 3)}\n' \
                 f'\n' \
                 f'Best model: {model_name}'

        if report_vebosity == 2:
            print(report)
        self.raw_data['mhcv_prob'] = list(self.predictions)
        self.raw_data['mhcv_q-value'] = list(self.qs)
        self.raw_data['mhcv_label'] = list(self.labels)
        self.raw_data['mhcv_peptide'] = list(self.peptides)
        if visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir,
                                    stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)
        elif not visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir,
                                    save_only=True,
                                    stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)
        elif visualize:
            self.visualize_training(stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)

        if report_dir is not None:
            with open(Path(report_dir, 'training_report.txt'), 'w') as f:
                f.write(report)
        if return_model:
            return model_name

        #self.run(initial_model_weights=model_name, learning_rate=second_fit_learning_rate)

    def new_run(self,
                encode_peptide_sequences: bool = False,
                epochs: int = 30,
                batch_size: int = 256,
                loss_fn=tf.losses.BinaryCrossentropy(),
                holdout_split: float = 0.25,
                validation_split: float = 0.25,
                initial_learning_rate: float = 0.001,
                subsequent_iterations_learning_rate: float = 0.0001,
                iterations: int = 1,
                target_fdr: float = 0.01,
                early_stopping_patience: int = 15,
                dropout: float = 0.5,
                hidden_layers: int = 3,
                subset_by_feature_qvalue: bool = False,
                subset_by_mhc_prediction_only: bool = False,
                subset_by_prior_qvalues: bool = False,
                n_features_to_meet_cutoff: int = 1,
                qvalue_cutoff: float = 0.1,
                stratify_based_on_MHC_presentation: bool = False,
                weight_by_inverse_peptide_counts: bool = True,
                visualize: bool = True,
                log_yscale: bool = False,
                log_xscale: bool = True,
                report_dir: Union[str, PathLike] = None,
                random_seed: int = None,
                return_model: bool = False,
                fit_verbosity: int = 2,
                verbosity: int = 2,
                clear_session: bool = True,
                model_to_load: str = None,
                keep_best_loss: bool = True,
                plot_accuracy: bool = False,
                weight_on_decoys: float = 1,
                weight_on_targets: float = 1):

        if clear_session:
            K.clear_session()

        self._set_seed(random_seed)
        random_seed = self.random_seed

        # prepare data for training
        self.prepare_data(validation_split=validation_split,
                          holdout_split=holdout_split,
                          random_seed=random_seed,
                          stratification_dimensions=2 if stratify_based_on_MHC_presentation else 1,
                          subset_by_feature_qvalue=subset_by_feature_qvalue,
                          subset_by_mhc_features_only=subset_by_mhc_prediction_only,
                          subset_by_prior_qvalues=subset_by_prior_qvalues,
                          n_features_to_meet_cutoff=n_features_to_meet_cutoff,
                          qvalue_cutoff=qvalue_cutoff,
                          verbosity=verbosity)

        self.model, model_name, callbacks_list = self._initialize_model(encode_peptide_sequences,
                                                                        hidden_layers,
                                                                        dropout,
                                                                        initial_learning_rate,
                                                                        early_stopping_patience,
                                                                        loss_fn)

        if model_to_load is not None:
            self.model.load_weights(model_to_load)
        # if we are encoding peptide sequences, add them to the input
        if encode_peptide_sequences:
            X_train = [self.X_train, self.X_train_encoded_peps]
            X_val = [self.X_val, self.X_val_encoded_peps]
            X_test = [self.X_test, self.X_test_encoded_peps]
            X = [self.X, self.X_encoded_peps]
        else:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
            X = self.X

        #
        peptide_counts = Counter(self.X_train_peps)
        if weight_by_inverse_peptide_counts:
            weights = np.array([1 / np.sqrt(peptide_counts[x]) for x in self.X_train_peps])
        else:
            weights = np.ones_like(self.y_train)
        print('Fitting model')
        self.fit_history = self.model.fit(X_train,
                                          self.y_train,
                                          validation_data=(X_val, self.y_val),
                                          sample_weight=weights,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          verbose=fit_verbosity,
                                          callbacks=callbacks_list,
                                          class_weight={0: weight_on_decoys, 1: weight_on_targets},
                                          use_multiprocessing=self.max_threads > 1,
                                          workers=self.max_threads)

        # load the best model
        if keep_best_loss:
            self.model.load_weights(model_name)

        self.predictions = self.model.predict(X).flatten()
        self.training_predictions = self.model.predict(X_train).flatten()
        self.testing_predictions = self.model.predict(X_test).flatten()
        self.validation_predictions = self.model.predict(X_val).flatten()
        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        self.qs = np.asarray(self.qs, dtype=float)
        self.roc = calculate_roc(self.qs, self.labels)
        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)

        min_val_loss = np.min(self.fit_history.history['val_loss'])
        stopping_idx = self.fit_history.history['val_loss'].index(min_val_loss) if keep_best_loss else \
            len(self.fit_history.history['val_loss']) - 1
        pep_qs, pep_xs, pep_ys, peps = calculate_peptide_level_qs(self.predictions, self.y, self.peptides,
                                                                  higher_better=True)
        psm_target_mask = (self.qs <= target_fdr) & (self.y == 1)
        n_psm_targets = np.sum(psm_target_mask)
        n_unique_psms = len(np.unique(self.peptides[psm_target_mask]))
        n_unique_peps = np.sum((pep_qs <= target_fdr) & (pep_ys == 1))
        evaluate = self.model.evaluate(X_test, self.y_test, batch_size=batch_size, verbose=0)

        report = '\n========== REPORT ==========\n\n' \
                 f'Total PSMs: {len(self.labels)}\n' \
                 f'Labeled as targets: {n_targets}\n' \
                 f'Labeled as decoys: {n_decoys}\n' \
                 f'Global FDR: {round(n_decoys / n_targets, 3)}\n' \
                 f'Theoretical number of possible true positives (PSMs): {n_targets - n_decoys}\n' \
                 f'Theoretical maximum global accuracy: {round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)}\n' \
                 f'  --Be wary if the accuracies below are higher than this value.\n' \
                 '----- CONFIDENT PSMS AND PEPTIDES -----\n' \
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n' \
                 f'Unique peptides at {target_fdr * 100}% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at {target_fdr * 100}% peptide-level FDR: {n_unique_peps}\n' \
                 f'\n' \
                 f'----- MODEL TRAINING, VALIDATION, TESTING -----\n' \
                 f'Training loss: {round(self.fit_history.history["loss"][stopping_idx], 3)} - ' \
                 f'Validation loss: {round(self.fit_history.history["val_loss"][stopping_idx], 3)} - ' \
                 f'Testing loss: {round(evaluate[0], 3)}\n' \
                 f'Training accuracy: {round(self.fit_history.history["global_accuracy"][stopping_idx], 3)} - ' \
                 f'Validation accuracy: {round(self.fit_history.history["val_global_accuracy"][stopping_idx], 3)} - ' \
                 f'Testing accuracy: {round(evaluate[1], 3)}\n' \
                 f'\n' \
                 f'Best model: {model_name}'

        if verbosity == 2:
            print(report)
        elif verbosity == 1:
            print(f'Targets validated at {target_fdr * 100}% FDR: {n_psm_targets} PSMs '
                  f'- {n_unique_psms} peptides (PSM level) '
                  f'- {n_unique_peps} peptides (peptide level)')
        self.raw_data['mhcv_prob'] = list(self.predictions)
        self.raw_data['mhcv_q-value'] = list(self.qs)
        self.raw_data['mhcv_label'] = list(self.labels)
        self.raw_data['mhcv_peptide'] = list(self.peptides)
        if visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir,
                                    stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)
        elif not visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir,
                                    save_only=True,
                                    stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)
        elif visualize:
            self.visualize_training(stopping_idx=stopping_idx,
                                    log_xscale=log_xscale,
                                    log_yscale=log_yscale,
                                    plot_accuracy=plot_accuracy)

        for i in range(iterations - 1):
            self.new_run(encode_peptide_sequences=encode_peptide_sequences,
                         epochs=epochs,
                         batch_size=batch_size,
                         loss_fn=loss_fn,
                         holdout_split=holdout_split,
                         validation_split=validation_split,
                         initial_learning_rate=subsequent_iterations_learning_rate,
                         iterations=1,
                         target_fdr=target_fdr,
                         early_stopping_patience=early_stopping_patience,
                         dropout=dropout,
                         hidden_layers=hidden_layers,
                         subset_by_feature_qvalue=subset_by_feature_qvalue,
                         subset_by_mhc_prediction_only=subset_by_mhc_prediction_only,
                         n_features_to_meet_cutoff=n_features_to_meet_cutoff,
                         subset_by_prior_qvalues=True,
                         qvalue_cutoff=qvalue_cutoff,
                         stratify_based_on_MHC_presentation=stratify_based_on_MHC_presentation,
                         weight_by_inverse_peptide_counts=weight_by_inverse_peptide_counts,
                         visualize=visualize,
                         log_yscale=log_yscale,
                         log_xscale=log_xscale,
                         report_dir=report_dir,
                         random_seed=random_seed+i,
                         return_model=return_model,
                         fit_verbosity=fit_verbosity,
                         verbosity=verbosity,
                         clear_session=clear_session,
                         model_to_load=model_name,
                         keep_best_loss=keep_best_loss,
                         plot_accuracy=plot_accuracy)

        if report_dir is not None:
            with open(Path(report_dir, 'training_report.txt'), 'w') as f:
                f.write(report)
        if return_model:
            return model_name

        # self.run(initial_model_weights=model_name, learning_rate=second_fit_learning_rate)

    def coteach(self,
                encode_peptide_sequences: bool = False,
                epochs: int = 30,
                batch_size: int = 256,
                loss_fn=tf.losses.BinaryCrossentropy(),
                holdout_split: float = 0.25,
                validation_split: float = 0.25,
                initial_learning_rate: float = 0.001,
                target_fdr: float = 0.01,
                early_stopping_patience: int = 15,
                dropout: float = 0.5,
                hidden_layers: int = 3,
                subset_by_feature_qvalue: bool = False,
                subset_by_mhc_prediction_only: bool = False,
                subset_by_prior_qvalues: bool = False,
                n_features_to_meet_cutoff: int = 1,
                qvalue_cutoff: float = 0.1,
                stratify_based_on_MHC_presentation: bool = False,
                weight_by_inverse_peptide_counts: bool = True,
                visualize: bool = True,
                log_yscale: bool = False,
                log_xscale: bool = True,
                report_dir: Union[str, PathLike] = None,
                random_seed: int = None,
                return_model: bool = False,
                fit_verbosity: int = 2,
                verbosity: int = 2,
                clear_session: bool = True,
                model_to_load: str = None,
                keep_best_loss: bool = True,
                plot_accuracy: bool = False,
                weight_on_decoys: float = 1,
                weight_on_targets: float = 1,
                forget_rate: float=0.9):

        if clear_session:
            K.clear_session()

        self._set_seed(random_seed)
        random_seed = self.random_seed

        rate_schedule = np.ones(epochs) * forget_rate
        rate_schedule[:5] = np.linspace(0, forget_rate, 5)

        # prepare data for training
        self.prepare_data(validation_split=validation_split,
                          holdout_split=holdout_split,
                          random_seed=random_seed,
                          stratification_dimensions=2 if stratify_based_on_MHC_presentation else 1,
                          subset_by_feature_qvalue=subset_by_feature_qvalue,
                          subset_by_mhc_features_only=subset_by_mhc_prediction_only,
                          subset_by_prior_qvalues=subset_by_prior_qvalues,
                          n_features_to_meet_cutoff=n_features_to_meet_cutoff,
                          qvalue_cutoff=qvalue_cutoff,
                          verbosity=verbosity)

        self.model1, model_name1, callbacks_list1 = self._initialize_model(encode_peptide_sequences,
                                                                        hidden_layers,
                                                                        dropout,
                                                                        initial_learning_rate,
                                                                        early_stopping_patience,
                                                                        loss_fn)
        self.model2, model_name2, callbacks_list2 = self._initialize_model(encode_peptide_sequences,
                                                                        hidden_layers,
                                                                        dropout,
                                                                        initial_learning_rate,
                                                                        early_stopping_patience,
                                                                        loss_fn)


        # if we are encoding peptide sequences, add them to the input
        if encode_peptide_sequences:
            X_train = [self.X_train, self.X_train_encoded_peps]
            X_val = [self.X_val, self.X_val_encoded_peps]
            X_test = [self.X_test, self.X_test_encoded_peps]
            X = [self.X, self.X_encoded_peps]
        else:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
            X = self.X

        #
        peptide_counts = Counter(self.X_train_peps)
        if weight_by_inverse_peptide_counts:
            weights = np.array([1 / np.sqrt(peptide_counts[x]) for x in self.X_train_peps])
        else:
            weights = np.ones_like(self.y_train)

        # Instantiate an optimizer.
        optimizer1 = keras.optimizers.Adam(learning_rate=1e-3)
        optimizer2 = keras.optimizers.Adam(learning_rate=1e-3)
        # Instantiate a loss function.
        # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the training dataset.
        batch_size = batch_size
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(self.y_train)).batch(batch_size)

        loss_history1 = []
        loss_history2 = []

        print('Fitting models')

        estimated_true_positives = np.sum(self.y_train == 1) - np.sum(self.y_train == 0)

        # train the model
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            losses_1 = []
            losses_2 = []

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape(persistent=True) as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits1 = self.model1(x_batch_train, training=True)  # model1 logits for this minibatch
                    logits2 = self.model2(x_batch_train, training=True)  # model2 logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value1, loss_value2, = loss_coteaching(logits1,
                                                                logits2,
                                                                y_batch_train,
                                                                forget_rate=rate_schedule[epoch])

                    losses_1.append(loss_value1)
                    losses_2.append(loss_value2)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads1 = tape.gradient(loss_value1, self.model1.trainable_weights)
                grads2 = tape.gradient(loss_value2, self.model2.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer1.apply_gradients(zip(grads1, self.model1.trainable_weights))
                optimizer2.apply_gradients(zip(grads2, self.model2.trainable_weights))

                # Log every 10 batches.
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: model1=%.4f - model2=%.4f"
                        % (step, float(loss_value1), float(loss_value2))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))

            loss_history1.append(np.mean(losses_1))
            loss_history2.append(np.mean(losses_2))

        plt.plot(loss_history1, label='model_1')
        plt.plot(loss_history2, label='model_2')
        plt.legend()
        plt.show()
        plt.close()

        self.predictions = (self.model1.predict(X).flatten() + self.model2.predict(X).flatten()) / 2
        self.training_predictions = (self.model1.predict(X_train).flatten() + self.model2.predict(X_train).flatten()) / 2
        self.testing_predictions = (self.model1.predict(X_test).flatten() + self.model2.predict(X_test).flatten()) / 2
        self.validation_predictions = (self.model1.predict(X_val).flatten() + self.model2.predict(X_val).flatten()) / 2
        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        self.qs = np.asarray(self.qs, dtype=float)
        self.roc = calculate_roc(self.qs, self.labels)


    def test_alt_fit(self,
                     encode_peptide_sequences: bool = False,
                     epochs: int = 30,
                     batch_size: int = 256,
                     loss_fn=tf.losses.BinaryCrossentropy(),
                     holdout_split: float = 0.25,
                     validation_split: float = 0.25,
                     learning_rate: float = 0.001,
                     early_stopping_patience: int = 15,
                     dropout: float = 0.5,
                     hidden_layers: int = 3,
                     stratify_based_on_MHC_presentation: bool = False,
                     weight_by_inverse_peptide_counts: bool = True,
                     visualize: bool = True,
                     report_dir: Union[str, PathLike] = None,
                     random_seed: int = None,
                     return_model: bool = False,
                     fit_verbosity: int = 2,
                     report_vebosity: int = 1,
                     clear_session: bool = True,
                     alternate_labels=None,
                     initial_model_weights: str = None,
                     keep_best_loss: bool = True):
        if clear_session:
            K.clear_session()

        if random_seed is None:
            random_seed = self.random_seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        X1, X2, y1, y2, peps1, peps2, encoded_peps1, encoded_peps2 = self.train_validate_split(random_seed)

        # CREATE CALLBACKS
        now = str(datetime.now()).replace(' ', '_').replace(':', '-')

        self.model = {}
        self.fit_history = {}
        training_predictions = {}
        validation_predictions = {}
        val_qs = {}

        peptide_sets = []

        for i in range(2):
            if encode_peptide_sequences:
                get_model = get_model_with_peptide_encoding
            else:
                get_model = get_model_without_peptide_encoding

            model_name = str(self.model_dir / f'mhcvalidator_{now}_model{i}.h5')
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
            self.model[i] = get_model(self.feature_matrix.shape[1],
                                      dropout=dropout,
                                      hidden_layers=hidden_layers,
                                      max_pep_length=encoded_pep_length)
            self.model[i].compile(loss=loss_fn,
                                  optimizer=optimizer,
                                  metrics=[global_accuracy])

            # load weights from an existing model if specified
            # if initial_model_weights is not None:
            #    self.model.load_weights(initial_model_weights)

            # if we are encoding peptide sequences, add them to the input
            if encode_peptide_sequences:
                X_train = [[X1, X2][i % 2], [encoded_peps1, encoded_peps2][i % 2]]
                X_val = [[X1, X2][(i + 1) % 2], [encoded_peps1, encoded_peps2][(i + 1) % 2]]
            else:
                X_train = [X1, X2][i % 2]
                X_val = [X1, X2][(i + 1) % 2]
            y_train = [y1, y2][i % 2]
            y_val = [y1, y2][(i + 1) % 2]
            peps_train = [peps1, peps2][i % 2]
            peps_val = [peps1, peps2][(i + 1) % 2]

            peptide_counts = Counter(peps_train)
            if weight_by_inverse_peptide_counts:
                weights = np.array([1 / np.sqrt(peptide_counts[x]) for x in [peps1, peps2][i % 2]])
            else:
                weights = np.ones_like([peps1, peps2][i % 2], dtype=np.float32)
            self.fit_history[i] = self.model[i].fit(X_train,
                                                    y_train,
                                                    validation_data=(X_val, y_val),
                                                    sample_weight=weights,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    verbose=fit_verbosity,
                                                    callbacks=callbacks_list)

            # load the best model
            if keep_best_loss:
                self.model[i].load_weights(model_name)

            training_predictions[i] = self.model[i].predict(X_train).flatten()
            validation_predictions[i] = self.model[i].predict(X_val).flatten()
            val_qs[i] = calculate_qs(validation_predictions[i], y_val)

            n_targets = np.sum(y_val == 1)
            n_decoys = np.sum(y_val == 0)
            max_accuracy = round((n_decoys + (n_targets - n_decoys)) / len(y_val), 3)

            n_epochs = len(self.fit_history[i].epoch)

            xs = range(1, len(self.fit_history[i].history['val_loss']) + 1)

            val_loss = np.min(self.fit_history[i].history['val_loss'])
            if keep_best_loss is None:
                stopping_idx = self.fit_history[i].history['val_loss']
            else:
                stopping_idx = len(self.fit_history[i].history['val_loss']) - 1
            n_psms = np.sum((val_qs[i] <= 0.01) & (y_val == 1))
            n_uniqe_peps = len(np.unique(peps_val[(val_qs[i] <= 0.01) & (y_val == 1)]))

            peptide_sets.append(set(peps_val[(val_qs[i] <= 0.01) & (y_val == 1)]))

            fig, ax = plt.subplots()

            tl = ax.plot(xs, self.fit_history[i].history['loss'], c='#3987bc', label='Training loss')
            vl = ax.plot(xs, self.fit_history[i].history['val_loss'], c='#ff851a', label='Validation loss')
            ax.set_ylabel('Loss')
            ax2 = ax.twinx()
            ta = ax2.plot(xs, self.fit_history[i].history['global_accuracy'], c='#3987bc', label='Training accuracy',
                          ls='--')
            va = ax2.plot(xs, self.fit_history[i].history['val_global_accuracy'], c='#ff851a',
                          label='Validation accuracy',
                          ls='--')
            ax2.set_ylabel('Global accuracy (targets and decoys)')
            ax.plot(xs, [val_loss] * n_epochs, ls=':', c='gray')
            ma = ax2.plot(xs, [max_accuracy] * n_epochs, ls='-.', c='k', zorder=0,
                          label='Predicted max accuracy')
            bm = ax.plot(stopping_idx + 1, val_loss,
                         marker='o', mec='red', mfc='none', ms='12', ls='none', label='best model')

            lines = tl + vl + bm + ta + va + ma
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels, bbox_to_anchor=(0, -.12, 1, 0), loc='upper center',
                       mode='expand', ncol=2)

            ax.set_xlabel('Epoch')
            ylim = ax.get_ylim()

            ax.plot([stopping_idx + 1, stopping_idx + 1], [0, 1], ls=':', c='gray')
            ax.set_ylim(ylim)
            ax.set_xlim((1, n_epochs))
            ax2.set_xlim((1, n_epochs))
            ax.set_title(f'Training curves\n#PSMs={n_psms} - #Peptides={n_uniqe_peps}')
            plt.tight_layout()
            plt.show()
            plt.clf()

        return peptide_sets

    def run_twice_test(self,
                       encode_peptide_sequences: bool = False,
                       lstm_model: bool = False,
                       epochs: int = 16,
                       batch_size: int = 256,
                       loss_fn=tf.losses.BinaryCrossentropy(),  # =weighted_bce(10, 2, 0.5),
                       holdout_split: float = 0.33,
                       validation_split: float = 0.33,
                       learning_rate: float = 0.001,
                       early_stopping_patience: int = 15,
                       subset: np.array = None,
                       weight_samples: bool = False,
                       decoy_factor=1,
                       target_factor=1,
                       decoy_bias=1,
                       target_bias=1,
                       visualize: bool = True,
                       report_dir: Union[str, PathLike] = None,
                       random_seed: int = None,
                       feature_qvalue_cutoff_for_training: float = None,
                       mhc_only_for_training_cutoff: bool = False,
                       n: int = 2,
                       return_report: bool = False,
                       fit_verbosity: int = 2,
                       report_vebosity: int = 1,
                       clear_session: bool = True):

        # run the fit once
        initial_model = self.run(encode_peptide_sequences=encode_peptide_sequences,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 loss_fn=loss_fn,
                                 holdout_split=holdout_split,
                                 validation_split=validation_split,
                                 learning_rate=learning_rate,
                                 early_stopping_patience=early_stopping_patience,
                                 visualize=visualize,
                                 report_dir=report_dir,
                                 random_seed=random_seed,
                                 return_model=True,
                                 fit_verbosity=fit_verbosity,
                                 report_vebosity=report_vebosity,
                                 clear_session=clear_session)

        # make a backup of the labels, because we are going to change them
        # label_backup = deepcopy(self.labels)

        # determine the number of labels to assign as false positives (i.e. set to 0)
        n_decoys = np.sum(self.labels == 0)
        n_false_positives = round(n_decoys * 2.1)

        # get the indices of these examples
        false_positive_indices = np.argpartition(self.predictions, n_false_positives)[:n_false_positives]

        # set them to 0
        alt_labels = deepcopy(self.labels)
        alt_labels[false_positive_indices] = 0

        # run again
        self.run(encode_peptide_sequences=encode_peptide_sequences,
                 epochs=4,
                 batch_size=batch_size,
                 loss_fn=loss_fn,
                 holdout_split=holdout_split,
                 validation_split=validation_split,
                 learning_rate=learning_rate,
                 early_stopping_patience=early_stopping_patience,
                 visualize=visualize,
                 report_dir=report_dir,
                 random_seed=random_seed,
                 return_model=False,
                 fit_verbosity=fit_verbosity,
                 report_vebosity=report_vebosity,
                 clear_session=clear_session,
                 alternate_labels=alt_labels,
                 initial_model_weights=initial_model)

        # set the labels back to the original values
        # self.labels = label_backup

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

        def plot_histogram(predictions, labels, title, log_xscale, log_yscale):
            predictions = np.log10(predictions) if log_xscale else predictions
            fig, ax = plt.subplots()
            D, bins, _ = ax.hist(x=np.array(predictions[labels == 0]).flatten(), label='Decoy', bins=100, alpha=0.6,
                                 zorder=999)
            T, bins, _ = ax.hist(x=np.array(predictions[labels == 1]).flatten(), label='Target', bins=100, alpha=0.6,
                                 range=(bins[0], bins[-1]))
            plt.title(title)
            #ax2 = ax.twinx()
            #decoy_cum_probabilities = D.cumsum() / (float(sum(D)) + float(sum(T)))
            #target_cum_probabilities = T.cumsum() / (float(sum(D)) + float(sum(T)))
            #ax2.plot(bins[:-1], decoy_cum_probabilities, linewidth=1, color="r", label='Decoy probability', ls='--')
            #tp_cum_probabilities = target_cum_probabilities - decoy_cum_probabilities
            #pep = decoy_cum_probabilities / (decoy_cum_probabilities + tp_cum_probabilities)
            #ax2.plot(bins[:-1], target_cum_probabilities, linewidth=1, color="b", label='Target probability', ls='--')
            #ax2.plot(bins[:-1], pep, linewidth=1, color="g", label='PEP', ls='--')
            #ax2.set_ylim((0, 1.1))
            if log_yscale:
                ax.set_yscale('log')
            ax.legend()
            #ax2.legend()
            ax.set_xlabel("log10(target probability)") if log_xscale else ax.set_xlabel("target probability")
            ax.set_ylabel("PSM count")
            if outdir is not None:
                plt.savefig(str(Path(outdir, f'{title.replace(" ", "_")}.svg')))
            if not save_only:
                plt.show()
            plt.clf()

        plot_histogram(self.training_predictions, self.y_train, 'Predictions for training data', log_xscale, log_yscale)
        plot_histogram(self.validation_predictions, self.y_val, 'Predictions for validation data', log_xscale, log_yscale)
        plot_histogram(self.testing_predictions, self.y_test, 'Predictions for testing data', log_xscale, log_yscale)
        plot_histogram(self.predictions, self.y, 'Predictions for all data', log_xscale, log_yscale)

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

    def plot_roc(self, q_cutoff=0.05):
        qs = self.qs[self.y == 1]  # get q-values of targets
        qs = qs[qs <= q_cutoff]
        roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
        plt.plot(qs, roc, ls='none', marker='.', ms=1)
        plt.xlabel('FDR')
        plt.ylabel('Number of PSMs')
        plt.title('ROC')
        plt.xlim((0, q_cutoff))
        plt.show()

    def get_peptide_list(self, fdr: float, label: int = 1):
        return self.peptides[(self.qs <= fdr) & (self.labels == label)]
