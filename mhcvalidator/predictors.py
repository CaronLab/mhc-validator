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
from mhcvalidator.constants import COMMON_AA, SUPERTYPES
from mhcvalidator.losses_and_metrics import weighted_bce, total_fdr, precision_m, n_psms_at_1percent_fdr
from mhcvalidator.fdr import calculate_qs, calculate_peptide_level_qs, calculate_roc
import matplotlib.pyplot as plt
from mhcflurry.encodable_sequences import EncodableSequences
from mhcvalidator.models import get_model_without_peptide_encoding, get_bigger_model_with_peptide_encoding2, get_model_with_lstm_peptide_encoding
from mhcvalidator.peptides import clean_peptide_sequences
from mhcnames import normalize_allele_name
from copy import deepcopy
from scipy.stats import percentileofscore
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from mhcvalidator.encoding import pad_and_encode_multiple_aa_seq
# from mhcvalidator.mhcnugget_helper import get_mhcnuggets_preds  # mhcnuggets seems to be broken with current versions of tensorflow
from mhcvalidator.libraries import load_library, filter_library
import tempfile
from collections import Counter
# This can be uncommented to prevent the GPU from getting used.
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from scipy.stats import gmean as geom_mean

#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.config.set_soft_device_placement(enabled=True)


DEFAULT_TEMP_MODEL_DIR = str(Path(tempfile.gettempdir()) / 'validator_models')


class MhcValidator:
    def __init__(self, random_seed: int = 0, model_dir: Union[str, PathLike] = DEFAULT_TEMP_MODEL_DIR):
        self.filename: str = None
        self.filepath: Path = None
        self.model = None
        self.raw_data: Union[pd.DataFrame, None] = None
        self.feature_matrix: Union[pd.DataFrame, None] = None
        self.feature_names: Union[List[str], None] = None
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
        self.prior_qs = None
        self.mhc_class: str = None
        self.alleles: List[str] = None
        self.min_len: int = 5
        self.max_len: int = 100
        self.model_dir = Path(model_dir)

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
        self.feature_names = list(self.feature_matrix.columns)

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
        #self.raw_data.drop(columns=['Label'], inplace=True)
        self.loaded_filetype = 'pout'
        self.filename = (Path(targets_pout).name, Path(decoys_pout).name)
        self.filepath = (Path(targets_pout).expanduser().resolve(), Path(decoys_pout).expanduser().resolve())

        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_features)
        self.feature_names = list(self.feature_matrix.columns)


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
        self.prior_qs = self.raw_data['q-value'].to_numpy(np.float32)
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
        self.feature_names = list(self.feature_matrix.columns)

    def prepare_data(self, validation_split: float = 0.33, holdout_split: float = 0.33, random_seed: int = None,
                     stratification_dimensions: int = 2):
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
        peptide is predicted to be presented by any alleles or not (i.e. %rank  > 2% for any alleles is class 1, else
        is class 2). This ensures there are predicted binders and non-binders in the training, validation, and testing
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
        (X_train, X_the_rest,  y_train, y_the_rest,  X_train_peps,
         X_the_rest_peps,  X_train_encoded_peps, X_the_rest_encoded_peps, _, stratification) = train_test_split(
            X, y, peptides, encoded_peps, stratification,
            random_state=random_seed,
            stratify=stratification,
            test_size=validation_split+holdout_split
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
            random_state=random_seed+1,
            stratify=stratification,
            test_size=validation_split/(validation_split+holdout_split)
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
        print('Running MhcFlurry')
        with tf.compat.v1.Session():
            from mhcflurry import Class1PresentationPredictor
            predictor = Class1PresentationPredictor.load()
            preds = predictor.predict(peptides=self.peptides,
                                      alleles={x: [x] for x in self.alleles},
                                      #n_flanks=self.n_flanks,
                                      #c_flanks=self.c_flanks,
                                      include_affinity_percentile=True,
                                      )
        self.feature_matrix = add_mhcflurry_to_feature_matrix(self.feature_matrix, preds)

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
        print(f'Running NetMHC{"II" if self.mhc_class=="II" else ""}pan')
        netmhcpan = NetMHCpanHelper(peptides=self.peptides,
                                    alleles=alleles,
                                    mhc_class=self.mhc_class,
                                    n_threads=n_processes)
        preds = netmhcpan.predict_df()
        to_drop = [x for x in preds.columns if 'rank' in x.lower()]
        preds.drop(columns=to_drop, inplace=True)
        self.feature_matrix = add_netmhcpan_to_feature_matrix(self.feature_matrix, preds)
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

    def get_quantile_ranks(self, decoy_scores, target_scores, decoy_only=False):
        """
        Returns the quantile ranks of the target scores against the decoy scores. Closer to 1 is "better".
        :param decoy_scores:
        :param target_scores:
        :param decoy_only:
        :return:
        """
        if decoy_only:
            target_ranks = np.array([percentileofscore(decoy_scores, x) for x in decoy_scores]) / 100
        else:
            target_ranks = np.array([percentileofscore(decoy_scores, x) for x in target_scores]) / 100
        if np.mean(decoy_scores) > np.mean(target_ranks):
            # the decoy scores are higher
            target_ranks = 1 - target_ranks
        return target_ranks

    def get_sample_weights(self, X, y, decoy_factor=1, target_factor=1, decoy_bias=1, target_bias=1):
        assert X.shape[0] == y.shape[0]
        columns = list(self.feature_matrix.columns)
        col_idx = [columns.index(x) for x in columns]
        summed_weights = np.zeros(y.shape)
        for i in col_idx:
            if len(np.unique(X[:, i])) < 10:  # the data is too quantized
                continue
            if 'mass' in columns[i].lower() or 'length' in columns[i].lower():  # we won't weight based on mass or length
                continue
            target_data = X[y == 1, i]
            decoy_data = X[y == 0, i]
            weights = np.zeros(y.shape)
            weights[y == 0] = self.get_quantile_ranks(decoy_data, decoy_data,
                                                      decoy_only=True) * decoy_factor + decoy_bias
            weights[y == 1] = self.get_quantile_ranks(decoy_scores=decoy_data,
                                                      target_scores=target_data) * target_factor + target_bias
            summed_weights = summed_weights + weights
        summed_weights = summed_weights / len(col_idx)
        return summed_weights

    def get_qvalue_mask(self, X, y, cutoff: float = 0.05, n: int = 4, mhc_only: bool = True):
        assert X.shape[0] == y.shape[0]
        columns = list(self.feature_matrix.columns)
        col_idx = [columns.index(x) for x in columns]
        passes = np.zeros(X.shape)
        for i in col_idx:
            col = columns[i].lower()
            if len(np.unique(X[:, i])) < 100:  # the data is too quantized
                continue
            if 'mass' in col or 'length' in col  or 'mz' in col:  # we won't rank based on mass or length
                continue
            if mhc_only:
                if not ('netmhcpan' in col or 'mhcflurry' in col or 'mhcnuggets' in col or 'netmhciipan' in col):
                    continue
            print(f'Calculating q-values: {columns[i]}')
            higher_better = np.median(X[y==1, i]) > np.median(X[y==0, i])
            qs = calculate_qs(X[:, i], y, higher_better)
            passes[:, i] = (y == 1) & (qs <= cutoff)
            passes[y == 0, i] = True
            print(f'  Target PSMs at {cutoff} FDR: {np.sum((y == 1) & (qs <= cutoff))}')
        mask = np.sum(passes, axis=1) >= n
        print(f'Total target PSMs with {n} or more features passing {cutoff} q-value cutoff: {np.sum(mask & y==1)}')
        return mask

    def get_rank_subset_mask(self, X, y, cutoff: float = 0.95, n: int = 2):
        assert X.shape[0] == y.shape[0]
        columns = list(self.feature_matrix.columns)
        col_idx = [columns.index(x) for x in columns]
        ranks = np.zeros(X.shape)
        for i in col_idx:
            if len(np.unique(X[:, i])) < 10:  # the data is too quantized
                continue
            if 'mass' in columns[i].lower() or 'length' in columns[i].lower():  # we won't rank based on mass or length
                continue
            target_data = X[y == 1, i]
            decoy_data = X[y == 0, i]
            ranks[y == 1, i] = self.get_quantile_ranks(decoy_scores=decoy_data,
                                                       target_scores=target_data) >= cutoff
            ranks[y == 0, i] = True
        mask = np.sum(ranks, axis=1) >= n
        return mask

    """
    This can be added into the training function to save model progress
    def get_model_name(k):
            return 'model_' + str(k) + '.h5'
            
    # CREATE CALLBACKS
    checkpoint = keras.callbacks.ModelCheckpoint(str(MODEL_DIR / get_model_name(fold_var)),
                                                 monitor='val_loss', verbose=0,
                                                 save_best_only=True, mode='min')

    callbacks_list = [checkpoint, SimpleEpochProgressMonitor()]
    # load best model weights and append to list
    model.load_weights(str(MODEL_DIR / get_model_name(fold_var)))
    """

    def run(self,
            encode_peptide_sequences: bool = False,
            lstm_model: bool = False,
            epochs: int = 16,
            batch_size: int = 256,
            loss_fn=tf.losses.BinaryCrossentropy(),  # =weighted_bce(10, 2, 0.5),
            holdout_split: float = 0.33,
            validation_split: float = 0.33,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 15,
            weight_samples: bool = False,
            decoy_factor=1,
            target_factor=1,
            decoy_bias=1,
            target_bias=1,
            visualize: bool = True,
            report_dir: Union[str, PathLike] = None,
            random_seed: int = None,
            return_model: bool = False,
            fit_verbosity: int = 2,
            report_vebosity: int = 1,
            clear_session: bool = True,
            alternate_labels=None,
            initial_model_weights: str = None):

        #with self.graph.as_default():
        #tf.compat.v1.enable_eager_execution()

        if clear_session:
            K.clear_session()

        if random_seed is None:
            random_seed = self.random_seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)


        if weight_samples:
            print('Calculating sample weights')
            if isinstance(self.X_train, list):
                self.training_weights = self.get_sample_weights(self.X_train[0], self.y_train,
                                                                decoy_factor=decoy_factor,
                                                                decoy_bias=decoy_bias,
                                                                target_factor=target_factor,
                                                                target_bias=target_bias)
            else:
                self.training_weights = self.get_sample_weights(self.X_train, self.y_train,
                                                                decoy_factor=decoy_factor,
                                                                decoy_bias=decoy_bias,
                                                                target_factor=target_factor,
                                                                target_bias=target_bias)
        else:
            self.training_weights = np.ones(self.y_train.shape[0])

        if encode_peptide_sequences:
            get_model = get_bigger_model_with_peptide_encoding2
        else:
            get_model = get_model_without_peptide_encoding

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
        self.model = get_model(self.feature_matrix.shape[1], max_pep_length=encoded_pep_length)
        self.model.compile(loss=loss_fn,
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # load weights from an existing model if specified
        if initial_model_weights is not None:
            self.model.load_weights(initial_model_weights)

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
        weights = np.array([1/peptide_counts[x] for x in self.X_train_peps])

        self.fit_history = self.model.fit(X_train,
                                          self.y_train,
                                          validation_data=(X_val, self.y_val),
                                          sample_weight=weights,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          verbose=fit_verbosity,
                                          callbacks=callbacks_list)

        # load the best model
        self.model.load_weights(model_name)

        self.predictions = self.model.predict(X).flatten()
        self.training_predictions = self.model.predict(X_train).flatten()
        self.testing_predictions = self.model.predict(X_test).flatten()
        self.validation_predictions = self.model.predict(X_val).flatten()
        if report_vebosity > 0:
            print('Calculating PSM-level q-values')
        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        self.qs = np.asarray(self.qs, dtype=float)
        if report_vebosity > 0:
            print('Calculating peptide-level q-values')
        self.roc = calculate_roc(self.qs, self.labels)
        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)

        min_val_loss = np.min(self.fit_history.history['val_loss'])
        stopping_idx = self.fit_history.history['val_loss'].index(min_val_loss)
        pep_qs, pep_xs, pep_ys, peps = calculate_peptide_level_qs(self.predictions, self.y, self.peptides, higher_better=True)
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
                 f'Theoretical number of possible true positives: {n_targets - n_decoys}\n' \
                 f'Theoretical maximum global accuracy: {round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)}\n' \
                 f'  --Be wary if the testing accuracy is much higher than this value.\n'\
                 '----- CONFIDENT PSMS AND PEPTIDES -----\n'\
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n'\
                 f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n' \
                 f'\n' \
                 f'----- MODEL TRAINING, VALIDATION, TESTING -----\n'\
                 f'Training loss: {round(self.fit_history.history["loss"][stopping_idx], 3)} - '\
                 f'Validation loss: {round(self.fit_history.history["val_loss"][stopping_idx], 3)} - '\
                 f'Testing loss: {round(evaluate[0], 3)}\n'\
                 f'Training accuracy: {round(self.fit_history.history["accuracy"][stopping_idx], 3)} - '\
                 f'Validation accuracy: {round(self.fit_history.history["val_accuracy"][stopping_idx], 3)} - '\
                 f'Testing accuracy: {round(evaluate[1], 3)}\n' \
                 f'\n' \
                 f'Best model: {model_name}'

        if report_vebosity > 0:
            print(report)
        self.raw_data['mhcv_prob'] = list(self.predictions)
        self.raw_data['mhcv_q-value'] = list(self.qs)
        self.raw_data['mhcv_label'] = list(self.labels)
        self.raw_data['mhcv_peptide'] = list(self.peptides)
        if visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir)
        elif not visualize and report_dir is not None:
            self.visualize_training(outdir=report_dir, save_only=True)
        elif visualize:
            self.visualize_training()
        if report_dir is not None:
            with open(Path(report_dir, 'training_report.txt'), 'w') as f:
                f.write(report)
        if return_model:
            return model_name

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
                                 lstm_model=lstm_model,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 loss_fn=loss_fn,
                                 holdout_split=holdout_split,
                                 validation_split=validation_split,
                                 learning_rate=learning_rate,
                                 early_stopping_patience=early_stopping_patience,
                                 subset=subset,
                                 weight_samples=weight_samples,
                                 decoy_factor=decoy_factor,
                                 target_factor=target_factor,
                                 decoy_bias=decoy_bias,
                                 target_bias=target_bias,
                                 visualize=visualize,
                                 report_dir=report_dir,
                                 random_seed=random_seed,
                                 feature_qvalue_cutoff_for_training=feature_qvalue_cutoff_for_training,
                                 mhc_only_for_training_cutoff=mhc_only_for_training_cutoff,
                                 n=n,
                                 return_model=True,
                                 fit_verbosity=fit_verbosity,
                                 report_vebosity=report_vebosity,
                                 clear_session=clear_session)

        # make a backup of the labels, because we are going to change them
        #label_backup = deepcopy(self.labels)

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
                 lstm_model=lstm_model,
                 epochs=4,
                 batch_size=batch_size,
                 loss_fn=loss_fn,
                 holdout_split=holdout_split,
                 validation_split=validation_split,
                 learning_rate=learning_rate,
                 early_stopping_patience=early_stopping_patience,
                 subset=subset,
                 weight_samples=weight_samples,
                 decoy_factor=decoy_factor,
                 target_factor=target_factor,
                 decoy_bias=decoy_bias,
                 target_bias=target_bias,
                 visualize=visualize,
                 report_dir=report_dir,
                 random_seed=random_seed,
                 feature_qvalue_cutoff_for_training=feature_qvalue_cutoff_for_training,
                 mhc_only_for_training_cutoff=mhc_only_for_training_cutoff,
                 n=n,
                 return_model=False,
                 fit_verbosity=fit_verbosity,
                 report_vebosity=report_vebosity,
                 clear_session=clear_session,
                 alternate_labels=alt_labels,
                 initial_model_weights=initial_model)

        # set the labels back to the original values
        #self.labels = label_backup

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

    def iterate_training(self,
                         iterations: int = 2,
                         fdr_for_iterations: float = 0.05,
                         encode_peptide_sequences: bool = False,
                         epochs: int = 30,
                         batch_size: int = 64,
                         loss_fn=tf.losses.BinaryCrossentropy(),  # =weighted_bce(10, 2, 0.5),
                         holdout_split: float = 0.25,
                         validation_split: float = 0.25,
                         weight_samples: bool = False,
                         decoy_factor=1,
                         target_factor=1,
                         decoy_bias=1,
                         target_bias=1,
                         # conf_threshold: float = 0.33,
                         visualize: bool = True,
                         report_dir: Union[str, PathLike] = None,
                         random_seed: int = None,
                         decoy_rank_for_training: float = None):
        if random_seed is None:
            random_seed = self.random_seed
        reports = []
        mask = self.get_rank_subset_mask(self.feature_matrix.to_numpy(np.float32), np.array(self.labels),
                                         cutoff=decoy_rank_for_training)
        print('Iteration 1')
        report = self.run(encode_peptide_sequences=encode_peptide_sequences,
                          epochs=epochs,
                          batch_size=batch_size,
                          loss_fn=loss_fn,
                          holdout_split=holdout_split,
                          validation_split=validation_split,
                          weight_samples=weight_samples,
                          decoy_factor=decoy_factor,
                          target_factor=target_factor,
                          decoy_bias=decoy_bias,
                          target_bias=target_bias,
                          visualize=visualize,
                          report_dir=report_dir,
                          random_seed=random_seed, subset=mask)
        reports.append(report)
        for iteration in range(2, iterations + 1):
            print(f'Iteration {iteration}')
            #random_seed += 1
            random_mask = np.random.RandomState(random_seed).choice(a=[True, False], size=self.y.shape[0], p=[0.15, 0.85])
            subset = (self.y == 0) | ((self.qs <= fdr_for_iterations) & (self.y == 1)) | ((self.y == 1) & random_mask)
            report = self.run(encode_peptide_sequences=encode_peptide_sequences,
                              epochs=epochs,
                              batch_size=batch_size,
                              loss_fn=loss_fn,
                              holdout_split=holdout_split,
                              validation_split=validation_split,
                              weight_samples=weight_samples,
                              decoy_factor=decoy_factor,
                              target_factor=target_factor,
                              decoy_bias=decoy_bias,
                              target_bias=target_bias,
                              visualize=visualize,
                              report_dir=report_dir,
                              random_seed=random_seed,
                              subset=subset)
            reports.append(report)
        for i in range(len(reports)):
            print(f'########## Iteration {i+1} ##########')
            print(reports[i])
            print()

    def grid_search(self,
                    batch_sizes: List[int] = (64, 128, 256, 512, 1028, 2056, 4112),
                    epochs: List[int] = (15, 30, 60, 120),
                    early_stopping_patiences: List[int] = (15,),
                    learning_rates: List[float] = (0.001,),
                    holdout_splits: List[float] = (0.33,),
                    validation_splits: List[float] = (0.33,),
                    output_dir: str = None,
                    encode_peptide_sequences: bool = False,
                    visualize: bool = False,
                    title: str = None):
        """
        Run a simple grid search of the following hyperparameters: batch_size, epochs, learning_rate, early
        stopping patience. The training curves (loss and accuracy) are plotted for each, along with the number
        of PSMs and unique peptides identified at 1% FDR. Note that using many values for each parameter will
        result in a very long run time and generate many many plots. It is best to start with a coarse grid and
        refine later.

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

        total = len(batch_sizes) * len(epochs)
        i = 1

        print(f'Saving plots to {pdf_file}')

        with plt_pdf.PdfPages(pdf_file) as pdf:
            for learning_rate in learning_rates:
                for max_epochs in epochs:
                    for batch_size in batch_sizes:
                        for validation_split in validation_splits:
                            for holdout_split in holdout_splits:
                                for early_stopping_patience in early_stopping_patiences:
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
                                             holdout_split=holdout_split)

                                    xs = range(1, len(self.fit_history.history['val_loss']) + 1)

                                    val_loss = np.min(self.fit_history.history['val_loss'])
                                    stopping_idx = self.fit_history.history['val_loss'].index(val_loss)

                                    n_psms_01 = np.sum((self.qs <= 0.01) & (self.labels == 1))
                                    n_uniqe_peps_01 = len(np.unique(self.peptides[(self.qs <= 0.01) & (self.labels == 1)]))

                                    n_psms_05 = np.sum((self.qs <= 0.05) & (self.labels == 1))
                                    n_uniqe_peps_05 = len(np.unique(self.peptides[(self.qs <= 0.05) & (self.labels == 1)]))

                                    text = f'Estimated max possible target PSMs: {theoretical_possible_targets}\n' \
                                           f'  Target PSMs at 1% FDR: {n_psms_01}\n' \
                                           f'  Target peptides at 1% FDR: {n_uniqe_peps_01}\n' \
                                           f'  Target PSMs at 5% FDR: {n_psms_05}\n' \
                                           f'  Target peptides at 5% FDR: {n_uniqe_peps_05}\n\n'\
                                           f'Title: {title}\n' \
                                           f'  epochs: {max_epochs}\n'\
                                           f'  batch size: {batch_size}\n'\
                                           f'  early stopping patience: {early_stopping_patience}\n'\
                                           f'  holdout split: {holdout_split}\n'\
                                           f'  validation split: {validation_split}\n'\
                                           f'  learning rate: {learning_rate}'

                                    fig, (ax, text_ax) = plt.subplots(2, 1, figsize=(8, 10))
                                    text_ax.axis('off')

                                    tl = ax.plot(xs, self.fit_history.history['loss'], c='#3987bc', label='Training loss')
                                    vl = ax.plot(xs, self.fit_history.history['val_loss'], c='#ff851a', label='Validation loss')
                                    ax.set_ylabel('Loss')
                                    ax2 = ax.twinx()
                                    ta = ax2.plot(xs, self.fit_history.history['accuracy'], c='#3987bc', label='Training accuracy', ls='--')
                                    va = ax2.plot(xs, self.fit_history.history['val_accuracy'], c='#ff851a', label='Validation accuracy', ls='--')
                                    ax2.set_ylabel('Accuracy')
                                    ax.plot(range(1, max_epochs+1), [val_loss] * max_epochs, ls=':', c='gray')
                                    ma = ax2.plot(range(1, max_epochs+1), [max_accuracy] * max_epochs, ls='-.', c='k', zorder=0,
                                                  label='Predicted max accuracy')
                                    bm = ax.plot(self.fit_history.history['val_loss'].index(val_loss) + 1, val_loss,
                                                 marker='o', mec='red', mfc='none', ms='12', ls='none', label='best model')

                                    lines = tl + vl + bm + ta + va + ma
                                    labels = [l.get_label() for l in lines]
                                    plt.legend(lines, labels, bbox_to_anchor=(0, -.12, 1, 0), loc='upper center',
                                               mode='expand', ncol=2)

                                    ax.set_xlabel('Epoch')
                                    ylim = ax.get_ylim()

                                    ax.plot([stopping_idx + 1, stopping_idx + 1], [0, 1], ls=':', c='gray')
                                    ax.set_ylim(ylim)
                                    ax.set_xlim((1, max_epochs))
                                    ax2.set_xlim((1, max_epochs))

                                    text_ax.text(0, 0.1, text, transform=text_ax.transAxes, size=14)

                                    plt.tight_layout()
                                    if visualize:
                                        plt.show()

                                    pdf.savefig(fig)

                                    #if output_dir:
                                    #    self.raw_data.to_csv(str(Path(output_dir) / f'{self.filename}_MhcV.txt'),
                                    #                         index=False)

    def visualize_training(self, outdir: Union[str, PathLike] = None, log_yscale: bool = False, save_only: bool = False):
        if self.fit_history is None or self.X_test is None or self.y_test is None:
            raise AttributeError("Model has not yet been trained. Use run to train.")
        if outdir is not None:
            if not Path(outdir).exists():
                Path(outdir).mkdir(parents=True)

        n_targets = np.sum(self.labels == 1)
        n_decoys = np.sum(self.labels == 0)
        max_accuracy = round((n_decoys + (n_targets - n_decoys)) / len(self.labels), 3)

        n_epochs = len(self.fit_history.epoch)

        xs = range(1, len(self.fit_history.history['val_loss']) + 1)

        val_loss = np.min(self.fit_history.history['val_loss'])
        stopping_idx = self.fit_history.history['val_loss'].index(val_loss)
        n_psms = np.sum((self.qs <= 0.01) & (self.labels == 1))
        n_uniqe_peps = len(np.unique(self.peptides[(self.qs <= 0.01) & (self.labels == 1)]))

        fig, ax = plt.subplots()

        tl = ax.plot(xs, self.fit_history.history['loss'], c='#3987bc', label='Training loss')
        vl = ax.plot(xs, self.fit_history.history['val_loss'], c='#ff851a', label='Validation loss')
        ax.set_ylabel('Loss')
        ax2 = ax.twinx()
        ta = ax2.plot(xs, self.fit_history.history['accuracy'], c='#3987bc', label='Training accuracy', ls='--')
        va = ax2.plot(xs, self.fit_history.history['val_accuracy'], c='#ff851a', label='Validation accuracy', ls='--')
        ax2.set_ylabel('Accuracy')
        ax.plot(xs, [val_loss] * n_epochs, ls=':', c='gray')
        ma = ax2.plot(xs, [max_accuracy] * n_epochs, ls='-.', c='k', zorder=0,
                      label='Predicted max accuracy')
        bm = ax.plot(self.fit_history.history['val_loss'].index(val_loss) + 1, val_loss,
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
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'training_history.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        train_predictions = self.training_predictions
        _, bins, _ = plt.hist(x=np.array(train_predictions[self.y_train == 0]).flatten(),
                              label='Decoy', bins=30, alpha=0.6)
        plt.hist(x=np.array(train_predictions[self.y_train == 1]).flatten(),
                 label='Target', bins=30, alpha=0.6, range=(bins[0], bins[-1]))
        plt.title('Training data')
        if log_yscale:
            plt.yscale('log')
        plt.legend()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'training_distribution.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        val_predictions = self.validation_predictions
        _, bins, _ = plt.hist(x=np.array(val_predictions[self.y_val == 0]).flatten(),
                              label='Decoy', bins=30, alpha=0.6)
        plt.hist(x=np.array(val_predictions[self.y_val == 1]).flatten(),
                 label='Target', bins=30, alpha=0.6, range=(bins[0], bins[-1]))
        plt.title('Validation data')
        if log_yscale:
            plt.yscale('log')
        plt.legend()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'validation_distribution.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        test_predictions = self.testing_predictions
        _, bins, _ = plt.hist(x=np.array(test_predictions[self.y_test == 0]).flatten(),
                              label='Decoy', bins=30, alpha=0.6)
        plt.hist(x=np.array(test_predictions[self.y_test == 1]).flatten(),
                 label='Target', bins=30, alpha=0.6, range=(bins[0], bins[-1]))
        plt.title('Testing data')
        if log_yscale:
            plt.yscale('log')
        plt.legend()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'testing_distribution.svg')))
        if not save_only:
            plt.show()
        plt.clf()

        predictions = self.predictions
        _, bins, _ = plt.hist(x=np.array(predictions[self.y == 0]).flatten(), label='Decoy', bins=30, alpha=0.6)
        plt.hist(x=np.array(predictions[self.y == 1]).flatten(), label='Target', bins=30, alpha=0.6,
                 range=(bins[0], bins[-1]))
        plt.title('All data')
        if log_yscale:
            plt.yscale('log')
        plt.legend()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'all_data_distribution.svg')))
        if not save_only:
            plt.show()
        plt.clf()

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
