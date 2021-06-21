import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from typing import Union, List
from os import PathLike
from pathlib import Path
from Validator.data_loaders import load_file, load_pout_data
from Validator.features import prepare_features
from Validator.predictions_parsers import add_mhcflurry_to_feature_matrix, add_netmhcpan_to_feature_matrix
from Validator.netmhcpan_helper import NetMHCpanHelper, format_class_II_allele
from Validator.constants import COMMON_AA, SUPERTYPES
from Validator.losses_and_metrics import weighted_bce, total_fdr, precision_m
from Validator.fdr import calculate_qs, calculate_peptide_level_qs
import matplotlib.pyplot as plt
from mhcflurry.encodable_sequences import EncodableSequences
from Validator.models import get_model_without_peptide_encoding, get_bigger_model_with_peptide_encoding2, get_model_with_lstm_peptide_encoding
from Validator.peptides import clean_peptide_sequences
from mhcnames import normalize_allele_name, parse_allele_name
from copy import deepcopy
from scipy.stats import percentileofscore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from Validator.callbacks import SimpleEpochProgressMonitor
from Validator.encoding import pad_and_encode_multiple_aa_seq
from Validator.mhcnugget_helper import get_mhcnuggets_preds
import tempfile
# This can be uncommented to prevent the GPU from getting used.
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from scipy.stats import gmean as geom_mean

#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.config.set_soft_device_placement(enabled=True)


DEFAULT_TEMP_MODEL_DIR = str(Path(tempfile.gettempdir()) / 'validator_models')


class Validator:
    def __init__(self, random_seed: int = 1234, model_dir: Union[str, PathLike] = DEFAULT_TEMP_MODEL_DIR):
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
        self.predicted_probabilities: Union[List[float], None] = None
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
        self.training_weights = None
        self.search_score_names: List[str] = []
        self.predictions = None
        self.qs = None
        self.roc = None
        self.prior_qs = None
        self.mhc_class: str = None
        self.alleles: List[str] = None
        self.min_len: int = 5
        self.max_len: int = 100
        self.model_dir = Path(model_dir)
        #self.graph = tf.Graph()

    def set_mhc_params(self, alleles: List[str] = ('HLA-A0201', 'HLA-B0702', 'HLA-C0702'), mhc_class: str = 'I'):
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
                  filename: Union[str, PathLike],
                  filetype='auto',
                  decoy_tag='rev_',
                  protein_column: str = None,
                  tag_is_prefix: bool = True,
                  file_delimiter: str = '\t',
                  use_features: Union[List[str], None] = None):

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file...')
        self.raw_data = load_file(filename=filename, filetype=filetype, decoy_tag=decoy_tag,
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
        self.filename = Path(filename).name
        self.filepath = Path(filename).expanduser().resolve()

        print('Preparaing training features')
        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_features)
        self.feature_names = list(self.feature_matrix.columns)

    def load_pout_data(self,
                       targets_pout: Union[str, PathLike],
                       decoys_pout: Union[str, PathLike],
                       use_features: Union[List[str], None] = None):

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file')
        self.raw_data = load_pout_data(targets_pout, decoys_pout, self.min_len, self.max_len)
        self.labels = self.raw_data['Label']
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
                             ):
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

    def prepare_data(self, use_feature: Union[List[str], None] = None):
        if self.raw_data is None:
            raise AttributeError("Data has not yet been loaded.")
        self.feature_matrix = prepare_features(self.raw_data,
                                               filetype=self.loaded_filetype,
                                               use_features=use_feature)

    def add_mhcflurry_predictions(self):
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using Validator.set_mhc_params')
        if self.mhc_class == 'II':
            raise RuntimeError('MhcFlurry is only compatible with MHC class I')
        if self.feature_matrix is None:
            self.prepare_data()
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

    def add_netmhcpan_predictions(self):
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using Validator.set_mhc_params')
        if self.mhc_class == 'II':
            alleles = [format_class_II_allele(x) for x in self.alleles]
        else:
            alleles = self.alleles

        if self.feature_matrix is None:
            self.prepare_data()
        print(f'Running NetMHC{"II" if self.mhc_class=="II" else ""}pan')
        netmhcpan = NetMHCpanHelper(peptides=self.peptides,
                                    alleles=alleles,
                                    mhc_class=self.mhc_class)
        preds = netmhcpan.predict_df()
        to_drop = [x for x in preds.columns if 'rank' in x.lower()]
        preds.drop(columns=to_drop, inplace=True)
        self.feature_matrix = add_netmhcpan_to_feature_matrix(self.feature_matrix, preds)

    def add_mhcnuggets_predictions(self):
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using Validator.set_mhc_params')
        if self.mhc_class == 'II':
            alleles = [format_class_II_allele(x) for x in self.alleles]
        else:
            alleles = self.alleles

        preds = get_mhcnuggets_preds(self.mhc_class, alleles, self.peptides)
        self.feature_matrix = self.feature_matrix.join(preds)

    def add_all_available_predictions(self, verbose_errors: bool = False):
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
        try:
            self.add_mhcnuggets_predictions()
        except Exception as e:
            if verbose_errors:
                print(e)
            print(f'Unable to run MhcNuggets.'
                  f'{" See exception information above." if verbose_errors else ""}')

    def get_confident_indices(self, index_for_training, quantile: float = 0.1):
        indices = set()
        # get the initial training set
        X = self.feature_matrix.iloc[index_for_training, :].to_numpy(dtype=np.float32)
        y = self.labels[index_for_training]
        # get the columns which contain the scores for confidence, both search engines and MHC predictors
        columns = list(self.feature_matrix.columns)
        search_idx = [columns.index(x) for x in self.search_score_names]
        predictor_index = [columns.index(x) for x in columns if ('NetMHC' in x) or ('MhcFlurry' in x)]
        col_idx = search_idx + predictor_index
        # go through those and get the indices of the PSMs which pass the threshold for the score
        for i in col_idx:
            if len(np.unique(X[:, i])) < 10:
                continue
            target_data = X[y == 1, i]
            decoy_data = X[y == 0, i]
            target_score = np.median(target_data)
            decoy_score = np.median(decoy_data)
            if target_score > decoy_score:
                cutoff = np.quantile(target_data, 1 - quantile)
                indices = indices | set(np.argwhere(target_data >= cutoff).flatten())
            else:
                cutoff = np.quantile(target_data, quantile)
                indices = indices | set(np.argwhere(target_data <= cutoff).flatten())
        decoy_indices = set(np.argwhere(y == 0).flatten())
        return np.array(list(indices | decoy_indices))

    def get_train_test_indices(self, y, peptides, test_ratio: float = 0.25, rs: RandomState = None):
        """
        Get random training and testing indices while ensuring there are no sequence duplicates between training and
        testing sets.
        :param y:
        :param peptides:
        :param test_ratio:
        :param rs:
        :return:
        """
        assert len(peptides) == len(y)

        if rs is None:
            rs = RandomState()

        peps = np.array(peptides)
        peps = np.unique(peps)  # unique sequences
        random_index = rs.permutation(len(peps))
        peps = peps[random_index]  # order is now randomized
        test_size = int(len(peps) * test_ratio)  # number of unique peptide sequences in test set

        test_peps, train_peps = set(peps[:test_size]), set(peps[test_size:])

        train_idx = []
        test_idx = []

        for i in range(len(peptides)):
            if peptides[i] in test_peps:
                test_idx.append(i)
            else:
                train_idx.append(i)

        return np.array(train_idx), np.array(test_idx)

    def get_quantile_ranks(self, decoy_scores, target_scores, decoy_only=False):
        """
        Returns the quantile ranks of the target scores against the decoy scores. Closer to 1 is "better".
        :param decoy_scores:
        :param target_scores:
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

    def train_validation_model(self,
                               encode_peptide_sequences: bool = False,
                               lstm_model: bool = False,
                               epochs: int = 10,
                               batch_size: int = 64,
                               loss_fn=tf.losses.BinaryCrossentropy(),  # =weighted_bce(10, 2, 0.5),
                               holdout_split: float = 0.5,
                               validation_split: float = 0.1,
                               subset: np.array = None,
                               weight_samples: bool = False,
                               decoy_factor=1,
                               target_factor=1,
                               decoy_bias=1,
                               target_bias=1,
                               #conf_threshold: float = 0.33,
                               visualize: bool = True,
                               report_dir: Union[str, PathLike] = None,
                               random_seed: int = None,
                               feature_qvalue_cutoff_for_training: float = None,
                               mhc_only_for_training_cutoff: bool = False,
                               n: int = 2):
        #with self.graph.as_default():
        #tf.compat.v1.enable_eager_execution()
        X = self.feature_matrix.to_numpy(dtype=np.float32)
        y = np.array(self.labels, dtype=np.float32)
        peptides = np.array(self.peptides)

        assert X.shape[0] == y.shape[0]

        if random_seed is None:
            random_seed = self.random_seed

        input_scalar = MinMaxScaler()

        # save X and y before we do any shuffling. We will need this in the original order for predictions later
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        #self.X = keras.utils.normalize(self.X, 0)
        input_scalar.fit(self.X)
        self.X = input_scalar.transform(self.X)

        if encode_peptide_sequences:
            # add encoded sequences to self.X
            if lstm_model:
                encoded_peps = pad_and_encode_multiple_aa_seq(list(self.peptides), padding='post',
                                                              max_length=self.max_len)
                encoded_pep_length = self.max_len
            else:
                encoder = EncodableSequences(list(self.peptides))
                padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
                encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                                       max_length=self.max_len,
                                                                                       alignment_method=padding)
                if self.mhc_class == 'I':
                    encoded_pep_length = self.max_len
                else:
                    encoded_pep_length = 2 * self.max_len
            encoded_peps = keras.utils.normalize(encoded_peps, 0)
            self.X = [self.X, encoded_peps]
        else:
            encoded_pep_length = self.max_len
        if subset is not None and feature_qvalue_cutoff_for_training is not None:
            raise ValueError("'subset' and 'decoy_training_rank' cannot both be defined.")
        if subset is not None:
            X = X[subset]
            y = y[subset]
            peptides = peptides[subset]
        if feature_qvalue_cutoff_for_training is not None:
            mask = self.get_qvalue_mask(self.feature_matrix.to_numpy(np.float32), np.array(self.labels),
                                        feature_qvalue_cutoff_for_training, n, mhc_only_for_training_cutoff)
            X = X[mask]
            y = y[mask]
            peptides = peptides[mask]

        rs = RandomState(seed=random_seed)

        # first get training and testing sets
        random_index = rs.permutation(X.shape[0])
        X = X[random_index]
        y = y[random_index]
        shuffled_peps = peptides[random_index]

        # to ensure we don't have common peptide sequences between train and test
        #train_idx, test_idx = self.get_train_test_indices(y, shuffled_peps, holdout_split, rs)
        #X_train, X_test = X[train_idx], X[test_idx]
        #y_train, y_test = y[train_idx], y[test_idx]
        #X_train_peps, X_test_peps = shuffled_peps[train_idx], shuffled_peps[test_idx]

        test_size = int(X.shape[0] * holdout_split)
        X_train, X_test = X[test_size:, :], X[:test_size, :]
        y_train, y_test = y[test_size:], y[:test_size]
        X_train_peps, X_test_peps = shuffled_peps[test_size:], shuffled_peps[:test_size]

        assert X_train.shape[0] == y_train.shape[0] == X_train_peps.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

        # Then get the good scoring targets and all decoys for training set
        #idx = self.get_confident_indices(random_index[test_size:], quantile=conf_threshold)

        #X_train, y_train, X_train_peps = X_train[idx], y_train[idx], X_train_peps[idx]

        # normalize everything
        input_scalar.fit(X_train)
        X_train = input_scalar.transform(X_train)  # keras.utils.normalize(X_train, 0)
        input_scalar.fit(X_test)
        X_test = input_scalar.transform(X_test)  # keras.utils.normalize(X_test, 0)
        if encode_peptide_sequences:
            # training peptides
            if lstm_model:
                encoded_peps = pad_and_encode_multiple_aa_seq(list(X_train_peps), padding='post',
                                                              max_length=self.max_len)
            else:
                encoder = EncodableSequences(list(X_train_peps))
                padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
                encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                                       max_length=self.max_len,
                                                                                       alignment_method=padding)
            X_train_peps = keras.utils.normalize(encoded_peps, 0)
            training_set = [X_train, X_train_peps]

            # testing peptides
            if lstm_model:
                encoded_peps = pad_and_encode_multiple_aa_seq(list(X_test_peps), padding='post',
                                                              max_length=self.max_len)
            else:
                encoder = EncodableSequences(list(X_test_peps))
                padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
                encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                                       max_length=self.max_len,
                                                                                       alignment_method=padding)
            X_test_peps = keras.utils.normalize(encoded_peps, 0)
            testing_set = [X_test, X_test_peps]
        else:
            training_set = X_train
            testing_set = X_test
        self.X_test = testing_set
        self.y_test = y_test

        # shuffle the X_train and y_train one last time
        assert y_train.shape[0] == X_train.shape[0]
        random_index = rs.permutation(y_train.shape[0])
        if encode_peptide_sequences:
            training_set = [training_set[0][random_index], training_set[1][random_index]]
        else:
            training_set = training_set[random_index]
        y_train = y_train[random_index]
        self.X_train = training_set
        self.y_train = y_train
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

        tf.random.set_seed(random_seed)

        if encode_peptide_sequences:
            if lstm_model:
                get_model = get_model_with_lstm_peptide_encoding
            else:
                get_model = get_bigger_model_with_peptide_encoding2
        else:
            get_model = get_model_without_peptide_encoding

        self.model = get_model(self.feature_matrix.shape[1], max_pep_length=encoded_pep_length)
        self.model.compile(loss=loss_fn,
                           optimizer='adam',
                           metrics=['accuracy'])

        self.fit_history = self.model.fit(self.X_train,
                                          self.y_train,
                                          sample_weight=self.training_weights,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          verbose=2,
                                          validation_split=validation_split)

        self.predictions = self.model.predict(self.X).flatten()
        print('Calculating PSM-level q-values')
        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        print('Calculating peptide-level q-values')
        pep_qs, pep_ys, peps = calculate_peptide_level_qs(self.predictions, self.y, self.peptides)
        qs = self.qs[self.y == 1]
        self.roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
        psm_target_mask = (self.qs <= 0.01) & (self.y == 1)
        n_psm_targets = np.sum(psm_target_mask)
        n_unique_psms = len(np.unique(self.peptides[psm_target_mask]))
        n_unique_peps = np.sum((pep_qs <= 0.01) & (pep_ys == 1))
        evaluate = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size, verbose=0)
        report = '----- PSMS AND PEPTIDES -----\n'\
                 f'Target PSMs at 1% FDR: {n_psm_targets}\n'\
                 f'Unique peptides at 1% PSM-level FDR: {n_unique_psms}\n' \
                 f'Unique peptides at 1% peptide-level FDR: {n_unique_peps}\n' \
                 f'\n' \
                 f'----- MODEL TRAINING, VALIDATION, TESTING -----\n'\
                 f'Training loss: {round(self.fit_history.history["loss"][-1], 3)} - '\
                 f'Validation loss: {round(self.fit_history.history["val_loss"][-1], 3)} - '\
                 f'Testing loss: {round(evaluate[0], 3)}\n'\
                 f'Training accuracy: {round(self.fit_history.history["accuracy"][-1], 3)} - '\
                 f'Validation accuracy: {round(self.fit_history.history["val_accuracy"][-1], 3)} - '\
                 f'Testing accuracy: {round(evaluate[1], 3)}\n'
        print(report)
        self.raw_data['v_prob'] = list(self.predictions)
        self.raw_data['q_value'] = list(self.qs)
        if visualize:
            self.visualize_training(outdir=report_dir)
        if report_dir is not None:
            with open(Path(report_dir, 'training_report.txt'), 'w') as f:
                f.write(report)
        return report

    def train_validation_model_with_kfold(self,
                               encode_peptide_sequences: bool = False,
                               epochs: int = 15,
                               batch_size: int = 128,
                               loss_fn=tf.losses.BinaryCrossentropy(),  # =weighted_bce(10, 2, 0.5),
                               holdout_split: float = 0.25,
                               subset: np.array = None,
                               visualize: bool = True,
                               report_dir: Union[str, PathLike] = None,
                               random_seed: int = None,
                               decoy_rank_for_training: float = None,
                                          k: int = 5):
        MODEL_DIR = self.model_dir / str(datetime.now()).replace(' ', '_').replace(':', '-')
        MODEL_DIR.mkdir(parents=True)

        X = self.feature_matrix.to_numpy(dtype=np.float32)
        y = np.array(self.labels, dtype=np.float32)
        peptides = np.array(self.peptides)

        assert X.shape[0] == y.shape[0]

        if random_seed is None:
            random_seed = self.random_seed

        # save X and y before we do any shuffling. We will need this in the original order for predictions later
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.X = keras.utils.normalize(self.X, 0)

        if encode_peptide_sequences:
            # add encoded sequences to self.X
            encoder = EncodableSequences(list(self.peptides))
            encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62')
            encoded_peps = keras.utils.normalize(encoded_peps, 0)
            self.X = [self.X, encoded_peps]
        if subset is not None and decoy_rank_for_training is not None:
            raise ValueError("'subset' and 'decoy_training_rank' cannot both be defined.")
        if subset is not None:
            X = X[subset]
            y = y[subset]
            peptides = peptides[subset]
        if decoy_rank_for_training is not None:
            mask = self.get_rank_subset_mask(self.feature_matrix.to_numpy(np.float32), np.array(self.labels),
                                             decoy_rank_for_training)
            X = X[mask]
            y = y[mask]
            peptides = peptides[mask]

        rs = RandomState(seed=random_seed)

        # first shuffle everything
        random_index = rs.permutation(X.shape[0])
        X = X[random_index]
        y = y[random_index]
        shuffled_peps = peptides[random_index]

        # get training and testing sets
        test_size = int(X.shape[0] * holdout_split)
        X_train, X_test = X[test_size:, :], X[:test_size, :]
        y_train, y_test = y[test_size:], y[:test_size]
        X_train_peps, X_test_peps = shuffled_peps[test_size:], shuffled_peps[:test_size]

        assert X_train.shape[0] == y_train.shape[0] == X_train_peps.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

        # normalize everything
        X_train = keras.utils.normalize(X_train, 0)
        X_test = keras.utils.normalize(X_test, 0)

        # encode peptides if needed
        if encode_peptide_sequences:
            # training peptides
            encoder = EncodableSequences(list(X_train_peps))
            encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62')
            X_train_peps = keras.utils.normalize(encoded_peps, 0)
            training_set = [X_train, X_train_peps]

            # testing peptides
            encoder = EncodableSequences(list(X_test_peps))
            encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62')
            X_test_peps = keras.utils.normalize(encoded_peps, 0)
            testing_set = [X_test, X_test_peps]
        else:
            training_set = X_train
            testing_set = X_test
        self.X_test = testing_set
        self.y_test = y_test

        # shuffle the X_train and y_train one last time
        assert y_train.shape[0] == X_train.shape[0]
        random_index = rs.permutation(y_train.shape[0])
        if encode_peptide_sequences:
            training_set = [training_set[0][random_index], training_set[1][random_index]]
        else:
            training_set = training_set[random_index]
        y_train = y_train[random_index]
        self.X_train = training_set
        self.y_train = y_train

        # set up the kfolds splits
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)

        tf.random.set_seed(random_seed)

        def get_model_name(k):
            return 'model_' + str(k) + '.h5'

        fold_var = 1
        if encode_peptide_sequences:
            predictions_from_all_models = np.empty((self.X[0].shape[0], k))
        else:
            predictions_from_all_models = np.empty((self.X.shape[0], k))
        reports = []
        # iterate through the splits
        for train_indices, val_indices in kfold.split(np.zeros(self.y_train.shape[0]), self.y_train):
            print(f'############### Fold {fold_var}/{k} ###############')
            if not encode_peptide_sequences:
                X_train, X_val = self.X_train[train_indices], self.X_train[val_indices]
                X_train = keras.utils.normalize(X_train, 0)
                X_val = keras.utils.normalize(X_val, 0)
            else:
                X_train = [self.X_train[0][train_indices], self.X_train[1][train_indices]]
                X_val = [self.X_train[0][val_indices], self.X_train[1][val_indices]]
                X_train[0] = keras.utils.normalize(X_train[0], 0)
                X_train[1] = keras.utils.normalize(X_train[1], 0)
                X_val[0] = keras.utils.normalize(X_val[0], 0)
                X_val[1] = keras.utils.normalize(X_val[1], 0)
            y_train, y_val = self.y_train[train_indices], self.y_train[val_indices]

            # CREATE CALLBACKS
            checkpoint = keras.callbacks.ModelCheckpoint(str(MODEL_DIR / get_model_name(fold_var)),
                                                         monitor='val_loss', verbose=0,
                                                         save_best_only=True, mode='min')

            callbacks_list = [checkpoint, SimpleEpochProgressMonitor()]

            if encode_peptide_sequences:
                get_model = get_bigger_model_with_peptide_encoding2
            else:
                get_model = get_model_without_peptide_encoding

            model = get_model(self.feature_matrix.shape[1])
            model.compile(loss=loss_fn,
                          optimizer='adam',
                          metrics=['accuracy'])

            fit_history = model.fit(X_train,
                                    y_train,
                                    sample_weight=self.training_weights,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=0,
                                    callbacks=callbacks_list,
                                    validation_data=(X_val, y_val))

            # load best model weights and append to list
            model.load_weights(str(MODEL_DIR / get_model_name(fold_var)))

            predictions = model.predict(self.X).flatten()
            predictions_from_all_models[:, fold_var-1] = predictions
            self.predictions = deepcopy(predictions)
            self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
            qs = self.qs[self.y == 1]
            self.roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
            target_mask = (self.qs <= 0.01) & (self.y == 1)
            n_targets = np.sum(target_mask)
            n_unique = len(np.unique(self.peptides[target_mask]))
            evaluate = model.evaluate(self.X_test, self.y_test, batch_size=batch_size, verbose=0)
            report = '----- PSMS AND PEPTIDES -----\n'\
                     f'Target PSMs at 1% FDR: {n_targets}\n'\
                     f'Unique peptides at 1% FDR: {n_unique}\n' \
                     f'\n' \
                     f'----- MODEL TRAINING, VALIDATION, TESTING -----\n'\
                     f'Training loss: {round(fit_history.history["loss"][-1], 3)} - '\
                     f'Validation loss: {round(fit_history.history["val_loss"][-1], 3)} - '\
                     f'Testing loss: {round(evaluate[0], 3)}\n'\
                     f'Training accuracy: {round(fit_history.history["accuracy"][-1], 3)} - '\
                     f'Validation accuracy: {round(fit_history.history["val_accuracy"][-1], 3)} - '\
                     f'Testing accuracy: {round(evaluate[1], 3)}\n\n'
            print(report)
            reports.append(report)
            fold_var += 1
            K.clear_session()

        # now average the predictions from all models and calculate other things
        self.predictions = np.mean(predictions_from_all_models, axis=1)

        self.qs = calculate_qs(self.predictions, self.y, higher_better=True)
        qs = self.qs[self.y == 1]
        self.roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
        target_mask = (self.qs <= 0.01) & (self.y == 1)
        n_targets = np.sum(target_mask)
        n_unique = len(np.unique(self.peptides[target_mask]))

        report = '############### FINAL REPORT ###############\n' \
                 f'Target PSMs at 1% FDR: {n_targets}\n' \
                 f'Unique peptides at 1% FDR: {n_unique}\n' \
                 f'############################################'

        #for rep in reports:
        #    print(rep)
        #    print()
        print(report)

        self.raw_data['v_prob'] = list(self.predictions)
        self.raw_data['q_value'] = list(self.qs)

        predictions = self.predictions
        plt.hist(x=np.array(predictions[self.y == 1]).flatten(), label='Target', bins=30, alpha=0.6)
        plt.hist(x=np.array(predictions[self.y == 0]).flatten(), label='Decoy', bins=30, alpha=0.6)
        if isinstance(self.filename, (list, tuple)):
            name = Path(self.filename[0]).stem
        else:
            name = Path(self.filename).stem
        plt.title('Validator scores\n'
                  f'{name}')
        plt.legend()
        plt.show()

        self.plot_roc(0.05)

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
        report = self.train_validation_model(encode_peptide_sequences=encode_peptide_sequences,
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
            report = self.train_validation_model(encode_peptide_sequences=encode_peptide_sequences,
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

    def visualize_training(self, outdir: Union[str, PathLike] = None, log_yscale: bool = False):
        if self.fit_history is None or self.X_test is None or self.y_test is None:
            raise AttributeError("Model has not yet been trained. Use train_validation_model to train.")
        if outdir is not None:
            if not Path(outdir).exists():
                Path(outdir).mkdir(parents=True)
        plt.plot(self.fit_history.history['loss'])
        plt.plot(self.fit_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper left')
        plt.tight_layout()
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'loss_history.svg')))
        plt.show()

        '''plt.plot(self.fit_history.history['total_fdr'])
        plt.plot(self.fit_history.history['val_total_fdr'])
        plt.title('total_fdr')
        plt.ylabel('fdr')
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper left')
        plt.show()'''

        train_predictions = self.model.predict(self.X_train)
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
        plt.show()

        test_predictions = self.model.predict(self.X_test)
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
        plt.show()

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
        plt.show()

        qs = self.qs[self.y == 1]  # get q-values of targets
        qs = qs[qs <= 0.05]
        roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
        plt.plot(qs, roc, ls='none', marker='.', ms=1)
        plt.xlim((0, 0.05))
        plt.xlabel('FDR')
        plt.ylabel('Number of PSMs')
        plt.title('ROC')
        if outdir is not None:
            plt.savefig(str(Path(outdir, 'ROC.svg')))
        plt.show()

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
