from mhcvalidator import MhcValidator
import argparse
from pathlib import Path
from mhcvalidator import __version__

description = f"""
MhcValidator v{__version__} (https://github.com/CaronLab/mhc-validator)
Copyright 2021 Kevin Kovalchik under GNU General Public License v3.0

MhcValidator is a tool for validating peptide-spectrum matches from mass spectrometry database searches. It is 
intended for use with data from immunopeptidomics experiments, though it can be use for most types of 
proteomics experiments as well.
"""

parser = argparse.ArgumentParser(description=description)

general = parser.add_argument_group('general parameters')
general.add_argument('-i',
                     '--input',
                     required=True,
                     nargs='+',
                     type=str,
                     help='Input file(s) for MhcValidator. Must be comma- or tab-separated files or pepXML. Note that '
                          'MhcValidator has only been thoroughly tested using PIN files as input '
                          '(Percolator input files). You can pass multiple files as a space-separated list. If you '
                          'pass a generic tabular file, it must contain a column titled "Peptide" or "peptide" which '
                          'contains the peptide sequences. For generic tabular files, you should also use the '
                          '--prot_column, --decoy_tag, --tag_is_prefix, and --features_to_use arguments so '
                          'MhcValidator can figure out which PSMs are targets and which are decoys.')

general.add_argument('--pep_column',
                     type=str,
                     help='The header of the column containing peptide sequences. Generally not required unless '
                          'the input is a generic text file (e.g. a CSV export from a search engine).')

general.add_argument('--prot_column',
                     type=str,
                     help='The header of the column containing protein identifications. Used '
                          'for inferring which PSMs are targets and which are decoys. Generally not required unless '
                          'the input is a generic text file (e.g. a CSV export from a search engine).')

general.add_argument('--decoy_tag',
                     type=str,
                     help='The tag indicating decoy hits in the protein column, e.g. rev_ or decoy_ are common. Used '
                          'for inferring which PSMs are targets and which are decoys. Usually not required for '
                          'PIN files.')

general.add_argument('--tag_is_prefix',
                     type=bool,
                     default=True,
                     help='Whether the decoy tag is a prefix or not. If not, it is assumed to be a suffix. Used '
                          'for inferring which PSMs are targets and which are decoys. Usually not required for '
                          'PIN files.')

general.add_argument('--features_to_use',
                     type=str,
                     nargs='+',
                     help='A list of headers for columns to be used as training features. If your tabular data '
                          'contains a column which indicates the target/decoy label of each PSM, DO NOT INCLUDE THIS '
                          'COLUMN HERE!!')

general.add_argument('--delimiter',
                     type=str,
                     default='\t',
                     help='The delimiter of the file, if it is tabular data.')

general.add_argument('-o',
                     '--output_dir',
                     type=str,
                     help='Output directory for MhcValidator. If not indicated, the input directory will be used.')

general.add_argument('-n',
                     '--n_processes',
                     type=int,
                     default=0,
                     help='The number of threads to be used concurrently when running NetMHCpan. Uses all available '
                          'CPUs if < 1.')

mhc_params = parser.add_argument_group('MHC parameters', 'Immunopeptidomics-specific parameters.')
mhc_params.add_argument('-c',
                        '--mhc_class',
                        type=str,
                        choices=('I', 'II'),
                        help='MHC class of the experiment. Not required if you are not running MhcFlurry or NetMHCpan. '
                             'Note that MhcValidator has only been thoroughly tested on class I data.')

mhc_params.add_argument('-a',
                        '--alleles',
                        nargs='+',
                        type=str,
                        help='MHC allele(s) of the sample of interest. If there is more than one, pass them as a space-'
                             'separated list. Not required if you are not running MhcFlurry or NetMHCpan.')

mhc_params.add_argument('-p',
                        '--predictors',
                        nargs='+',
                        type=str,
                        choices=('netmhcpan', 'mhcflurry'),
                        help='The algorithms whose predictions you want to be considered by the discriminant function.')

training = parser.add_argument_group('training parameters', 'Related to the training of the artificial neural network.')

training.add_argument('-v',
                      '--verbose_training',
                      type=int,
                      default=0,
                      help='The verbosity level of tensorflow during training. Should be one of {0, 1, 2}.')

training.add_argument('-e',
                      '--epochs',
                      type=int,
                      default=40,
                      help='The maximum number of epochs for training.')

training.add_argument('-b',
                      '--batch_size',
                      type=int,
                      default=512,
                      help='The batch size used in training.')

training.add_argument('-k',
                      '--k_folds',
                      type=int,
                      default=3,
                      help='The number of splits used in training and predictions, as in K-fold cross-validation.')

training.add_argument('--stopping_patience',
                      type=int,
                      default=10,
                      help='Early stopping patience used in model training. If the loss fails to improve after this '
                           'number of epochs, the training stops and the best performing model is kept.')

training.add_argument('-s',
                      '--encode_peptide_sequences',
                      action='store_true',
                      help='Encode peptide sequences as features for the training algorithm.')

model_params = parser.add_argument_group('model parameters', 'Related to the architecture of the artificial neural '
                                                             'network to be trained.')

model_params.add_argument('-d',
                          '--dropout',
                          type=float,
                          default=0.6,
                          help='Dropout to be applied between the input and output of all layers in the artificial '
                               'neural network.')

model_params.add_argument('-w',
                          '--width_ratio',
                          type=float,
                          default=5.0,
                          help='The ratio of the width of all dense hidden layers to the number of features.')

model_params.add_argument('-l',
                          '--hidden_layers',
                          type=int,
                          default=2,
                          help='The number of hidden layers in the artificial neural network (post-sequence encoding, '
                               'if applicable).')

model_params.add_argument('--convolutional_layers',
                          type=int,
                          default=1,
                          help='The number of convolutional layers used in the sequence-encoding network.')

model_params.add_argument('--filter_size',
                          type=int,
                          default=4,
                          help='The filter size used in the convolutional sequence-encoding network.')

model_params.add_argument('--n_filters',
                          type=int,
                          default=12,
                          help='The number of filters used in the convolutional sequence-encoding network.')

model_params.add_argument('--filter_stride',
                          type=int,
                          default=4,
                          help='The filter stride used in the convolutional sequence-encoding network.')

model_params.add_argument('-f',
                          '--n_sequence_features',
                          type=int,
                          default=6,
                          help='The number of nodes in the output layer of the sequence-encoding network. i.e. '
                               'the number of numerical features in which to encode each peptide sequence as part of '
                               'the model training.')

model_params.add_argument('--learning_rate',
                          type=float,
                          default=0.001,
                          help='The learning rate used when training the neural network.')


def run():
    args = parser.parse_args()

    if args.predictors:
        if not args.mhc_class and args.alleles and args.predictors:
            raise parser.error(message="'--predictors' requires that '--mhc_class' and '--alleles' be set.")
    v = MhcValidator()
    v.set_mhc_params(alleles=args.alleles,
                     mhc_class=args.mhc_class)
    for input_file in args.input:
        if args.output_dir is None:
            args.output_dir = Path(input_file).parent
        v.load_data(input_file,
                    peptide_column=args.pep_column,
                    protein_column=args.prot_column,
                    decoy_tag=args.decoy_tag,
                    tag_is_prefix=args.tag_is_prefix,
                    file_delimiter=args.delimiter)
        v.encode_peptide_sequences()

        if args.predictors:
            if 'netmhcpan' in args.predictors:
                v.add_netmhcpan_predictions(n_processes=args.n_processes)
            if 'mhcflurry' in args.predictors:
                v.add_mhcflurry_predictions()

        if args.encode_peptide_sequences:
            model = v.get_nn_model_with_sequence_encoding(learning_rate=args.learning_rate,
                                                          dropout=args.dropout,
                                                          hidden_layers=args.hidden_layers,
                                                          width_ratio=args.width_ratio,
                                                          convolutional_layers=args.convolutional_layers,
                                                          filter_size=args.filter_size,
                                                          filter_stride=args.filter_stride,
                                                          n_filters=args.n_filters,
                                                          n_encoded_sequence_features=args.n_sequence_features)
            additional_data = v.encoded_peptides
        else:
            model = v.get_nn_model(learning_rate=args.learning_rate,
                                   dropout=args.dropout,
                                   hidden_layers=args.hidden_layers,
                                   width_ratio=args.width_ratio)
            additional_data = None

        v.run(model=model,
              additional_training_data_for_model=additional_data,
              epochs=args.epochs,
              batch_size=args.batch_size,
              early_stopping_patience=args.stopping_patience,
              n_splits=args.k_folds,
              visualize=False,
              report_directory=args.output_dir,
              verbose=args.verbose_training)

