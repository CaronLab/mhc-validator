import os

import pandas as pd
from pyteomics import mzml
import re
import numpy as np
from typing import List, Union, Iterable
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from subprocess import Popen
from datetime import datetime
from mhcvalidator.peptides import encode_peptide_modifications
from copy import deepcopy

AUTORT_DIR = Path('~/.autort/').expanduser()
AUTORT = AUTORT_DIR / 'autort.py'
TMP_DIR = Path('/tmp/mhcvalidator_autort')
if not TMP_DIR.exists():
    TMP_DIR.mkdir()


def autort_files():
    files = ['https://github.com/bzhanglab/AutoRT/raw/master/autort.py',
             'https://github.com/bzhanglab/AutoRT/raw/master/requirements.txt']

    model_files = ['https://github.com/bzhanglab/AutoRT/raw/master/models/base_model/aa.tsv',
                   'https://github.com/bzhanglab/AutoRT/raw/master/models/base_model/readme.txt',
                   'https://github.com/bzhanglab/AutoRT/raw/master/models/base_model/model.json']
    model_files += [f'https://github.com/bzhanglab/AutoRT/raw/master/models/base_model/model_{x}.h5'
                    for x in range(10)]

    gen_model_files = ['https://github.com/bzhanglab/AutoRT/raw/master/models/general_base_model/aa.tsv',
                       'https://github.com/bzhanglab/AutoRT/raw/master/models/general_base_model/readme.txt',
                       'https://github.com/bzhanglab/AutoRT/raw/master/models/general_base_model/model.json']
    gen_model_files += [f'https://github.com/bzhanglab/AutoRT/raw/master/models/general_base_model/model_{x}.h5'
                        for x in range(10)]

    module_files = ['https://github.com/bzhanglab/AutoRT/raw/master/autort/DataIO.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/Metrics.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/ModelT.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/ModelUpdate.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/PeptideEncode.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/RTModels.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/RegCallback.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/Utils.py',
                    'https://github.com/bzhanglab/AutoRT/raw/master/autort/__init__.py',]

    return files, model_files, gen_model_files, module_files


def do_autort_files_exist() -> bool:
    files = [AUTORT, AUTORT_DIR / 'requirements.txt']
    for model in ['base_model', 'general_base_model']:
        files += [AUTORT_DIR / model / 'model.json']
        files += [AUTORT_DIR / model / 'aa.tsv']
        files += [AUTORT_DIR / model / 'readme.txt']
        files += [AUTORT_DIR / model / f'model_{x}.h5' for x in range(10)]
    files += [AUTORT_DIR / 'autort/DataIO.py',
              AUTORT_DIR / 'autort/Metrics.py',
              AUTORT_DIR / 'autort/ModelT.py',
              AUTORT_DIR / 'autort/ModelUpdate.py',
              AUTORT_DIR / 'autort/PeptideEncode.py',
              AUTORT_DIR / 'autort/RTModels.py',
              AUTORT_DIR / 'autort/RegCallback.py',
              AUTORT_DIR / 'autort/Utils.py',
              AUTORT_DIR / 'autort/__init__.py']

    for f in files:
        if not f.exists():
            return False
    return True


def check_for_autort():
    # check if autort exists. if not, download!
    if not do_autort_files_exist():
        AUTORT_DIR.mkdir(exist_ok=True)
        (AUTORT_DIR / "base_model").mkdir(exist_ok=True)
        (AUTORT_DIR / "general_base_model").mkdir(exist_ok=True)
        (AUTORT_DIR / "autort").mkdir(exist_ok=True)
        files = autort_files()
        for f in files[0]:
            name = f.split('/')[-1]
            print(f'Downloading {f}')
            command = f'curl -L -o {AUTORT_DIR / name} {f}'
            _ = Popen(command.split()).communicate()
        for f in files[1]:
            name = f.split('/')[-1]
            print(f'Downloading {f}')
            command = f'curl -L -o {AUTORT_DIR / "base_model" / name} {f}'
            _ = Popen(command.split()).communicate()
        for f in files[2]:
            name = f.split('/')[-1]
            print(f'Downloading {f}')
            command = f'curl -L -o {AUTORT_DIR / "general_base_model" / name} {f}'
            _ = Popen(command.split()).communicate()
        for f in files[3]:
            name = f.split('/')[-1]
            print(f'Downloading {f}')
            command = f'curl -L -o {AUTORT_DIR / "autort" / name} {f}'
            _ = Popen(command.split()).communicate()
        command = f'python -m pip install -r {AUTORT_DIR / "requirements.txt"}'
        _ = Popen(command.split()).communicate()


def extract_rt(scans: Union[List[int], np.array],
               mzml_file: Union[str, PathLike]):

    mzml_data = mzml.read(mzml_file)
    rts = {}

    for i, spec in enumerate(tqdm(mzml_data, desc='Extracting retention times')):
        scan_idx = spec['id'].rindex('=')
        scan = int(spec['id'][scan_idx + 1:])
        rt = float(spec['scanList']['scan'][0]['scan start time'])
        rts[scan] = rt

    return np.array([rts[int(scan)] for scan in scans])


def write_training_file(peptides: Union[List[str], np.array],
                        retention_times: Union[List[float], np.array],
                        path: Union[str, PathLike],
                        encode_modifications: bool = True,
                        modification_encodings: dict = None):
    if encode_modifications:
        peptides = deepcopy(peptides)
        peptides = encode_peptide_modifications(peptides, modification_encoding=modification_encodings)

    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = peptides
    df['y'] = retention_times

    grouped = df.groupby(['x']).mean()

    df2 = pd.DataFrame(columns=['x', 'y'])
    df2['x'] = list(grouped.index)
    df2['y'] = grouped['y']

    with open(path, 'w') as f:
        f.write('x\ty\n')
        for pep, rt in zip(peptides, retention_times):
            f.write(f'{pep}\t{rt}\n')


def train(training_file: Union[str, PathLike], model_file: Union[str, PathLike] = None, **kwargs):

    check_for_autort()
    output_dir = TMP_DIR / str(datetime.now()).replace(':', '-').replace(' ', '_')

    if not output_dir.exists():
        output_dir.mkdir()

    if model_file is None:
        model_file = AUTORT_DIR / "general_base_model" / "model.json"

    additional_args = ' '.join([f'--{key} {value}' for key, value in kwargs.items()])

    command = f'python {AUTORT} train ' \
              f'-i {training_file} ' \
              f'-o {output_dir} ' \
              f'-m {model_file} ' \
              f'-rlr ' \
              f'{additional_args}'

    _ = Popen(command.split()).communicate()

    return output_dir / 'model.json'


def predict(peptides: Union[List[str], np.array],
            model_file: Union[str, PathLike],
            output_dir: Union[str, PathLike] = None,
            encode_modifications: bool = True,
            modification_encodings: dict = None):

    if encode_modifications:
        peptides = deepcopy(peptides)
        peptides = encode_peptide_modifications(peptides, modification_encodings)

    model_file = Path(model_file)
    if output_dir is None:
        output_dir = TMP_DIR / (str(datetime.now()).replace(':', '-').replace(' ', '_') + '_predict')
    if not output_dir.exists():
        output_dir.mkdir()
    peptide_file = output_dir / 'predictions_input.tsv'
    with open(peptide_file, 'w') as f:
        f.write('x\n')
        f.write('\n'.join(peptides))

    command = f'python {AUTORT} predict ' \
              f'-t {peptide_file} ' \
              f'-o {output_dir} ' \
              f'-s {model_file} ' \
              f'-p predictions'

    _ = Popen(command.split()).communicate()

    with open(output_dir / 'predictions.tsv', 'r') as f:
        header = f.readline().strip().split()
        idx = header.index('y_pred')
        predictions = [x.strip().split() for x in f.readlines()]
        rt_dict = {x[0]: x[idx] for x in predictions}

    return np.array([float(rt_dict[pep]) for pep in peptides])


def train_predict(predict_peptides: Iterable[str],
                  train_peptides: Iterable[str],
                  train_retention_times: Iterable[float],
                  max_retention_time: int = 0,
                  encode_modifications: bool = True):

    if encode_modifications:
        predict_peptides = deepcopy(predict_peptides)
        train_peptides = deepcopy(train_peptides)
        predict_peptides, modification_encodings = encode_peptide_modifications(predict_peptides,
                                                                                return_encoding_dictionary=True)
        train_peptides = encode_peptide_modifications(train_peptides)

    output_dir = TMP_DIR / str(datetime.now()).replace(':', '-').replace(' ', '_')
    if not output_dir.exists():
        output_dir.mkdir()
    write_training_file(train_peptides, train_retention_times, output_dir / 'training_peptides.tsv')
    model_file = train(output_dir / 'training_peptides.tsv', unit='m', max_rt=max_retention_time)
    predicted_rts = predict(predict_peptides, model_file, output_dir)

    return predicted_rts
