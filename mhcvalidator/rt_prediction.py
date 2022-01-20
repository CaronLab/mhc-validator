import os
from pyteomics import mzml
import re
import numpy as np
from typing import List, Union
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from subprocess import Popen


def extract_rt(scans: Union[List[int], np.array],
               mzml_file: Union[str, PathLike]):

    mzml_data = mzml.read(mzml_file)
    rts = {}

    for i, spec in enumerate(tqdm(mzml_data, desc='Extracting retention times')):
        scan_idx = spec['id'].rindex('=')
        scan = int(spec['id'][scan_idx + 1:])
        rt = float(spec['scanList']['scan'][0]['scan start time'])
        rts[scan] = rt

    return np.array([rts[scan] for scan in scans])


def write_training_file(peptides: Union[List[str], np.array],
                        retention_times: Union[List[float], np.array],
                        path: Union[str, PathLike]):

    with open(path, 'w') as f:
        f.write('x\ty\n')
        for pep, rt in zip(peptides, retention_times):
            f.write(f'{pep}\t{rt}\n')


def transfer_train(training_file: Union[str, PathLike], model_file: Union[str, PathLike] = None, **kwargs):

    # check if autort exists. if not, download!
    autort_dir = Path('~/.autort/').expanduser()
    if not autort_dir.exists():
        autort_dir.mkdir()
        command = f'wget https://github.com/bzhanglab/AutoRT/archive/refs/heads/master.zip -P {autort_dir}'
        _ = Popen(command.split()).communicate()


