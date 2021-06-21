import pandas as pd
from mhcnuggets.src.predict import predict
import tempfile
from mhcnames import normalize_allele_name
from typing import Iterable
from numpy import log


def get_mhcnuggets_preds(mhc_class: str, alleles: Iterable[str], peptides: Iterable[str]):
    df = pd.DataFrame()
    for allele in alleles:
        tp = tempfile.NamedTemporaryFile('w', delete=False)
        tp.write('\n'.join(list(peptides)) + '\n')
        tp.flush()
        tp.close()
        out = tempfile.NamedTemporaryFile(delete=False)
        predict(mhc_class, tp.name, allele, output=out.name, mass_spec=True, binary_preds=True)
        out.close()
        with open(out.name, 'r') as f:
            preds = [float(x.strip().split(',')[1]) for x in f.readlines()[1:]]
        df[f'log_{allele}_mhcnuggets'] = log(preds)
    return df
