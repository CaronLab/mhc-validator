import pandas as pd
import numpy as np
from mhcvalidator.constants import EPSILON


def add_mhcflurry_to_feature_matrix(feature_matrix: pd.DataFrame, mhcflurry_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from mhcflurry_predictions to feature_matrix. Affinity predictions added as log values.
    All non-log value clipped to a minimum of 1e-7.
    :param feature_matrix:
    :param mhcflurry_predictions:
    :return:
    """

    predictions = pd.DataFrame()
    alleles = list(mhcflurry_predictions.loc[:, 'sample_name'].unique())
    for allele in alleles:
        df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == allele, :]
        predictions[f'{allele}_MhcFlurry_PresentationScore'] = df['presentation_score'].clip(lower=EPSILON).to_numpy()
        predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['affinity'].clip(lower=EPSILON)).to_numpy()
    df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == alleles[0], :]
    predictions[f'MhcFlurry_ProcessingScore'] = df['processing_score'].clip(lower=EPSILON).to_numpy()   # we only need one processing score column

    return feature_matrix.join(predictions)


def add_netmhcpan_to_feature_matrix(feature_matrix: pd.DataFrame, netmhcpan_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from netmhcpan_predictions to feature_matrix. Affinity predictions added as log values.
    All non-log value clipped to a minimum of 1e-7.
    :param feature_matrix:
    :param mhcflurry_predictions:
    :return:
    """

    predictions = pd.DataFrame()
    alleles = list(netmhcpan_predictions.loc[:, 'Allele'].unique())
    for allele in alleles:
        df = netmhcpan_predictions.loc[netmhcpan_predictions['Allele'] == allele, :]
        for pred in ['EL_score', 'Aff_Score', 'Aff_nM']:
            predictions[f'{allele}_NetMHCpan_{pred}'] = df[pred].clip(lower=EPSILON).to_numpy()
        predictions[f'{allele}_logNetMHCpan_Aff_nM'] = np.log(predictions[f'{allele}_NetMHCpan_Aff_nM'].to_numpy())
        predictions.drop(columns=[f'{allele}_NetMHCpan_Aff_nM'], inplace=True)

    return feature_matrix.join(predictions)
