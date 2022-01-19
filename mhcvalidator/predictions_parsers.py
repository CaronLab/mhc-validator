import pandas as pd
import numpy as np
from mhcvalidator.constants import EPSILON
from typing import List


def add_mhcflurry_to_feature_matrix(feature_matrix: pd.DataFrame, mhcflurry_predictions: pd.DataFrame,
                                    peptide_list: List[str], from_file: bool = False) -> pd.DataFrame:
    """
    Add features from mhcflurry_predictions to feature_matrix. Affinity predictions added as log values.
    All non-log value clipped to a minimum of 1e-7.

    :param feature_matrix:
    :param mhcflurry_predictions:
    :param peptide_list: The list of peptides that should be in the predictions. This is used to ensure the order is
    correct.
    :param from_file: If the dataframe came from a commandline usage of mhcflurry that wrote an output csv,
    e.g. `mhcflurry-predict`, set this to True.
    :return:
    """

    predictions = pd.DataFrame()
    if from_file:
        alleles = list(mhcflurry_predictions.loc[:, 'allele'].unique())
        for allele in alleles:
            df = mhcflurry_predictions.loc[mhcflurry_predictions['allele'] == allele, :]
            assert list(df['peptide']) == list(peptide_list)
            predictions[f'{allele}_MhcFlurry_PresentationScore'] = df['mhcflurry_presentation_score'].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['mhcflurry_affinity'].clip(lower=EPSILON)).to_numpy()
        df = mhcflurry_predictions.loc[mhcflurry_predictions['allele'] == alleles[0], :]
        predictions[f'MhcFlurry_ProcessingScore'] = df['mhcflurry_processing_score'].clip(lower=EPSILON).to_numpy()   # we only need one processing score column
    else:
        alleles = list(mhcflurry_predictions.loc[:, 'sample_name'].unique())
        for allele in alleles:
            df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == allele, :]
            assert list(df['peptide']) == list(peptide_list)
            predictions[f'{allele}_MhcFlurry_PresentationScore'] = df['presentation_score'].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['affinity'].clip(lower=EPSILON)).to_numpy()
        df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == alleles[0], :]
        predictions[f'MhcFlurry_ProcessingScore'] = df['processing_score'].clip(lower=EPSILON).to_numpy()   # we only need one processing score column
    return feature_matrix.join(predictions)


def format_mhcflurry_predictions_dataframe(mhcflurry_predictions: pd.DataFrame,
                                           peptide_list: List[str],
                                           from_file: bool = False) -> pd.DataFrame:
    """
    convert the mhcflurry predictions to a wide-format dataframe and add columns for "best" predictions per peptide.

    :param mhcflurry_predictions:
    :param peptide_list: The list of peptides that should be in the predictions. This is used to ensure the order is
    correct.
    :param from_file: If the dataframe came from a commandline usage of mhcflurry that wrote an output csv,
    e.g. `mhcflurry-predict`, set this to True.
    :return:
    """

    predictions = pd.DataFrame()
    predictions['Peptide'] = np.array(peptide_list)
    if from_file:
        alleles = list(mhcflurry_predictions.loc[:, 'allele'].unique())
        for allele in alleles:
            df = mhcflurry_predictions.loc[mhcflurry_predictions['allele'] == allele, :]
            assert list(df['peptide']) == list(peptide_list)
            for pred in df.columns:
                if pred in ['allele', 'peptide', 'mhcflurry_processing_score']:
                    continue
                predictions[f'{allele}_MhcFlurry_{pred}'] = df[pred].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['mhcflurry_affinity'].clip(lower=EPSILON)).to_numpy()
        df = mhcflurry_predictions.loc[mhcflurry_predictions['allele'] == alleles[0], :]
        predictions[f'MhcFlurry_ProcessingScore'] = df['mhcflurry_processing_score'].clip(lower=EPSILON).to_numpy()   # we only need one processing score column
    else:
        alleles = list(mhcflurry_predictions.loc[:, 'sample_name'].unique())
        for allele in alleles:
            df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == allele, :]
            assert list(df['peptide']) == list(peptide_list)
            for pred in df.columns:
                if pred in ['allele', 'peptide', 'processing_score']:
                    continue
                predictions[f'{allele}_MhcFlurry_{pred}'] = df[pred].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['affinity'].clip(lower=EPSILON)).to_numpy()
        df = mhcflurry_predictions.loc[mhcflurry_predictions['sample_name'] == alleles[0], :]
        predictions[f'MhcFlurry_ProcessingScore'] = df['processing_score'].clip(lower=EPSILON).to_numpy()   # we only need one processing score column
    return predictions


def add_netmhcpan_to_feature_matrix(feature_matrix: pd.DataFrame,
                                    netmhcpan_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from netmhcpan_predictions to feature_matrix. Affinity predictions added as log values.
    All non-log value clipped to a minimum of 1e-7.
    :param feature_matrix:
    :param netmhcpan_predictions:
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


def format_netmhcpan_prediction_dataframe(netmhcpan_predictions: pd.DataFrame,
                                          peptide_list: List[str]) -> pd.DataFrame:
    """
    convert the netmhcpan predictions to a wide-format dataframe and add columns for "best" predictions per peptide.
    :param netmhcpan_predictions:
    :return:
    """

    def binder_type(x):
        x=list(x)
        if 'Strong' in x:
            return 'Strong'
        elif 'Weak' in x:
            return 'Weak'
        else:
            return 'Non-binder'

    predictions = pd.DataFrame()
    predictions['Peptide'] = np.array(peptide_list)
    alleles = list(netmhcpan_predictions.loc[:, 'Allele'].unique())
    for allele in alleles:
        df = netmhcpan_predictions.loc[netmhcpan_predictions['Allele'] == allele, :].copy(deep=True)
        assert list(df['Peptide']) == list(peptide_list)
        for pred in df.columns:
            if pred in ['Allele', 'Peptide', 'Binder']:
                continue
            predictions[f'{allele}_NetMHCpan_{pred}'] = df[pred].to_numpy()
        predictions[f'{allele}_NetMHCpan_Binder'] = df['Binder'].to_numpy()
        predictions[f'{allele}_logNetMHCpan_Aff_nM'] = np.log(predictions[f'{allele}_NetMHCpan_Aff_nM'].to_numpy())
    binder_columns = [x for x in predictions.columns if 'Binder' in x]
    predictions['Binder'] = predictions.loc[:, binder_columns].apply(binder_type, axis=1).to_numpy()

    return predictions
