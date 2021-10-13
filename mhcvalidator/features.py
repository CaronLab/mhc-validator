import pandas as pd
import numpy as np
from typing import List, Union
from mhcvalidator.peptides import clean_peptide_sequences
from mhcvalidator.constants import PROTON_MASS
from collections import Counter


pepxml_numerical_features = ['num_tot_proteins',
                             'num_matched_ions',
                             'massdiff',
                             'hyperscore',
                             'nextscore',
                             'expect'
                             ]
"""
Index(['uncalibrated_precursor_neutral_mass', 'assumed_charge', 'native_id',
       'spectrum', 'precursor_neutral_mass', 'retention_time_sec',
       'start_scan', 'end_scan', 'index', 'protein', 'protein_descr',
       'peptide_prev_aa', 'peptide_next_aa', 'num_tol_term', 'hyperscore',
       'nextscore', 'expect', 'modifications', 'peptide',
       'num_missed_cleavages', 'num_tot_proteins', 'tot_num_ions', 'hit_rank',
       'num_matched_ions', 'massdiff', 'calc_neutral_pep_mass', 'is_rejected',
       'modified_peptide', 'Label'],
      dtype='object')
"""


def prepare_features(data, filetype, use_features: Union[List[str], None]):
    """
    Doesn't do anything for now. Eventually this is supposed to select the appropriate features depending
    on filetype.
    :param data:
    :param filetype:
    :param use_features:
    :return:
    """
    if filetype not in ['pin', 'pout', 'pepxml', 'tabular', 'mzid', 'tandem', 'spectromine']:
        raise ValueError("filetype must be one of ['pin', 'pout', 'pepxml', 'tabular', 'mzid', 'tandem', 'spectromine']")
    if filetype == 'pepxml':
        return prepare_pepxml_features(data, use_features)
    elif filetype == 'mzid':
        return prepare_mzid_features(data, use_features)
    elif filetype == 'pin':
        return prepare_pin_features(data, use_features)
    elif filetype == 'pout':
        return prepare_pout_features(data, use_features)
    elif filetype == 'tandem':
        return prepare_tandem_features(data, use_features)
    elif filetype == 'tabular':
        return prepare_tabular_features(data, use_features)
    elif filetype == 'spectromine':
        return prepare_spectromine_features(data, use_features)
    else:
        return prepare_pout_features(data, use_features)


def prepare_spectromine_features(data,
                                 use_features: Union[List[str], None],
                                 include_cleavage_metrics: bool = False):
    features = pd.DataFrame()
    if use_features:
        for f in use_features:
            features[f] = data[f]
    else:
        use_columns = ['PEP.PeptideLength', 'PEP.RunEvidenceCount', 'PEP.Score', 'PSM.Search',
                       'PSM.DeltaMS1MZ(Theor-Cali)', 'PSM.DeltaMS1MZ(Theor-Meas)',
                       'PSM.NrOfMatchedMS2Ions', 'PSM.Score', 'PSM.CalibratedMS1MZ', 'PSM.MeasuredMS1MZ']
        for f in use_columns:
            features[f] = data[f].astype(np.float32)

        charges = pd.get_dummies(data['PP.Charge'], prefix='PP.Charge')
        features = features.join(charges)

        if 'R.FileName' in data.columns:
            filenames = pd.get_dummies(data['R.FileName'], prefix='R.FileName')
            features = features.join(filenames)

        peptide_counts = Counter(data['PEP.StrippedSequence'])
        features['V.NPeptideMatches'] = \
            np.vectorize(lambda x: peptide_counts[x])(data['PEP.StrippedSequence']).astype(np.float32)

        features['V.MatchedIonFraction'] = features['PSM.NrOfMatchedMS2Ions'] / (features['PEP.PeptideLength'] * 2)

        features['V.CalibratedAbsPPM'] = features['PSM.DeltaMS1MZ(Theor-Cali)'] / features['PSM.CalibratedMS1MZ'] * 1e6
        features['V.MeasuredAbsPPM'] = features['PSM.DeltaMS1MZ(Theor-Meas)'] / features['PSM.MeasuredMS1MZ'] * 1e6

    return features


def prepare_pepxml_features(data,
                            use_features: Union[List[str], None],
                            include_cleavage_metrics: bool = False):
    features = pd.DataFrame()
    if use_features:
        for f in use_features:
            features[f] = data[f]
        search_scores = []
    else:
        if len(data['hit_rank'].unique()) > 1:
            data = data.loc[data['hit_rank'] == 1]
            print('INFO: Searches with more than one peptide hit per spectrum are not '
                  'yet supported. Only the top ranking PSMs will be kept.')

        common_columns = ['spectrum', 'spectrumNativeID', 'native_id', 'precursor_neutral_mass',
                          'uncalibrated_precursor_neutral_mass',
                          'assumed_charge', 'retention_time_sec', 'start_scan', 'end_scan',
                          'index', 'protein', 'protein_descr', 'peptide_prev_aa',
                          'peptide_next_aa', 'num_tol_term', 'modifications', 'hit_rank', 'peptide',
                          'num_tot_proteins', 'num_matched_ions', 'tot_num_ions', 'is_rejected',
                          'num_missed_cleavages', 'calc_neutral_pep_mass', 'massdiff',
                          'num_matched_peptides', 'modified_peptide', 'nmc', 'ntt', 'peptideprophet_ntt_prob']
        DO_NOT_USE = ['Label', 'q-value', 'q_value']
        df_columns = list(data.columns)
        search_scores = list(set(df_columns) - set(common_columns + DO_NOT_USE))
        for score in search_scores:
            # this catches different search scores, also peptideprophet and iprophet values
            if score == 'expect':
                features['log10_evalue'] = np.log10(data['expect'].astype(np.float32))
            try:
                features[score] = data[score].astype(np.float32)
            except ValueError as e:
                print(score)
                raise e

        if include_cleavage_metrics:
            features['num_tol_term'] = data['num_tol_term'].astype(np.float32)
            features['num_missed_cleavages'] = data['num_missed_cleavages'].astype(np.float32)

        features['precursor_neutral_mass'] = data['precursor_neutral_mass'].astype(np.float32)
        features['calc_neutral_pep_mass'] = data['calc_neutral_pep_mass'].astype(np.float32)

        features['abs_ppm_massdiff'] = (np.abs(data['calc_neutral_pep_mass'].astype(np.float32) -
                                               data['precursor_neutral_mass'].astype(np.float32)) /
                                        data['precursor_neutral_mass'].astype(np.float32)) * 1e6

        features['matched_ion_fraction'] = data['num_matched_ions'].astype(np.float32) / data['tot_num_ions'].astype(np.float32)

        features['massdiff'] = data['massdiff'].astype(np.float32)

        features['num_matched_ions'] = data['num_matched_ions'].astype(np.float32)

        cleaned_peps = clean_peptide_sequences(list(data['peptide']))
        features['length'] = np.vectorize(len)(cleaned_peps)

        peptide_counts = Counter(cleaned_peps)
        features['n_peptide_matches'] = np.vectorize(lambda x: peptide_counts[x])(cleaned_peps).astype(np.float32)
        charges = pd.get_dummies(data['assumed_charge'], prefix='charge')
        features = features.join(charges)

    return features


def prepare_mzid_features(data,
                          use_features: Union[List[str], None]):
    features = pd.DataFrame()
    if use_features:
        for f in use_features:
            features[f] = data[f]
        search_scores = []
    else:
        if len(data['rank'].unique()) > 1:
            data = data.loc[data['rank'] == 1]
            print('INFO: Searches with more than one peptide hit per spectrum are not '
                  'yet supported. Only the top ranking PSMs will be kept.')

        common_columns = ['spectrumID', 'scan number(s)', 'scan start time', 'location', 'name',
                          'FileFormat', 'SpectrumIDFormat', 'chargeState',
                          'experimentalMassToCharge', 'calculatedMassToCharge', 'rank',
                          'passThreshold', 'IsotopeError',
                          'AssumedDissociationMethod', 'PeptideSequence', 'start', 'end', 'pre',
                          'post', 'isDecoy', 'length', 'accession', 'protein description',
                          'numDatabaseSequences', 'Modification']
        DO_NOT_USE = ['Label', 'q-value', 'q_value']
        columns_to_use = ['scan start time', 'chargeState',
                          'experimentalMassToCharge', 'calculatedMassToCharge', 'IsotopeError']
        df_columns = list(data.columns)
        search_scores = list(set(df_columns) - set(common_columns + DO_NOT_USE))
        for score in search_scores:
            # this catches different search scores, also peptideprophet and iprophet values
            if ('expect' in score.lower() or 'evalue' in score.lower()) and ('log' not in score.lower()):
                features[f'log10_{score}'] = np.log10(data[score].astype(np.float32))
            features[score] = data[score].astype(np.float32)
        for col in columns_to_use:
            features[col] = data[col].astype(np.float32)

        # calculate abs ppm error, taking into account isotope error
        ppm = []
        for i in data.index:
            err = data.loc[i, 'IsotopeError']
            charge = data.loc[i, 'chargeState']
            exp = data.loc[i, 'experimentalMassToCharge']
            calc = data.loc[i, 'calculatedMassToCharge']
            if data.loc[i, 'IsotopeError'] != 0:
                calc = calc + (err * PROTON_MASS / charge)
            ppm_error = np.abs(calc - exp) / exp * 1e6
            ppm.append(ppm_error)

        features['abs_ppm_massdiff'] = ppm

        cleaned_peps = clean_peptide_sequences(list(data['PeptideSequence']))
        peptide_counts = Counter(cleaned_peps)
        features['n_peptide_matches'] = np.vectorize(lambda x: peptide_counts[x])(cleaned_peps).astype(np.float32)
        charges = pd.get_dummies(data['chargeState'], prefix='charge')
        features = features.join(charges)

        features['length'] = np.vectorize(len)(cleaned_peps).astype(np.float32)

    return features


def prepare_tandem_features(data, use_features: Union[List[str], None]):
    raise NotImplementedError()


def prepare_pin_features(data, use_features: Union[List[str], None]):
    features = pd.DataFrame()
    if use_features:
        for f in use_features:
            features[f] = data[f]
    else:
        if 'rank' in data or 'Rank' in data:
            if len(data['rank'].unique()) > 1:
                raise NotImplementedError('Sorry, searches with more than one peptide hit per spectrum are not '
                                          'yet supported.')
        do_not_use = ['SpecId', 'Label', 'ScanNr', 'Peptide', 'Proteins']
        to_use = [x for x in data.columns if x not in do_not_use]
        for feature in to_use:
            features[feature] = data[feature].astype(np.float32)
        charges = [x for x in data.columns if 'charge' in x.lower()]
        for x in charges:
            if len(data[x].unique()) == 1:
                features.drop(columns=[x], inplace=True)

    return features


def prepare_pout_features(data, use_features: Union[List[str], None]):
    features = pd.DataFrame()
    if use_features:
        for f in use_features:
            features[f] = data[f]
    else:
        features['score'] = data['score'].astype(np.float32)
        features['posterior_error_prob'] = data['posterior_error_prob'].astype(np.float32)
    return features

def prepare_tabular_features(data, use_features: Union[List[str], None]):
    # this one will probably require input to specify which columns are important
    raise NotImplementedError()
