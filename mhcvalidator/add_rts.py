from mhcvalidator.validator import MhcValidator
import sys
import os
from contextlib import contextmanager
from typing import List, Union
from pyteomics import mzml
from os import PathLike
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from koinapy import Koina
import numpy as np
import pandas as pd
from mhcvalidator.peptides import clean_peptide_sequences
from remove_nonstandard_peps import remove_nonstandard_peps


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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

def adjustment_function(x_data, y_data,deg):
    # Fit a quadratic function to the data
    coefficients = np.polyfit(x_data, y_data, deg=deg)
    function = np.poly1d(coefficients)
    return function


def adjustment_plot(x_data, y_data, function):
    # Assuming df is already defined and is a pandas DataFrame
    # Define the columns to be plotted
    overlap_df = pd.DataFrame()
    overlap_df['real_rts'] = x_data
    overlap_df['iRTs'] = y_data

    x_column = 'real_rts'
    y_column = 'iRTs'

    # Create the point plot
    sns.pointplot(data=overlap_df, x=x_column, y=y_column)

    # Display the plot
    matplotlib.use('TkAgg')
    plt.show()
    plt.close()

    # Generate x values for plotting the fitted function
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = function(x_fit)

    # Plot the original data points
    plt.scatter(x_data, y_data, label='Data Points')

    # Plot the fitted quadratic function
    plt.plot(x_fit, y_fit, color='red', label='Fitted Function')

    # Add labels and legend
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()

    # Display the plot
    plt.show()


def extract_peps(pins):
    peptides_for_RT = []
    for pin in pins:
        df = pd.read_table(pin, usecols=['Label', 'Peptide'], index_col=False)
        peptides = clean_peptide_sequences(list(df['Peptide']))
        peptides_for_RT += remove_nonstandard_peps(peptides)
    unique_peptides_for_RT = list(set(peptides_for_RT))
    return unique_peptides_for_RT

def predict_rts(peptide_list):


    inputs = pd.DataFrame()
    inputs['peptide_sequences'] = np.array(peptide_list)

    # Make predictions using prosit_2019_irt (Wilhelm/Kuster lab: https://github.com/kusterlab/prosit) :
    # If you are unsure what inputs your model requires run `model.model_inputs`
    model_prosit = Koina("Prosit_2019_irt", "koina.wilhelmlab.org:443")
    predictions_prosit = model_prosit.predict(inputs, debug=True)

    # Make predictions using AlphaPeptDeep_rt_generic (Mann lab: https://github.com/MannLabs/alphapeptdeep):
    model_prosit24 = model = Koina("Prosit_2024_irt_cit", "koina.wilhelmlab.org:443")
    predictions_prosit24 = model_prosit24.predict(inputs)

    predictions = pd.DataFrame()
    predictions['Peptide'] = predictions_prosit['peptide_sequences']
    predictions['iRT_prosit'] = predictions_prosit['irt']
    predictions['iRT_prosit24'] = predictions_prosit24['irt']

    return predictions


def add_dRTs(raw_data,observed_rts,predicted_rts,qvalue_for_training):
    # create pseudo irts using predicted rts from the best peptides and raw rts:
    raw_data['iRT_prosit24'] = predicted_rts
    raw_data['observed_rts'] = observed_rts
    filtered_v_results = raw_data[
        (raw_data['mhcv_pep-level_q-value'] <= qvalue_for_training) & (raw_data['mhcv_label'] == 1)]
    filtered_v_results_sub = filtered_v_results[filtered_v_results['observed_rts'] > 15]
    # Now create the adjustment function and adjust rts values to adjusted_rts
    x_data = filtered_v_results_sub['observed_rts']
    y_data = filtered_v_results_sub['iRT_prosit24']
    function = adjustment_function(x_data, y_data, 1)
    adjustment_plot(x_data, y_data, function)
    x_data = filtered_v_results['observed_rts']
    y_data = filtered_v_results['iRT_prosit24']
    function_low_rts = adjustment_function(x_data, y_data, 10)
    adjustment_plot(x_data, y_data, function_low_rts)
    # Use the function to adjust all rts in the raw data pin file
    raw_data['adjusted_rts'] = raw_data['observed_rts'].apply(lambda x: function_low_rts(x) if x < 15 else function(x))
    raw_data['iRT_prosit24'] = predicted_rts['iRT_prosit24']
    raw_data['delta_rt'] = abs(raw_data['adjusted_rts'] - raw_data['iRT_prosit24'])
    raw_data['rel_delta_rt'] = (abs(raw_data['adjusted_rts'] - raw_data['iRT_prosit24']))/raw_data['adjusted_rts']
    return raw_data