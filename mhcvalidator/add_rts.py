from typing import List, Union
from pyteomics import mzml
from os import PathLike
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from koinapy import Koina
import numpy as np
import pandas as pd
from mhcvalidator.peptides import clean_peptide_sequences



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
    # Create the point plot
    matplotlib.use('TkAgg')
    overlap_df = pd.DataFrame()
    overlap_df['real_rts'] = x_data
    overlap_df['iRTs'] = y_data

    x_column = 'real_rts'
    y_column = 'iRTs'

    # Create the point plot
    #sns.pointplot(data=overlap_df, x=x_column, y=y_column)

    # Generate x values for plotting the fitted function
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = function(x_fit)

    # Plot the original data points
    plt.scatter(x_data, y_data, label='Data Points')

    # Plot the fitted function
    plt.plot(x_fit, y_fit, color='red', label='Fitted Function')

    # Add labels and legend
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()

    # Display the plot
    plt.show()


# def extract_peps(pins):
#     peptides_for_RT = []
#     for pin in pins:
#         df = pd.read_table(pin, usecols=['Label', 'Peptide'], index_col=False)
#         peptides = clean_peptide_sequences(list(df['Peptide']))
#         peptides_for_RT += remove_nonstandard_peps(peptides)
#     unique_peptides_for_RT = list(set(peptides_for_RT))
#     return unique_peptides_for_RT

def predict_rts(peptide_list,rt_prediction_method):

    peptide_list_clean = clean_peptide_sequences(list(peptide_list))
    inputs = pd.DataFrame()
    inputs['peptide_sequences'] = np.array(peptide_list_clean)
    print(inputs)
    # Make predictions using prosit_2019_irt (Wilhelm/Kuster lab: https://github.com/kusterlab/prosit) :
    # If you are unsure what inputs your model requires run `model.model_inputs`
    if rt_prediction_method == 'iRT_prosit24':
        model_prosit24 = model = Koina("Prosit_2024_irt_cit", "koina.wilhelmlab.org:443")
        predictions_prosit = model_prosit24.predict(inputs)
        predictions = pd.DataFrame()
        predictions['Peptide'] = predictions_prosit['peptide_sequences']
        predictions['iRT_prosit'] = predictions_prosit['irt']
    else:
        model_prosit = Koina("Prosit_2019_irt", "koina.wilhelmlab.org:443")
        predictions_prosit = model_prosit.predict(inputs, debug=True)
        predictions = pd.DataFrame()
        predictions['Peptide'] = predictions_prosit['peptide_sequences']
        predictions['iRT_prosit'] = predictions_prosit['irt']

    return predictions


def add_dRTs(raw_data,observed_rts,predicted_rts,qvalue_for_training,visualize_rts):
    # create pseudo irts using predicted rts from the best peptides and raw rts:
    raw_data['iRT_prosit'] = predicted_rts
    raw_data['observed_rts'] = observed_rts
#Filter data
    filtered_v_results = raw_data[
        (raw_data['mhcv_pep-level_q-value'] <= qvalue_for_training) & (raw_data['mhcv_label'] == 1)]
    x_data = filtered_v_results['observed_rts']
    y_data = filtered_v_results['iRT_prosit']
    function = adjustment_function(x_data, y_data, 1)
    min_observed_rts = filtered_v_results['observed_rts'].min()
    max_observed_rts = filtered_v_results['observed_rts'].max()

 # Use the function to adjust all rts in the raw data pin file
    raw_data['adjusted_rts'] = raw_data['observed_rts'].apply(
        lambda x: function(x) if min_observed_rts <= x <= max_observed_rts else x)
    raw_data['delta_rt'] = abs(raw_data['adjusted_rts'] - raw_data['iRT_prosit'])
    raw_data['rel_delta_rt'] = abs((raw_data['adjusted_rts'] - raw_data['iRT_prosit'])/raw_data['adjusted_rts'])
    adjustment_plot(x_data, y_data, function)
    if visualize_rts:
        plot_rts(raw_data)
    return raw_data




def plot_rts(raw_data):
    matplotlib.use('TkAgg')
    filtered_v_raw = raw_data[raw_data['mhcv_q-value'] < 0.01]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(raw_data['adjusted_rts'], raw_data['iRT_prosit'], label='All Data', alpha=0.5)
    plt.scatter(filtered_v_raw['adjusted_rts'], filtered_v_raw['iRT_prosit'], color='red',
                label='FDR < 0.01', alpha=0.5)

    # Add a line representing a slope of 1
    max_val = min(raw_data['adjusted_rts'].max(), raw_data['iRT_prosit'].max())
    plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', label='Slope 1')

    # Add labels and title
    plt.xlabel('Observed RTs (Adjusted)')
    plt.ylabel('predicted iRTs')
    plt.title('Observed RTs vs predicted RTs')
    plt.legend()
    # Display the plot
    plt.show()