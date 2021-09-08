import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mhcvalidator.fdr import calculate_qs
from mhcvalidator.predictors import MhcValidator
from matplotlib_venn import venn2, venn2_unweighted, venn3, venn3_unweighted
from mhcvalidator.peptides import clean_peptide_sequences
import pandas as pd
with open(
        '/Data/Development/mhcvalidator/validation_experiments/search_with_crap_database/JY_Human/JY_301120_S3_target.pout',
        'r') as f:
    header = f.readline().strip().split()
pout = pd.read_table(
    '/Data/Development/mhcvalidator/validation_experiments/search_with_crap_database/JY_Human/JY_301120_S3_target.pout',
    usecols=header, index_col=False)
decoy_pout = pd.read_table(
    '/Data/Development/mhcvalidator/validation_experiments/search_with_crap_database/JY_Human/JY_301120_S3_decoy.pout',
    usecols=header, index_col=False)

qs = v.qs[v.y == 1]  # get q-values of targets
roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
plt.plot(qs, roc, ls='none', marker='.', ms=1, label='mhcvalidator')
qs = pout['q-value'].to_numpy(dtype=float)
roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
plt.plot(qs, roc, ls='none', marker='.', ms=1, label='Percolator')
plt.legend()
plt.xlabel('q-value')
plt.ylabel('Number of PSMs')
plt.title('ROC')
plt.xlim((0, 1.1))
plt.tight_layout()
plt.show()

# plot percolator PEP distribution
plt.hist(x=pout['posterior_error_prob'], label='Targets', alpha=0.6, bins=30)
plt.hist(x=decoy_pout['posterior_error_prob'], label='Decoys', alpha=0.6, bins=30)
plt.legend()
plt.yscale('log')
plt.title('Percolator PEP')
plt.show()

# I will recalculate the Percolator q-values
perc_peps = np.asarray(list(pout['posterior_error_prob']) + list(decoy_pout['posterior_error_prob']))
perc_labels = np.asarray([1]*len(pout) + [0]*len(decoy_pout))
perc_qs = calculate_qs(perc_peps, perc_labels, higher_better=False)[perc_labels==1]

# and plot then alongside mhcvalidator
qs = v.qs[v.y == 1]  # get q-values of targets
roc = np.sum(qs <= qs[:, np.newaxis], axis=1)
plt.plot(qs, roc, ls='none', marker='.', ms=1, label='mhcvalidator')

roc = np.sum(perc_qs <= perc_qs[:, np.newaxis], axis=1)
plt.plot(perc_qs, roc, ls='none', marker='.', ms=1, label='Percolator')

plt.xlabel('q-value')
plt.ylabel('Number of PSMs')
plt.title('mhcvalidator q-values vs recalculated Percolator q-values')
plt.xlim((0, 1))
plt.legend()
plt.show()

# run on all Purcell PIN files
pin_files = list(Path('/media/labcaron/Elements/validator_validation/data/PXD023044-purcell-triple_negative_bc/IFN-positive_class_I').glob('*.pin'))
validator_peps_w_encoding_wo_mhcflurry = []
validator_peps_wo_encoding_wo_mhcflurry = []
validator_peps_w_encoding_w_mhcflurry = []
validator_peps_wo_encoding_w_mhcflurry = []
validator_peps_w_encoding_w_mhcflurry_w_netmhcpan = []
validator_peps_wo_encoding_w_mhcflurry_w_netmhcpan = []
percolator_peps = []
v = MhcValidator()
v.set_mhc_params(['A0201', 'A0217', 'B4002', 'B4101', 'C0202', 'C1701'], 'I')
for pin in pin_files:
    print(f'PIN file: {pin.stem}')

    # open the percolator POUT files
    target_pout = pd.read_table(f'{pin}_targets.pout',
        usecols=header, index_col=False)
    perc_qs = target_pout['q-value'].to_numpy(dtype=float)
    perc_roc = np.sum(perc_qs <= perc_qs[:, np.newaxis], axis=1)
    percolator_peps += list(clean_peptide_sequences(list(target_pout.loc[perc_qs <= 0.01, 'peptide'])))

    v.load_data(pin, filetype='pin')
    v.run(encode_peptide_sequences=True, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/with_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_w_encoding_wo_mhcflurry += list(peps)
    validator_w_encoding_wo_mhcflurry_qs = v.qs[v.y == 1]
    validator_w_encoding_wo_mhcflurry_roc = np.sum(
        validator_w_encoding_wo_mhcflurry_qs <= validator_w_encoding_wo_mhcflurry_qs[:, np.newaxis], axis=1)


    v.run(encode_peptide_sequences=False, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/without_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_wo_encoding_wo_mhcflurry += list(peps)
    validator_wo_encoding_wo_mhcflurry_qs = v.qs[v.y == 1]
    validator_wo_encoding_wo_mhcflurry_roc = np.sum(
        validator_wo_encoding_wo_mhcflurry_qs <= validator_wo_encoding_wo_mhcflurry_qs[:, np.newaxis], axis=1)


    v.add_mhcflurry_predictions()
    v.run(encode_peptide_sequences=True, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/with_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_w_encoding_w_mhcflurry += list(peps)
    validator_w_encoding_w_mhcflurry_qs = v.qs[v.y == 1]
    validator_w_encoding_w_mhcflurry_roc = np.sum(
        validator_w_encoding_w_mhcflurry_qs <= validator_w_encoding_w_mhcflurry_qs[:, np.newaxis], axis=1)

    v.run(encode_peptide_sequences=False, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/without_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_wo_encoding_w_mhcflurry += list(peps)
    validator_wo_encoding_w_mhcflurry_qs = v.qs[v.y == 1]
    validator_wo_encoding_w_mhcflurry_roc = np.sum(
        validator_wo_encoding_w_mhcflurry_qs <= validator_wo_encoding_w_mhcflurry_qs[:, np.newaxis], axis=1)


    v.add_netmhcpan_predictions()
    v.run(encode_peptide_sequences=True, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/with_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_w_encoding_w_mhcflurry_w_netmhcpan += list(peps)
    validator_w_encoding_w_mhcflurry_qs_w_netmhcpan = v.qs[v.y == 1]
    validator_w_encoding_w_mhcflurry_roc_w_netmhcpan = np.sum(
        validator_w_encoding_w_mhcflurry_qs_w_netmhcpan <= validator_w_encoding_w_mhcflurry_qs_w_netmhcpan[:, np.newaxis], axis=1)

    v.run(encode_peptide_sequences=False, epochs=15, visualize=False,
          report_dir=f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/{pin.stem}/without_sequence_encoding')
    peps = v.peptides[(v.qs <= 0.01) & (v.labels == 1)]
    validator_peps_wo_encoding_w_mhcflurry_w_netmhcpan += list(peps)
    validator_wo_encoding_w_mhcflurry_qs_w_netmhcpan = v.qs[v.y == 1]
    validator_wo_encoding_w_mhcflurry_roc_w_netmhcpan = np.sum(
        validator_wo_encoding_w_mhcflurry_qs_w_netmhcpan <= validator_wo_encoding_w_mhcflurry_qs_w_netmhcpan[:, np.newaxis], axis=1)

    plt.plot(perc_qs, perc_roc, ls='none', marker='.', ms=1.5, label="Percolator")
    plt.plot(validator_w_encoding_wo_mhcflurry_qs, validator_w_encoding_wo_mhcflurry_roc,
             ls='none', marker='.', ms=1.5, label="mhcvalidator (Encoding)")
    plt.plot(validator_wo_encoding_wo_mhcflurry_qs, validator_wo_encoding_wo_mhcflurry_roc,
             ls='none', marker='.', ms=1.5, label="mhcvalidator")
    plt.plot(validator_w_encoding_w_mhcflurry_qs, validator_w_encoding_w_mhcflurry_roc,
             ls='none', marker='.', ms=1.5, label="mhcvalidator (Encoding, MhcFlurry)")
    plt.plot(validator_wo_encoding_w_mhcflurry_qs, validator_wo_encoding_w_mhcflurry_roc,
             ls='none', marker='.', ms=1.5, label="mhcvalidator (MhcFlurry)")
    plt.plot(validator_w_encoding_w_mhcflurry_qs_w_netmhcpan, validator_w_encoding_w_mhcflurry_roc_w_netmhcpan,
             ls='none', marker='.', ms=1.5, label="mhcvalidator (Encoding, MhcFlurry, NetMHCpan)")
    plt.plot(validator_wo_encoding_w_mhcflurry_qs_w_netmhcpan, validator_wo_encoding_w_mhcflurry_roc_w_netmhcpan,
             ls='none', marker='.', ms=1.5, label="mhcvalidator (MhcFlurry, NetMHCpan)")
    plt.xlim((0, 0.05))
    plt.legend()
    plt.xlabel('FDR')
    plt.ylabel('Number of PSMs')
    plt.title(pin.stem)
    plt.tight_layout()
    plt.savefig(f'/media/labcaron/Elements/validator_validation/results/validator_runs/purcell/IFN_pos_class_I_pins/ROCs/{pin.stem}_roc.svg')
    plt.show()

with open('/media/labcaron/Elements/validator_validation/data/PXD023044-purcell-triple_negative_bc/pos_class_I_peps.txt', 'r') as f:
    purcell_peps = [x.strip() for x in f.readlines()]

venn3_unweighted([set(purcell_peps), set(validator_peps_w_encoding_wo_mhcflurry), set(validator_peps_wo_encoding_wo_mhcflurry)],
      ['Purcell master list', 'mhcvalidator with\nencoding', 'mhcvalidator without\nencoding'])
plt.title('mhcvalidator without MhcFlurry predictions')
plt.show()

venn3_unweighted([set(purcell_peps), set(validator_peps_w_encoding_w_mhcflurry), set(validator_peps_wo_encoding_w_mhcflurry)],
      ['Purcell master list', 'mhcvalidator with\nencoding', 'mhcvalidator without\nencoding'])
plt.title('mhcvalidator with MhcFlurry predictions')
plt.show()

plt.bar(['Purcell', 'Percolator', 'V',  'V w/ encoding', 'V w/ predictors', 'V w/ all'], [len(purcell_peps), len(set(percolator_peps)), len(set(validator_peps_wo_encoding_wo_mhcflurry)), len(set(validator_peps_w_encoding_wo_mhcflurry)), len(set(validator_peps_wo_encoding_w_mhcflurry_w_netmhcpan)), len(set(validator_peps_w_encoding_w_mhcflurry_w_netmhcpan))])
plt.xticks(rotation=45)
plt.ylabel('Number of unique peptides')
plt.tight_layout()
plt.show()
