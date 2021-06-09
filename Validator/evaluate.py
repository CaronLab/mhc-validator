from Validator.predictors import Validator
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3

figdir = '/Data/Development/Validator/Validator/evaluations/different_features/120706_AR_CoOpNO_JY_Normal_W_10%_Rep#1_msms14-comet'
reports = {}
stats = {}
peps = {}
# load data
v = Validator()
v.set_mhc_params(['A0201', 'B0702', ' C0702'], 'I')
v.load_data('/Data/Development/SysteMHC_Test/SYSMHC00001/Kowalewskid_160207_Rammensee_Germany_JY/120706_AR_CoOpNO_JY_Normal_W_10%_Rep#1_msms14-comet.pep.xml',
    filetype='pepxml')

# train with no MHC features
summary = v.train_validation_model(report_dir=figdir + '/no_mhc_predictions-no_seq_encoding')
peps['no_mhc_predictions-no_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['no_mhc_predictions-no_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['no_mhc_predictions-no_seq_encoding'] = summary

summary = v.train_validation_model(encode_peptide_sequences=True,
                                       report_dir=figdir + '/no_mhc_predictions-yes_seq_encoding')
peps['no_mhc_predictions-yes_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['no_mhc_predictions-yes_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['no_mhc_predictions-yes_seq_encoding'] = summary


# MhcFlurry only
v.add_mhcflurry_predictions()
summary = v.train_validation_model(report_dir=figdir + '/mhcflurry-no_seq_encoding')
peps['mhcflurry-no_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['mhcflurry-no_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['mhcflurry-no_seq_encoding'] = summary

summary = v.train_validation_model(encode_peptide_sequences=True,
                                       report_dir=figdir + '/mhcflurry-yes_seq_encoding')
peps['mhcflurry-yes_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['mhcflurry-yes_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['mhcflurry-yes_seq_encoding'] = summary


# NetMHCpan only
v = Validator()
v.set_mhc_params(['A0201', 'B0702', ' C0702'], 'I')
v.load_data('/Data/Development/SysteMHC_Test/SYSMHC00001/Kowalewskid_160207_Rammensee_Germany_JY/120706_AR_CoOpNO_JY_Normal_W_10%_Rep#1_msms14-comet.pep.xml',
    filetype='pepxml')
v.add_netmhcpan_predictions()
summary = v.train_validation_model(report_dir=figdir + '/netmhcpan-no_seq_encoding')
peps['netmhcpan-no_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['netmhcpan-no_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['netmhcpan-no_seq_encoding'] = summary

summary = v.train_validation_model(encode_peptide_sequences=True,
                                       report_dir=figdir + '/netmhcpan-yes_seq_encoding')
peps['netmhcpan-yes_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['netmhcpan-yes_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['netmhcpan-yes_seq_encoding'] = summary

# add MhcFlurry
v.add_mhcflurry_predictions()
summary = v.train_validation_model(report_dir=figdir + '/mhcflurry+netmhcpan-no_seq_encoding')
peps['mhcflurry+netmhcpan-no_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['mhcflurry+netmhcpan-no_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['mhcflurry+netmhcpan-no_seq_encoding'] = summary

summary = v.train_validation_model(encode_peptide_sequences=True,
                                       report_dir=figdir + '/mhcflurry+nemhcpan-yes_seq_encoding')
peps['mhcflurry+nemhcpan-yes_seq_encoding'] = set(np.asarray(v.peptides)[(v.qs <= 0.01) & (v.labels == 1)])
stats['mhcflurry+nemhcpan-yes_seq_encoding'] = int(summary.split('\n')[2].split()[-1])
reports['mhcflurry+nemhcpan-yes_seq_encoding'] = summary

with open(Path(figdir, 'training_summaries.txt'), 'w') as f:
    for key, value in reports.items():
        f.write('##### ' +str(key) + ' #####\n')
        f.write(str(value) + '\n\n')

x = 0
for key, value in stats.items():
    plt.bar(x, value, label=key)
    x += 1
plt.legend()
plt.ylabel('Unique peptide sequences at 1% FDR')
plt.tight_layout()
plt.savefig(str(Path(figdir, 'unique_peptides.svg')))
plt.show()

from matplotlib_venn import venn3_unweighted
to_plot = ['netmhcpan-no_seq_encoding', 'mhcflurry-no_seq_encoding', 'no_mhc_predictions-yes_seq_encoding']
venn3_unweighted([peps[x] for x in to_plot], to_plot)
plt.show()
