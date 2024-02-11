# mhc-validator
Validation of peptide-spectrum matches from mass spectrometry-based immunopeptidomics experiments integrating both database search metrics and
MHC interaction/presentation predictors into the discriminant function.

### Installing MHCvalidator
On your Linux machine, create a virtual envirmonment with the python IDE (Integrated Develoment Enviroment) of your choice. In the example PyCharm was used. The you can clone MHcvalidator from the CaronLab github page, install required packages and MHCvalidator as follows:

```python
## in terminal console run
git clone https://github.com/CaronLab/mhc-validator
pip install -r mhc-validator/requirements.txt
pip install ./mhc-validator
```
The above should install MHCvalidator. 
Now in order to fully use MHCvalidators capacity to implement MHC binding affinities you will have to manually install NetMHCpan4.1. You can run MHCvalidator without it by using only MHCflurry (Which was installed from the requirments.txt), but as described in our publication we suggest to also use NetMHCpan4.1 to achieve the full potential of MHCvalidator. 
Do install NetMHCpan4.1 you have to go to 'https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux' and download NetMHCpan4.1 for Linux as described on the website. A general overview of  NetMHCpan4.1 can be found here: 'https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/'. 

After downloading NetMHCpan4.1 install as instructed in the readme file provided in the NetMHCpan4.1 download folder. When installed and tested on your local machine, copy the 'netMHCpan' file to the following folder: ./YOUR_MHCvalidator_FOLDER/venv/bin , for example you can do as follows:

```terminal
cp ./NetMHCpan4.1_directory/netMHCpan ./PATH/TO/YOUR_MHCvalidator_FOLDER/venv/bin
```
Now MHCvalidator can use NetMHCpan4.1 to make predictions and be used to its full potential.

### Tutorial to use MHCvalidator based on an example file

The first step is to set parameters and load data (database search results in PIN format)

```python
from mhcvalidator import MhcValidator

validator = MhcValidator()
validator.set_mhc_params(['A0201', 'B0702'])  # optional if MhcFlurry and NetMHCpan are not being used
validator.load_data('/path/to/pin_file.pin')
```

Next is to run:
```python
# To run in "MV" configuration (fully connected neural 
# network with no extra features added to input), do this:
validator.run()

# To run in "MV+MHC" configuration (fully connected neural network 
# with NetMHCpan and/or MHCFlurry predictions added to the standard PIN features), do this:
validator.run(netmhcpan=True, mhcflurry=True)

# To run in MV+PE configuration (convolutional neural network which encodes peptides sequences
# and then feeds into a fully connected neural network with the standard PIN features), do this:
validator.run(sequence_encoding=True)

# To run in MV+MHC+PE configuration (same as MV+PE, but also with NetMHCpan and/or MhcFlurry predictions added 
# to the standard PIN features), do this:
validator.run(sequence_encoding=True, netmhcpan=True, mhcflurry=True)

# An important argument for the `run` function is "report_directory". Setting this tells MhcValidator to
# save information about the model fitting as well as the predictions results into this directory
# For example:
validator.run(report_directory="/path/to/results/directory")
```
