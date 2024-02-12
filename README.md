# mhc-validator
Validation of peptide-spectrum matches from mass spectrometry-based immunopeptidomics experiments integrating both database search metrics and
MHC interaction/presentation predictors into the discriminant function.

### Installing MHCvalidator
On your Linux machine, create a virtual envirmonment with the python IDE (Integrated Develoment Enviroment) of your choice. In the example PyCharm was used. Then you can clone MHcvalidator from the CaronLab github page, install required packages and MHCvalidator as follows:

```python
## in terminal console run
git clone https://github.com/CaronLab/mhc-validator
pip install -r mhc-validator/requirements.txt
pip install ./mhc-validator
```
The above should install MHCvalidator and the installation should not take more than 10minutes, depending on your interent connection it could tak elonger to pull the folder. 
Now in order to fully use MHCvalidators capacity to implement MHC binding affinities you will have to manually install NetMHCpan4.1 on your machine. You can run MHCvalidator without it by using only MHCflurry (Which was installed from the requirements.txt), but as described in our publication we suggest to also use NetMHCpan4.1 to achieve the full potential of MHCvalidator. 
To install NetMHCpan4.1, you have to go to 'https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux' and download NetMHCpan4.1 for Linux (Version 4.1b) as described on the website. A general overview of  NetMHCpan4.1 can be found here: 'https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/'. 

After downloading NetMHCpan4.1, install the software as instructed in the readme file provided in the NetMHCpan4.1 download folder. When installed and tested on your local machine, copy the 'netMHCpan' file to the following folder: ./YOUR_MHCvalidator_FOLDER/venv/bin , for example you can do as follows:

```terminal
cp ./NetMHCpan4.1_directory/netMHCpan ./PATH/TO/YOUR_MHCvalidator_FOLDER/venv/bin
```
Now MHCvalidator can use NetMHCpan4.1 to make predictions and be used to its full potential.

### Tutorial to use MHCvalidator based on an example experiment

The first step is to create a new python file (example: mhcvalidator_test.py) in you venv, then set parameters and load data (database search results in PIN format). In this example we use the data provided in the github.com/CaronLab/mhc-validator master branch that you pulled earlier. In other words you have the data already downloaded and everything shoul dbe ready to go!

Note: The example data are from the JY serial dilution experiment (Dilution point 3, 4 fold dilution, Figure 3D) described in our publication ('Integrating Machine Learning-Enhanced Immunopeptidomics and SARS-CoV-2 Population-Scale Analyses Unveils Novel Antigenic Features for Next-Generation COVID-19 Vaccines')


Below is the tutorial explaining how to run MHCvalidator in python 3.10 using the example .pin data that you automatically downloaded with the MHCvalidator package:

1. Access example data and import MHCvalidator:
```python
# import requirements
from mhcvalidator.validator import MhcValidator
from pathlib import Path
import os

# Query the sample data from the mhc-validator folder that you pulled from GitHub which contains the data:
sample_folder = Path(os.getcwd()+f'/mhc-validator')
pins = [p for p in sample_folder.glob('*.pin') ]
```
2. Set alleles:
```python
# Set alleles that are applicable to your experiments/data, in our case the following three are applicable:
alleles = ['HLA-A0201', 'HLA-B0702', 'HLA-C0702']
```

3. Run MHCvalidator for each pin file seperately, this might take up to 5minutes for each pin file:
```python
for pin in pins:
    validator = MhcValidator() # Open a MHCvalidator instance, a new one has to be opened for each .pin file
    validator.load_data(pin) # Load the pin file
    validator.set_mhc_params(alleles=alleles) # Load the alleles you specified above
    validator.run(sequence_encoding=True, netmhcpan=True, mhcflurry=True, report_directory=sample_folder / f'{pin.stem}_MhcValidator') #Run MHCvalidator, note that we added all available predictions by setting all configurations to 'True'. You can change these configurations as detailed below if for some reason you want to.
```
If you get a warning that your GPU is not connected (CUUDA warning) from MHCflurry, you can simply ignore that wanring since the gain in analysis speed is minimal.

4. Now you should find the results in the mhc-validator folder that you pulled from GitHub. Here you find two seperate folders named after the .pin files you analyzed. The peptide results table that you will be mostly interested in (.tsv format) is named 'PIN_FILE_NAME.MhcValidator_annotated.tsv'. These peptide results can now be used to study your samples as you normally would. You can also find a .pdf file depicting the training report.

5. You can compare your results to our results, our results folders for each pin file are annotated with the suffix '_default_analysis'.

### Additional information

Note that in step 3. , we used the default configuration that implies MHCflurry and NetMHCpan4.1 predictions. MHCvalidator is built to be used with a multitude of configurations which are described below:

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
