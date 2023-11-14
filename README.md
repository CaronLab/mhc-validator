# mhc-validator
Validation of peptide-spectrum matches from mass spectrometry-based immunopeptidomics experiments integrating both database search metrics and
MHC interaction/presentation predictors into the discriminant function.

This projet is in development. More information will be added soon.

### Very very basic tutorial

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
```