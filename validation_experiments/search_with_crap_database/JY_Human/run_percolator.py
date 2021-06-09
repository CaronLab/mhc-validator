#!/bin/python3

from pathlib import Path
from os import system

pin_files = Path('.').glob('*.pin')

for pin in pin_files:
    target_out = Path(pin).stem + '_target.pout'
    decoy_out = Path(pin).stem + '_decoy.pout'
    system(f'percolator -U -m {target_out} -M {decoy_out} {pin}')
