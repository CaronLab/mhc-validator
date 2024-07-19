#!/bin/bash

COMET="./comet.linux.exe"
PARAMS="./immunopeptidomics_example_comet.params"

DIRECTORY="."
FASTA="./*.fasta"
$COMET -P$PARAMS -D$FASTA "$DIRECTORY"/*.mgf

