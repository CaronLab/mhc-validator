#!/bin/bash

COMET="./comet.linux.exe"
PARAMS="./immunopeptidomics_example_comet.params"

DIRECTORY="."
FASTA="./FASTA_FILE_NAME.fasta"
$COMET -P$PARAMS -D$FASTA "$DIRECTORY"/*.mgf

