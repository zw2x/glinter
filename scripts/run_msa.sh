#! /usr/bin/bash

src=$1
$GLINT_ROOT/preprocess/MSA/run_hhblits.sh $src
$GLINT_ROOT/preprocess/MSA/msa_to_hhm.sh $src
