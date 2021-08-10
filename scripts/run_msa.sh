#! /usr/bin/bash

src=$1
name=$(basename $src)
python $GLINT_ROOT/preprocess/pdbseq.py $src/$name.pdb $src/$name.seq
$GLINT_ROOT/preprocess/MSA/run_hhblits.sh $src
$GLINT_ROOT/preprocess/MSA/msa_to_hhm.sh $src
