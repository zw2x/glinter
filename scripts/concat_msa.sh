#! /usr/bin/bash

srcdir=$1
bash $GLINT_ROOT/preprocess/MSA/concat_msa.sh $srcdir
bash $GLINT_ROOT/preprocess/MSA/filter_msa.sh $srcdir
