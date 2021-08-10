#! /usr/bin/bash

srcdir=$(readlink -f $1)
echo $srcdir

echo "build msms feature"
python $GLINT_ROOT/preprocess/msms_builder.py --srcdir $srcdir
echo "build monomer tensor"
python $GLINT_ROOT/preprocess/mten_builder.py --hhr $srcdir --msms $srcdir --tgtdir $srcdir
echo "build msa tensor"
python $GLINT_ROOT/preprocess/msa_builder.py --a3mdir $srcdir --tgtdir $srcdir --use-concat --use-hhfilter
echo "collect tensors"
name=$(basename $srcdir)
python $GLINT_ROOT/preprocess/feat_verifier.py --msa-dir $srcdir --mten-dir $srcdir --repo $srcdir/$name.pkl
