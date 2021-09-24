#! /usr/bin/bash

srcdir=$(readlink -f $1)
echo $srcdir
mode=$2 # 1-hetero, otherwise homo

echo "build msms feature"

if [ $mode -eq 1 ]; then
    python $GLINT_ROOT/preprocess/msms_builder.py --srcdir $srcdir
else
    python $GLINT_ROOT/preprocess/msms_builder.py --srcdir $srcdir --seqmap $srcdir/map.txt --refseq $srcdir/ref.seq
fi

echo "build monomer tensor"
python $GLINT_ROOT/preprocess/mten_builder.py --hhr $srcdir --msms $srcdir --tgtdir $srcdir
echo "build msa tensor"

if [ $mode -eq 1 ]; then
    python $GLINT_ROOT/preprocess/msa_builder.py --a3mdir $srcdir --tgtdir $srcdir --use-concat --use-hhfilter
else
    python $GLINT_ROOT/preprocess/msa_builder.py --a3mdir $srcdir --tgtdir $srcdir --use-hhfilter
fi

echo "collect tensors"
name=$(basename $srcdir)
if [ $mode -eq 1 ]; then
    python $GLINT_ROOT/preprocess/feat_verifier.py --msa-dir $srcdir --mten-dir $srcdir --repo $srcdir/$name.pkl
else
    python $GLINT_ROOT/preprocess/feat_verifier.py --msa-dir $srcdir --mten-dir $srcdir --repo $srcdir/$name.pkl --model $srcdir/model.txt
fi
