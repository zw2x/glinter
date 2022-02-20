#! /usr/bin/bash

srcdir=$1
ESM_PATH=$2

GLINT_ROOT=..

CKPT_PATH=$GLINT_ROOT/ckpts/glinter1.pt

pkl=$srcdir/ranked_0.pkl

if [ ! -f $pkl ]; then
    exit 1
fi

MSA_CMD=$GLINT_ROOT/glinter/models/msa_model.py
echo "generate esm for $pkl"
python $MSA_CMD --dimer-root $pkl --feature esm --ckpt-path $ESM_PATH --generate-esm-attention

echo "predict $pkl"
python $MSA_CMD --dimer-root $pkl --esm-root $srcdir --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --ckpt-path $CKPT_PATH

python $GLINT_ROOT/scripts/compute_score.py $srcdir A B
