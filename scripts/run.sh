#! /usr/bin/bash

# replace by your checkpoints paths
CKPT_PATH=$GLINT_ROOT/ckpts/esm-ca-atom-sur-1-8-6-6-best.pt
ESM_PATH=/mnt/data/ESM/esm_msa1_t12_100M_UR50S.pt


rfile=$1
lfile=$2
root=$3

receptor=$(basename $rfile .pdb)
ligand=$(basename $lfile .pdb)


name=$receptor:$ligand
srcdir=$root/$name

bash $GLINT_ROOT/scripts/format_input_paths.sh $receptor $ligand $rfile $lfile $srcdir

# build msa
$GLINT_ROOT/scripts/run_msa.sh $srcdir/$receptor

if [ ! -f $srcdir/$receptor/$receptor.a3m ]; then
    echo "Cannot build $srcdir/$receptor/$receptor.a3m "
    exit 1
fi

$GLINT_ROOT/scripts/run_msa.sh $srcdir/$ligand

if [ ! -f $srcdir/$ligand/$ligand.a3m ]; then
    echo "Cannot build $srcdir/$ligand/$ligand.a3m "
    exit 1
fi

$GLINT_ROOT/scripts/concat_msa.sh $srcdir

# build msms
$GLINT_ROOT/scripts/run_msms.sh $receptor $srcdir
$GLINT_ROOT/scripts/run_msms.sh $ligand $srcdir

# build feature
$GLINT_ROOT/scripts/build_features.sh $srcdir

pkl=$srcdir/$name.pkl

if [ ! -f $pkl ]; then
    exit 1
fi

MSA_CMD=$GLINT_ROOT/glinter/models/msa_model.py
python $MSA_CMD --dimer-root $pkl --feature esm --ckpt-path $ESM_PATH --generate-esm-attention
python $MSA_CMD --dimer-root $pkl --feature atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --ckpt-path $CKPT_PATH
