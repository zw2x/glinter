#! /usr/bin/bash

# replace by your checkpoints paths
CKPT_PATH=$GLINT_ROOT/ckpts/glinter1.pt
ESM_PATH=$GLINT_ROOT/scratch/esm_msa1_t12_100M_UR50S.pt
ConcatMSA="$GLINT_ROOT/scripts/concat_msa.sh"

rfile=$1
lfile=$2
root=$3 # output root
receptor=$(basename $rfile .pdb)
ligand=$(basename $lfile .pdb)

if [ $# -eq 4 ]; then
    ConcatMSA=$4
fi

if [ ! -x $ConcatMSA ]; then
    echo "[ERROR] Cannot execute $ConcatMSA"
    exit -1
fi

name=$receptor:$ligand
srcdir=$root/$name

bash $GLINT_ROOT/scripts/format_input_paths.sh $receptor $ligand $rfile $lfile $srcdir

# fetch pdbseqs
python $GLINT_ROOT/preprocess/pdbseq.py $srcdir/$receptor/$receptor.pdb $srcdir/$receptor/$receptor.seq $srcdir/$receptor/$receptor.pos
python $GLINT_ROOT/preprocess/pdbseq.py $srcdir/$ligand/$ligand.pdb $srcdir/$ligand/$ligand.seq $srcdir/$ligand/$ligand.pos

# build msa
if [ ! -f $srcdir/$receptor/$receptor.a3m ]; then
    $GLINT_ROOT/scripts/run_msa.sh $srcdir/$receptor
    if [ ! -f $srcdir/$receptor/$receptor.a3m ]; then
        echo "Cannot build $srcdir/$receptor/$receptor.a3m "
        exit 1
    fi
fi

if [ ! -f $srcdir/$ligand/$ligand.a3m ]; then
    $GLINT_ROOT/scripts/run_msa.sh $srcdir/$ligand
    if [ ! -f $srcdir/$ligand/$ligand.a3m ]; then
        echo "Cannot build $srcdir/$ligand/$ligand.a3m "
        exit 1
    fi
fi

# concat msa
$ConcatMSA $srcdir

# build msms
$GLINT_ROOT/scripts/run_msms.sh $receptor $srcdir
$GLINT_ROOT/scripts/run_msms.sh $ligand $srcdir

# build feature
$GLINT_ROOT/scripts/build_features.sh $srcdir 1

pkl=$srcdir/$name.pkl

if [ ! -f $pkl ]; then
    exit 1
fi

MSA_CMD=$GLINT_ROOT/glinter/models/msa_model.py
echo "generate esm for $pkl"
python $MSA_CMD --dimer-root $pkl --feature esm --ckpt-path $ESM_PATH --generate-esm-attention
echo "predict $pkl"
python $MSA_CMD --dimer-root $pkl --esm-root $srcdir --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --ckpt-path $CKPT_PATH

python $GLINT_ROOT/scripts/compute_score.py $srcdir $receptor $ligand
