#! /usr/bin/bash

# replace by your checkpoints paths
CKPT_PATH=$GLINT_ROOT/ckpts/glinter1.pt
ESM_PATH=$GLINT_ROOT/scratch/esm_msa1_t12_100M_UR50S.pt


rfile=$1 # receptor file
lfile=$2 # ligand file
root=$3 # output root
repr=$4 # repr name

receptor=$(basename $rfile .pdb)
ligand=$(basename $lfile .pdb)


name=$receptor:$ligand
srcdir=$root/$name

bash $GLINT_ROOT/scripts/format_input_paths.sh $receptor $ligand $rfile $lfile $srcdir

# fetch pdbseqs
python $GLINT_ROOT/preprocess/pdbseq.py $srcdir/$receptor/$receptor.pdb $srcdir/$receptor/$receptor.seq $srcdir/$receptor/$receptor.pos
python $GLINT_ROOT/preprocess/pdbseq.py $srcdir/$ligand/$ligand.pdb $srcdir/$ligand/$ligand.seq $srcdir/$ligand/$ligand.pos

# align pdbseq to representative
python $GLINT_ROOT/preprocess/align.py $srcdir/$ligand/$ligand.seq $srcdir/$repr/$repr.seq > $srcdir/map.txt
python $GLINT_ROOT/preprocess/align.py $srcdir/$receptor/$receptor.seq $srcdir/$repr/$repr.seq >> $srcdir/map.txt

# build model.txt
cp $srcdir/$repr/$repr.seq $srcdir/ref.seq
echo "$receptor:$ligand  $repr:$repr" > $srcdir/model.txt

# build msa

if [ ! -f $srcdir/$repr/$repr.a3m ]; then
    $GLINT_ROOT/scripts/run_msa.sh $srcdir/$repr
    if [ ! -f $srcdir/$repr/$repr.a3m ]; then
        echo "Cannot build $srcdir/$repr/$repr.a3m "
        exit 1
    fi
fi
bash $GLINT_ROOT/preprocess/MSA/filter_msa.sh $srcdir/$repr
cp $srcdir/$repr/$repr.hh.a3m $srcdir/"$repr.hh.a3m"

# build msms
$GLINT_ROOT/scripts/run_msms.sh $receptor $srcdir
$GLINT_ROOT/scripts/run_msms.sh $ligand $srcdir

# build features for homodimers
$GLINT_ROOT/scripts/build_features.sh $srcdir 2


pkl=$srcdir/$name.pkl

if [ ! -f $pkl ]; then
    echo "ERROR cannot build $pkl"
    exit 1
fi

MSA_CMD=$GLINT_ROOT/glinter/models/msa_model.py
echo "generate esm for $pkl"
python $MSA_CMD --dimer-root $pkl --feature esm --ckpt-path $ESM_PATH --generate-esm-attention
echo "predict $pkl"
python $MSA_CMD --dimer-root $pkl --esm-root $srcdir --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --ckpt-path $CKPT_PATH

python $GLINT_ROOT/scripts/compute_score.py $srcdir $receptor $ligand
