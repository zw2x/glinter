#! /usr/bin/bash

pdbid=$1 #$(basename $pdb .pdb)
pdbroot=$2
pdb=$pdbroot/$pdbid/$pdbid.pdb

tmpdir="/tmp/$pdbid-tmp"

tgtdir=$pdbroot/$pdbid

if [ -f "$tgtdir/$pdbid.face" ]; then
    exit
fi

if [ ! -d $tmpdir ]; then
    mkdir -p $tmpdir
fi

file_base=$tmpdir/$pdbid
reduced=$tmpdir/$pdbid.reduced.pdb

# reduce 
$REDUCE_PATH/reduce -Trim $pdb > $tmpdir/tmp.pdb
$REDUCE_PATH/reduce -HIS $tmpdir/tmp.pdb > $reduced

# MSMS
xyzrn=$tmpdir/$pdbid.xyzrn
echo $GLINT_ROOT
python $GLINT_ROOT/glinter/points/xyzrn.py $reduced $xyzrn

$MSMS_BIN -density 3.0 -hdensity 3.0 -probe 1.5 -if $xyzrn -of $file_base -af $file_base

if [ ! -f $file_base.area ]; then
    rm -rf $tmpdir
    exit
fi

# re-compute normal
# python $glint/preprocess/export_ply.py $file_base
 
if [ ! -d $tgtdir ]; then
    mkdir -p $tgtdir
fi

mv $reduced $tgtdir
mv $file_base.vert $tgtdir
mv $file_base.face $tgtdir
mv $file_base.area $tgtdir
mv $xyzrn $tgtdir
# rm -rf $tmpdir
