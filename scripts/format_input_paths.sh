#! /usr/bin/bash

receptor=$1
ligand=$2
rfile=$3
lfile=$4
root=$5

rdir=$root/$receptor
ldir=$root/$ligand

if [ ! -d $rdir ]; then
    mkdir -p $rdir
fi

if [ ! -d $ldir ]; then
    mkdir -p $ldir
fi

cp $rfile $rdir/$receptor.pdb
cp $lfile $ldir/$ligand.pdb
