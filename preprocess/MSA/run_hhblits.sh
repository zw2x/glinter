#! /usr/bin/bash

base=$GLINT_ROOT/external

meff_cdhit=$base/meff_cdhit
verify_fasta=$base/verify_fasta

HHBLITS=$HHBLITS_BIN/hhblits

if [ ! -f $meff_cdhit ]; then
    echo "ERROR: invalid utility $meff_cdhit"
    exit 1
fi

MSADir=$1
relnam=$(basename $MSADir)
input_fasta=$MSADir/$relnam.seq

evalue=0.001
cpu_num=2

tgtdir=$(readlink -f $MSADir)

tmpWorkDir=`mktemp -d /tmp/${relnam}.tmpWork4HHBlits.XXXXXX`
cp $input_fasta $tmpWorkDir/$relnam.fasta
cd $tmpWorkDir

if [ ! -f "$tgtdir/$relnam.a3m" ]; then
    echo "Building MSAs for $relnam from $DB by hhsuite..."
    
    # verifty fasta
    seqfile=$relnam.seq
    $verify_fasta $relnam.fasta $seqfile
    if [ $? -ne 0 ]; then
        echo "ERROR: failed to run verify $relnam.fasta"
        exit 1
    fi
    
    # calculate minimum alignment coverage
    ## the default minimum coverage percentage is set to 60%
    a=60
    ## b is the minimum percentage of coverage such that at least 80 residues of the query sequence are covered by homologs
    ## $3 in the below sentence is the query sequence length
    b=`tail -n1 $seqfile | wc | awk '{print int(7000/($3-1))}'`
    if [ $a -gt $b ]; then
        coverage=$b
    else
        coverage=$a
    fi

    # run hhblits for 3 iterations
    iteration=3
    a3mfile=$relnam.a3m
    hhrfile=$relnam.hhr
    hhmfile=$relnam.hhm
    echo "Running HHblits with -maxfilt 500000 -diff inf -id 99 -cov $coverage..."
    $HHBLITS -i $seqfile -cpu $cpu_num -d $HHDB -o $hhrfile -ohhm $hhmfile -oa3m $a3mfile -n $iteration -e $evalue -maxfilt 500000 -diff inf -id 99 -cov $coverage
    ret=$?

    if [ $ret -ne 0 ]; then
        echo "ERROR: failed to hhblits $seqfile"
        rm -rf $tmpWorkDir
        exit 1
    fi

    wait

    # calculate meff
    mefffile=$relnam.meff
    numLines=`wc -l $a3mfile | cut -f1 -d' ' `
    if [ $numLines -ge 200000 ]; then
        echo 11 > $mefffile
    else
        $meff_cdhit -i $a3mfile > $mefffile
    fi

    wait
    mv $hhmfile $mefffile $a3mfile $seqfile $tgtdir/
fi

rm -rf $tmpWorkDir
