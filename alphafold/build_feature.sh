#! /usr/bin/bash
base=$1
output=$2
chain=$(basename ${base})

if [ ! -f ${base}/${chain}.hhm.pkl ]; then
    hhmake -i ${base}/uniclust_hits.a3m -o ${base}/${chain}.hhm
    python ../glinter/hhm/LoadHHM.py ${base}/${chain}.hhm
fi


input_pdb=${output}/ranked_0_${chain}.pdb
tgtdir=${output}/ranked_0/${chain}

if [ -f "${tgtdir}/${chain}.face" ]; then
    exit
fi

tmpdir=$(mktemp -d /tmp/glinter-XXXXXX)
if [ ! -d ${tmpdir} ]; then
    mkdir -p ${tmpdir}
fi

file_base=${tmpdir}/${chain}
reduced=${tmpdir}/${chain}.reduced.pdb

# reduce 
reduce -Trim ${input_pdb} > ${tmpdir}/tmp.pdb
reduce -HIS ${tmpdir}/tmp.pdb > ${reduced}

# MSMS
xyzrn=$tmpdir/${chain}.xyzrn
python ../glinter/points/xyzrn.py ${reduced} ${xyzrn}

msms -density 3.0 -hdensity 3.0 -probe 1.5 -if ${xyzrn} -of ${file_base} \
    -af ${file_base}

if [ ! -f ${file_base}.area ]; then
    rm -rf ${tmpdir}
    exit 1
fi

# re-compute normal
# python $glint/preprocess/export_ply.py $file_base
 
if [ ! -d ${tgtdir} ]; then
    mkdir -p ${tgtdir}
fi

mv ${reduced} ${tgtdir}
mv ${file_base}.vert ${tgtdir}
mv ${file_base}.face ${tgtdir}
mv ${file_base}.area ${tgtdir}
mv ${xyzrn} ${tgtdir}

rm -rf ${tmpdir}
