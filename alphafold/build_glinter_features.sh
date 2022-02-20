#! /usr/bin/bash

rootdir=$1
srcdir=${rootdir}/ranked_0
msadir=${rootdir}/msas

python ../preprocess/msms_builder.py --srcdir ${srcdir}

python ../preprocess/mten_builder.py --hhr ${msadir} --msms $srcdir \
    --tgtdir $srcdir

python get_concat_a3m.py ${rootdir}/features.pkl > ${srcdir}/ranked_0.hh.a3m

python ../preprocess/msa_builder.py --a3mdir $srcdir --tgtdir $srcdir \
    --use-concat --use-hhfilter --no-check

mv ${srcdir}/ranked_0.hh.msa ${srcdir}/A:B.msa

python ../preprocess/feat_verifier.py --msa-dir $srcdir --mten-dir $srcdir \
    --repo $srcdir/ranked_0.pkl --from-path
