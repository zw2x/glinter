#! /usr/bin/bash
srcdir=$1

paths=$(find $srcdir -type f -name "*.a3m" -a ! -name "\.*")
for p in $paths; do
    base=${p%.a3m}
    if [ ! -f $base.hhm.pkl ]; then
        echo $base
        $HHBLITS_BIN/hhmake -i $base.a3m -o $base.hhm
        python $GLINT_ROOT/glinter/hhm/LoadHHM.py $base.hhm
    fi
done
