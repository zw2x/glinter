srcdir=$1
name=$(basename $srcdir)

if [ -f $srcdir/$name.hh.a3m ]; then
    exit 0
fi
if [ -f $srcdir/$name.a3m_cc ]; then
    $HHBLITS_BIN/hhfilter -diff 200 -cov 20 -i $srcdir/$name.a3m_cc -o $srcdir/$name.hh.a3m
fi
if [ -f $srcdir/$name.a3m ]; then
    $HHBLITS_BIN/hhfilter -diff 200 -cov 20 -i $srcdir/$name.a3m -o $srcdir/$name.hh.a3m
fi
