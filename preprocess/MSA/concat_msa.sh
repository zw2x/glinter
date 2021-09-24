#! /usr/bin
srcdir=$1
name=$(basename $srcdir)

rec=$(echo $name | cut -d ':' -f1)
lig=$(echo $name | cut -d ':' -f2)

if [ ! -f $srcdir/$name.a3m_cc ]; then
    if [ ! -f $srcdir/$rec/$rec.a3m_sb ]; then
        $GLINT_ROOT/external/A3M_NoGap $srcdir/$rec/$rec.a3m $srcdir/$rec/$rec.a3m_ng
        $GLINT_ROOT/external/A3M_SpecBloc $srcdir/$rec/$rec.a3m_ng $GLINT_ROOT/external/TaxTree $srcdir/$rec/$rec.a3m_sb > /tmp/$rec.sb
        rm /tmp/$rec.sb
    fi

    if [ ! -f $srcdir/$lig/$lig.a3m_sb ]; then
        $GLINT_ROOT/external/A3M_NoGap $srcdir/$lig/$lig.a3m $srcdir/$lig/$lig.a3m_ng
        $GLINT_ROOT/external/A3M_SpecBloc $srcdir/$lig/$lig.a3m_ng $GLINT_ROOT/external/TaxTree $srcdir/$lig/$lig.a3m_sb > /tmp/$lig.sb
        rm /tmp/$lig.sb
    fi

    $GLINT_ROOT/external/MSA_ConCat $srcdir/$rec/$rec.a3m_sb $srcdir/$lig/$lig.a3m_sb $srcdir/$name.a3m_cc
    $GLINT_ROOT/external/meff_cdhit -S 0.65 -c 0 -i $srcdir/$name.a3m_cc > $srcdir/$name.meff
fi
