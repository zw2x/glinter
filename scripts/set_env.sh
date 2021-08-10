export GLINT_ROOT=$(cd $(dirname $(readlink -f $0)) && cd .. && pwd)
echo $GLINT_ROOT
export REDUCE_PATH=$GLINT_ROOT/external/reduce
export PATH=$PATH:$REDUCE_PATH
export REDUCE_HET_DICT=$REDUCE_PATH/reduce_wwPDB_het_dict.txt
export MSMS_BIN=$GLINT_ROOT/external/msms
export HHBLITS_BIN=$GLINT_ROOT/external/hhblits-bin

# replace with your own hh-suite database
export HHDB=/mnt/data/hhsuite-data/uniclust30_2016_09/uniclust30_2016_09

if [ ! -f ${HHDB}_hhm.ffindex ]; then
    echo "ERROR: invalid or damaged sequence database: $HHDB"
    exit 1
fi
