# !/bin/bash
# Fracture the ShapeNet dataset
#

# Set the input arguments
ROOTDIR="$1"         # eg:
if [ -z "$1" ]; then
    echo "Must pass: ROOTDIR"; exit
fi
SPLITSFILE="$2"     # eg: 
if [ -z "$2" ]; then
    echo "Must pass: SPLITSFILE"; exit
fi
OUTFILE="$3"        # eg:
if [ -z "$3" ]; then
    echo "Must pass: OUTFILE"; exit
fi

# Number of breaks
NUMBREAKS="10"

# Build the pkl files
python python/build.py \
    "$ROOTDIR" \
    "$SPLITSFILE" \
    --test_out "$OUTFILE"_test.pkl \
    --breaks "$NUMBREAKS" \
    --load_models
# --train_out "$OUTFILE"_train.pkl \