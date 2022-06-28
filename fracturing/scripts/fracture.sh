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
OPERATION="$3"      # eg:
if [ -z "$3" ]; then
    echo "Must pass: OPERATION"; exit
fi
NUMBREAKS="$4"      # eg:
if [ -z "$4" ]; then
    NUMBREAKS="10"
fi
SUBSAMPLE="6000"

# Run 
case $OPERATION in
	"0")
        # Waterproof
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            WATERPROOF \
            --instance_subsample "$SUBSAMPLE" \
            --outoforder \
            -t 6
        ;;
    
	"1")
        # Clean
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            CLEAN \
            --outoforder \
            -t "$MENDNETTHREADSFRACTURE"
        ;;

	"2")
        # Break
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            BREAK \
            --breaks "$NUMBREAKS" \
            --break_all \
            --min_break 0.05 \
            --max_break 0.20 \
            --break_method surface-area \
            --outoforder \
            -t "$MENDNETTHREADSFRACTURE"
        ;;

	"3")
        # Compute the sample points 
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            SAMPLE \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$MENDNETTHREADSFRACTURE"
        ;;

	"4")
        # Compute SDF
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            SDF PARTIAL_SDF \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t 3
        ;;

	"5")
        # Compute Voxels
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            VOXEL_32 \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$MENDNETTHREADSFRACTURE"
        ;;

	"6")
        # Compute Uniform Occupancy
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            UNIFORM_OCC \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t 3
        ;;
    
    "7")
        # Compute SDF
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            PARTIAL_SDF \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t 3
        ;;
esac