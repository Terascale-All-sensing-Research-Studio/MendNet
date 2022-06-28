
CHK="2000"
SPLITNUM="$2"
if [ -z "$SPLITNUM" ]; then
    SPLITNUM="1"
fi
if [ -z "$MENDNETTHREADS" ]; then
    MENDNETTHREADS="5"
fi
if [ -z "$MENDNETTHREADSRENDER" ]; then
    MENDNETTHREADSRENDER="5"
fi
# NME="deepsdf_Z"
NME="full_slippage"

echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct_deepsdf.py \
	-e "$1" \
	-c "$CHK" \
	--name "$NME" \
	--threads "$MENDNETTHREADS" \
	--render_threads "$MENDNETTHREADSRENDER" \
	--slippage 1.0 \
	--slippage_method fake-classifier \
	--save_values \
	--out_of_order \
	--seed 1 \
    --split "$SPLITNUM"