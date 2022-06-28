EXPERIMENT="$1"
LAMBDA_EQUALITY="1.0"
LAMBDA_SUPRESS="0.5"
SUPRESS_LEVEL="5.1"
SUPRESS_COOLDOWN="1600"
UNIFORM_RATIO="0.2"
EQUALITY_WARMUP="800"
CHK="2000"
NUMITS="1600"
LREG="0.0001"
LR="0.005"
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
NME="ours"

echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct.py \
    -e "$1" \
    -c "$CHK" \
    --name "$NME" \
    --threads "$MENDNETTHREADS" \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --eq_warmup "$EQUALITY_WARMUP" \
    --lambda_eq "$LAMBDA_EQUALITY" \
    --lambda_sup $LAMBDA_SUPRESS \
    --sup_cooldown "$SUPRESS_COOLDOWN" \
    --render_threads "$MENDNETTHREADSRENDER" \
    --uniform_ratio "$UNIFORM_RATIO" \
    --out_of_order \
    --stop 100000 \
    --seed 1 \
    --split "$SPLITNUM"