#!/usr/bin/env bash
# Limiar "pensar protege/amplifica": garante, para o denso e cada compressor, (a) traces pensando 16k,
# (b) teacher forcing sobre os traces do denso (TV/token) e (c) resposta sem raciocinio (swap b=0).
#   sbatch --job-name=xo-14b --export=ALL,BASE=Qwen/Qwen3-14B,SPECS="rtn-w4g128 wanda-0.5" scripts/xover.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job${SLURM_JOB_ID:-direct}; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
DATASET=${DATASET:-math500}; NPROB=${NPROB:-300}; MEM=${MEM:-0.85}
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); SUF=$DATASET-n$NPROB-s4-b16384; DTR=$DXS_DATA/traces/${P}__$SUF.parquet
LOCKS=$DXS_DATA/locks; mkdir -p "$LOCKS"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
nothink() { [ -f "$DXS_DATA/swap/${2}__${SUF}__ans-${2}.parquet" ] || {
  log "$2: sem raciocinio"; $PY scripts/swap_answer.py --traces "$1" --answerer "$3" --budgets 0 --gpu-mem $MEM \
      > "logs/xo_nothink_${2}_$DATASET.log" 2>&1 || log "  FALHOU nothink $2"; }; }

# denso: traces pensando (uma fila so, por trava) e sem raciocinio
if [ ! -f "$DTR" ]; then
  if mkdir "$LOCKS/${P}__${SUF}__dense_traces" 2>/dev/null; then
    log "$P: traces densos"; $PY scripts/gen_traces.py --model "$BASE" --dataset "$DATASET" --n-problems "$NPROB" --samples 4 \
        --max-tokens 16384 --gpu-mem $MEM > "logs/xo_dense_${P}_$DATASET.log" 2>&1 || { log "FALHOU denso"; exit 1; }
  else log "$P: esperando traces densos"; until [ -f "$DTR" ]; do sleep 60; done; fi
fi
nothink "$DTR" "$P" "$BASE"

for SPEC in $SPECS; do
  C=${P}__$SPEC; QTR=$DXS_DATA/traces/${C}__$SUF.parquet; TFF=$DXS_DATA/tf/${P}__${SUF}__vs__${C}__v2.parquet
  if [ -f "$QTR" ] && [ -f "$TFF" ] && [ -f "$DXS_DATA/swap/${C}__${SUF}__ans-${C}.parquet" ]; then log "[skip] $C"; continue; fi
  log "$C: comprimir"
  $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/xo_make_$C.log" 2>&1 || { log "  FALHOU make $C"; continue; }
  [ -f "$QTR" ] || { log "$C: traces pensando"; $PY scripts/gen_traces.py --model "$C" --dataset "$DATASET" --n-problems "$NPROB" \
      --samples 4 --max-tokens 16384 --gpu-mem $MEM > "logs/xo_gen_${C}_$DATASET.log" 2>&1 || log "  FALHOU gen $C"; }
  [ -f "$TFF" ] || { log "$C: teacher forcing"; $PY scripts/teacher_force.py --traces "$DTR" --dense "$BASE" --comp "$C" --tag v2 --chunk ${TFCHUNK:-1024} \
      > "logs/xo_tf_${C}_$DATASET.log" 2>&1 || log "  FALHOU tf $C"; }
  nothink "$QTR" "$C" "$C"
  rm -rf "$DXS_CKPTS/$C"
done
log "concluido $P/$DATASET"
