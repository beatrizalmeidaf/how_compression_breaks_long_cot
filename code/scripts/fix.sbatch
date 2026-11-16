#!/usr/bin/env bash
# Intervencoes anti-loop para um compressor: rollback (margens 64 e half) e repetition_penalty.
#   sbatch --job-name=fix-rtn3 --export=ALL,BASE=Qwen/Qwen3-4B,SPEC=rtn-w3g128,RPS="1.05 1.1" scripts/fix.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job$SLURM_JOB_ID; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
DATASET=${DATASET:-math500}; NPROB=${NPROB:-300}; B=16384
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); SUF=$DATASET-n$NPROB-s4-b$B
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
if [ "$SPEC" = dense ]; then C=$BASE; else C=${P}__$SPEC; $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/fix_make_$C.log" 2>&1 || { log "FALHOU make"; exit 1; }; fi
QTR=$DXS_DATA/traces/$(basename "$C" | tr 'A-Z' 'a-z')__$SUF.parquet
if [ "$SPEC" != dense ]; then for M in 64 half; do
  log "$C: rollback margem $M"
  $PY scripts/rollback.py --traces "$QTR" --model "$C" --margin $M > "logs/fix_rb${M}_$(basename $C)_$DATASET.log" 2>&1 || log "  FALHOU rollback $M"
done; fi
for RP in ${RPS:-}; do
  log "$C: repetition_penalty $RP"
  $PY scripts/gen_traces.py --model "$C" --dataset "$DATASET" --n-problems "$NPROB" --samples 4 --max-tokens $B --rep-penalty $RP \
      > "logs/fix_rp${RP}_$(basename $C)_$DATASET.log" 2>&1 || log "  FALHOU rp $RP"
done
log "concluido $C"
