#!/usr/bin/env bash
# Hipotese da temperatura. SPEC=dense|<compressor>; TEMPS="0.3 1.0" gera traces nessas temperaturas;
# SHIFT="spec1 spec2 ..." mede a temperatura efetiva (teacher forcing) desses compressores.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job$SLURM_JOB_ID; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
DATASET=${DATASET:-math500}; NPROB=${NPROB:-300}
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); SUF=$DATASET-n$NPROB-s4-b16384; DTR=$DXS_DATA/traces/${P}__$SUF.parquet
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for S in ${SHIFT:-}; do
  C=${P}__$S; log "$C: temperatura efetiva"
  $PY scripts/make_compressed.py --base "$BASE" "$S" > "logs/temp_make_$C.log" 2>&1 || { log "  FALHOU make"; continue; }
  $PY scripts/temp_shift.py --traces "$DTR" --dense "$BASE" --comp "$C" >> "logs/temp_shift_$P.log" 2>&1 || log "  FALHOU shift"
  rm -rf "$DXS_CKPTS/$C"
done
if [ -n "${SPEC:-}" ]; then
  if [ "$SPEC" = dense ]; then C=$BASE; else C=${P}__$SPEC; $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/temp_make_$C.log" 2>&1 || { log "FALHOU make"; exit 1; }; fi
  for T in ${TEMPS:-}; do
    log "$C: traces em T=$T"
    $PY scripts/gen_traces.py --model "$C" --dataset "$DATASET" --n-problems "$NPROB" --samples 4 --max-tokens 16384 --temperature $T \
        > "logs/temp_gen_$(basename $C)_T$T.log" 2>&1 || log "  FALHOU T=$T"
  done
fi
log "concluido"
