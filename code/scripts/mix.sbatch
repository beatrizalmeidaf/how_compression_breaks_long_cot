#!/usr/bin/env bash
# Decodificacao intercalada D/Q para um compressor e varios esquemas.
#   sbatch --job-name=mix-rtn3 --export=ALL,BASE=Qwen/Qwen3-4B,SPEC=rtn-w3g128,SCHEDS="warm1024 per100/900" scripts/mix.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job${SLURM_JOB_ID:-direct}; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
DATASET=${DATASET:-math500}; NPROB=${NPROB:-300}
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); C=${P}__$SPEC
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
$PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/mix_make_$C.log" 2>&1 || { log "FALHOU make"; exit 1; }
for S in $SCHEDS; do
  log "$C: esquema $S"
  $PY scripts/mix_decode.py --dense "$BASE" --comp "$C" --schedule "$S" --dataset "$DATASET" --n-problems "$NPROB" \
      > "logs/mix_${C}_${S//\//-}_$DATASET.log" 2>&1 || log "  FALHOU $S"
done
log "concluido"
