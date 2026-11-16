#!/usr/bin/env bash
# Resposta forcada para uma lista de compressores (e "dense").
#   sbatch --job-name=force-4b --export=ALL,BASE=Qwen/Qwen3-4B,SPECS="dense rtn-w3g128" scripts/force.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job$SLURM_JOB_ID; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
DATASET=${DATASET:-math500}; NPROB=${NPROB:-300}
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); SUF=$DATASET-n$NPROB-s4-b16384
for SPEC in $SPECS; do
  if [ "$SPEC" = dense ]; then C=$BASE; N=$P; else C=${P}__$SPEC; N=$C
    $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/force_make_$C.log" 2>&1 || { echo "FALHOU make $C"; continue; }; fi
  echo "[$(date '+%m-%d %H:%M:%S')] $N: resposta forcada"
  $PY scripts/force_answer.py --traces "$DXS_DATA/traces/${N}__$SUF.parquet" --model "$C" > "logs/force_${N}_$DATASET.log" 2>&1 || echo "  FALHOU force $N"
  [ "$SPEC" = dense ] || rm -rf "$DXS_CKPTS/$C"
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido"
