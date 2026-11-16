#!/usr/bin/env bash
# Troca de papeis pensador x respondedor entre o denso e cada compressor.
#   sbatch --job-name=swap-4b --export=ALL,BASE=Qwen/Qwen3-4B,SPECS="rtn-w3g128 wanda-0.5" scripts/swap.sbatch
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
P=$(basename "$BASE" | tr 'A-Z' 'a-z'); SUF=$DATASET-n$NPROB-s4-b16384; DTR=$DXS_DATA/traces/${P}__$SUF.parquet
run() { echo "[$(date '+%m-%d %H:%M:%S')] pensa=$(basename $1 .parquet | cut -c1-40) conclui=$2"; $PY scripts/swap_answer.py --traces "$1" --answerer "$2" ${SHORT:+--short} >> "logs/swap_${P}_$DATASET.log" 2>&1 || echo "  FALHOU"; }
run "$DTR" "$BASE"
for SPEC in $SPECS; do
  C=${P}__$SPEC; QTR=$DXS_DATA/traces/${C}__$SUF.parquet
  $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/swap_make_$C.log" 2>&1 || { echo "FALHOU make $C"; continue; }
  run "$QTR" "$C"; run "$DTR" "$C"; run "$QTR" "$BASE"
  rm -rf "$DXS_CKPTS/$C"
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido"
