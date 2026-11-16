#!/usr/bin/env bash
# Saida antecipada na deteccao de loop (scripts/early_exit.py) para cada item BASE:SPEC:DS:NP.
#   sbatch --job-name=exit-a --export=ALL,LISTFILE=scripts/exit_list_a.txt scripts/exit.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
export DXS_CKPTS=$DXS_CKPTS/job$SLURM_JOB_ID; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
L=logs/exit_$SLURM_JOB_NAME.log
LIST=${LIST:-$(cat "$LISTFILE")}
for item in $LIST; do
  IFS=: read BASE SPEC DS NP <<< "$item"
  case "$BASE" in deepseek-ai/*) export HF_HUB_CACHE=/path/to/storage;; *) export HF_HUB_CACHE=/path/to/storage;; esac
  P=$(basename "$BASE" | tr 'A-Z' 'a-z')
  if [ "$SPEC" = dense ]; then C=$BASE; N=$P; else C=${P}__$SPEC; N=$C
    $PY scripts/make_compressed.py --base "$BASE" "$SPEC" >> "$L" 2>&1 || { echo "FALHOU make $C" | tee -a $L; continue; }; fi
  echo "[$(date '+%m-%d %H:%M:%S')] $N $DS" | tee -a $L
  $PY scripts/early_exit.py --traces "$DXS_DATA/traces/${N}__$DS-n$NP-s4-b16384.parquet" --model "$C" 2>&1 | grep -vE " INFO |WARNING|it/s\]" >> "$L" || echo "  FALHOU $N $DS" | tee -a $L
  [ "$SPEC" = dense ] || rm -rf "$DXS_CKPTS/$C"
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido" | tee -a $L
bash scripts/update_paper.sh >> "$L" 2>&1
