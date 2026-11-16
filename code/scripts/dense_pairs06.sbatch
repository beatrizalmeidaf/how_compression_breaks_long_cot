#!/usr/bin/env bash
# M1 (complemento): Qwen3-0.6B denso, na faixa de acuracia dos comprimidos severos.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
SUF=math500-n300-s4-b16384
echo "[$(date '+%m-%d %H:%M:%S')] traces Qwen3-0.6B"
$PY scripts/gen_traces.py --model Qwen/Qwen3-0.6B --dataset math500 --n-problems 300 --samples 4 --max-tokens 16384 --gpu-mem 0.75 \
    > logs/dense_pairs06_gen.log 2>&1 || { echo FALHOU gen; exit 1; }
run() { echo "[$(date '+%m-%d %H:%M:%S')] pensa=$1 conclui=$2"
  $PY scripts/swap_answer.py --traces "$DXS_DATA/traces/$1__$SUF.parquet" --answerer "$2" --short --gpu-mem 0.75 >> logs/dense_pairs.log 2>&1 || echo "  FALHOU"; }
run qwen3-0.6b Qwen/Qwen3-0.6B
for S in "qwen3-4b Qwen/Qwen3-4B" "qwen3-8b Qwen/Qwen3-8B" "qwen3-1.7b Qwen/Qwen3-1.7B"; do
  set -- $S
  run qwen3-0.6b "$2"
  run "$1" Qwen/Qwen3-0.6B
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido"
