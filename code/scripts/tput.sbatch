#!/usr/bin/env bash
# Velocidade de decodificacao dos modelos bf16 (referencia para os AWQ reais de realk.sbatch).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
L=logs/realk.log
for X in 1.7B:qwen3-1.7b 4B:qwen3-4b 8B:qwen3-8b; do
  IFS=: read M LAB <<< "$X"
  $PY scripts/throughput.py --model Qwen/Qwen3-$M --label $LAB 2>&1 | grep -E "^\[throughput\]|Error" >> $L
done
bash scripts/update_paper.sh >> "$L" 2>&1
