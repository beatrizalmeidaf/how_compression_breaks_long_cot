#!/usr/bin/env bash
# Kernels reais de 4 bits: traces de Qwen3-4B/8B-AWQ (vLLM com pesos inteiros) + velocidade de decodificacao.
#   sbatch --job-name=realk scripts/realk.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
set -o pipefail
L=logs/realk.log
filt() { grep -vE " INFO |WARNING|it/s\]"; }
for M in 4B 8B; do
  m=qwen3-$(echo $M | tr A-Z a-z)
  echo "[$(date '+%m-%d %H:%M:%S')] traces $m-awq" | tee -a $L
  $PY scripts/gen_traces.py --model Qwen/Qwen3-$M-AWQ --dataset math500 --n-problems 300 --samples 4 --max-tokens 16384 2>&1 | filt >> $L \
    || DXS_DTYPE=float16 $PY scripts/gen_traces.py --model Qwen/Qwen3-$M-AWQ --dataset math500 --n-problems 300 --samples 4 --max-tokens 16384 2>&1 | filt >> $L
  src=$DXS_DATA/traces/$m-awq__math500-n300-s4-b16384.parquet
  [ -f "$src" ] && ln -sf "$src" "$DXS_DATA/traces/${m}__awq-real__math500-n300-s4-b16384.parquet"
done
for X in 1.7B:qwen3-1.7b 4B:qwen3-4b 8B:qwen3-8b 4B-AWQ:qwen3-4b-awq 8B-AWQ:qwen3-8b-awq; do
  IFS=: read M LAB <<< "$X"
  $PY scripts/throughput.py --model Qwen/Qwen3-$M --label $LAB 2>&1 | grep -E "^\[throughput\]|Error" >> $L \
    || DXS_DTYPE=float16 $PY scripts/throughput.py --model Qwen/Qwen3-$M --label $LAB 2>&1 | grep -E "^\[throughput\]|Error" >> $L
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido" | tee -a $L
bash scripts/update_paper.sh >> "$L" 2>&1
