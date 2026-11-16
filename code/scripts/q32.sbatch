#!/usr/bin/env bash
# Escala: Qwen3-32B denso + 4 compressores, MATH-500 (100 dos 300 problemas x 4, 16k), em 1 GPU.
# Checkpoints comprimidos (64 GB cada) em /raid (passa a cota leve por algumas horas); apagados apos cada geracao.
#   sbatch --job-name=q32 scripts/q32.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
set -o pipefail
# /local ficou sem espaco (disco compartilhado)
export DXS_CKPTS=/path/to/storage$SLURM_JOB_ID; mkdir -p "$DXS_CKPTS"; trap 'rm -rf "$DXS_CKPTS"' EXIT
mkdir -p /path/to/storage; L=/path/to/storage   # /home tem cota e ja encheu durante um job
echo "[$(date '+%m-%d %H:%M:%S')] download" | tee -a $L
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt','vocab.json'])" >> $L 2>&1 \
  || { echo "FALHOU download" | tee -a $L; exit 1; }
gen() { $PY scripts/gen_traces.py --model "$1" --dataset math500 --n-problems 300 --subset 100 --samples 4 --max-tokens 16384 \
          --gpu-mem 0.93 2>&1 | grep -vE " INFO |WARNING|it/s\]" >> $L || echo "  FALHOU gen $1" | tee -a $L; }
# espera os processos do vLLM liberarem as GPUs (um vLLM que quebrou pode segurar memoria por alguns segundos)
waitgpu() { for _ in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$u" -lt 2000 ] && return; sleep 5; done; echo "  GPU ainda ocupada ($u MiB)" | tee -a $L; }
echo "[$(date '+%m-%d %H:%M:%S')] qwen3-32b dense" | tee -a $L
gen Qwen/Qwen3-32B
# GPTQ nao cabe numa GPU com o 32B (modelo de 64 GB + Hessianas): os severos sao Wanda-50% e RTN-W3
for SPEC in rtn-w4g128 wanda-0.5 rtn-w3g128; do
  C=qwen3-32b__$SPEC
  [ -f "$DXS_DATA/early_exit/${C}__math500-n100s-s4-b16384__exit.parquet" ] && continue   # ja feito
  echo "[$(date '+%m-%d %H:%M:%S')] $C" | tee -a $L
  waitgpu
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/make_compressed.py --base Qwen/Qwen3-32B "$SPEC" >> $L 2>&1 || { echo "FALHOU make $C" | tee -a $L; continue; }
  gen "$C"
  $PY scripts/early_exit.py --traces "$DXS_DATA/traces/${C}__math500-n100s-s4-b16384.parquet" --model "$C" --gpu-mem 0.93 2>&1 \
    | grep -vE " INFO |WARNING|it/s\]" >> $L
  rm -rf "$DXS_CKPTS/$C"
  bash scripts/update_paper.sh >> "$L" 2>&1
done
$PY scripts/early_exit.py --traces "$DXS_DATA/traces/qwen3-32b__math500-n100s-s4-b16384.parquet" --model Qwen/Qwen3-32B --gpu-mem 0.93 2>&1 | grep -vE " INFO |WARNING|it/s\]" >> $L
echo "[$(date '+%m-%d %H:%M:%S')] concluido" | tee -a $L
bash scripts/update_paper.sh >> "$L" 2>&1
