#!/usr/bin/env bash
# Reforco da comparacao de modo de falha: modelos densos pequenos em GPQA/BBH e outra familia (R1-Distill-Qwen-1.5B).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
L=logs/failure_more.log
gen() { echo "[$(date '+%m-%d %H:%M:%S')] gera $1 $2 n=$3" | tee -a $L
  [ -n "$4" ] && export HF_HUB_CACHE=$4
  $PY scripts/gen_traces.py --model "$1" --dataset "$2" --n-problems "$3" --samples 4 --max-tokens 16384 --gpu-mem 0.75 >> $L 2>&1 || echo "  FALHOU" | tee -a $L
  export HF_HUB_CACHE=/path/to/storage; }
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
  gen $m gpqa_diamond 198
  gen $m bbh 300
done
gen deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B math500 300 /path/to/storage

echo "[$(date '+%m-%d %H:%M:%S')] analise" | tee -a $L
comp() { ls $DXS_DATA/traces | grep -E "^$1__[a-z].*__$2-s4-b16384.parquet" | sed "s/__$2-s4-b16384.parquet//"; }
for ds in gpqa_diamond-n198 bbh-n300; do
  $PY scripts/failure_mode.py --dataset $ds --tag ${ds%%-*} --models qwen3-0.6b qwen3-1.7b qwen3-4b $(comp qwen3-4b $ds) >> $L 2>&1
done
$PY scripts/failure_mode.py --dataset math500-n300 --tag r1distill --models deepseek-r1-distill-qwen-1.5b deepseek-r1-distill-llama-8b $(comp deepseek-r1-distill-llama-8b math500-n300) >> $L 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] concluido" | tee -a $L
