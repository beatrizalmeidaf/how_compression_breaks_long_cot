#!/usr/bin/env bash
# Controle M1: a assimetria escrita x leitura vale entre modelos DENSOS de tamanhos diferentes?
# Troca de papeis com resposta curta entre Qwen3 densos (mesmo tokenizer), nos dois sentidos.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"; source env.sh
SUF=math500-n300-s4-b16384
run() { echo "[$(date '+%m-%d %H:%M:%S')] pensa=$1 conclui=$2"
  $PY scripts/swap_answer.py --traces "$DXS_DATA/traces/$1__$SUF.parquet" --answerer "$2" --short --gpu-mem 0.75 >> logs/dense_pairs.log 2>&1 || echo "  FALHOU"; }
for pair in "qwen3-1.7b Qwen/Qwen3-1.7B qwen3-8b Qwen/Qwen3-8B" "qwen3-1.7b Qwen/Qwen3-1.7B qwen3-4b Qwen/Qwen3-4B" "qwen3-4b Qwen/Qwen3-4B qwen3-8b Qwen/Qwen3-8B"; do
  set -- $pair
  run "$1" "$4"    # fraco escreve, forte le
  run "$3" "$2"    # forte escreve, fraco le
done
echo "[$(date '+%m-%d %H:%M:%S')] concluido"
