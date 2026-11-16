#!/usr/bin/env bash
# Uma fila do A2 sob o Slurm (sobrevive ao fim da sessao SSH; processos soltos morrem com Linger=no).
#   sbatch --job-name=a2-8b --export=ALL,BASE=Qwen/Qwen3-8B,DATASET=math500,NPROB=300,SPECS="rtn-w8g128 awq-official" scripts/a2.sbatch
# So gpu: o cache de modelos (/local) e deste no.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"
trap 'rm -rf /path/to/storage$SLURM_JOB_ID' EXIT
bash scripts/run_a2.sh slurm $SPECS
