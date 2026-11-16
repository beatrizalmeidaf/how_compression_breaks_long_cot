#!/usr/bin/env bash
# A1: lei de dilatacao do orcamento. Por config: cria o checkpoint comprimido em RAM (DXS_CKPTS),
# gera os traces de Q (16k), faz teacher forcing sobre os traces de D e APAGA o checkpoint.
# Nao ha rollouts. Etapas pulam o que ja existe. Pico de armazenamento: ~8 GB por fila, em /dev/shm.
#
#   [MEM=0.7] [KEEP=1] bash scripts/run_a1.sh <gpu> <spec ...>      spec: rtn-w4g128, awq-official, gptq-w3g128, wanda-0.5
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

GPU=${1:?gpu}; shift
DENSE=Qwen/Qwen3-4B
MEM=${MEM:-0.85}
B=${B:-16384}
SUF=math500-n300-s4-b$B
DTR=$DXS_DATA/traces/qwen3-4b__$SUF.parquet
export CUDA_VISIBLE_DEVICES=$GPU

until [ -f "$DTR" ]; do sleep 30; done            # traces densos sao pre-requisito do TF
for SPEC in "$@"; do
  C=qwen3-4b__$SPEC
  QTR=$DXS_DATA/traces/${C}__$SUF.parquet
  TFF=$DXS_DATA/tf/qwen3-4b__${SUF}__vs__$C.parquet
  if [ -f "$QTR" ] && [ -f "$TFF" ]; then echo "[skip] $C"; continue; fi
  echo "[$(date +%H:%M:%S)] $C: comprimir"
  $PY scripts/make_compressed.py --base $DENSE "$SPEC" > "logs/a1_make_$C.log" 2>&1 || { echo "  FALHOU make $C"; continue; }
  echo "[$(date +%H:%M:%S)] $C: traces"
  $PY scripts/gen_traces.py --model "$C" --dataset math500 --n-problems 300 --samples 4 --max-tokens $B \
      --gpu-mem $MEM > "logs/a1_gen_$C.log" 2>&1 || echo "  FALHOU gen $C"
  echo "[$(date +%H:%M:%S)] $C: teacher forcing"
  $PY scripts/teacher_force.py --traces "$DTR" --dense $DENSE --comp "$C" > "logs/a1_tf_$C.log" 2>&1 || echo "  FALHOU tf $C"
  [ "${KEEP:-0}" = 1 ] || rm -rf "${DXS_CKPTS:?}/$C"
done
echo "[$(date +%H:%M:%S)] fila concluida"
