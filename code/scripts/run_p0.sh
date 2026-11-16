#!/usr/bin/env bash
# Piloto P0: Qwen3-4B vs compressores, 300 problemas do MATH-500, 4 amostras, orcamento 8k.
# Cada etapa pula o que ja existe, entao pode ser re-executado apos falha.
#
#   bash scripts/run_p0.sh <gpu> [config ...]
#
# Etapas por config: traces de Q -> TF(D-traces, Q) -> rollouts D (barato) -> rollouts Q (exato A)
#                    -> TF(Q-traces, Q) -> rollouts D (exato B)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

GPU=${1:?gpu}; shift
CONFIGS=("${@:-qwen3-4b__rtn-w8g128 qwen3-4b__awq-official qwen3-4b__rtn-w4g128 qwen3-4b__gptq-w3g128 qwen3-4b__wanda-0.5 qwen3-4b__rtn-w3g128}")
read -r -a CONFIGS <<< "${CONFIGS[*]}"
DENSE=Qwen/Qwen3-4B
MEM=${MEM:-0.85}   # fracao da GPU para o vLLM (baixar se a GPU e compartilhada)
ROLL="--rollouts 1 --max-forks 4 --gpu-mem $MEM"
SUF=math500-n300-s4-b8192
DTR=$DXS_DATA/traces/qwen3-4b__$SUF.parquet
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs

run() { echo "[$(date +%H:%M:%S)] $*"; "$@"; }

for C in "${CONFIGS[@]}"; do
  QTR=$DXS_DATA/traces/${C}__$SUF.parquet
  run $PY scripts/gen_traces.py --model "$C" --dataset math500 --n-problems 300 --samples 4 --gpu-mem $MEM > "logs/gen_$C.log" 2>&1
  run $PY scripts/teacher_force.py --traces "$DTR" --dense $DENSE --comp "$C" > "logs/tf_$C.log" 2>&1
  TFD=$DXS_DATA/tf/qwen3-4b__${SUF}__vs__$C.parquet
  run $PY scripts/fork_rollouts.py --tf "$TFD" --traces "$DTR" --rollout-model $DENSE $ROLL > "logs/roll_cheap_$C.log" 2>&1
  run $PY scripts/fork_rollouts.py --tf "$TFD" --traces "$DTR" --rollout-model "$C" $ROLL > "logs/roll_exactA_$C.log" 2>&1
  run $PY scripts/teacher_force.py --traces "$QTR" --dense $DENSE --comp "$C" --layers "" > "logs/tfq_$C.log" 2>&1
  TFQ=$DXS_DATA/tf/${C}__${SUF}__vs__$C.parquet
  run $PY scripts/fork_rollouts.py --tf "$TFQ" --traces "$QTR" --rollout-model $DENSE $ROLL > "logs/roll_exactB_$C.log" 2>&1
  $PY scripts/p0_analyze.py --dense-traces "$DTR" --configs "${CONFIGS[@]}" --rollouts 1 > logs/p0_analyze.log 2>&1 || true
done
echo "[$(date +%H:%M:%S)] P0 concluido"
