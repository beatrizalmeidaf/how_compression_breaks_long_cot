#!/usr/bin/env bash
# A2: as tres frentes de uma vez, para um (modelo base, dataset):
#   (3) consolidacao  traces de D e de cada compressor com orcamento B
#   (1) critico       features de D -> critico de comprimento -> termo crit no teacher forcing
#   (2) loops         vies de copia no teacher forcing + sonda curta de loops (D e cada Q)
# Varias filas podem rodar para o mesmo (base, dataset): as etapas compartilhadas usam travas (mkdir).
# Checkpoints comprimidos ficam em DXS_CKPTS (/raid) e sao apagados depois de usados.
#
#   BASE=Qwen/Qwen3-8B DATASET=math500 NPROB=300 [B=16384] [MEM=0.85] bash scripts/run_a2.sh <gpu> <spec ...>
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

GPU=${1:?gpu}; shift
BASE=${BASE:-Qwen/Qwen3-4B}
DATASET=${DATASET:-math500}
NPROB=${NPROB:-300}
B=${B:-16384}
MEM=${MEM:-0.85}
P=$(basename "$BASE" | tr 'A-Z' 'a-z')
SUF=$DATASET-n$NPROB-s4-b$B
DTR=$DXS_DATA/traces/${P}__$SUF.parquet
FEATS=$DXS_DATA/critic/${P}__${SUF}__feats.npz
CRIT=$DXS_DATA/critic/${P}__${SUF}__critic.pt
LOCKS=$DXS_DATA/locks
mkdir -p "$LOCKS" logs
# <gpu> = "slurm": usa a GPU que o Slurm alocou (nao mexe em CUDA_VISIBLE_DEVICES)
if [ "$GPU" = slurm ]; then QID=job${SLURM_JOB_ID:-$$}; else export CUDA_VISIBLE_DEVICES=$GPU; QID=gpu$GPU; fi
export DXS_CKPTS=$DXS_CKPTS/$QID     # checkpoints por fila: filas do mesmo modelo nao apagam os da outra
mkdir -p "$DXS_CKPTS"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# etapa compartilhada: so quem pega a trava executa; os demais esperam o arquivo de saida
shared() {  # shared <nome> <arquivo esperado> <comando...>
  local name=$1 want=$2; shift 2
  [ -f "$want" ] && return 0
  local fail="$LOCKS/${P}__${SUF}__$name.failed"
  if mkdir "$LOCKS/${P}__${SUF}__$name" 2>/dev/null; then
    log "$P/$DATASET: $name"
    "$@" > "logs/a2_${P}_${DATASET}_$name.log" 2>&1 || { log "  FALHOU $name"; touch "$fail"; }
  else
    log "$P/$DATASET: esperando $name"
    until [ -f "$want" ] || [ -f "$fail" ]; do sleep 30; done
  fi
  [ -f "$want" ]
}

shared dense_traces "$DTR" $PY scripts/gen_traces.py --model "$BASE" --dataset "$DATASET" --n-problems "$NPROB" \
    --samples 4 --max-tokens "$B" --gpu-mem "$MEM" || { log "sem traces densos, abortando"; exit 1; }
[ -f "$CRIT" ] || shared feats "$FEATS" $PY scripts/extract_feats.py --traces "$DTR" --dense "$BASE"
shared critic "$CRIT" $PY scripts/train_length_critic.py --feats "$FEATS" --dense "$BASE"
CRIT_ARG=(); [ -f "$CRIT" ] && CRIT_ARG=(--critic "$CRIT")      # sem critico: TF roda sem o termo crit
shared probe_dense "$DXS_DATA/probes/${P}__${SUF}__probe__${P}.parquet" \
    $PY scripts/probe_loops.py --traces "$DTR" --model "$BASE" --gpu-mem "$MEM"

for SPEC in "$@"; do
  C=${P}__$SPEC
  QTR=$DXS_DATA/traces/${C}__$SUF.parquet
  TFF=$DXS_DATA/tf/${P}__${SUF}__vs__${C}__v2.parquet
  PRB=$DXS_DATA/probes/${P}__${SUF}__probe__$C.parquet
  if [ -f "$QTR" ] && [ -f "$TFF" ] && [ -f "$PRB" ]; then log "[skip] $C"; continue; fi
  log "$C: comprimir"
  $PY scripts/make_compressed.py --base "$BASE" "$SPEC" > "logs/a2_make_$C.log" 2>&1 \
      || { log "  FALHOU make $C (ex.: sem checkpoint AWQ publicado)"; continue; }
  log "$C: traces"
  $PY scripts/gen_traces.py --model "$C" --dataset "$DATASET" --n-problems "$NPROB" --samples 4 --max-tokens "$B" \
      --gpu-mem "$MEM" > "logs/a2_gen_${C}_$DATASET.log" 2>&1 || log "  FALHOU gen $C"
  log "$C: teacher forcing v2 (copia + critico)"
  $PY scripts/teacher_force.py --traces "$DTR" --dense "$BASE" --comp "$C" "${CRIT_ARG[@]}" --tag v2 \
      > "logs/a2_tf_${C}_$DATASET.log" 2>&1 || log "  FALHOU tf $C"
  log "$C: sonda de loops"
  $PY scripts/probe_loops.py --traces "$DTR" --model "$C" --gpu-mem "$MEM" \
      > "logs/a2_probe_${C}_$DATASET.log" 2>&1 || log "  FALHOU probe $C"
  [ "${KEEP:-0}" = 1 ] || rm -rf "${DXS_CKPTS:?}/$C"
done
log "fila concluida: $P/$DATASET $*"
