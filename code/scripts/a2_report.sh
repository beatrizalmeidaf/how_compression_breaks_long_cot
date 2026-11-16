#!/usr/bin/env bash
# Roda a analise de orcamento (alpha, kappa, preditores) para cada (modelo, dataset) com traces densos.
cd "$(dirname "$0")/.." && source env.sh
SPECS="rtn-w8g128 awq-official gptq-w4g128 rtn-w4g128 wanda-0.3 gptq-w3g128 wanda-0.5 rtn-w3g128"
for DTR in $DXS_DATA/traces/qwen3-{1.7b,4b,8b}__*-b16384.parquet; do
  [ -f "$DTR" ] || continue
  [ $(basename "$DTR" | grep -o "__" | wc -l) -eq 1 ] || continue     # so traces densos
  P=$(basename "$DTR" | cut -d_ -f1)
  echo "=================== $(basename "$DTR" .parquet)"
  DXS_RESULTS=$PWD/results $PY scripts/a1_budget_tax.py --dense-traces "$DTR" --configs $(for s in $SPECS; do echo ${P}__$s; done)
done
