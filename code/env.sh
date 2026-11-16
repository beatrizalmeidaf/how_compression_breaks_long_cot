# source env.sh -- set these paths for your machine before running any script
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/hf_cache}        # Hugging Face model cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export DXS_DATA=${DXS_DATA:-$HOME/dxs_data}                 # datasets, traces and derived outputs
export DXS_CKPTS=${DXS_CKPTS:-$HOME/dxs_ckpts}              # simulated compressed checkpoints (large, re-creatable)
export C4_VALIDATION_FILE=${C4_VALIDATION_FILE:-$HF_HUB_CACHE/c4/en/c4-validation.00000-of-00008.json.gz}  # GPTQ/Wanda calibration
export PY=${PY:-python}
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$DXS_DATA" "$DXS_CKPTS"
