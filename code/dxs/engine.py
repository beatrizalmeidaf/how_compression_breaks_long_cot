"""vLLM compartilhado por geracao de traces e rollouts de bifurcacao."""

from __future__ import annotations

from dxs.sampling import SamplerCfg


def make_llm(model: str, max_model_len: int, gpu_mem: float = 0.85, seed: int = 0):
    import os
    from vllm import LLM
    # DXS_DTYPE: kernels reais de baixa precisao podem exigir float16; DXS_TP: tensor parallel (modelos grandes)
    return LLM(model=model, dtype=os.environ.get("DXS_DTYPE", "bfloat16"), max_model_len=max_model_len,
               gpu_memory_utilization=gpu_mem, seed=seed, enable_prefix_caching=True, generation_config="vllm",
               tensor_parallel_size=int(os.environ.get("DXS_TP", "1")),
               # a fusao all-reduce+RMSNorm do flashinfer compila um kernel incompativel com o CUDA local
               compilation_config={"pass_config": {"fuse_allreduce_rms": False}} if os.environ.get("DXS_TP", "1") != "1" else None)


def sampling_params(cfg: SamplerCfg, n: int, max_tokens: int, seed: int | None = None, **extra):
    from vllm import SamplingParams
    return SamplingParams(n=n, max_tokens=max_tokens, seed=seed, **cfg.as_dict(), **extra)
