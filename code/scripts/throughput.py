"""Velocidade de decodificacao com kernels reais: tokens/s em lote (128 pedidos) e com um pedido por vez.

Comprimento fixo (ignore_eos) para que modelos que param em pontos diferentes sejam comparaveis.
Uso: python scripts/throughput.py --model Qwen/Qwen3-4B-AWQ --label qwen3-4b-awq
Saida: linha acrescentada a results/throughput.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dxs.data import chat_prompt, load_problems
from dxs.engine import make_llm
from dxs.paths import model_path

ROOT = Path(__file__).resolve().parents[1]


def weight_gb(path: str) -> float:
    import os
    if Path(path).exists():
        p = Path(path)
    else:   # snapshot no cache do HF (pode faltar README/LICENSE, que nao importam)
        root = Path(os.environ["HF_HUB_CACHE"]) / f"models--{path.replace('/', '--')}" / "snapshots"
        p = max(root.iterdir(), key=lambda d: d.stat().st_mtime)
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    path = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    probs = load_problems("math500", 128, seed=0)
    ids = [tok(chat_prompt(tok, p.question), add_special_tokens=False).input_ids for p in probs]
    llm = make_llm(path, max_model_len=4096 + 1024, gpu_mem=args.gpu_mem)
    sp = lambda n: SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=n, ignore_eos=True)
    llm.generate([{"prompt_token_ids": ids[0]}], sp(64))   # aquecimento
    t0 = time.time()
    r = llm.generate([{"prompt_token_ids": i} for i in ids], sp(2048))
    batch = sum(len(o.outputs[0].token_ids) for o in r) / (time.time() - t0)
    single = []
    for i in ids[:8]:
        t0 = time.time()
        r = llm.generate([{"prompt_token_ids": i}], sp(1024))
        single.append(len(r[0].outputs[0].token_ids) / (time.time() - t0))
    row = {"label": args.label, "model": args.model, "weights_gb": round(weight_gb(path), 2),
           "tok_s_batch128": round(batch, 1), "tok_s_single": round(sum(single) / len(single), 1)}
    f = ROOT / "results" / "throughput.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    old = old[old.label != args.label] if len(old) else old
    pd.concat([old, pd.DataFrame([row])]).to_csv(f, index=False)
    print(f"[throughput] {row}")


if __name__ == "__main__":
    main()
