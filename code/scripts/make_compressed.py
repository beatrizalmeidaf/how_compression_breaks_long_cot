"""Gera checkpoints comprimidos SIMULADOS (pesos desquantizados em bf16).

Salvar em bf16 faz o vLLM (geracao) e o HF (teacher forcing) verem exatamente os
mesmos pesos, sem depender de kernels AWQ/GPTQ diferentes em cada engine. O que
se estuda e a mudanca de DISTRIBUICAO, nao a velocidade.

Specs:
    awq-official          checkpoint publicado <base>-AWQ, desquantizado
    rtn-w<b>g<g>          RTN afim por grupo (g=0 -> por canal)
    gptq-w<b>g<g>         GPTQ sequencial, calibracao C4 (128 x 512)
    wanda-<s>             Wanda nao estruturado, esparsidade s (ex.: 0.5)

Uso: python scripts/make_compressed.py --base Qwen/Qwen3-4B rtn-w4g128 rtn-w3g128 awq-official
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dxs.paths import CKPTS, HF_CACHE



def ckpt_name(base: str, spec: str) -> str:
    return f"{base.split('/')[-1].lower()}__{spec}"


def c4_calibration(tok, n_seq: int = 128, seq_len: int = 512, seed: int = 0) -> torch.Tensor:
    """Mesma amostragem de dxs.compressors.c4_calibration, lendo o shard do C4 direto do arquivo
    (o cache do `datasets` gravado por outra versao nao abre em modo offline)."""
    import os
    from datasets import load_dataset
    ds = load_dataset("json", data_files=os.environ["C4_VALIDATION_FILE"], split="train")
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in torch.randperm(len(ds), generator=g).tolist():
        ids = tok(ds[i]["text"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(ids) > seq_len:
            s = int(torch.randint(0, len(ids) - seq_len, (1,), generator=g))
            rows.append(ids[s:s + seq_len])
        if len(rows) == n_seq:
            break
    batch = torch.stack(rows)
    if tok.bos_token_id is not None:
        batch = torch.cat([torch.full((n_seq, 1), tok.bos_token_id), batch[:, :-1]], dim=1)
    return batch


def apply_spec(model, tok, bank, spec: str) -> dict:
    from dxs.compressors import calibrated_compress, load_official_quantized, rtn

    layers = model.model.layers
    if spec == "awq-official":
        return load_official_quantized(bank, f"{model.name_or_path}-AWQ", cache_dir=str(HF_CACHE))
    if m := re.fullmatch(r"rtn-w(\d+)g(\d+)", spec):
        bits, group = int(m[1]), int(m[2])
        bank.map(lambda w, t: rtn(w, bits, group if group else w.shape[-1]))
        return {"method": "rtn", "bits": bits, "group": group}
    calib = c4_calibration(tok, n_seq=128, seq_len=512, seed=0).to(model.device)
    if m := re.fullmatch(r"gptq-w(\d+)g(\d+)", spec):
        return calibrated_compress(model, bank, layers, calib, "gptq", bits=int(m[1]), group=int(m[2]))
    if m := re.fullmatch(r"wanda-([0-9.]+)", spec):
        return calibrated_compress(model, bank, layers, calib, "wanda", sparsity=float(m[1]))
    raise ValueError(f"spec desconhecida: {spec}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dxs.compressors import WeightBank

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    model.eval()
    bank = WeightBank(model.model)
    for spec in args.specs:
        out = CKPTS / ckpt_name(args.base, spec)
        if (out / "dxs_meta.json").exists():
            print(f"[skip] {out}")
            continue
        t0 = time.time()
        bank.restore()
        info = apply_spec(model, tok, bank, spec)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        meta = {"base": args.base, "spec": spec, "info": {k: v for k, v in info.items() if k != "alphas"},
                "seconds": round(time.time() - t0, 1)}
        (out / "dxs_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[ok] {out} ({meta['seconds']} s) {meta['info']}")
    bank.restore()


if __name__ == "__main__":
    main()
