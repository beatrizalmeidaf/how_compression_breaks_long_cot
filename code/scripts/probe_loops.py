"""Sonda barata de colapso em loop: continuacoes CURTAS de um modelo a partir de prefixos de D.

Para cada trace de D sorteia --per-trace posicoes j (uniformes no trace) e gera --tokens tokens com o
modelo avaliado a partir de prompt + output[:j]. Mede a repeticao no fim da continuacao (fracao de
20-gramas de caracteres repetidos). Rodar tambem com o proprio D da a linha de base.

Custo: n_traces * per_trace * tokens (ex.: 1200 * 2 * 1024 = 2.5M tokens, ~1% de uma avaliacao completa).

Uso: python scripts/probe_loops.py --traces <traces de D> --model qwen3-4b__rtn-w3g128
Saida: DXS_DATA/probes/<traces>__probe__<modelo>.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dxs.engine import make_llm, sampling_params
from dxs.paths import DATA, model_path
from dxs.sampling import SamplerCfg

PROBES = DATA / "probes"


def loopiness(text: str, chars: int = 1600) -> float:
    t = text[-chars:]
    grams = [t[i:i + 20] for i in range(0, len(t) - 20, 5)]
    return 1 - len(set(grams)) / max(1, len(grams))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-trace", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = Path(args.traces)
    out = PROBES / f"{traces.stem}__probe__{args.model.split('/')[-1].lower()}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    df = pd.read_parquet(traces)
    rng = np.random.default_rng(args.seed)          # mesma semente -> mesmos prefixos para todo modelo
    jobs = []
    for r in df.itertuples():
        out_ids = [int(x) for x in r.output_ids]
        for j in rng.integers(0, max(1, len(out_ids)), size=args.per_trace):
            jobs.append((r.pid, r.sample, int(j), [int(x) for x in r.prompt_ids] + out_ids[:j]))
    llm = make_llm(model_path(args.model), max_model_len=max(len(j[3]) for j in jobs) + args.tokens + 8,
                   gpu_mem=args.gpu_mem, seed=args.seed)
    t0 = time.time()
    res = llm.generate([{"prompt_token_ids": j[3]} for j in jobs],
                       sampling_params(SamplerCfg(), n=1, max_tokens=args.tokens, seed=None))
    rows = [{"pid": j[0], "sample": j[1], "pos": j[2], "n_gen": len(o.outputs[0].token_ids),
             "finish": o.outputs[0].finish_reason, "loop": loopiness(o.outputs[0].text)}
            for j, o in zip(jobs, res)]
    PROBES.mkdir(parents=True, exist_ok=True)
    p = pd.DataFrame(rows)
    if (~p.finish.isin(["stop", "length"])).any():
        raise RuntimeError(f"continuacoes abortadas: {p.finish.value_counts(dropna=False).to_dict()}")
    p.to_parquet(out)
    print(f"[ok] {out} ({(time.time() - t0) / 60:.1f} min)  loop medio={p.loop.mean():.4f}  "
          f"frac loop>0.5={(p.loop > 0.5).mean():.4f}")


if __name__ == "__main__":
    main()
