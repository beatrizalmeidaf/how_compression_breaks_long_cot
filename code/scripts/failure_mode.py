"""Como os modelos erram: resposta errada terminada vs. nao terminar dentro do orcamento (loop ou nao).

Para cada modelo (denso pequeno ou comprimido), sobre os traces de 16k do MATH-500:
    acc@16k, fracao dos erros que sao nao-terminacao, que sao loops, e que sao traces longos sem loop,
com IC 95% por bootstrap sobre problemas (a fracao e razao de somas na reamostra).
Saida: results/failure_mode_dense_vs_compressed.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from dxs.paths import DATA
from loop_hazard import loop_onset

import argparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-1.7b__wanda-0.3", "qwen3-1.7b__gptq-w4g128",
          "qwen3-8b__gptq-w3g128", "qwen3-14b__wanda-0.5", "qwen3-4b__rtn-w4g0", "qwen3-4b__gptq-w3g128",
          "qwen3-8b__wanda-0.5", "qwen3-4b__wanda-0.5", "qwen3-4b__rtn-w3g128"]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="math500-n300")
ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
ap.add_argument("--tag", default="", help="sufixo do csv de saida (vazio = arquivo principal do MATH)")
args = ap.parse_args()
S = f"{args.dataset}-s4-b16384"
MODELS = args.models
rng = np.random.default_rng(0)
rows = []
for n in MODELS:
    if not (DATA / "traces" / f"{n}__{S}.parquet").exists():
        print("[falta]", n)
        continue
    d = pd.read_parquet(DATA / "traces" / f"{n}__{S}.parquet", columns=["pid", "finish", "correct", "output_ids"])
    d["err"] = ~((d.finish == "stop") & d.correct)
    d["trunc"] = d.finish == "length"
    d["loop"] = [loop_onset(np.asarray(x)) is not None if f == "length" else False for x, f in zip(d.output_ids, d.finish)]
    g = d.groupby("pid").agg(err=("err", "sum"), trunc=("trunc", "sum"), loop=("loop", "sum"), n=("err", "size"))
    a = g[["err", "trunc", "loop"]].to_numpy(float)
    idx = rng.integers(0, len(a), (2000, len(a)))
    E, Tr, L = a[idx, 0].sum(1), a[idx, 1].sum(1), a[idx, 2].sum(1)
    ci = lambda v: f"{np.median(v):.2f} [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]"
    rows.append({"model": n, "acc@16k": float(1 - (g.err / g.n).mean()), "truncated": float(d.trunc.mean()),
                 "errors not finished": ci(Tr / E), "errors in loops": ci(L / E), "errors long without loop": ci((Tr - L) / E)})
res = pd.DataFrame(rows).sort_values("acc@16k")
res.insert(1, "task", args.dataset.split("-")[0])
res.to_csv(ROOT / "results" / f"failure_mode_dense_vs_compressed{('_' + args.tag) if args.tag else ''}.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
