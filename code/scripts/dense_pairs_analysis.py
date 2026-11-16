"""M1: a assimetria escrita x leitura e especifica da compressao?

Para pares de Qwen3 DENSOS (fraco W, forte S), mesmo protocolo da troca de papeis (resposta curta, 4096 tokens):
    dano de escrita = acc(S escreve -> S le) - acc(W escreve -> S le)
    dano de leitura = acc(S escreve -> S le) - acc(S escreve -> W le)
e a razao escrita/leitura comparada com versoes comprimidas do forte de acuracia sozinha parecida.
Saida: results/dense_pairs.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
DS = "math500-n300"
rng = np.random.default_rng(0)


def pp(t, a, b=4096):
    f = DATA / "swap" / f"{t}__{DS}-s4-b16384__ans-{a}__short.parquet"
    if not f.exists():
        return None
    x = pd.read_parquet(f)
    return x[x.b == b].groupby("pid").correct.mean()


def decomp(S, W, kind):
    parts = [pp(S, S), pp(W, S), pp(S, W), pp(W, W)]
    if any(p is None for p in parts):
        return None
    x = pd.concat(parts, axis=1).dropna().to_numpy()
    bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    w, r = bt[:, 0] - bt[:, 1], bt[:, 0] - bt[:, 2]
    ci = lambda v: f"{np.median(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    return {"kind": kind, "strong": S, "weak": W, "weak alone": x[:, 3].mean(), "strong alone": x[:, 0].mean(),
            "writing damage": ci(w), "reading damage": ci(r), "writing - reading": ci(w - r),
            "reading / writing": float(np.median(r) / np.median(w)) if np.median(w) > 0 else np.nan}


rows = []
for S, W in (("qwen3-8b", "qwen3-1.7b"), ("qwen3-4b", "qwen3-1.7b"), ("qwen3-8b", "qwen3-4b"),
             ("qwen3-8b", "qwen3-0.6b"), ("qwen3-4b", "qwen3-0.6b"), ("qwen3-1.7b", "qwen3-0.6b")):
    r = decomp(S, W, "dense smaller model")
    if r:
        rows.append(r)
for S in ("qwen3-4b", "qwen3-8b"):
    for spec in ("rtn-w4g128", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"):
        r = decomp(S, f"{S}__{spec}", "compressed copy")
        if r:
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "dense_pairs.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
