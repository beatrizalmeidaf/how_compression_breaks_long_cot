"""Efeito da compressao no modo de falha, (A) dentro do mesmo pai e (B) contra modelos densos menores de mesma acuracia.

Todos os modelos de uma tarefa respondem aos mesmos problemas, entao o bootstrap reamostra problemas em conjunto
(pareado). Taxas sobre TODOS os traces (nao so sobre os erros, cujo denominador varia):
    nao-terminacao = fracao dos traces cortados no orcamento
    errado-terminado = fracao dos traces que terminaram com resposta errada
    fracao dos erros que nao terminaram (como na Tabela failure_mode)

(A) para cada variante comprimida: delta (comprimido - pai) das tres quantidades, IC 95%.
(B) pares (denso menor, comprimido) com |delta acc| <= MATCH, mesma tarefa: delta da fracao dos erros que nao terminam.

Saidas: results/failure_paired.csv, results/failure_matched.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
T = DATA / "traces"
B = 16384
MATCH = 0.03   # regra de pareamento por acuracia, fixada antes de olhar os pares
PARENTS = [("qwen3-1.7b", "math500-n300"), ("qwen3-4b", "math500-n300"), ("qwen3-8b", "math500-n300"),
           ("qwen3-14b", "math500-n300"), ("qwen3-4b", "gpqa_diamond-n198"), ("qwen3-4b", "bbh-n300"),
           ("deepseek-r1-distill-llama-8b", "math500-n300")]
SMALL = {"math500-n300": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "deepseek-r1-distill-qwen-1.5b"],
         "gpqa_diamond-n198": ["qwen3-0.6b", "qwen3-1.7b"], "bbh-n300": ["qwen3-0.6b", "qwen3-1.7b"]}
N_BOOT = 2000
rng = np.random.default_rng(0)


def per_problem(name, ds):
    f = T / f"{name}__{ds}-s4-b{B}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=["pid", "finish", "correct"])
    d["ok"] = (d.finish == "stop") & d.correct
    d["trunc"] = d.finish == "length"
    d["wrong"] = (d.finish == "stop") & ~d.correct
    return d.groupby("pid")[["ok", "trunc", "wrong"]].sum().astype(float).assign(n=d.groupby("pid").size())


def stats(g):
    """Linhas de reamostra -> (acc, nao-terminacao, errado-terminado, fracao dos erros nao terminados)."""
    n = g[..., 3].sum(-1)
    tr, wr = g[..., 1].sum(-1), g[..., 2].sum(-1)
    return g[..., 0].sum(-1) / n, tr / n, wr / n, tr / np.maximum(tr + wr, 1e-9)


def fmt(v, pt):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def compare(a, b):
    """b - a, pareado por problema; a e b sao DataFrames por pid (mesmos problemas)."""
    a, b = a.align(b, join="inner", axis=0)
    A, Bm = a[["ok", "trunc", "wrong", "n"]].to_numpy(), b[["ok", "trunc", "wrong", "n"]].to_numpy()
    idx = rng.integers(0, len(A), (N_BOOT, len(A)))
    sa, sb = stats(A[idx]), stats(Bm[idx])
    pa, pb = stats(A[None]), stats(Bm[None])
    return [(float(x[0]), float(y[0]), sy - sx) for x, y, sx, sy in zip(pa, pb, sa, sb)]


rows = []
for parent, ds in PARENTS:
    P = per_problem(parent, ds)
    if P is None:
        continue
    for f in sorted(T.glob(f"{parent}__*__{ds}-s4-b{B}.parquet")):
        spec = f.name.split("__")[1]
        if "+" in spec:   # variantes de temperatura / penalidade: fora
            continue
        Q = per_problem(f"{parent}__{spec}", ds)
        (aP, aQ, dA), (tP, tQ, dT), (wP, wQ, dW), (sP, sQ, dS) = compare(P, Q)
        rows.append({"model": parent, "task": ds.split("-")[0], "compressor": spec, "acc parent": aP, "acc compr.": aQ,
                     "non-term. parent": tP, "non-term. compr.": tQ, "delta non-term.": fmt(dT, tQ - tP),
                     "wrong-term. parent": wP, "wrong-term. compr.": wQ, "delta wrong-term.": fmt(dW, wQ - wP),
                     "share parent": sP, "share compr.": sQ, "delta share": fmt(dS, sQ - sP),
                     "share up (CI > 0)": bool(np.percentile(dS, 2.5) > 0),
                     "non-term. up (CI > 0)": bool(np.percentile(dT, 2.5) > 0)})
pa = pd.DataFrame(rows)
pa.to_csv(ROOT / "results" / "failure_paired.csv", index=False)
pd.set_option("display.width", 250)
print(pa[["model", "task", "compressor", "acc parent", "acc compr.", "delta non-term.", "delta wrong-term.",
          "delta share"]].round(3).to_string(index=False))
sev = pa.compressor.isin(["gptq-w3g128", "gptq-w3g64", "wanda-0.5", "rtn-w3g128"])
for lab, g in (("severos", pa[sev]), ("leves", pa[~sev])):
    print(f"[{lab}] nao-terminacao sobe (IC>0): {g['non-term. up (CI > 0)'].sum()}/{len(g)}; "
          f"fracao dos erros sobe (IC>0): {g['share up (CI > 0)'].sum()}/{len(g)}")

# (B) pares de mesma acuracia
rows = []
for ds, smalls in SMALL.items():
    for s in smalls:
        S = per_problem(s, ds)
        if S is None:
            continue
        accS = float(S.ok.sum() / S.n.sum())
        fam = "r1" if s.startswith("deepseek") else "qwen3"
        for f in sorted(T.glob(f"*__*__{ds}-s4-b{B}.parquet")):
            name = f.name.replace(f"__{ds}-s4-b{B}.parquet", "")
            if "+" in name or name.startswith("deepseek") != (fam == "r1"):
                continue
            Q = per_problem(name, ds)
            accQ = float(Q.ok.sum() / Q.n.sum())
            if abs(accQ - accS) > MATCH:
                continue
            (_, _, _), (tS, tQ, dT), (_, _, _), (sS, sQ, dS) = compare(S, Q)
            rows.append({"task": ds.split("-")[0], "smaller dense": s, "compressed": name, "acc smaller": accS,
                         "acc compr.": accQ, "share smaller": sS, "share compr.": sQ, "delta share": fmt(dS, sQ - sS),
                         "delta non-term.": fmt(dT, tQ - tS), "share higher (CI > 0)": bool(np.percentile(dS, 2.5) > 0)})
mt = pd.DataFrame(rows)
mt.to_csv(ROOT / "results" / "failure_matched.csv", index=False)
print(f"\n(B) pares com |delta acc| <= {MATCH}")
print(mt.round(3).to_string(index=False))
