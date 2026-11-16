"""Controle de dilatacao para o dano de escrita.

Um prefixo de 4096 tokens de um escritor com kappa > 1 cobre menos do raciocinio do que 4096 tokens do denso.
Dano de escrita ajustado = acc(D escreve ~4096/kappa tokens -> D le) - acc(Q escreve 4096 -> D le),
com acc(D -> D) interpolado (em log b) entre os orcamentos medidos 1024/2048/4096. Para kappa <= 1 o ajuste
usaria b > 4096 (nao medido): usamos 4096, o que e conservador (subestima o dano ajustado).
kappa vem do ajuste da decomposicao (results/fit_validation.csv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
fv = pd.read_csv(ROOT / "results" / "fit_validation.csv")
TASKDS = {"math500": "math500-n300", "gpqa_diamond": "gpqa_diamond-n198", "bbh": "bbh-n300"}


def pp(t, a, ds, b):
    f = DATA / "swap" / f"{t}__{ds}-s4-b16384__ans-{a}__short.parquet"
    if not f.exists():
        return None
    x = pd.read_parquet(f)
    return x[x.b == b].groupby("pid").correct.mean()


rows = []
for _, r in fv.iterrows():
    m, ds, s, k = r.model, TASKDS[r.task], r.compressor, r.kappa
    q = f"{m}__{s}"
    dd = {b: pp(m, m, ds, b) for b in (1024, 2048, 4096)}
    qd, dq = pp(q, m, ds, 4096), pp(m, q, ds, 4096)
    if qd is None or dq is None or any(v is None for v in dd.values()):
        continue
    beq = min(4096.0, 4096.0 / k)
    bs = [1024, 2048, 4096]
    lo = max([b for b in bs if b <= beq], default=1024)
    hi = min([b for b in bs if b >= beq], default=4096)
    w = 0.0 if hi == lo else (np.log(beq) - np.log(lo)) / (np.log(hi) - np.log(lo))
    ref = (1 - w) * dd[lo] + w * dd[hi]
    x = pd.concat([dd[4096], ref, qd, dq], axis=1).dropna().to_numpy()
    bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    raw, adj, read = bt[:, 0] - bt[:, 2], bt[:, 1] - bt[:, 2], bt[:, 0] - bt[:, 3]
    ci = lambda v: f"{np.median(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    rows.append({"model": m, "task": r.task, "compressor": s, "kappa": k, "dense budget matched": beq,
                 "writing damage (raw)": ci(raw), "writing damage (dilation-matched)": ci(adj), "reading damage": ci(read),
                 "adj - reading > 0": bool(np.percentile(adj - read, 2.5) > 0)})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "dilation_control.csv", index=False)
pd.set_option("display.width", 250)
print(res.drop(columns=["dense budget matched"]).to_string(index=False))
k1 = res[res.kappa > 1]
print(f"\nkappa>1 (onde o ajuste importa): escrita ajustada > leitura com IC fora de zero em {k1['adj - reading > 0'].sum()}/{len(k1)}")
print(f"todas: {res['adj - reading > 0'].sum()}/{len(res)}")
