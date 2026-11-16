"""Avalia a regra pre-registrada (prereg/2026-10-14_crossover.md) em todos os pares disponiveis.

Para cada (modelo, dataset, compressor) com traces pensando, resposta sem raciocinio e teacher forcing:
    retencao_pensando, retencao_sem_pensar, dif (IC 95% por bootstrap nos problemas), TV/token,
    conjunto: dev (16 pares ja vistos) | teste (14B e compressores novos) | visto-TV (TV ja conhecido)
    previsao da regra (TV < 0.114 -> dif > 0) e se acertou; exclusao "no chao" (retencao < 0.2).
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dxs.paths import DATA

THR = 0.114
NEW_SPECS = {"gptq-w3g64", "rtn-w4g0", "wanda-0.4"}
DEV = {("qwen3-8b", "math500", s) for s in ["rtn-w4g128", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]} | \
      {("qwen3-1.7b", "math500", s) for s in ["gptq-w4g128", "wanda-0.3", "rtn-w4g128", "gptq-w3g128"]} | \
      {("qwen3-4b", "math500", s) for s in ["rtn-w4g128", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]} | \
      {("qwen3-4b", "gpqa_diamond", s) for s in ["rtn-w4g128", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]}
rng = np.random.default_rng(0)


def think_acc(f):
    d = pd.read_parquet(f, columns=["pid", "finish", "correct"])
    return ((d.finish == "stop") & d.correct).groupby(d.pid).mean()


def nothink_acc(f):
    d = pd.read_parquet(f)
    return d[d.b == 0].groupby("pid").correct.mean()


rows = []
for dtr in sorted(glob.glob(str(DATA / "traces" / "qwen3-*__*-s4-b16384.parquet"))):
    stem = Path(dtr).stem
    if stem.count("__") != 1:
        continue
    model, suf = stem.split("__")
    ds = suf.split("-n")[0]
    dN = DATA / "swap" / f"{stem}__ans-{model}.parquet"
    if not dN.exists():
        continue
    dT, dNa = think_acc(dtr), nothink_acc(dN)
    for qtr in sorted(glob.glob(str(DATA / "traces" / f"{model}__*__{suf}.parquet"))):
        c = Path(qtr).stem.split("__")[1]
        if "+" in c:
            continue
        qN = DATA / "swap" / f"{model}__{c}__{suf}__ans-{model}__{c}.parquet"
        tf = DATA / "tf" / f"{stem}__vs__{model}__{c}__v2.parquet"
        if not (qN.exists() and tf.exists()):
            continue
        t = pd.read_parquet(tf, columns=["n_out", "sum_tv"])
        tv = t.sum_tv.sum() / t.n_out.sum()
        x = pd.concat([dNa, nothink_acc(qN), dT, think_acc(qtr)], axis=1).dropna().to_numpy()
        bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
        rN, rT = bt[:, 1] / bt[:, 0], bt[:, 3] / bt[:, 2]
        dif = rT - rN
        key = (model, ds, c)
        conj = "dev" if key in DEV else ("teste" if (model == "qwen3-14b" or c in NEW_SPECS) else "visto-TV")
        rows.append({"conjunto": conj, "modelo": model, "dataset": ds, "config": c, "tv_tok": tv,
                     "ret_sem": float(np.median(rN)), "ret_pens": float(np.median(rT)), "dif": float(np.median(dif)),
                     "dif_lo": float(np.percentile(dif, 2.5)), "dif_hi": float(np.percentile(dif, 97.5)),
                     "chao": bool(np.median(rN) < 0.2 or np.median(rT) < 0.2)})

r = pd.DataFrame(rows).sort_values(["conjunto", "tv_tok"])
r["prev"] = np.where(r.tv_tok < THR, "+", "-")
r["acertou"] = np.where(r.chao, None, (r.dif > 0) == (r.tv_tok < THR))
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
for conj in ["dev", "teste", "visto-TV"]:
    g = r[(r.conjunto == conj) & ~r.chao]
    if len(g):
        rho = spearmanr(g.tv_tok, g.dif).statistic if len(g) > 2 else float("nan")
        print(f"\n{conj:9s}: n={len(g)}  sinal certo={g.acertou.mean():.2f}  Spearman(TV, dif)={rho:+.3f}  "
              f"ICs que excluem zero: {((g.dif_lo > 0) | (g.dif_hi < 0)).sum()}")
g = r[(r.conjunto == "teste") & ~r.chao]
if len(g) >= 3:
    ok = g.acertou.mean() >= 0.8 and spearmanr(g.tv_tok, g.dif).statistic <= -0.6
    print(f"\nCRITERIO PRE-REGISTRADO (teste): {'PASSOU' if ok else 'FALHOU'}")
r.to_csv(Path(__file__).resolve().parents[1] / "results" / "xover_all.csv", index=False)
