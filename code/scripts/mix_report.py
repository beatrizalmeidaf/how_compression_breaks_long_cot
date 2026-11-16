"""Quanto da acuracia o comprimido recupera com uma fracao de tokens escritos pelo denso?

Para cada compressor: acc(16k) de Q sozinho, de D sozinho e de cada esquema de mistura, a fracao de
tokens escrita por D e a fracao da distancia Q->D recuperada, com IC 95% por bootstrap nos problemas.
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dxs.paths import DATA, TRACES

B = 16384
rng = np.random.default_rng(0)


def per_problem(f):
    d = pd.read_parquet(f)
    return d.assign(ok=(d.finish == "stop") & d.correct).groupby("pid").ok.mean(), d


rows = []
dense, _ = per_problem(TRACES / "qwen3-4b__math500-n300-s4-b16384.parquet")
for spec in ["rtn-w3g128", "wanda-0.5", "gptq-w3g128"]:
    q, _ = per_problem(TRACES / f"qwen3-4b__{spec}__math500-n300-s4-b16384.parquet")
    for f in sorted(glob.glob(str(TRACES / f"qwen3-4b__{spec}+mix-*__math500-n300-s2-b16384.parquet"))):
        m, d = per_problem(f)
        x = pd.concat([dense, q, m], axis=1).dropna().to_numpy()
        boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
        rec = (boot[:, 2] - boot[:, 1]) / (boot[:, 0] - boot[:, 1])
        rows.append({"config": spec, "esquema": Path(f).stem.split("+mix-")[1].split("__")[0],
                     "frac_denso": d.dense_tokens.sum() / d.n_out.sum(), "acc_D": x[:, 0].mean(), "acc_Q": x[:, 1].mean(),
                     "acc_mix": x[:, 2].mean(), "recupera": np.median(rec),
                     "IC95": f"[{np.percentile(rec, 2.5):.2f}, {np.percentile(rec, 97.5):.2f}]",
                     "cortados": (d.finish == "length").mean()})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
res.to_csv(Path(__file__).resolve().parents[1] / "results" / "mix_summary.csv", index=False)
