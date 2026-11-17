"""kappa so nos problemas que os dois modelos resolvem: kappa < 1 vem de sobrarem so os problemas faceis?

Quando alpha < 1, o comprimido acerta um subconjunto dos problemas. Se esse subconjunto for o dos problemas que o
denso resolve rapido, a curva agregada do comprimido "parece" mais rapida (kappa < 1) sem que nada fique mais rapido.
Aqui o ajuste alpha * acc_D(b / kappa) e refeito restringindo os DOIS modelos aos problemas em que ambos acertam pelo
menos uma das 4 amostras em 16k. Se kappa muda pouco, a selecao nao explica kappa.

Saidas: results/kappa_common.csv, acl/tables/kappa_common_note.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fit_validation import GRID, FastCurve, families, fit

ROOT = Path(__file__).resolve().parents[1]
from dxs.paths import DATA

T = DATA / "traces"
B = 16384
DS = {"math500": "math500-n300", "gpqa_diamond": "gpqa_diamond-n198", "bbh": "bbh-n300"}


def one(x):
    ds = DS[x["task"]]
    cols = ["pid", "finish", "n_out", "correct"]
    d = pd.read_parquet(T / f"{x['model']}__{ds}-s4-b{B}.parquet", columns=cols)
    q = pd.read_parquet(T / f"{x['model']}__{x['compressor']}__{ds}-s4-b{B}.parquet", columns=cols)
    ok = lambda f: ((f.finish == "stop") & f.correct).groupby(f.pid).any()
    both = ok(d) & ok(q)
    keep = set(both[both].index)
    d, q = d[d.pid.isin(keep)], q[q.pid.isin(keep)]
    dc, qc = FastCurve(d), FastCurve(q)
    starts, pred = families(lambda b: dc.acc(b))["dilation"]
    p, rmse = fit(pred, starts, qc.acc(GRID), GRID, nonneg_alpha=True)
    return {**x, "n_common": len(keep), "alpha_common": float(p[0]), "kappa_common": float(np.exp(p[1])),
            "rmse_common": rmse}


if __name__ == "__main__":
    fv = pd.read_csv(ROOT / "results" / "fit_validation.csv")
    fv = fv[fv.kappa_identified][["model", "task", "compressor", "alpha", "kappa"]]
    with Pool(min(len(fv), 48)) as pool:
        r = pd.DataFrame(pool.map(one, fv.to_dict("records")))
    r.to_csv(ROOT / "results" / "kappa_common.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    r["shift"] = r.kappa_common / r.kappa
    low = r[r.kappa < 1]
    side = ((r.kappa < 1) == (r.kappa_common < 1)).sum()
    note = (f"Refitting on the problems that both models solve at least once, $\\kappa$ stays on the same side of 1 in "
            f"{side} of {len(r)} variants, with a median ratio of {r['shift'].median():.2f} to the full-set estimate "
            f"(range {r['shift'].min():.2f}--{r['shift'].max():.2f}); among the {len(low)} variants with $\\kappa<1$, "
            f"{(low.kappa_common < 1).sum()} keep $\\kappa<1$")
    (ROOT / "acl" / "tables" / "kappa_common_note.tex").write_text(note + "%\n")
    print(note)
