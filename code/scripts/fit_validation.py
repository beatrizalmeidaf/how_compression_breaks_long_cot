"""Validacao da decomposicao acc_Q(b) = alpha * acc_D(b / kappa).

(1) Familias alternativas de 2 parametros sobre a curva do denso:
      dilation  alpha * acc_D(b / kappa)          (a do artigo: escala de tempo multiplicativa)
      shift     alpha * acc_D(b - delta)          (deslocamento aditivo em tokens)
      power     alpha * acc_D(b) ** gamma
      affine    a * acc_D(b) + c
    RMSE dentro da amostra (grade inteira) e FORA da amostra (ajuste em b <= 8192, erro em 8192 < b <= 16384).
(2) ICs 95% de alpha e kappa por bootstrap pareado sobre problemas (D e Q reamostrados juntos).

Saidas: results/fit_validation.csv (por config) e resumo no stdout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
T = DATA / "traces"
B = 16384
GRID = np.unique(np.round(np.geomspace(512, B, 24)))
TRAIN = GRID[GRID <= 8192]
TEST = GRID[GRID > 8192]
RUNS = [("qwen3-1.7b", "math500-n300"), ("qwen3-4b", "math500-n300"), ("qwen3-8b", "math500-n300"),
        ("qwen3-14b", "math500-n300"), ("qwen3-4b", "gpqa_diamond-n198"), ("deepseek-r1-distill-llama-8b", "math500-n300"),
        ("qwen3-4b", "bbh-n300")]
SPECS = ["rtn-w8g128", "awq-official", "gptq-w4g128", "rtn-w4g128", "rtn-w4g0", "wanda-0.3", "wanda-0.4", "gptq-w3g64",
         "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]
N_BOOT = 200
ALPHA_MIN = 0.3   # abaixo disso a curva comprimida e quase plana e kappa nao e identificavel
rng = np.random.default_rng(0)


class FastCurve:
    """acc(b) com pesos por problema (permite bootstrap sem reconstruir DataFrames)."""

    def __init__(self, df):
        self.pids, inv = np.unique(df.pid.to_numpy(), return_inverse=True)
        self.inv = inv
        self.t = np.where(df.finish.to_numpy() == "stop", df.n_out.to_numpy(float), np.inf)
        self.ok = df.correct.to_numpy(bool)
        self.n_per = np.bincount(inv)

    def acc(self, b, mult=None):
        mult = np.ones(len(self.pids)) if mult is None else mult
        w = mult[self.inv] / self.n_per[self.inv] / mult.sum()
        b = np.atleast_1d(np.asarray(b, float))
        return ((self.t[None, :] <= b[:, None]) & self.ok[None, :]) @ w


def families(dacc):
    """Cada familia: (params iniciais, funcao pred(params, b))."""
    def interp(x):
        return np.interp(np.log(np.clip(x, 1, None)), np.log(GRID_FINE), dacc(GRID_FINE), left=0.0)
    return {
        "dilation": ([[a, np.log(k)] for a in (0.5, 1.0) for k in (1, 2, 4)],
                     lambda p, b: p[0] * interp(b / np.exp(p[1]))),
        "shift": ([[a, d] for a in (0.5, 1.0) for d in (-1000, 0, 1000, 4000)],
                  lambda p, b: p[0] * interp(b - p[1])),
        "power": ([[a, np.log(g)] for a in (0.5, 1.0) for g in (0.5, 1, 2, 4)],
                  lambda p, b: p[0] * interp(b) ** np.exp(p[1])),
        "affine": ([[a, c] for a in (0.5, 1.0) for c in (-0.1, 0.0, 0.1)],
                   lambda p, b: p[0] * interp(b) + p[1]),
    }


GRID_FINE = np.unique(np.round(np.geomspace(64, B, 400)))


def fit(pred, starts, target, b, nonneg_alpha=False):
    def loss(p):
        if nonneg_alpha and p[0] < 0:
            return 1e3 + p[0] ** 2
        return float(np.mean((pred(p, b) - target) ** 2))
    best = min((minimize(loss, s, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000}) for s in starts), key=lambda r: r.fun)
    return best.x, float(np.sqrt(best.fun))


def one(job):
    m, ds, spec = job
    rng = np.random.default_rng(zlib.crc32(f"{m}|{ds}|{spec}".encode()))
    ddf = pd.read_parquet(T / f"{m}__{ds}-s4-b{B}.parquet", columns=["pid", "finish", "n_out", "correct"])
    f = T / f"{m}__{spec}__{ds}-s4-b{B}.parquet"
    qdf = pd.read_parquet(f, columns=["pid", "finish", "n_out", "correct"])
    dc, qc = FastCurve(ddf), FastCurve(qdf)
    assert (dc.pids == qc.pids).all()
    fam = families(lambda b: dc.acc(b))
    y_all, y_tr, y_te = qc.acc(GRID), qc.acc(TRAIN), qc.acc(TEST)
    r = {"model": m, "task": ds.split("-")[0], "compressor": spec, "acc_Q": float(qc.acc(B)[0])}
    degenerate = y_all.max() < 0.01   # curva nula: alpha = 0 e kappa indefinido
    for name, (starts, pred) in fam.items():
        p, rmse = fit(pred, starts, y_all, GRID, nonneg_alpha=name == "dilation")
        p_tr, _ = fit(pred, starts, y_tr, TRAIN)
        r[f"rmse_{name}"] = rmse
        r[f"oos_{name}"] = float(np.sqrt(np.mean((pred(p_tr, TEST) - y_te) ** 2)))
        if name == "dilation":
            r["alpha"], r["kappa"] = float(p[0]), float(np.exp(p[1]))
    if degenerate:
        r["alpha"], r["kappa"] = 0.0, np.nan
    r["kappa_identified"] = bool(r["alpha"] >= ALPHA_MIN)
    # bootstrap pareado de alpha e kappa (varias partidas, como no ajuste pontual)
    a_s, k_s = [], []
    n = len(dc.pids)
    for _ in range(N_BOOT if not degenerate else 0):
        mult = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        fam_b = families(lambda b, mult=mult: dc.acc(b, mult))
        starts, pred = fam_b["dilation"]
        p, _ = fit(pred, [[r["alpha"], np.log(r["kappa"])]] + starts, qc.acc(GRID, mult), GRID, nonneg_alpha=True)
        a_s.append(p[0])
        k_s.append(np.exp(p[1]))
    r["alpha_lo"], r["alpha_hi"] = np.percentile(a_s, [2.5, 97.5]) if a_s else (0.0, 0.0)
    r["kappa_lo"], r["kappa_hi"] = np.percentile(k_s, [2.5, 97.5]) if k_s else (np.nan, np.nan)
    if not r["kappa_identified"]:
        r["kappa"] = r["kappa_lo"] = r["kappa_hi"] = np.nan
    pass
    print(f"{m:30s} {ds.split('-')[0]:12s} {spec:12s} rmse dil={r['rmse_dilation']:.3f} shift={r['rmse_shift']:.3f} "
          f"pow={r['rmse_power']:.3f} aff={r['rmse_affine']:.3f} | oos dil={r['oos_dilation']:.3f} shift={r['oos_shift']:.3f} "
          f"pow={r['oos_power']:.3f} aff={r['oos_affine']:.3f} | alpha {r['alpha']:.2f} [{r['alpha_lo']:.2f},{r['alpha_hi']:.2f}] "
          f"kappa {r['kappa']:.2f} [{r['kappa_lo']:.2f},{r['kappa_hi']:.2f}]", flush=True)
    return r


if __name__ == "__main__":
    JOBS = [(m, ds, spec) for m, ds in RUNS for spec in SPECS
            if (T / f"{m}__{ds}-s4-b{B}.parquet").exists() and (T / f"{m}__{spec}__{ds}-s4-b{B}.parquet").exists()]
    from multiprocessing import Pool
    with Pool(min(len(JOBS), 56)) as pool:
        rows = pool.map(one, JOBS)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "fit_validation.csv", index=False)
    fams = ["dilation", "shift", "power", "affine"]
    print("\nResumo (n = %d configs)" % len(res))
    for kind in ("rmse", "oos"):
        med = {f: res[f"{kind}_{f}"].median() for f in fams}
        mx = {f: res[f"{kind}_{f}"].max() for f in fams}
        wins = res[[f"{kind}_{f}" for f in fams]].idxmin(axis=1).str.replace(f"{kind}_", "").value_counts().to_dict()
        print(f"  {kind:4s} mediana: " + "  ".join(f"{f}={med[f]:.4f}" for f in fams) +
              " | max: " + "  ".join(f"{f}={mx[f]:.3f}" for f in fams) + f" | melhor em: {wins}")
    sev = res[res.acc_Q < 0.8 * res.groupby(["model", "task"]).acc_Q.transform("max")]
    print(f"  oos dilation vs shift nas configs com mais dano (n={len(sev)}): mediana dil={sev.oos_dilation.median():.4f} shift={sev.oos_shift.median():.4f}")
    idf = res[res.kappa_identified]
    ci_k = idf[(idf.kappa_lo > 1) | (idf.kappa_hi < 1)]
    print(f"  kappa identificado (alpha >= {ALPHA_MIN}): {len(idf)}/{len(res)}; IC excluindo 1: {len(ci_k)}/{len(idf)}")
    inside = idf[(idf.kappa >= idf.kappa_lo) & (idf.kappa <= idf.kappa_hi)]
    print(f"  kappa pontual dentro do proprio IC: {len(inside)}/{len(idf)}")
