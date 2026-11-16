"""A1: a compressao age como um "imposto" sobre o orcamento de raciocinio?

Lei de dilatacao:  acc_Q(b) ~= acc_D(b / kappa)            (1 parametro)
Com colapso:       acc_Q(b) ~= alpha * acc_D(b / kappa)    (2 parametros)
    1 - alpha = fracao de problemas perdida (traces que entram em loop e nunca terminam)
    kappa     = dilatacao do tempo de pensar nos traces que terminam

Um trace gerado com orcamento B da a acuracia em QUALQUER orcamento b <= B sem gerar de
novo: ele conta como acerto em b se terminou em ate b tokens e a resposta esta certa.

Por compressor:
  acc(b), fracao cortada, acuracia condicional a terminar (o modelo erra ou so nao termina?)
  kappa_fit   minimiza o erro quadratico entre acc_Q(b) e acc_D(b/kappa) na grade de orcamentos
  kappa_q50   razao das medianas do comprimento (censurado em B)
  RMSE da dilatacao vs. baselines de 1 parametro (deslocamento constante, fator multiplicativo)
  preditores so de teacher forcing (se o arquivo existir):
      k_stop = sum p_stop / sum q_stop   (razao de risco de parada, sobre traces de D)
      k_hes  = sum q_hes / sum p_hes     (inflacao da taxa de hesitacao)
      TV/token, KL/token
      k_copy  = sum q(c) / sum p(c)       vies de copia (arquivo __v2)
      k_crit  = 1 + sum crit / sum n_out  kappa previsto pelo critico de comprimento (arquivo __v2)
  sonda curta de loops (DXS_DATA/probes): probe_loop = loop medio de Q - loop medio de D

Uso: python scripts/a1_budget_tax.py --dense-traces <arquivo> --configs qwen3-4b__rtn-w4g128 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr

from dxs.paths import DATA, RESULTS, TF, TRACES


def finish_time(df: pd.DataFrame) -> np.ndarray:
    """Tokens ate terminar; inf se foi cortado pelo orcamento."""
    return np.where(df.finish.to_numpy() == "stop", df.n_out.to_numpy(float), np.inf)


class Curve:
    """acc(b) = media sobre problemas da fracao de amostras que terminam em ate b tokens E acertam."""

    def __init__(self, df: pd.DataFrame, budget: int):
        self.budget = budget
        self.pid = df.pid.to_numpy()
        self.t = finish_time(df)
        self.ok = df.correct.to_numpy(bool)
        self.n_prob = df.pid.nunique()
        self._w = 1.0 / df.groupby("pid").pid.transform("size").to_numpy() / self.n_prob

    def acc(self, b) -> np.ndarray:
        b = np.atleast_1d(np.asarray(b, float))
        hit = (self.t[None, :] <= b[:, None]) & self.ok[None, :]
        return (hit * self._w[None, :]).sum(1)

    def finished(self, b: float) -> float:
        return float(((self.t <= b) * self._w).sum())

    def cond_acc(self, b: float) -> float:
        f = self.finished(b)
        return float(self.acc(b)[0] / f) if f > 0 else float("nan")

    def quantile(self, q: float) -> float:
        v = float(np.quantile(self.t, q, method="lower"))
        return v if np.isfinite(v) else float("nan")


def fit_kappa(d: Curve, q: Curve, grid: np.ndarray) -> tuple[float, float]:
    def loss(logk: float) -> float:
        k = np.exp(logk)
        g = grid[grid / k <= d.budget]            # so onde acc_D(b/kappa) e observavel
        return float(np.mean((q.acc(g) - d.acc(g / k)) ** 2)) if len(g) >= 3 else 1e9
    r = minimize_scalar(loss, bounds=(np.log(0.5), np.log(64.0)), method="bounded", options={"xatol": 1e-3})
    return float(np.exp(r.x)), float(np.sqrt(r.fun))


def fit_alpha_kappa(d: Curve, q: Curve, grid: np.ndarray) -> tuple[float, float, float]:
    def loss(x) -> float:
        a, k = x[0], np.exp(x[1])
        g = grid[grid / k <= d.budget]
        return float(np.mean((q.acc(g) - a * d.acc(g / k)) ** 2)) if len(g) >= 3 else 1e9
    best = min((minimize(loss, [a0, np.log(k0)], method="Nelder-Mead") for a0 in (0.5, 1.0) for k0 in (1, 2, 4)),
               key=lambda r: r.fun)
    return float(best.x[0]), float(np.exp(best.x[1])), float(np.sqrt(best.fun))


def loopiness(text: str, chars: int = 1600) -> float:
    """Fracao de 20-gramas de caracteres repetidos no fim do texto (proxy de loop)."""
    t = text[-chars:]
    grams = [t[i:i + 20] for i in range(0, len(t) - 20, 5)]
    return 1 - len(set(grams)) / max(1, len(grams))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-traces", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tag", default="a1")
    args = ap.parse_args()

    dtr = Path(args.dense_traces)
    suffix = dtr.stem.split("__", 1)[1]
    B = int(suffix.rsplit("-b", 1)[1])
    grid = np.unique(np.round(np.geomspace(512, B, 24)))
    d = Curve(pd.read_parquet(dtr), B)
    print(f"denso: acc({B})={d.acc(B)[0]:.4f} cortados={1 - d.finished(B):.3f} acc|terminou={d.cond_acc(B):.4f} "
          f"mediana T={d.quantile(0.5):.0f}  ({d.n_prob} problemas)")

    rows = []
    for c in args.configs:
        f = TRACES / f"{c}__{suffix}.parquet"
        if not f.exists():
            continue
        qdf = pd.read_parquet(f)
        q = Curve(qdf, B)
        k, rmse = fit_kappa(d, q, grid)
        a2, k2, rmse2 = fit_alpha_kappa(d, q, grid)
        cut = qdf[qdf.finish == "length"]
        diff = q.acc(grid) - d.acc(grid)
        ratio = float((q.acc(grid) * d.acc(grid)).sum() / (d.acc(grid) ** 2).sum())
        row = {"config": c.split("__", 1)[1], "acc": q.acc(B)[0], "delta": q.acc(B)[0] - d.acc(B)[0],
               "cortados": 1 - q.finished(B), "acc|term": q.cond_acc(B),
               "T50": q.quantile(0.5), "k_q50": q.quantile(0.5) / d.quantile(0.5),
               "k_q25": q.quantile(0.25) / d.quantile(0.25), "k_fit": k, "rmse_dil": rmse,
               "alpha2": a2, "k2": k2, "rmse2": rmse2,
               "loop_cut": float(np.mean([loopiness(t) for t in cut.text])) if len(cut) else float("nan"),
               "rmse_shift": float(np.sqrt(np.mean((diff - diff.mean()) ** 2))),
               "rmse_mult": float(np.sqrt(np.mean((q.acc(grid) - ratio * d.acc(grid)) ** 2)))}
        tf2 = TF / f"{dtr.stem}__vs__{c}__v2.parquet"
        tf = tf2 if tf2.exists() else TF / f"{dtr.stem}__vs__{c}.parquet"
        if tf.exists():
            t = pd.read_parquet(tf)
            n = t.n_out.sum()
            row.update(k_stop=t.sum_p_stop.sum() / t.sum_q_stop.sum(), k_hes=t.sum_q_hes.sum() / t.sum_p_hes.sum(),
                       tv_tok=t.sum_tv.sum() / n, kl_tok=t.sum_kl_raw.sum() / n)
            if "sum_q_copy" in t:
                row["k_copy"] = t.sum_q_copy.sum() / t.sum_p_copy.sum()
            if "sum_crit" in t and t.sum_crit.abs().sum() > 0:
                row["k_crit"] = 1 + t.sum_crit.sum() / n
        probes = DATA / "probes"
        pq = probes / f"{dtr.stem}__probe__{c}.parquet"
        pd_ = probes / f"{dtr.stem}__probe__{dtr.stem.split('__')[0]}.parquet"
        if pq.exists() and pd_.exists():
            a, b = pd.read_parquet(pq), pd.read_parquet(pd_)
            row["probe_loop"] = a.loop.mean() - b.loop.mean()
            row["probe_frac"] = (a.loop > 0.5).mean() - (b.loop > 0.5).mean()
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "k_stop" in res and len(res.dropna(subset=["k_stop"])) >= 4:
        r = res.dropna(subset=["k_stop"])
        print("\nSpearman dos preditores de teacher forcing com kappa, 1-alpha e -delta:")
        for col in [c for c in ("k_stop", "k_hes", "k_copy", "k_crit", "probe_loop", "probe_frac", "tv_tok", "kl_tok")
                    if c in r and r[c].notna().sum() >= 4]:
            print(f"  {col:8s} vs k2 rho={spearmanr(r.k2, r[col]).statistic:+.3f}   "
                  f"vs 1-alpha rho={spearmanr(1 - r.alpha2, r[col]).statistic:+.3f}   "
                  f"vs -delta rho={spearmanr(-r.delta, r[col]).statistic:+.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag != "a1" else f"a2_{dtr.stem}"
    out = RESULTS / f"{tag}_summary.csv"
    res.to_csv(out, index=False)
    curves = pd.DataFrame({"budget": grid, "dense": d.acc(grid)})
    for c in args.configs:
        f = TRACES / f"{c}__{suffix}.parquet"
        if f.exists():
            curves[c.split("__", 1)[1]] = Curve(pd.read_parquet(f), B).acc(grid)
    curves.to_csv(RESULTS / f"{tag}_curves.csv", index=False)
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
