"""Analise do piloto P0: o preditor divergencia x sensibilidade acerta o Delta de acuracia?

Para cada compressor Q:
  medido      Delta = acc(Q) - acc(D), media por problema, IC 95% por bootstrap pareado nos problemas
  barato      prefixos de D + rollouts de D     (so teacher forcing de Q + rollouts do denso)
  exato A     prefixos de D + rollouts de Q     (lema de diferenca de desempenho, sem aproximacao)
  exato B     prefixos de Q + rollouts de D     (idem, na outra direcao)
  baselines   sum TV, sum KL cru, TV/token (nao estao em unidades de acuracia: comparar por ranking)

Estimativa por trace = sum_TV * media_k s_k  (bifurcacoes sorteadas com prob. TV_j / sum TV).

Uso: python scripts/p0_analyze.py --dense-traces <arquivo> --configs qwen3-4b__rtn-w4g128 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dxs.paths import RESULTS, SENS, TF, TRACES

N_BOOT = 2000


def per_problem(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("pid")[col].mean()


def boot_ci(values: dict[str, pd.Series], fn, rng) -> tuple[float, float, float]:
    """fn(dict de series alinhadas por pid) -> escalar; bootstrap reamostrando problemas."""
    pids = sorted(set.intersection(*(set(v.index) for v in values.values())))
    aligned = {k: v.loc[pids].to_numpy() for k, v in values.items()}
    point = fn(aligned)
    idx = rng.integers(0, len(pids), size=(N_BOOT, len(pids)))
    boots = [fn({k: v[i] for k, v in aligned.items()}) for i in idx]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def trace_estimates(sens_file: Path) -> pd.DataFrame | None:
    if not sens_file.exists():
        return None
    s = pd.read_parquet(sens_file)
    g = s.groupby(["pid", "sample"]).agg(sum_tv=("sum_tv", "first"), s_mean=("s", "mean")).reset_index()
    g["est"] = g.sum_tv * g.s_mean
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-traces", required=True)
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--tag", default="p0")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    dtr = Path(args.dense_traces)
    dense = pd.read_parquet(dtr)
    acc_d = per_problem(dense, "correct")
    dname = args.dense.split("/")[-1].lower()
    suffix = dtr.stem.split("__", 1)[1]            # <dataset>-n..-s..-b..
    rows = []
    for c in args.configs:
        row = {"config": c}
        qtr = TRACES / f"{c}__{suffix}.parquet"
        if qtr.exists():
            acc_q = per_problem(pd.read_parquet(qtr), "correct")
            row["acc_q"] = float(acc_q.mean())
            row["delta"], row["delta_lo"], row["delta_hi"] = boot_ci(
                {"q": acc_q.astype(float), "d": acc_d.astype(float)}, lambda a: (a["q"] - a["d"]).mean(), rng)
        tf_d = TF / f"{dtr.stem}__vs__{c}.parquet"
        if tf_d.exists():
            t = pd.read_parquet(tf_d)
            n = t.n_out.sum()
            row.update(sum_tv=float(t.sum_tv.mean()), tv_per_tok=float(t.sum_tv.sum() / n),
                       tvraw_per_tok=float(t.sum_tv_raw.sum() / n), kl_per_tok=float(t.sum_kl_raw.sum() / n),
                       sum_kl=float(t.sum_kl_raw.mean()))
        estimators = {
            "cheap": SENS / f"{dtr.stem}__vs__{c}__roll-{dname}-r{args.rollouts}.parquet",
            "exactA": SENS / f"{dtr.stem}__vs__{c}__roll-{c}-r{args.rollouts}.parquet",
            "exactB": SENS / f"{qtr.stem}__vs__{c}__roll-{dname}-r{args.rollouts}.parquet",
        }
        for name, f in estimators.items():
            est = trace_estimates(f)
            if est is None:
                continue
            pp = per_problem(est, "est")
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = boot_ci({"e": pp}, lambda a: a["e"].mean(), rng)
        rows.append(row)
        print(json.dumps(row, default=float))

    res = pd.DataFrame(rows)
    res.insert(1, "acc_d", float(acc_d.mean()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.tag}_summary.csv"
    res.to_csv(out, index=False)

    print(f"\nacc(D) = {acc_d.mean():.4f}   ({len(acc_d)} problemas)\n")
    show = [c for c in ["config", "acc_q", "delta", "delta_lo", "delta_hi", "cheap", "cheap_lo", "cheap_hi",
                        "exactA", "exactB", "sum_tv", "tv_per_tok", "kl_per_tok"] if c in res]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if "delta" in res and len(res.dropna(subset=["delta"])) >= 3:
        r = res.dropna(subset=["delta"])
        print("\nSpearman com -Delta medido (quanto maior, melhor preve o dano):")
        for col in ["cheap", "exactA", "exactB", "sum_tv", "tv_per_tok", "kl_per_tok", "sum_kl"]:
            if col in r and r[col].notna().sum() >= 3:
                sign = 1 if col in ("cheap", "exactA", "exactB") else -1
                print(f"  {col:12s} rho={spearmanr(-r.delta, -sign * r[col]).statistic:+.3f}")
        for col in ["cheap", "exactA", "exactB"]:
            if col in r and r[col].notna().any():
                err = (r[col] - r.delta).abs()
                print(f"  {col:12s} erro absoluto medio vs medido = {err.mean():.4f}")
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
