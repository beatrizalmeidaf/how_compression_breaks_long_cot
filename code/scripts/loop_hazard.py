"""Quando os loops comecam? Analise de sobrevivencia do colapso em loop nos traces de Q.

Loop: a cauda do trace e (quase) periodica nos TOKENS, com periodo p <= 1024: ids[t] == ids[t-p] em
>= 90% dos ultimos 2048 tokens. Inicio do loop: primeira posicao a partir da qual a taxa de
coincidencia em janelas de 64 tokens fica >= 0.8 ate o fim.

Riscos competitivos por trace: terminar (stop), entrar em loop (evento) ou ser cortado sem loop (censura).
Saidas por compressor:
    frac dos cortados que sao loops, mediana do inicio do loop
    risco de loop por 1000 tokens em faixas de posicao (o risco e constante ou cresce com a posicao?)
    previsao sem parametro livre de alpha:  acc_Q(b) ~= sum_{traces de D} 1[kappa*T <= b] * correto * S_loop(kappa*T)
        S_loop = Kaplan-Meier do tempo ate o loop;  kappa = razao das medianas de comprimento dos traces terminados

Uso: python scripts/loop_hazard.py --dense-traces <arquivo> --configs qwen3-4b__rtn-w3g128 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dxs.paths import RESULTS, TRACES

TAIL, MAXP, WIN = 2048, 1024, 64


def loop_onset(ids: np.ndarray) -> int | None:
    """Posicao em que o loop comeca, ou None se a cauda nao e periodica."""
    n = len(ids)
    if n < TAIL + MAXP:
        return None
    tail = ids[-TAIL:]
    best_p, best_r = 0, 0.0
    for p in range(1, MAXP + 1):
        r = float(np.mean(tail == ids[-TAIL - p:-p]))
        if r > best_r + 1e-9:
            best_p, best_r = p, r
        if r >= 0.999:
            break
    if best_r < 0.9:
        return None
    m = (ids[best_p:] == ids[:-best_p]).astype(np.float32)      # m[i] refere-se a posicao i + p
    run = np.convolve(m, np.ones(WIN) / WIN, mode="valid")       # janela que comeca em i
    bad = np.nonzero(run < 0.8)[0]
    start = (bad[-1] + 1 if len(bad) else 0) + best_p
    return int(start)


def km(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sobrevivencia de Kaplan-Meier avaliada em `grid`."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(1 - e / at_risk)
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(t) - 1)], 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-traces", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    dtr = Path(args.dense_traces)
    suffix = dtr.stem.split("__", 1)[1]
    B = int(suffix.rsplit("-b", 1)[1])
    d = pd.read_parquet(dtr)
    d_fin = d[d.finish == "stop"]
    wd = 1.0 / d.groupby("pid").pid.transform("size").to_numpy() / d.pid.nunique()
    grid_b = np.array([2048, 4096, 8192, B])
    bands = [(0, 2000), (2000, 4000), (4000, 8000), (8000, B)]
    rows = []
    for c in args.configs:
        f = TRACES / f"{c}__{suffix}.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f)
        onset = [loop_onset(np.asarray(x)) if fin == "length" else None for x, fin in zip(q.output_ids, q.finish)]
        is_loop = np.array([o is not None for o in onset])
        time = np.where(is_loop, [o if o is not None else 0 for o in onset], q.n_out.to_numpy()).astype(float)
        cut = (q.finish == "length").to_numpy()
        row = {"config": c.split("__", 1)[1], "cortados": cut.mean(), "loops/cortados": is_loop.sum() / max(1, cut.sum()),
               "loop_frac": is_loop.mean(), "inicio_med": float(np.median(time[is_loop])) if is_loop.any() else np.nan}
        for lo, hi in bands:      # risco por 1000 tokens: eventos na faixa / tokens-em-risco na faixa
            exposure = np.clip(time, lo, hi) - lo
            ev = (is_loop & (time >= lo) & (time < hi)).sum()
            row[f"h{lo // 1000}-{hi // 1000}k"] = 1000 * ev / max(1.0, exposure.sum())
        # previsao sem parametro livre da curva de acuracia
        q_fin = q[q.finish == "stop"]
        kappa = q_fin.n_out.median() / d_fin.n_out.median() if len(q_fin) else np.nan
        row["kappa_med"] = kappa
        td = np.where(d.finish.to_numpy() == "stop", d.n_out.to_numpy(float), np.inf) * kappa
        s = km(time, is_loop.astype(float), np.where(np.isfinite(td), td, B))
        ok = d.correct.to_numpy(bool)
        wq = 1.0 / q.groupby("pid").pid.transform("size").to_numpy() / q.pid.nunique()
        tq = np.where(q.finish.to_numpy() == "stop", q.n_out.to_numpy(float), np.inf)
        for b in grid_b:
            row[f"acc@{b}"] = float(((tq <= b) & q.correct.to_numpy(bool)) @ wq)
            row[f"prev@{b}"] = float((((td <= b) & ok) * s) @ wd)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    print(res.round(3).to_string(index=False))
    err = np.concatenate([(res[f"prev@{b}"] - res[f"acc@{b}"]).to_numpy() for b in grid_b])
    print(f"\nerro da previsao sem parametro livre (acc prevista - medida): |medio|={np.abs(err).mean():.3f} max={np.abs(err).max():.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / f"hazard_{dtr.stem}.csv", index=False)


if __name__ == "__main__":
    main()
