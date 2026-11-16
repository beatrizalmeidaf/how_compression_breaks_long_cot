"""Sensibilidade do detector de loop a janela, limiar e frequencia de checagem, e custo da checagem.

Sem gerar nada: aplica o detector causal a cada trace gravado (16k) das 67 variantes da saida antecipada e mede,
por configuracao e regime (denso / leve / severo):
    recall     fracao dos traces cortados em loop (definicao retrospectiva do artigo) em que o detector dispara
    false fire fracao dos traces que terminariam com resposta correta em que o detector dispara antes do fim
               (o pior caso de perda: se a resposta forcada errasse sempre, a acuracia cairia isso)
    saved      fracao dos tokens de raciocinio que deixariam de ser gerados (sem contar a resposta forcada)
    lag        mediana de (disparo - inicio do loop) nos loops detectados
A acuracia com a politica so foi medida na configuracao padrao (scripts/early_exit.py), porque exige gerar.
Custo: tempo de CPU de uma checagem (uma janela), comparado ao tempo de gerar `stride` tokens.

Saidas: results/early_exit_sensitivity.csv, acl/tables/early_exit_sensitivity.tex, acl/tables/early_exit_sensitivity_note.tex
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from dxs.paths import DATA
from loop_hazard import loop_onset

ROOT = Path(__file__).resolve().parents[1]
SEVERE = {"gptq-w3g128", "gptq-w3g64", "wanda-0.5", "rtn-w3g128"}
# (rotulo, janela, periodo maximo, limiar, intervalo entre checagens); a primeira e a do artigo
SETTINGS = [("default", 2048, 1024, 0.90, 256), ("window 1024", 1024, 512, 0.90, 256),
            ("window 4096", 4096, 1024, 0.90, 256), ("threshold 0.80", 2048, 1024, 0.80, 256),
            ("threshold 0.95", 2048, 1024, 0.95, 256), ("every 128", 2048, 1024, 0.90, 128),
            ("every 512", 2048, 1024, 0.90, 512)]


def check(seg, tail, maxp, thr):
    win = sliding_window_view(seg[:-1], tail)
    return (win == seg[maxp:]).mean(1).max() >= thr


def detect(ids, tail, maxp, thr, stride):
    for t in range(tail + maxp, len(ids) + 1, stride):
        if check(ids[t - tail - maxp:t], tail, maxp, thr):
            return t
    return None


def one(path):
    stem = path.name.replace("__exit.parquet", "")
    name, ds = stem.rsplit("__", 1)
    spec = name.split("__")[1] if "__" in name else "dense"
    regime = "dense" if spec == "dense" else ("severe" if spec in SEVERE else "mild")
    d = pd.read_parquet(DATA / "traces" / f"{stem}.parquet", columns=["output_ids", "finish", "correct", "n_out"])
    out = []
    onsets = [loop_onset(np.asarray(o, dtype=np.int64)) if f == "length" else None for o, f in zip(d.output_ids, d.finish)]
    for lab, tail, maxp, thr, stride in SETTINGS:
        for o, f, c, n, on in zip(d.output_ids, d.finish, d.correct, d.n_out, onsets):
            det = detect(np.asarray(o, dtype=np.int64), tail, maxp, thr, stride)
            out.append({"setting": lab, "variant": name, "task": ds, "regime": regime, "loop": on is not None,
                        "ok": f == "stop" and bool(c), "fired": det is not None, "n": n,
                        "kept": n if det is None else min(det, n), "lag": (det - on) if (det and on) else np.nan})
    return out


def overhead():
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 150000, 2048 + 1024)
    t0 = time.perf_counter()
    for _ in range(200):
        check(seg, 2048, 1024, 0.9)
    return (time.perf_counter() - t0) / 200


if __name__ == "__main__":
    files = sorted((DATA / "early_exit").glob("*__exit.parquet"))
    with Pool(min(len(files), 64)) as pool:
        rows = [r for part in pool.map(one, files) for r in part]
    x = pd.DataFrame(rows)
    g = x.groupby(["setting", "regime"])
    s = pd.DataFrame({
        "recall": g.apply(lambda h: h[h.loop].fired.mean()),
        "false_fire": g.apply(lambda h: (h.fired & h.ok).sum() / max(h.ok.sum(), 1)),
        "saved": g.apply(lambda h: 1 - h.kept.sum() / h.n.sum()),
        "lag": g.lag.median()}).reset_index()
    # pior variante (falso disparo maximo por variante, entre denso e leve)
    worst = x[x.regime != "severe"].groupby(["setting", "variant", "task"]).apply(
        lambda h: (h.fired & h.ok).sum() / max(h.ok.sum(), 1)).groupby("setting").max().rename("worst_false_fire")
    s = s.merge(worst, on="setting", how="left")
    s["order"] = s.setting.map({lab: i for i, (lab, *_) in enumerate(SETTINGS)})
    s = s.sort_values(["order", "regime"]).drop(columns="order")
    s.to_csv(ROOT / "results" / "early_exit_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    ms = 1000 * overhead()

    W = s.pivot(index="setting", columns="regime")
    L = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "setting & recall (severe) & saved (severe) & lag (severe) & false fire (dense) & false fire (mild) & "
         "false fire (severe) & worst variant (dense/mild) \\\\", "\\midrule"]
    for lab, *_ in SETTINGS:
        r = W.loc[lab]
        L.append(f"{lab} & {r[('recall', 'severe')]:.3f} & {r[('saved', 'severe')]:.3f} & {r[('lag', 'severe')]:.0f} & "
                 f"{r[('false_fire', 'dense')]:.4f} & {r[('false_fire', 'mild')]:.4f} & {r[('false_fire', 'severe')]:.4f} & "
                 f"{r[('worst_false_fire', 'mild')]:.4f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "acl" / "tables" / "early_exit_sensitivity.tex").write_text("\n".join(L) + "\n")
    nd = s[s.regime != "severe"]
    sev = s[s.regime == "severe"]
    note = (f"Across the {len(SETTINGS)} detector settings, the share of looped traces detected ranges from "
            f"{sev.recall.min():.2f} to {sev.recall.max():.2f}, the share of reasoning tokens saved under severe "
            f"compression from {sev.saved.min():.2f} to {sev.saved.max():.2f}, and the detector fires on at most "
            f"{100 * nd.false_fire.max():.2f}\\% of the dense and mild traces that would have ended correctly "
            f"(at most {100 * s.worst_false_fire.max():.1f}\\% in any single dense or mild variant). One check of the "
            f"default window costs {ms:.1f}\\,ms of CPU time, against 0.7--1.8\\,s to generate the 256 tokens between "
            f"checks for a single request with Qwen3-1.7B to 8B in our setup (Section~\\ref{{sec:failure}})")
    (ROOT / "acl" / "tables" / "early_exit_sensitivity_note.tex").write_text(note + "%\n")
    print(note)
