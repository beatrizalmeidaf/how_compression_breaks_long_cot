"""Gera paper/tables.md a partir dos dados (nada transcrito a mao). Tabelas com dados ausentes sao puladas.

T1  decomposicao alpha/kappa (todas as configs com traces 16k)
T2  troca de papeis pensador x respondedor (resposta livre e, se existir, resposta curta)
T3  resposta forcada: conhecimento x controle (Qwen3-4B, MATH-500)
T4  temperatura: denso x comprimidos
T5  colapso em loop e risco por posicao
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from a1_budget_tax import Curve, fit_alpha_kappa
from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
T, SW, FO = DATA / "traces", DATA / "swap", DATA / "forced"
B = 16384
GRID = np.unique(np.round(np.geomspace(512, B, 24)))
RUNS = [("qwen3-1.7b", "math500-n300"), ("qwen3-4b", "math500-n300"), ("qwen3-8b", "math500-n300"),
        ("qwen3-14b", "math500-n300"), ("qwen3-4b", "gpqa_diamond-n198"), ("deepseek-r1-distill-llama-8b", "math500-n300"), ("qwen3-4b", "bbh-n300")]
NAMES = {"rtn-w8g128": "RTN-W8", "awq-official": "AWQ-W4", "gptq-w4g128": "GPTQ-W4", "rtn-w4g128": "RTN-W4",
         "rtn-w4g0": "RTN-W4 (per-ch.)", "wanda-0.3": "Wanda 30%", "wanda-0.4": "Wanda 40%", "gptq-w3g64": "GPTQ-W3 (g64)",
         "gptq-w3g128": "GPTQ-W3", "wanda-0.5": "Wanda 50%", "rtn-w3g128": "RTN-W3"}
rng = np.random.default_rng(0)
out = []


TABLES: dict[str, pd.DataFrame] = {}


def md(df: pd.DataFrame, key: str | None = None) -> str:
    if key:
        TABLES[key] = df.copy()
    return df.to_markdown(index=False, floatfmt=".3f")


def acc_pp(f):
    d = pd.read_parquet(f, columns=["pid", "finish", "correct"])
    return ((d.finish == "stop") & d.correct).groupby(d.pid).mean()


# ---------------------------------------------------------------- T1
rows = []
for m, ds in RUNS:
    dtr = T / f"{m}__{ds}-s4-b{B}.parquet"
    if not dtr.exists():
        continue
    d = Curve(pd.read_parquet(dtr), B)
    for spec, name in NAMES.items():
        f = T / f"{m}__{spec}__{ds}-s4-b{B}.parquet"
        if not f.exists():
            continue
        q = Curve(pd.read_parquet(f), B)
        a, k, r = fit_alpha_kappa(d, q, GRID)
        rows.append({"model": m, "task": ds.split("-")[0], "compressor": name, "acc dense": d.acc(B)[0],
                     "acc compr.": q.acc(B)[0], "truncated": 1 - q.finished(B), "alpha": min(max(a, 0), 1.05),
                     "kappa": k if q.acc(B)[0] > 0.02 else np.nan, "RMSE": r})
t1 = pd.DataFrame(rows)
out += ["## T1. Budget decomposition acc_Q(b) = alpha * acc_D(b / kappa)", md(t1, "alpha_kappa"),
        f"\nConfigs: {len(t1)}; RMSE max = {t1.RMSE.max():.3f}, median = {t1.RMSE.median():.3f}\n"]

# ---------------------------------------------------------------- T2
def swap_pp(think, ans, ds, b, short):
    f = SW / f"{think}__{ds}-s4-b{B}__ans-{ans}{'__short' if short else ''}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    return d[d.b == b].groupby("pid").correct.mean()


for short in (False, True):
    rows = []
    for m, ds in RUNS:
        for spec in ["rtn-w4g128", "gptq-w4g128", "wanda-0.3", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]:
            q = f"{m}__{spec}"
            parts = [swap_pp(m, m, ds, 4096, short), swap_pp(q, q, ds, 4096, short), swap_pp(m, q, ds, 4096, short),
                     swap_pp(q, m, ds, 4096, short), swap_pp(m, m, ds, 0, short)]
            if any(p is None for p in parts):
                continue
            x = pd.concat(parts, axis=1).dropna().to_numpy()
            bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
            rec = (bt[:, 2] - bt[:, 1]) / (bt[:, 0] - bt[:, 1])
            gain = bt[:, 3] - bt[:, 4]
            rows.append({"model": m, "task": ds.split("-")[0], "compressor": NAMES[spec], "D->D": x[:, 0].mean(),
                         "Q->Q": x[:, 1].mean(), "D thinks, Q answers": x[:, 2].mean(), "Q thinks, D answers": x[:, 3].mean(),
                         "D, no reasoning": x[:, 4].mean(),
                         "gap closed by D's reasoning": f"{np.median(rec):.2f} [{np.percentile(rec, 2.5):.2f}, {np.percentile(rec, 97.5):.2f}]",
                         "D's gain from Q's reasoning": f"{np.median(gain):+.3f} [{np.percentile(gain, 2.5):+.3f}, {np.percentile(gain, 97.5):+.3f}]"})
    if rows:
        title = "short answer (<= 32 tokens after '\\\\boxed{')" if short else "free answer (<= 2048 tokens)"
        out += [f"## T2{'b' if short else 'a'}. Thinker x answerer swap, 4096 reasoning tokens, {title}", md(pd.DataFrame(rows), f"swap_{'short' if short else 'free'}"), ""]

# ---------------------------------------------------------------- T2c
# decomposicao escrita x leitura (resposta curta: o leitor nao consegue resolver de novo)
rows = []
for m, ds in RUNS:
    for spec in ["rtn-w4g128", "gptq-w4g128", "wanda-0.3", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]:
        q = f"{m}__{spec}"
        parts = [swap_pp(m, m, ds, 4096, True), swap_pp(q, m, ds, 4096, True), swap_pp(m, q, ds, 4096, True),
                 swap_pp(m, m, ds, 0, True)]
        if any(p is None for p in parts):
            continue
        x = pd.concat(parts, axis=1).dropna().to_numpy()
        bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
        write, read, info = bt[:, 0] - bt[:, 1], bt[:, 0] - bt[:, 2], bt[:, 1] - bt[:, 3]
        ci = lambda v: f"{np.median(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
        dwr = write - read
        p_boot = min(1.0, 2 * min((dwr <= 0).mean(), (dwr >= 0).mean()))
        rows.append({"model": m, "task": ds.split("-")[0], "compressor": NAMES[spec], "p_boot": p_boot,
                     "writing damage (D reads)": ci(write), "reading damage (D's trace)": ci(read),
                     "writing - reading": ci(write - read),
                     "info left in Q's trace (vs no reasoning)": ci(info),
                     "share of D's trace info": f"{np.median(info / (bt[:, 0] - bt[:, 3])):.2f}"})
if rows:
    t2c = pd.DataFrame(rows)
    n_sig = sum(float(v.split("[")[1].split(",")[0]) > 0 for v in t2c["writing - reading"])
    # Holm (p do bootstrap tem resolucao 1/2000 -> p = 0 vira p < 0.001)
    order = np.argsort(t2c.p_boot.to_numpy())
    holm_ok = np.zeros(len(t2c), bool)
    for rank, i in enumerate(order):
        if max(t2c.p_boot.iloc[i], 0.0005) * (len(t2c) - rank) <= 0.05 and float(t2c["writing - reading"].iloc[i].split()[0]) > 0:
            holm_ok[i] = True
        else:
            break
    (ROOT / "acl" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "acl" / "tables" / "holm_note.tex").write_text(f"{int(holm_ok.sum())}/{len(t2c)}")
    print(f"[holm] escrita > leitura apos Holm: {int(holm_ok.sum())}/{len(t2c)}")
    t2c = t2c.drop(columns=["p_boot"])
    out += ["## T2c. Writing vs reading damage (short answer, 4096 reasoning tokens)", md(t2c, "write_read"),
            f"\nwriting damage > reading damage with CI excluding 0: {n_sig}/{len(t2c)} combinations\n"]

# ---------------------------------------------------------------- T2d
lk = ROOT / "results" / "leakage_check.csv"
if lk.exists():
    t = pd.read_csv(lk)
    t = t.rename(columns={"subset": "traces", "escrita": "writing damage", "leitura": "reading damage",
                          "escrita-leitura IC": "writing - reading 95% CI", "frac traces": "share of traces"})
    t["traces"] = t["traces"].map({"todos": "all", "sem vazamento": "no leakage"})
    out += ["## T2d. Writing vs reading damage by reasoning length, all traces vs traces whose dense prefix does not yet "
            "contain the answer (short answer; leakage = prefix contains \\boxed or the gold string; "
            "produced by scripts/leakage_check.py)", md(t, "leakage"), ""]

# ---------------------------------------------------------------- T3
rows = []
for spec in ["", "rtn-w4g128", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]:
    n = "qwen3-4b" + (f"__{spec}" if spec else "")
    f = FO / f"{n}__math500-n300-s4-b{B}__forced.parquet"
    if not f.exists():
        continue
    fo = pd.read_parquet(f)
    cu = Curve(pd.read_parquet(T / f"{n}__math500-n300-s4-b{B}.parquet"), B)
    r = {"model": NAMES.get(spec, "dense")}
    for b in (1024, 4096, 16384):
        r[f"natural@{b // 1024}k"] = cu.acc(b)[0]
        r[f"forced@{b // 1024}k"] = fo[fo.b == b].groupby("pid").correct.mean().mean()
    rows.append(r)
if rows:
    out += ["## T3. Forced answers (Qwen3-4B, MATH-500): natural vs forced accuracy at reasoning budget b", md(pd.DataFrame(rows), "forced"), ""]

# ---------------------------------------------------------------- T4
rows = []
for spec in ["", "gptq-w3g128", "wanda-0.5", "rtn-w3g128"]:
    n = "qwen3-4b" + (f"__{spec}" if spec else "")
    r = {"model": NAMES.get(spec, "dense")}
    for tag, lab in (("+t0.3", "T=0.3"), ("", "T=0.6"), ("+t1", "T=1.0"), ("+t1.3", "T=1.3")):
        f = T / f"{n}{tag}__math500-n300-s4-b{B}.parquet"
        r[lab] = acc_pp(f).mean() if f.exists() else np.nan
    rows.append(r)
out += ["## T4. Sampling temperature (Qwen3-4B, MATH-500, acc@16k)", md(pd.DataFrame(rows), "temperature"), ""]

# ---------------------------------------------------------------- T5
hz = sorted(glob.glob(str(ROOT / "results" / "hazard_qwen3-*.csv")))
rows = []
for f in hz:
    h = pd.read_csv(f)
    run = Path(f).stem[7:].split("-n")[0]
    for _, r in h.iterrows():
        if r["loop_frac"] >= 0.1:
            rows.append({"run": run, "compressor": NAMES.get(r.config, r.config), "truncated": r.cortados,
                         "loops among truncated": r["loops/cortados"], "median loop onset": r.inicio_med,
                         "hazard 0-2k": r["h0-2k"], "2-4k": r["h2-4k"], "4-8k": r["h4-8k"], "8-16k": r["h8-16k"]})
if rows:
    out += ["## T5. Loop collapse: loops per 1000 tokens by position band (configs with >= 10% looping traces)",
            md(pd.DataFrame(rows), "loops"), ""]

# ---------------------------------------------------------------- T6
# ablacao causal: os mesmos ~10% de tokens escritos pelo denso, em pedacos de 50/100/200/400 tokens
rows = []
dense_pp = acc_pp(T / "qwen3-4b__math500-n300-s4-b16384.parquet")
for spec in ["rtn-w3g128", "wanda-0.5", "gptq-w3g128"]:
    q_pp = acc_pp(T / f"qwen3-4b__{spec}__math500-n300-s4-b16384.parquet")
    mixes = {}
    for chunk, sched in ((50, "per50-450"), (100, "per100-900"), (200, "per200-1800"), (400, "per400-3600")):
        f = T / f"qwen3-4b__{spec}+mix-{sched}__math500-n300-s2-b16384.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["pid", "finish", "correct", "dense_tokens", "n_out"])
            mixes[chunk] = (((d.finish == "stop") & d.correct).groupby(d.pid).mean(), d.dense_tokens.sum() / d.n_out.sum())
    if len(mixes) < 4:
        continue
    r = {"compressor": NAMES[spec], "Q alone": q_pp.mean(), "dense": dense_pp.mean()}
    for chunk, (m, frac) in mixes.items():
        r[f"chunk {chunk}"] = m.mean()
        r[f"dense share ({chunk})"] = frac
    x = pd.concat([mixes[50][0], mixes[400][0]], axis=1).dropna().to_numpy()
    bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    dlt = bt[:, 0] - bt[:, 1]
    r["chunk 50 - chunk 400"] = f"{np.median(dlt):+.3f} [{np.percentile(dlt, 2.5):+.3f}, {np.percentile(dlt, 97.5):+.3f}]"
    rows.append(r)
if rows:
    out += ["## T6. Causal ablation: ~10% of tokens written by the dense model, in chunks of 50/100/200/400 tokens "
            "(Qwen3-4B, MATH-500, 2 samples/problem)", md(pd.DataFrame(rows), "ablation"), ""]

(ROOT / "paper").mkdir(exist_ok=True)
(ROOT / "paper" / "tables.md").write_text("# Tables (generated by scripts/paper_tables.py)\n\n" + "\n".join(out))
print("\n".join(out))


# ---------------------------------------------------------------- LaTeX (acl/tables/*.tex)
MODEL_TEX = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
             "deepseek-r1-distill-llama-8b": "R1-Llama-8B",
             "qwen3-0.6b": "Qwen3-0.6B", "deepseek-r1-distill-qwen-1.5b": "R1-Qwen-1.5B"}
TASK_TEX = {"math500": "MATH", "gpqa_diamond": "GPQA", "bbh": "BBH"}
SEVERE = {"GPTQ-W3", "GPTQ-W3 (g64)", "Wanda 50%", "RTN-W3"}   # 3 bits ou 50% esparso


def tidy(df):
    df = df.copy()
    for c in ("model", "run"):
        if c in df:
            df[c] = df[c].map(lambda v: MODEL_TEX.get(str(v).split("/")[0].split("__")[0], v))
    if "task" in df:
        df["task"] = df["task"].map(lambda v: TASK_TEX.get(v, v))
    return df


HEAD = {"acc@16k": "$\\acc$ (16k)", "truncated": "trunc.", "errors not finished": "non-term./err.",
        "errors in loops": "loop/err.", "errors long without loop": "long/err.", "alpha": "$\\alpha$", "kappa": "$\\kappa$",
        "alpha (mild)": "$\\alpha$ (mild)", "kappa (mild)": "$\\kappa$ (mild)", "alpha (severe)": "$\\alpha$ (severe)",
        "alpha 95\\% CI": "$\\alpha$ 95\\% CI", "kappa 95\\% CI": "$\\kappa$ 95\\% CI", "acc dense": "$\\acc_D$",
        "acc compr.": "$\\acc_Q$", "acc parent": "$\\acc$ parent", "acc smaller": "$\\acc$ smaller",
        **{f"{k}@{b}": f"{k} {b}" for k in ("natural", "forced") for b in ("1k", "4k", "16k")}}


def to_tex(df, name, fmt="%.2f"):
    tex = tidy(df).to_latex(index=False, escape=True, float_format=lambda v: "" if pd.isna(v) else fmt % v, na_rep="")
    tex = tex.replace("nan", "--")
    head, sep, body = tex.partition("\\midrule")   # so o cabecalho recebe simbolos
    lines = head.split("\n")
    for i, ln in enumerate(lines):
        if ln.endswith("\\\\") and "&" in ln:
            cells = [c.strip() for c in ln[:-2].split("&")]
            lines[i] = " & ".join(HEAD.get(c, c) for c in cells) + " \\\\"
    tex = "\n".join(lines) + sep + body
    (ROOT / "acl" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "acl" / "tables" / f"{name}.tex").write_text(tex)


if "alpha_kappa" in TABLES:
    t = TABLES["alpha_kappa"].copy()
    full = t.drop(columns=["truncated"]).copy()
    fvp_ = ROOT / "results" / "fit_validation.csv"
    if fvp_.exists():   # intervalos de bootstrap de alpha e kappa (scripts/fit_validation.py)
        fv_ = pd.read_csv(fvp_)
        fv_["model"] = fv_["model"]
        fv_["compressor"] = fv_.compressor.map(lambda v: NAMES.get(v, v))
        fv_["task"] = fv_["task"]
        # fonte unica: valores pontuais e ICs do mesmo ajuste; kappa vazio quando nao identificavel (alpha < 0.3)
        full = full.drop(columns=["alpha", "kappa", "RMSE"]).merge(
            fv_[["model", "task", "compressor", "alpha", "kappa", "rmse_dilation", "alpha_lo", "alpha_hi", "kappa_lo", "kappa_hi"]]
            .rename(columns={"rmse_dilation": "RMSE"}), on=["model", "task", "compressor"], how="left")
        t = t.drop(columns=["alpha", "kappa", "RMSE"]).merge(full[["model", "task", "compressor", "alpha", "kappa", "RMSE"]],
                                                             on=["model", "task", "compressor"], how="left")
        full["alpha 95% CI"] = full.apply(lambda r: f"[{r.alpha_lo:.2f}, {r.alpha_hi:.2f}]" if pd.notna(r.alpha_lo) else "", axis=1)
        full["kappa 95% CI"] = full.apply(lambda r: f"[{r.kappa_lo:.2f}, {r.kappa_hi:.2f}]" if pd.notna(r.kappa_lo) and pd.notna(r.kappa) else "", axis=1)
        full = full.drop(columns=["alpha_lo", "alpha_hi", "kappa_lo", "kappa_hi"])
        full = full[["model", "task", "compressor", "acc dense", "acc compr.", "alpha", "alpha 95% CI", "kappa", "kappa 95% CI", "RMSE"]]
    to_tex(full, "alpha_kappa_full", "%.3f")
    t["regime"] = np.where(t.compressor.isin(SEVERE), "severe", "mild")   # definicao a priori: 3 bits ou 50% esparso
    rows = []
    for (m, task), g in t.groupby(["model", "task"], sort=False):
        mi, se = g[g.regime == "mild"], g[g.regime == "severe"]
        rng_ = lambda v: f"{v.min():.2f}--{v.max():.2f}" if len(v) else "--"
        rows.append({"model": m, "task": task, "n": len(g), "alpha (mild)": rng_(mi.alpha), "kappa (mild)": rng_(mi.kappa.dropna()),
                     "alpha (severe)": rng_(se.alpha), "max RMSE": g.RMSE.max()})
    to_tex(pd.DataFrame(rows), "alpha_kappa_summary", "%.3f")
if "write_read" in TABLES:
    w = TABLES["write_read"][["model", "task", "compressor", "writing damage (D reads)", "reading damage (D's trace)",
                              "writing - reading"]].copy()
    w.columns = ["model", "task", "compressor", "writing damage", "reading damage", "writing $-$ reading"]
    tex = tidy(w).to_latex(index=False, escape=True)
    tex = tex.replace("writing \\$-\\$ reading", "writing $-$ reading")
    (ROOT / "acl" / "tables" / "write_read.tex").write_text(tex)
for key, name in (("forced", "forced"), ("temperature", "temperature"), ("leakage", "leakage"), ("loops", "loops"),
                  ("ablation", "ablation"), ("swap_free", "swap_free")):
    if key in TABLES:
        df = TABLES[key]
        if key == "leakage":
            df = df[["run", "compressor", "b", "traces", "share of traces", "writing damage", "reading damage", "writing - reading 95% CI"]]
        if key == "ablation":
            df = df[[c for c in df.columns if not c.startswith("dense share")]]
        if key == "swap_free":
            df = df[["model", "task", "compressor", "D->D", "Q->Q", "D thinks, Q answers", "Q thinks, D answers", "D, no reasoning"]]
        to_tex(df, name)
print("[latex]", sorted(p.name for p in (ROOT / "acl" / "tables").glob("*.tex")))

# ---------------------------------------------------------------- validacao do ajuste e controle de dilatacao
fvp = ROOT / "results" / "fit_validation.csv"
if fvp.exists():
    fv = pd.read_csv(fvp)
    fams = [("dilation", "dilation $\\alpha\\,\\mathrm{acc}_D(b/\\kappa)$"), ("power", "power $\\alpha\\,\\mathrm{acc}_D(b)^\\gamma$"),
            ("shift", "shift $\\alpha\\,\\mathrm{acc}_D(b-\\delta)$"), ("affine", "affine $a\\,\\mathrm{acc}_D(b)+c$")]
    best_in = fv[[f"rmse_{f}" for f, _ in fams]].idxmin(axis=1).str.replace("rmse_", "").value_counts()
    best_out = fv[[f"oos_{f}" for f, _ in fams]].idxmin(axis=1).str.replace("oos_", "").value_counts()
    rows = [{"family": lab, "median RMSE": fv[f"rmse_{f}"].median(), "max RMSE": fv[f"rmse_{f}"].max(),
             "median OOS": fv[f"oos_{f}"].median(), "max OOS": fv[f"oos_{f}"].max(),
             "best (in / out)": f"{best_in.get(f, 0)} / {best_out.get(f, 0)}"} for f, lab in fams]
    tex = pd.DataFrame(rows).to_latex(index=False, escape=False, float_format="%.3f")
    (ROOT / "acl" / "tables" / "fit_validation.tex").write_text(tex)
    idf = fv[fv.kappa_identified] if "kappa_identified" in fv else fv
    n_k = int(((idf.kappa_lo > 1) | (idf.kappa_hi < 1)).sum())
    (ROOT / "acl" / "tables" / "fit_validation_note.tex").write_text(f"{n_k} of the {len(idf)}")
    print(f"[fit] kappa identified in {len(idf)}/{len(fv)}; CI excludes 1 in {n_k}/{len(idf)}")
dcp = ROOT / "results" / "dilation_control.csv"
if dcp.exists():
    dc = pd.read_csv(dcp)
    dc = dc[["model", "task", "compressor", "kappa", "writing damage (raw)", "writing damage (dilation-matched)", "reading damage"]]
    dc["compressor"] = dc.compressor.map(lambda v: NAMES.get(v, v))
    to_tex(dc, "dilation_control")
dpp = ROOT / "results" / "dense_pairs.csv"
if dpp.exists():
    to_tex(pd.read_csv(dpp), "dense_pairs")
print("[latex]", sorted(p.name for p in (ROOT / "acl" / "tables").glob("*.tex")))
fmp = ROOT / "results" / "failure_mode_dense_vs_compressed.csv"
if fmp.exists():
    fm = pd.read_csv(fmp)
    def lab(n):
        m, _, s = n.partition("__")
        return MODEL_TEX.get(m, m) + (f" + {NAMES.get(s, s)}" if s else " (dense)")
    fm["model"] = fm.model.map(lab)
    to_tex(fm, "failure_mode", "%.3f")
    more = [ROOT / "results" / f"failure_mode_dense_vs_compressed_{t}.csv" for t in ("gpqa_diamond", "bbh", "r1distill")]
    more = [pd.read_csv(f) for f in more if f.exists()]
    if more:   # GPQA/BBH (Qwen3) e familia R1-Distill no MATH
        fx = pd.concat(more)
        fx["task"] = fx.task.map(lambda v: TASK_TEX.get(v, v))
        fx["model"] = fx.model.map(lab)
        to_tex(fx, "failure_mode_more", "%.3f")
dpp = ROOT / "results" / "dense_pairs.csv"
if dpp.exists():
    dp = pd.read_csv(dpp)
    dp["weak"] = dp.weak.map(lambda n: MODEL_TEX.get(n.split("__")[0], n) + (f" + {NAMES.get(n.split('__')[1], n.split('__')[1])}" if "__" in n else " (dense)"))
    dp["strong"] = dp.strong.map(lambda n: MODEL_TEX.get(n, n))
    dp = dp.drop(columns=["kind", "reading / writing"]).rename(columns={"weak alone": "weak alone (acc)", "strong alone": "strong alone (acc)"})
    to_tex(dp, "dense_pairs")
print("[latex+]", sorted(p.name for p in (ROOT / "acl" / "tables").glob("*.tex")))
# efeito pareado da compressao no modo de falha (scripts/failure_paired.py)
fpp = ROOT / "results" / "failure_paired.csv"
if fpp.exists():
    fp = pd.read_csv(fpp)
    fp["compressor"] = fp.compressor.map(lambda v: NAMES.get(v, v))
    fp["task"] = fp.task.map(lambda v: TASK_TEX.get(v, v))
    fp = fp[["model", "task", "compressor", "acc parent", "acc compr.", "delta non-term.", "delta wrong-term.", "delta share"]]
    fp.columns = ["model", "task", "compressor", "acc parent", "acc compr.", "$\\Delta$ non-term.", "$\\Delta$ wrong-term.",
                  "$\\Delta$ non-term./err."]
    tex = tidy(fp).to_latex(index=False, escape=True, float_format=lambda v: "%.3f" % v)
    for c in ("non-term.", "wrong-term.", "non-term./err."):
        tex = tex.replace(f"\\$\\textbackslash Delta\\$ {c}", f"$\\Delta$ {c}")
    tex = tex.replace("acc parent", "$\\acc$ parent").replace("acc compr.", "$\\acc_Q$")
    (ROOT / "acl" / "tables" / "failure_paired.tex").write_text(tex)
    sev = fp.compressor.isin(SEVERE)
    up_nt = pd.read_csv(fpp)["non-term. up (CI > 0)"]
    up_sh = pd.read_csv(fpp)["share up (CI > 0)"]
    (ROOT / "acl" / "tables" / "failure_paired_note.tex").write_text(
        f"{int(up_nt[sev].sum())} of {int(sev.sum())}")
    print(f"[paired] severe: non-term up {int(up_nt[sev].sum())}/{int(sev.sum())}, share up {int(up_sh[sev].sum())}/{int(sev.sum())}; "
          f"mild: non-term up {int(up_nt[~sev].sum())}/{int((~sev).sum())}, share up {int(up_sh[~sev].sum())}/{int((~sev).sum())}")
fmm = ROOT / "results" / "failure_matched.csv"
if fmm.exists():
    fm2 = pd.read_csv(fmm)
    fm2["task"] = fm2.task.map(lambda v: TASK_TEX.get(v, v))
    def lab2(n):
        m, _, s = n.partition("__")
        return MODEL_TEX.get(m, m) + (f" + {NAMES.get(s, s)}" if s else "")
    fm2["smaller dense"] = fm2["smaller dense"].map(lab2)
    fm2["compressed"] = fm2["compressed"].map(lab2)
    fm2 = fm2[["task", "smaller dense", "compressed", "acc smaller", "acc compr.", "share smaller", "share compr.", "delta share"]]
    fm2.columns = ["task", "dense", "compressed", "acc smaller", "acc compr.", "non-term./err. (dense)",
                   "non-term./err. (compr.)", "difference"]
    to_tex(fm2, "failure_matched", "%.3f")
