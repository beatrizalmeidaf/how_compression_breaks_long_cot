"""Figuras do paper (PDF vetorial em acl/figures/), geradas direto dos dados.

F1  curvas de acuracia por orcamento + ajuste alpha*acc_D(b/kappa)   (Qwen3-4B, MATH-500)
F2  dano de escrita x dano de leitura, 24 combinacoes (resposta curta, 4k tokens)
F3  risco de loop por faixa de posicao (configs colapsadas)            [apendice]
F4  ablacao causal: frequencia das correcoes do denso                  [apendice]

Paleta: slots 1-3 da paleta de referencia (validada para dispersao, todos os pares);
identidade nunca so por cor: marcadores e rotulos diretos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a1_budget_tax import Curve, fit_alpha_kappa
from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "acl" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T, SW = DATA / "traces", DATA / "swap"
B = 16384
GRID = np.unique(np.round(np.geomspace(512, B, 24)))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"gptq-w4g128": "GPTQ-W4", "rtn-w4g128": "RTN-W4", "wanda-0.3": "Wanda 30%", "gptq-w3g128": "GPTQ-W3",
         "wanda-0.5": "Wanda 50%", "rtn-w3g128": "RTN-W3"}
COL = 3.25   # largura de uma coluna ACL (pol.)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42,
})
rng = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok]", OUT / f"{name}.pdf")


# ---------------------------------------------------------------- F1
def fig_budget():
    d = Curve(pd.read_parquet(T / "qwen3-4b__math500-n300-s4-b16384.parquet"), B)
    specs = ["gptq-w4g128", "gptq-w3g128", "wanda-0.5"]
    fig, axes = plt.subplots(1, 3, figsize=(2 * COL + 0.3, 1.9), sharey=True)
    xs = np.unique(np.round(np.geomspace(512, B, 60)))
    for ax, s in zip(axes, specs):
        q = Curve(pd.read_parquet(T / f"qwen3-4b__{s}__math500-n300-s4-b16384.parquet"), B)
        a, k, rmse = fit_alpha_kappa(d, q, GRID)
        ax.plot(xs, d.acc(xs), color=MUTED, lw=1.4, label="dense")
        ax.plot(xs, q.acc(xs), color=BLUE, lw=1.6, label="compressed (measured)")
        xf = xs[xs / k <= B]
        ax.plot(xf, a * d.acc(xf / k), color=ORANGE, lw=1.4, ls=(0, (4, 2)), label=r"fit $\alpha\,\mathrm{acc}_D(b/\kappa)$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192, 16384], ["1k", "2k", "4k", "8k", "16k"])
        ax.set_title(f"{NAMES[s]}:  $\\alpha$={a:.2f}, $\\kappa$={k:.2f}", color=INK)
        ax.set_xlabel("reasoning budget $b$ (tokens)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy at budget $b$")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "budget_curves")


# ---------------------------------------------------------------- F2
def swap_pp(think, ans, ds, b):
    f = SW / f"{think}__{ds}-s4-b16384__ans-{ans}__short.parquet"
    if not f.exists():
        return None
    x = pd.read_parquet(f)
    return x[x.b == b].groupby("pid").correct.mean()


def fig_write_read():
    runs = [("qwen3-1.7b", "math500-n300", "MATH"), ("qwen3-4b", "math500-n300", "MATH"), ("qwen3-8b", "math500-n300", "MATH"),
            ("deepseek-r1-distill-llama-8b", "math500-n300", "MATH"), ("qwen3-4b", "gpqa_diamond-n198", "GPQA"),
            ("qwen3-4b", "bbh-n300", "BBH")]
    rows = []
    for m, ds, task in runs:
        for s in NAMES:
            q = f"{m}__{s}"
            parts = [swap_pp(m, m, ds, 4096), swap_pp(q, m, ds, 4096), swap_pp(m, q, ds, 4096)]
            if any(p is None for p in parts):
                continue
            x = pd.concat(parts, axis=1).dropna().to_numpy()
            bt = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
            w, r = bt[:, 0] - bt[:, 1], bt[:, 0] - bt[:, 2]
            severe = s in ("gptq-w3g128", "wanda-0.5", "rtn-w3g128")   # a priori: 3 bits ou 50% esparso
            rows.append(dict(task=task, severe=severe, w=np.median(w), wlo=np.percentile(w, 2.5), whi=np.percentile(w, 97.5),
                             r=np.median(r), rlo=np.percentile(r, 2.5), rhi=np.percentile(r, 97.5)))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    lim = (-0.15, 0.72)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.40, 0.33, "writing = reading", rotation=38, color=MUTED, fontsize=6.5, ha="center")
    markers = {"MATH": "o", "GPQA": "^", "BBH": "s"}
    for sev, color, lab in ((True, BLUE, "severe (3-bit / 50% sparse)"), (False, ORANGE, "mild (4-bit / 30% sparse)")):
        for task, mk in markers.items():
            g = t[(t.severe == sev) & (t.task == task)]
            if g.empty:
                continue
            ax.errorbar(g.r, g.w, xerr=[g.r - g.rlo, g.rhi - g.r], yerr=[g.w - g.wlo, g.whi - g.w], fmt="none",
                        ecolor=color, elinewidth=0.6, alpha=0.5)
            ax.scatter(g.r, g.w, marker=mk, s=26, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="severe (3-bit / 50% sparse)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="mild (4-bit / 30% sparse)")] + \
              [Line2D([], [], marker=mk, ls="", color=MUTED, label=task) for task, mk in markers.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=1, handletextpad=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("reading damage  (dense writes; compressed reads)")
    ax.set_ylabel("writing damage  (compressed writes; dense reads)")
    fig.tight_layout()
    save(fig, "write_read")
    return t


# ---------------------------------------------------------------- F3
def fig_hazard():
    sel = [("qwen3-4b__math500", "wanda-0.5", "Qwen3-4B, Wanda 50%, MATH", BLUE, "o"),
           ("qwen3-4b__math500", "rtn-w3g128", "Qwen3-4B, RTN-W3, MATH", ORANGE, "s"),
           ("qwen3-4b__gpqa_diamond", "wanda-0.5", "Qwen3-4B, Wanda 50%, GPQA", AQUA, "^")]
    bands = ["h0-2k", "h2-4k", "h4-8k", "h8-16k"]
    mids = [1000, 3000, 6000, 12000]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for run, spec, lab, color, mk in sel:
        f = ROOT / "results" / f"hazard_{run}-n{'300' if 'math' in run else '198'}-s4-b16384.csv"
        h = pd.read_csv(f)
        r = h[h.config == spec].iloc[0]
        ax.plot(mids, [r[b] for b in bands], color=color, marker=mk, ms=4, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(mids, ["0–2k", "2–4k", "4–8k", "8–16k"])
    ax.set_xlabel("position in the reasoning (tokens)")
    ax.set_ylabel("loop onsets per 1k tokens at risk")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "loop_hazard")


# ---------------------------------------------------------------- F4
def fig_ablation():
    def pp(f):
        x = pd.read_parquet(f, columns=["pid", "finish", "correct"])
        return ((x.finish == "stop") & x.correct).groupby(x.pid).mean().mean()
    chunks = [50, 100, 200, 400]
    sched = {50: "per50-450", 100: "per100-900", 200: "per200-1800", 400: "per400-3600"}
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for s, color, mk in (("rtn-w3g128", BLUE, "o"), ("wanda-0.5", ORANGE, "s"), ("gptq-w3g128", AQUA, "^")):
        ys = [pp(T / f"qwen3-4b__{s}+mix-{sched[c]}__math500-n300-s2-b16384.parquet") for c in chunks]
        ax.plot(chunks, ys, color=color, marker=mk, ms=4, label=NAMES[s])
        ax.text(chunks[-1] * 1.08, ys[-1], NAMES[s], color=INK, fontsize=6.5, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(c) for c in chunks])
    ax.set_xlim(40, 700)
    ax.set_xlabel("size of each dense-written chunk (tokens; 10% of all tokens)")
    ax.set_ylabel("accuracy (MATH-500)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3)
    fig.tight_layout()
    save(fig, "ablation")


# ---------------------------------------------------------------- F0 (sintese 2x2)
def fig_matrix():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cases = [("qwen3-4b", "math500-n300", "rtn-w3g128", "Qwen3-4B, MATH"),
             ("deepseek-r1-distill-llama-8b", "math500-n300", "rtn-w3g128", "R1-Llama-8B, MATH"),
             ("qwen3-4b", "bbh-n300", "rtn-w3g128", "Qwen3-4B, BBH")]
    fig, axes = plt.subplots(1, 3, figsize=(2 * COL + 0.3, 1.75))
    for ax, (m, ds, s, title) in zip(axes, cases):
        q = f"{m}__{s}"
        val = lambda w, r: swap_pp(w, r, ds, 4096).mean()
        M = np.array([[val(m, m), val(m, q)], [val(q, m), val(q, q)]])
        base = swap_pp(m, m, ds, 0).mean()
        ax.imshow(M, cmap=seq, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else INK, fontweight="bold")
        ax.set_xticks([0, 1], ["dense", "RTN-W3"])
        ax.set_yticks([0, 1], ["dense", "RTN-W3"])
        ax.set_xlabel("reader")
        ax.set_ylabel("writer")
        ax.set_title(f"{title}\n(dense, no reasoning: {base:.2f})", color=INK)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=1.2)
    save(fig, "matrix")


# ---------------------------------------------------------------- F5
def fig_early_exit():
    """Saida antecipada na deteccao de loop: ganho de acuracia x tokens economizados, por variante."""
    from matplotlib.lines import Line2D
    r = pd.read_csv(ROOT / "results" / "early_exit.csv")
    r["saved"] = 100 * (1 - r["token ratio"])
    markers = {"math500": ("o", "MATH"), "gpqa_diamond": ("s", "GPQA"), "bbh": ("^", "BBH")}
    regimes = (("dense", MUTED, "dense"), ("mild", ORANGE, "mild compression"), ("severe", BLUE, "severe compression"))
    fig, ax = plt.subplots(figsize=(1.45 * COL, 2.5))
    ax.axhline(0, color=MUTED, lw=0.8)
    for reg, color, _ in regimes:
        for task, (mk, _) in markers.items():
            g = r[(r.regime == reg) & (r.task == task)]
            if not len(g):
                continue
            ax.errorbar(g.saved, g.d_acc, yerr=[g.d_acc - g.d_acc_lo, g.d_acc_hi - g.d_acc], fmt="none",
                        ecolor=color, elinewidth=0.6, alpha=0.5)
            ax.scatter(g.saved, g.d_acc, marker=mk, s=24, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    short = {"qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3-14b": "14B", "qwen3-32b": "32B",
             "deepseek-r1-distill-llama-8b": "R1-8B"}
    task_s = {"math500": "", "gpqa_diamond": " GPQA", "bbh": " BBH"}
    left = {"8B Wanda 50%", "8B RTN-W3"}                       # rotulo a esquerda do ponto (evita colisao)
    nudge = {"1.7B RTN-W4": 4, "4B Wanda 50%": -4}               # deslocamento vertical (pt)
    for _, x in r[r.d_acc >= 0.1].iterrows():   # rotulos diretos so nos maiores ganhos
        name = f"{short.get(x.model, x.model)} {NAMES.get(x.compressor, x.compressor)}{task_s[x.task]}"
        if x.compressor == "rtn-w4g128" and x.model == "qwen3-1.7b":
            name += " (collapsed)"
        dx = -4 if name in left else 4
        ha = "right" if dx < 0 else "left"
        dy = nudge.get(name.split(" (")[0], 0)
        if "(collapsed)" in name:   # acima do ponto, centrado
            dx, dy, ha = 0, 11, "center"
        ax.annotate(name, (x.saved, x.d_acc), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, color=INK, va="center", ha=ha)
    dead = r[(r["acc natural"] < 0.01) & (r.saved > 50)]
    if len(dead):
        ax.annotate("Qwen3-1.7B, 3-bit / 50% sparse\n(accuracy 0 with or without exit)", (dead.saved.mean(), 0),
                    xytext=(0, 14), textcoords="offset points", fontsize=6, color=MUTED, ha="center")
    ax.set_xlabel("tokens saved by early exit (%)")
    ax.set_ylabel(r"$\Delta$ accuracy (early exit $-$ natural)")
    ax.set_xlim(-3, 100)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=lab) for _, c, lab in regimes] + \
              [Line2D([], [], marker=mk, ls="", color=MUTED, label=lab) for mk, lab in markers.values()]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "early_exit")


if __name__ == "__main__":
    if len(sys.argv) > 1:   # ex.: python scripts/paper_figures.py early_exit
        for name in sys.argv[1:]:
            globals()[f"fig_{name}"]()
        raise SystemExit
    fig_matrix()
    fig_budget()
    fig_write_read()
    fig_hazard()
    fig_ablation()
    fig_early_exit()
