"""Intervencao: o denso (D) escreve parte dos tokens do raciocinio do comprimido (Q).

Esquemas (--schedule):
    warm<W>          D escreve os primeiros W tokens; depois so Q (ex.: warm1024)
    per<Ld>/<Lq>     ciclos: Q escreve Lq tokens, D escreve Ld tokens (ex.: per100/900 = 10% denso)
    qfirst ou dfirst no fim do periodico escolhe quem comeca (padrao: Q comeca)

Os dois modelos ficam carregados ao mesmo tempo (dois motores vLLM na mesma GPU). A cada rodada,
todos os traces ativos geram o proximo trecho com o modelo da vez; o outro motor reaproveita o
prefixo pelo prefix caching. Orcamento total de --budget tokens, somando D e Q. A saida tem o formato de
traces (finish/n_out/correct) para reusar a analise de orcamento, mais `dense_tokens`.

Uso: python scripts/mix_decode.py --dense Qwen/Qwen3-4B --comp qwen3-4b__rtn-w3g128 --schedule per100/900
Saida: DXS_DATA/traces/<comp>+mix-<esquema>__<dataset>-n<N>-s<S>-b<B>.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dxs.data import chat_prompt, final_answer, is_correct, load_problems
from dxs.engine import make_llm, sampling_params
from dxs.paths import TRACES, model_path
from dxs.sampling import SamplerCfg


def schedule(spec: str):
    """Funcao posicao -> (modelo, tamanho maximo do trecho)."""
    if m := re.fullmatch(r"warm(\d+)", spec):
        w = int(m[1])
        return lambda pos: ("D", w - pos) if pos < w else ("Q", 1 << 30)
    if m := re.fullmatch(r"per(\d+)/(\d+)(dfirst)?", spec):
        ld, lq, dfirst = int(m[1]), int(m[2]), bool(m[3])
        cyc = ld + lq
        first, second = (("D", ld), ("Q", lq)) if dfirst else (("Q", lq), ("D", ld))

        def f(pos: int):
            r = pos % cyc
            return (first[0], first[1] - r) if r < first[1] else (second[0], cyc - r)
        return f
    raise ValueError(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--comp", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.42, help="fracao da GPU POR motor (sao dois)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sched = schedule(args.schedule)
    tag = args.schedule.replace("/", "-")
    out = TRACES / (f"{args.comp.split('/')[-1].lower()}+mix-{tag}__{args.dataset}-n{args.n_problems}"
                    f"-s{args.samples}-b{args.budget}.parquet")
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(args.dense))
    probs = load_problems(args.dataset, args.n_problems, seed=args.seed)
    prompts = [tok(chat_prompt(tok, p.question), add_special_tokens=False).input_ids for p in probs]
    max_len = max(map(len, prompts)) + args.budget + 16
    eng = {"D": make_llm(model_path(args.dense), max_model_len=max_len, gpu_mem=args.gpu_mem, seed=args.seed),
           "Q": make_llm(model_path(args.comp), max_model_len=max_len, gpu_mem=args.gpu_mem, seed=args.seed + 1)}

    traces = [{"pid": p.pid, "sample": k, "gold": p.gold, "prompt": prompts[i], "out": [], "dense": 0, "finish": None}
              for i, p in enumerate(probs) for k in range(args.samples)]
    t0, rounds = time.time(), 0
    while True:
        active = [t for t in traces if t["finish"] is None]
        if not active:
            break
        rounds += 1
        for who in ("D", "Q"):
            batch = []
            for t in active:
                m, n = sched(len(t["out"]))
                if m == who:
                    batch.append((t, min(n, args.budget - len(t["out"]))))
            if not batch:
                continue
            res = eng[who].generate([{"prompt_token_ids": t["prompt"] + t["out"]} for t, _ in batch],
                                    [sampling_params(SamplerCfg(), n=1, max_tokens=n) for _, n in batch])
            for (t, n), r in zip(batch, res):
                o = r.outputs[0]
                if o.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"geracao abortada: {o.finish_reason}")
                t["out"] += list(o.token_ids)
                if who == "D":
                    t["dense"] += len(o.token_ids)
                if o.finish_reason == "stop":
                    t["finish"] = "stop"
                elif len(t["out"]) >= args.budget:
                    t["finish"] = "length"
        print(f"  rodada {rounds}: ativos={len(active)}  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for t in traces:
        text = tok.decode(t["out"], skip_special_tokens=False)
        ans = final_answer(text)
        rows.append({"pid": t["pid"], "sample": t["sample"], "gold": t["gold"], "finish": t["finish"],
                     "n_out": len(t["out"]), "dense_tokens": t["dense"], "answer": ans,
                     "correct": is_correct(ans, t["gold"]), "text": text})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[ok] {out}\n  acc={df.correct.mean():.4f} cortados={(df.finish == 'length').mean():.3f} "
          f"frac denso={df.dense_tokens.sum() / df.n_out.sum():.3f} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
