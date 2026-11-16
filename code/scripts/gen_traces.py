"""Traces de raciocinio (modo thinking) com vLLM: tokens exatos + resposta + correcao.

Uso: python scripts/gen_traces.py --model Qwen/Qwen3-4B --dataset math500 --n-problems 300 --samples 4
     python scripts/gen_traces.py --model qwen3-4b__rtn-w4g128 ...   (checkpoint em DXS_DATA/ckpts)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dxs.data import chat_prompt, final_answer, is_correct, load_problems
from dxs.engine import make_llm, sampling_params
from dxs.paths import TRACES, model_path
from dxs.sampling import SamplerCfg


def trace_file(model: str, dataset: str, n: int | str, samples: int, max_tokens: int) -> Path:
    return TRACES / f"{model.split('/')[-1].lower()}__{dataset}-n{n}-s{samples}-b{max_tokens}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0,
                    help="usa K dos --n-problems (subconjunto pareavel com as rodadas completas)")
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="repetition_penalty do vLLM (baseline anti-loop)")
    ap.add_argument("--temperature", type=float, default=0.6, help="temperatura de amostragem (padrao do Qwen3: 0.6)")
    args = ap.parse_args()

    tag = f"+rp{args.rep_penalty:g}" if args.rep_penalty != 1.0 else ""
    tag += f"+T{args.temperature:g}" if args.temperature != 0.6 else ""
    nname = f"{args.subset}s" if args.subset else args.n_problems
    out = trace_file(args.model + tag, args.dataset, nname, args.samples, args.max_tokens)
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer

    path = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    probs = load_problems(args.dataset, args.n_problems, seed=args.seed)
    if args.subset:   # amostra fixa dos problemas (a ordem do arquivo segue as materias do MATH)
        import numpy as np
        keep = sorted(np.random.default_rng(1).choice(len(probs), size=args.subset, replace=False))
        probs = [probs[i] for i in keep]
    prompts = [chat_prompt(tok, p.question) for p in probs]
    prompt_ids = [tok(s, add_special_tokens=False).input_ids for s in prompts]
    max_len = max(map(len, prompt_ids)) + args.max_tokens + 16

    llm = make_llm(path, max_model_len=max_len, gpu_mem=args.gpu_mem, seed=args.seed)
    extra = {"repetition_penalty": args.rep_penalty} if args.rep_penalty != 1.0 else {}
    sp = sampling_params(SamplerCfg(temperature=args.temperature), n=args.samples, max_tokens=args.max_tokens, **extra)
    t0 = time.time()
    res = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sp)
    secs = time.time() - t0

    rows = []
    for prob, ids, r in zip(probs, prompt_ids, res):
        for k, o in enumerate(r.outputs):
            ans = final_answer(o.text)
            rows.append({"pid": prob.pid, "sample": k, "gold": prob.gold, "prompt_ids": ids,
                         "output_ids": list(o.token_ids), "n_out": len(o.token_ids),
                         "finish": o.finish_reason, "text": o.text, "answer": ans,
                         "correct": is_correct(ans, prob.gold)})
    df = pd.DataFrame(rows)
    bad = df[~df.finish.isin(["stop", "length"])]
    if len(bad):   # geracao interrompida (ex.: processo morto): nao salvar traces abortados como validos
        raise RuntimeError(f"{len(bad)} traces com finish invalido: {bad.finish.value_counts(dropna=False).to_dict()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    n_tok = int(df.n_out.sum())
    print(f"[ok] {out}\n  acc={df.correct.mean():.4f} cortados={(df.finish == 'length').mean():.3f} "
          f"tokens={n_tok:,} ({n_tok / secs:,.0f} tok/s, {secs / 60:.1f} min)")


if __name__ == "__main__":
    main()
