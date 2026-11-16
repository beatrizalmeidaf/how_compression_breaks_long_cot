"""Troca de papeis: quem PENSA (traces) e quem CONCLUI (--answerer) podem ser modelos diferentes.

Para cada trace do pensador e cada b em --budgets:
    prefixo = so o raciocinio, ate b tokens (corta no </think> se ele aparecer antes de b)
    b = 0   -> sem raciocinio: "<think>\\n\\n</think>\\n\\n" (formato no-think do Qwen3)
    o --answerer recebe prompt + prefixo + "\\n</think>\\n\\n" e gera a resposta (ate --answer-tokens)
Sempre forca (inclusive traces que terminaram), para os 4 pares pensador x respondedor serem comparaveis:
    D pensa / D conclui    referencia
    Q pensa / Q conclui    modelo comprimido inteiro
    D pensa / Q conclui    o comprimido sabe LER um bom raciocinio?
    Q pensa / D conclui    o raciocinio do comprimido SERVE para alguma coisa?
D e Q compartilham tokenizer (mesmo modelo base), entao os ids do trace valem para os dois.

Uso: python scripts/swap_answer.py --traces <traces do pensador> --answerer qwen3-4b__rtn-w3g128
Saida: DXS_DATA/swap/<traces>__ans-<respondedor>.parquet  (uma linha por trace x b)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dxs.data import final_answer, is_correct
from dxs.engine import make_llm, sampling_params
from dxs.markers import stop_id
from dxs.paths import DATA, model_path
from dxs.sampling import SamplerCfg

SWAP = DATA / "swap"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--answerer", required=True)
    ap.add_argument("--budgets", default="0,1024,2048,4096")
    ap.add_argument("--answer-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--short", action="store_true",
                    help="controle: resposta imediata (prefixo 'The final answer is $\\boxed{', ate 32 tokens) - "
                         "mede a informacao do raciocinio, sem deixar o respondedor resolver de novo")
    args = ap.parse_args()
    if args.short:
        args.answer_tokens = 32

    traces = Path(args.traces)
    out = SWAP / f"{traces.stem}__ans-{args.answerer.split('/')[-1].lower()}{'__short' if args.short else ''}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(args.answerer))
    close = tok.encode("\n</think>\n\n", add_special_tokens=False)
    nothink = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    lead = "The final answer is $\\boxed{" if args.short else ""
    if args.short:
        close = tok.encode("\n</think>\n\n" + lead, add_special_tokens=False)
        nothink = tok.encode("<think>\n\n</think>\n\n" + lead, add_special_tokens=False)
    stop = stop_id(tok)
    df = pd.read_parquet(traces)
    # templates que ja abrem o raciocinio no prompt (ex.: R1-Distill termina em "<think>\n"): sem raciocinio = so fechar
    if "<think>" in tok.decode([int(x) for x in df.prompt_ids.iloc[0]][-4:]):
        nothink = tok.encode("</think>\n\n" + lead, add_special_tokens=False)
    budgets = [int(b) for b in args.budgets.split(",")]
    jobs = []
    for r in df.itertuples():
        out_ids = [int(x) for x in r.output_ids]
        think = out_ids[:out_ids.index(stop)] if stop in out_ids else out_ids     # so o raciocinio
        for b in budgets:
            pre = nothink if b == 0 else think[:b] + close
            jobs.append((r.pid, r.sample, b, r.gold, pre, [int(x) for x in r.prompt_ids], min(b, len(think))))
    llm = make_llm(model_path(args.answerer), max_model_len=max(len(j[5]) + len(j[4]) for j in jobs) + args.answer_tokens + 8,
                   gpu_mem=args.gpu_mem)
    res = llm.generate([{"prompt_token_ids": j[5] + j[4]} for j in jobs],
                       sampling_params(SamplerCfg(), n=1, max_tokens=args.answer_tokens))
    rows = []
    for j, r in zip(jobs, res):
        o = r.outputs[0]
        if o.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"geracao abortada: {o.finish_reason}")
        text = "</think>" + ("\\boxed{" if args.short else "") + o.text
        ok = (args.short or o.finish_reason == "stop") and is_correct(final_answer(text), j[3])
        rows.append({"pid": j[0], "sample": j[1], "b": j[2], "think_used": j[6], "correct": bool(ok),
                     "ans_tokens": len(o.token_ids), "ans_finish": o.finish_reason})
    SWAP.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_parquet(out)
    g = res.groupby("b").agg(acc=("correct", "mean"), tokens_resposta=("ans_tokens", "mean"),
                             resposta_cortada=("ans_finish", lambda s: (s == "length").mean()))
    print(f"[ok] {out}\n{g.round(4).to_string()}")


if __name__ == "__main__":
    main()
