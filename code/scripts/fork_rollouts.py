"""Rollouts a partir das bifurcacoes: estima V(h.y_Q) e V(h.y_D) em cada posicao sorteada.

Para cada bifurcacao (trace, t, y_Q, y_D) do arquivo de teacher forcing, continua
prompt + output[:t] + [y] com --rollout-model, R vezes por ramo, com o mesmo
sampler e o MESMO orcamento total de tokens do trace original. V e a fracao de
continuacoes corretas.

  --rollout-model = D (padrao): preditor barato  (prefixos de D, valor de D)
  --rollout-model = Q         : forma exata A    (prefixos de D, valor de Q)
  TF feito sobre traces de Q + rollouts com D: forma exata B (prefixos de Q, valor de D)

Uso: python scripts/fork_rollouts.py --tf <tf.parquet> --traces <traces.parquet> --rollout-model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dxs.data import final_answer, is_correct
from dxs.engine import make_llm, sampling_params
from dxs.paths import SENS, model_path
from dxs.sampling import SamplerCfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--rollout-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--max-forks", type=int, default=0, help="usar so as primeiras k bifurcacoes de cada trace")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf_path = Path(args.tf)
    rm = args.rollout_model.split("/")[-1].lower()
    out = SENS / f"{tf_path.stem}__roll-{rm}-r{args.rollouts}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer

    tf = pd.read_parquet(tf_path)
    tr = pd.read_parquet(args.traces).set_index(["pid", "sample"])
    path = model_path(args.rollout_model)
    tok = AutoTokenizer.from_pretrained(path)

    jobs = []  # (row_idx, fork_idx, branch, prefix_ids, prefix_out_ids, gold)
    for i, r in enumerate(tf.itertuples()):
        base = tr.loc[(r.pid, r.sample)]
        prompt, output = [int(x) for x in base.prompt_ids], [int(x) for x in base.output_ids]
        nk = len(r.fork_t) if not args.max_forks else min(args.max_forks, len(r.fork_t))
        for k in range(nk):
            t = int(r.fork_t[k])
            for branch, y in (("q", int(r.y_q[k])), ("d", int(r.y_d[k]))):
                out_prefix = output[:t] + [y]
                jobs.append((i, k, branch, prompt + out_prefix, out_prefix, base.gold))

    max_len = max(len(j[3]) for j in jobs) + args.budget
    llm = make_llm(path, max_model_len=max_len, gpu_mem=args.gpu_mem, seed=args.seed)
    eos = set(tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    prompts, sps, keep = [], [], []
    for j in jobs:
        remaining = args.budget - len(j[4])
        if j[4][-1] in eos or remaining <= 0:   # forcou fim de sequencia / sem orcamento: sem rollout
            keep.append(False)
            continue
        keep.append(True)
        prompts.append({"prompt_token_ids": j[3]})
        sps.append(sampling_params(SamplerCfg(), n=args.rollouts, max_tokens=remaining))
    t0 = time.time()
    res = iter(llm.generate(prompts, sps))
    secs = time.time() - t0

    recs, n_tok = [], 0
    for j, kept in zip(jobs, keep):
        i, k, branch, _, out_prefix, gold = j
        prefix_text = tok.decode(out_prefix, skip_special_tokens=False)
        if kept:
            outs = next(res).outputs
            n_tok += sum(len(o.token_ids) for o in outs)
            vals = [is_correct(final_answer(prefix_text + o.text), gold) for o in outs]
        else:
            vals = [is_correct(final_answer(prefix_text), gold)]
        recs.append({"row": i, "fork": k, "branch": branch, "v": float(np.mean(vals)), "n": len(vals)})

    rec = pd.DataFrame(recs).pivot_table(index=["row", "fork"], columns="branch", values="v").reset_index()
    rec["s"] = rec["q"] - rec["d"]
    meta = tf[["pid", "sample", "sum_tv"]].reset_index(names="row")
    rec = rec.merge(meta, on="row")
    rec["fork_t"] = [tf.fork_t.iloc[r][f] for r, f in zip(rec.row, rec.fork)]
    rec["fork_tv"] = [tf.fork_tv.iloc[r][f] for r, f in zip(rec.row, rec.fork)]
    rec["fork_ent"] = [tf.fork_ent.iloc[r][f] for r, f in zip(rec.row, rec.fork)]
    SENS.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(out)
    print(f"[ok] {out}  {len(prompts)} prompts, {n_tok:,} tokens, {secs / 60:.1f} min ({n_tok / secs:,.0f} tok/s)\n"
          f"  mean s={rec.s.mean():+.4f}  frac s!=0={(rec.s != 0).mean():.3f}")


if __name__ == "__main__":
    main()
