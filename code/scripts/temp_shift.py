"""A compressao equivale a uma mudanca de temperatura? Medido so por teacher forcing.

Sobre os primeiros --n-traces traces de D, com os logits crus l_D e l_Q em cada posicao:
    H_D, H_Q        entropia media (T=1) de D e de Q               -> Q afia (H_Q < H_D) ou achata?
    tau*            argmin_tau  media KL( softmax(l_Q) || softmax(l_D / tau) )
                    tau* < 1: Q se parece com D em temperatura mais BAIXA (mais afiado); tau* > 1: mais alta
    kl_1, kl_tau    KL em tau=1 e em tau*; 1 - kl_tau/kl_1 = fracao da divergencia que e "so temperatura"
    o mesmo separado por tercil da entropia de D (posicoes de compromisso vs. posicoes de escolha)

Uso: python scripts/temp_shift.py --traces <traces de D> --dense Qwen/Qwen3-4B --comp qwen3-4b__rtn-w3g128
Saida: uma linha em DXS_DATA/results/temp_shift.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from dxs.paths import DATA
from teacher_force import final_hidden, load  # noqa: E402

TAUS = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--comp", required=True)
    ap.add_argument("--n-traces", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    out = DATA / "results" / "temp_shift.csv"
    key = (Path(args.traces).stem, args.comp)
    if out.exists() and ((pd.read_csv(out).traces == key[0]) & (pd.read_csv(out).comp == key[1])).any():
        print("[skip]", key)
        return
    df = pd.read_parquet(args.traces).iloc[::max(1, len(pd.read_parquet(args.traces, columns=['pid'])) // args.n_traces)][:args.n_traces]
    md, mq = load(args.dense), load(args.comp)
    HD, HQ, KL = [], [], []
    for r in df.itertuples():
        prompt, output = [int(x) for x in r.prompt_ids], [int(x) for x in r.output_ids]
        ids = torch.tensor(prompt + output, device="cuda")
        sl = slice(len(prompt) - 1, len(prompt) - 1 + len(output))
        hd, _ = final_hidden(md, ids)
        hq, _ = final_hidden(mq, ids)
        hd, hq = hd[sl], hq[sl]
        for s in range(0, hd.shape[0], args.chunk):
            ld, lq = md.lm_head(hd[s:s + args.chunk]).float(), mq.lm_head(hq[s:s + args.chunk]).float()
            logq = torch.log_softmax(lq, -1)
            q = logq.exp()
            HQ.append(-(q * logq).sum(-1))
            logp = torch.log_softmax(ld, -1)
            HD.append(-(logp.exp() * logp).sum(-1))
            KL.append(torch.stack([(q * (logq - torch.log_softmax(ld / t, -1))).sum(-1) for t in TAUS.tolist()], 1))
    HD, HQ, KL = torch.cat(HD), torch.cat(HQ), torch.cat(KL)
    row = {"traces": key[0], "comp": key[1], "n_tok": len(HD)}

    def stats(mask, tag):
        kl = KL[mask].mean(0)
        i = int(kl.argmin())
        i1 = int((TAUS == 1.0).nonzero()[0])
        row.update({f"H_D{tag}": float(HD[mask].mean()), f"H_Q{tag}": float(HQ[mask].mean()), f"tau{tag}": float(TAUS[i]),
                    f"kl1{tag}": float(kl[i1]), f"kltau{tag}": float(kl[i]), f"frac_temp{tag}": float(1 - kl[i] / kl[i1])})

    stats(torch.ones_like(HD, dtype=torch.bool), "")
    q1, q2 = torch.quantile(HD[:2_000_000].float(), torch.tensor([1 / 3, 2 / 3], device=HD.device))
    stats(HD <= q1, "_lowH")
    stats(HD > q2, "_highH")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print("[ok]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
