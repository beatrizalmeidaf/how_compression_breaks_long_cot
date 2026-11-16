"""Teacher forcing de D (denso) e Q (comprimido) sobre traces ja gerados.

Por trace e por posicao de saida j (token output_ids[j], previsto a partir do prefixo
prompt + output[:j]):
    TV_j        TV entre as distribuicoes de AMOSTRAGEM (temperatura/top-k/top-p) de D e Q
    tv_raw_j    TV entre softmax crus (T=1)
    kl_raw_j    KL(p_raw || q_raw)
    ent_j       entropia da distribuicao de amostragem de D
    p_stop/q_stop   prob. (de amostragem) de emitir </think> em D e em Q
    p_hes/q_hes     massa (de amostragem) em tokens de hesitacao em D e em Q

Bifurcacoes: K posicoes por trace sorteadas iid com prob. TV_j / sum TV (com reposicao).
Em cada uma, y_Q ~ (q-p)+ e y_D ~ (p-q)+. Assim
    sum_j TV_j s_j  ~=  (sum TV / K) * sum_k [V(h.y_Q) - V(h.y_D)]
e um estimador nao viesado do termo do lema de diferenca de desempenho no trace.

Vies de copia (preditor de loop): onde os ultimos K tokens ja apareceram antes no contexto, c_j e o
token que seguiu a ocorrencia anterior. Nas posicoes em que D NAO copiou (output[j] != c_j):
    sum_p_copy = sum p(c_j),  sum_q_copy = sum q(c_j)     ->  k_copy = sum_q_copy / sum_p_copy
Critico de comprimento (--critic): Qhat(h, y) = b(h) + e_y^T U V^T h preve tokens restantes depois de y;
    crit_j = sum_y (q - p)(y) Qhat(h_j, y)  (o b(h) cancela)   ->  sum_crit ~ mudanca de 1a ordem no comprimento

Saidas (DXS_DATA/tf/<traces>__vs__<Q>[__<tag>]):
    .parquet   uma linha por trace (somas) + listas das bifurcacoes
    .npz       arrays por token concatenados (+ offsets)
    _hid.npy   (opcional) hidden states de D nas bifurcacoes, camadas --layers

Uso: python scripts/teacher_force.py --traces <arquivo.parquet> --dense Qwen/Qwen3-4B --comp qwen3-4b__rtn-w4g128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dxs.paths import TF, model_path
from dxs.sampling import SamplerCfg, entropy, kl, processed_probs, sample_residual, tv_and_residuals


def load(name: str):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_path(name), dtype=torch.bfloat16).to("cuda").eval()
    return m


@torch.no_grad()
def final_hidden(model, ids: torch.Tensor, want_layers: list[int] | None = None):
    out = model.model(input_ids=ids[None], output_hidden_states=bool(want_layers), use_cache=False)
    last = out.last_hidden_state[0]
    extra = [out.hidden_states[l][0] for l in want_layers] if want_layers else None
    return last, extra


COLS = ("tv", "tv_raw", "kl_raw", "ent", "p_stop", "q_stop", "p_hes", "q_hes", "p_copy", "q_copy", "crit")
COPY_K = 8


def copy_candidates(ids: list[int], lo: int, n: int, k: int = COPY_K) -> np.ndarray:
    """c_j para j em [0, n): token que seguiu a ocorrencia anterior mais recente do k-grama que termina
    em lo+j (contexto de output[j]); -1 se o k-grama nao ocorreu antes."""
    last: dict[tuple, int] = {}
    out = np.full(n, -1, dtype=np.int64)
    for e in range(k - 1, lo + n):
        key = tuple(ids[e - k + 1:e + 1])
        if e >= lo:
            prev = last.get(key)
            if prev is not None:
                out[e - lo] = ids[prev + 1]
        last[key] = e
    return out


@torch.no_grad()
def per_token(md, mq, hd, hq, cfg: SamplerCfg, chunk: int, stop: int, hes: torch.Tensor,
              copy_c: torch.Tensor, critic: dict | None):
    """Metricas por posicao a partir dos hidden finais [L, H] de D e Q."""
    cols = {k: [] for k in COLS}
    for s in range(0, hd.shape[0], chunk):
        ld = md.lm_head(hd[s:s + chunk]).float()
        lq = mq.lm_head(hq[s:s + chunk]).float()
        p, q = processed_probs(ld, cfg), processed_probs(lq, cfg)
        tv, _, _ = tv_and_residuals(p, q)
        pr, qr = torch.softmax(ld, -1), torch.softmax(lq, -1)
        cols["tv"].append(tv)
        cols["tv_raw"].append(0.5 * (pr - qr).abs().sum(-1))
        cols["kl_raw"].append(kl(pr, qr))
        cols["ent"].append(entropy(p))
        cols["p_stop"].append(p[:, stop])
        cols["q_stop"].append(q[:, stop])
        cols["p_hes"].append(p[:, hes].sum(-1))
        cols["q_hes"].append(q[:, hes].sum(-1))
        c = copy_c[s:s + chunk]
        has = c >= 0
        cc = c.clamp_min(0)[:, None]
        cols["p_copy"].append(torch.where(has, p.gather(1, cc).squeeze(1), torch.zeros_like(tv)))
        cols["q_copy"].append(torch.where(has, q.gather(1, cc).squeeze(1), torch.zeros_like(tv)))
        if critic is not None:
            z = (hd[s:s + chunk].float() @ critic["V"] - critic["o"]) @ critic["U"].T   # [n, H]
            scores = z @ critic["E"].T                                             # [n, V]
            cols["crit"].append(((q - p) * scores).sum(-1) * critic["scale"])
        else:
            cols["crit"].append(torch.zeros_like(tv))
    return {k: torch.cat(v).cpu().numpy() for k, v in cols.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--comp", required=True)
    ap.add_argument("--k-forks", type=int, default=0, help="bifurcacoes por trace para fork_rollouts (0 = nenhuma)")
    ap.add_argument("--layers", default="", help="camadas de D salvas nas bifurcacoes ('' = nenhuma)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--critic", default="", help="critico de comprimento (train_length_critic.py)")
    ap.add_argument("--tag", default="", help="sufixo do arquivo de saida (ex.: v2)")
    args = ap.parse_args()

    traces = Path(args.traces)
    stem = f"{traces.stem}__vs__{args.comp.split('/')[-1].lower()}" + (f"__{args.tag}" if args.tag else "")
    out = TF / f"{stem}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    layers = [int(x) for x in args.layers.split(",") if x]
    cfg = SamplerCfg()
    df = pd.read_parquet(traces)
    from transformers import AutoTokenizer
    from dxs.markers import hesitation_ids, stop_id
    tok = AutoTokenizer.from_pretrained(model_path(args.dense))
    stop, hes = stop_id(tok), torch.tensor(hesitation_ids(tok), device="cuda")
    md, mq = load(args.dense), load(args.comp)
    critic = None
    if args.critic:
        ck = torch.load(args.critic, map_location="cuda")
        critic = {"U": ck["U"].float(), "V": ck["V"].float(), "o": ck["o"].float(), "scale": float(ck["scale"]),
                  "E": md.lm_head.weight.float() / float(ck["E_norm"])}
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    rows, arrays, hid = [], {k: [] for k in COLS}, []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        prompt = list(r.prompt_ids)
        output = list(r.output_ids)
        ids = torch.tensor(prompt + output, device="cuda")
        lo = len(prompt) - 1                      # hidden em lo+j preve output[j]
        sl = slice(lo, lo + len(output))
        hd_all, extra = final_hidden(md, ids, layers)
        hq_all, _ = final_hidden(mq, ids)
        hd, hq = hd_all[sl], hq_all[sl]
        copy_np = copy_candidates([int(x) for x in prompt + output], lo, len(output))
        copy_c = torch.from_numpy(copy_np).cuda()
        m = per_token(md, mq, hd, hq, cfg, args.chunk, stop, hes, copy_c, critic)
        nocopy = (copy_np >= 0) & (copy_np != np.asarray(output))
        for k in arrays:
            arrays[k].append(m[k].astype(np.float32))

        tv = torch.from_numpy(m["tv"]).cuda()
        fork_t, y_q, y_d = [], [], []
        if args.k_forks and tv.sum() > 0:
            pos = torch.multinomial(tv / tv.sum(), args.k_forks, replacement=True, generator=gen)
            ld, lq = md.lm_head(hd[pos]).float(), mq.lm_head(hq[pos]).float()
            _, pos_q, pos_p = tv_and_residuals(processed_probs(ld, cfg), processed_probs(lq, cfg))
            fork_t = pos.tolist()
            y_q = sample_residual(pos_q, gen).tolist()
            y_d = sample_residual(pos_p, gen).tolist()
            if layers:
                hid.append(torch.stack([e[sl][pos] for e in extra], 1).to(torch.float16).cpu().numpy())
        rows.append({"pid": r.pid, "sample": r.sample, "correct": r.correct, "n_out": len(output),
                     "sum_tv": float(m["tv"].sum()), "sum_tv_raw": float(m["tv_raw"].sum()),
                     "sum_kl_raw": float(m["kl_raw"].sum()), "mean_ent": float(m["ent"].mean()),
                     "finish": r.finish, **{f"sum_{k}": float(m[k].sum()) for k in ("p_stop", "q_stop", "p_hes", "q_hes", "crit")},
                     "n_copy_avail": int((copy_np >= 0).sum()), "n_nocopy": int(nocopy.sum()),
                     "sum_p_copy": float(m["p_copy"][nocopy].sum()), "sum_q_copy": float(m["q_copy"][nocopy].sum()),
                     "fork_t": fork_t, "fork_tv": [float(m["tv"][t]) for t in fork_t],
                     "fork_ent": [float(m["ent"][t]) for t in fork_t], "y_q": y_q, "y_d": y_d})
        if i % 100 == 0:
            print(f"  {i}/{len(df)} {time.time() - t0:.0f}s sumTV={rows[-1]['sum_tv']:.2f}", flush=True)

    TF.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    offsets = np.cumsum([0] + [len(a) for a in arrays["tv"]])
    np.savez_compressed(TF / f"{stem}.npz", offsets=offsets, **{k: np.concatenate(v) for k, v in arrays.items()})
    if hid:
        np.save(TF / f"{stem}_hid.npy", np.concatenate(hid))
    s = pd.DataFrame(rows)
    print(f"[ok] {out} ({time.time() - t0:.0f}s)  mean sumTV={s.sum_tv.mean():.2f} "
          f"mean TV/token={s.sum_tv.sum() / s.n_out.sum():.4f}  mean KLraw/token={s.sum_kl_raw.sum() / s.n_out.sum():.4f}")


if __name__ == "__main__":
    main()
