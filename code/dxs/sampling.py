"""Distribuicao de amostragem efetiva e as quantidades do lema de diferenca de desempenho.

A TV relevante e entre as distribuicoes que o sampler REALMENTE usa (temperatura,
top-k, top-p), nao entre os softmax crus. `processed_probs` reproduz a ordem do
sampler do vLLM: logits/T -> top-k -> top-p -> softmax.

Para uma posicao com distribuicoes p (denso) e q (comprimido):
    sum_y (q - p)(y) V(h.y) = TV * ( E_{y~r_Q} V(h.y) - E_{y~r_D} V(h.y) )
com r_Q = (q - p)+ / TV e r_D = (p - q)+ / TV (as duas partes residuais tem massa TV).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplerCfg:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}


def processed_probs(logits: torch.Tensor, cfg: SamplerCfg) -> torch.Tensor:
    """[..., V] logits -> [..., V] probabilidades efetivas (fp32)."""
    x = logits.float() / cfg.temperature
    if 0 < cfg.top_k < x.shape[-1]:
        kth = torch.topk(x, cfg.top_k, dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if cfg.top_p < 1.0:
        # igual ao vLLM: ordena crescente e remove a cauda com massa acumulada <= 1 - p,
        # mantendo sempre o token mais provavel
        sx, si = torch.sort(x, dim=-1, descending=False)
        cum = torch.softmax(sx, dim=-1).cumsum(dim=-1)
        drop = cum <= (1.0 - cfg.top_p)
        drop[..., -1] = False
        x = x.scatter(-1, si, sx.masked_fill(drop, float("-inf")))
    return torch.softmax(x, dim=-1)


def tv_and_residuals(p: torch.Tensor, q: torch.Tensor):
    """TV por posicao e as partes positivas (q-p)+ e (p-q)+ (nao normalizadas)."""
    d = q - p
    pos_q = d.clamp_min(0)
    pos_p = (-d).clamp_min(0)
    tv = 0.5 * d.abs().sum(-1)
    return tv, pos_q, pos_p


def sample_residual(pos: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Amostra de (x)+ / massa, uma por linha. `pos` [N, V] com massa > 0."""
    return torch.multinomial(pos / pos.sum(-1, keepdim=True), 1, generator=gen).squeeze(-1)


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p || q) com suporte truncado; tokens com p>0 e q=0 recebem q=eps (KL grande, finito)."""
    return (p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum(-1)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(eps))).sum(-1)
