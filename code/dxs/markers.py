"""Tokens que controlam o comprimento do raciocinio: parada e hesitacao.

Parada: `</think>` encerra o bloco de raciocinio (no Qwen3 e um token unico).
Hesitacao: palavras que abrem um novo ramo ("Wait", "But", "Alternatively"...), na linha
de Quantized Reasoning Models Think They Need to Think Longer (arXiv 2606.00206).
So entram variantes que o tokenizer codifica em UM token.
"""

from __future__ import annotations

HESITATION_WORDS = (
    "Wait", "But", "Alternatively", "However", "Hmm", "Actually", "Maybe", "Perhaps", "Hold",
    "Instead", "Though", "Although", "Still", "Otherwise", "Unless", "Again",
)


def stop_id(tok) -> int:
    ids = tok.encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"</think> nao e token unico neste tokenizer: {ids}")
    return ids[0]


def hesitation_ids(tok) -> list[int]:
    out = set()
    for w in HESITATION_WORDS:
        for form in (w, w.lower()):
            for prefix in ("", " ", "\n", "\n\n"):
                ids = tok.encode(prefix + form, add_special_tokens=False)
                if len(ids) == 1:
                    out.add(ids[0])
    return sorted(out)
