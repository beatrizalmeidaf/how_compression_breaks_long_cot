"""Problemas, prompts e correcao de respostas.

Formato de prompt e de resposta segue a recomendacao do Qwen3 para matematica:
"Please reason step by step, and put your final answer within \\boxed{}."
Um trace sem `</think>` (cortado pelo orcamento) nao tem resposta e conta como erro.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dxs.paths import DATASETS

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"
LETTERS = ("A", "B", "C", "D")
STR_GOLD = "§"   # prefixo do gabarito de tarefas com resposta textual curta (BBH): comparacao por string normalizada
BBH_EXCLUDE = {"multistep_arithmetic_two", "object_counting",   # matematica
               "dyck_languages", "word_sorting"}                # saida longa e livre


@dataclass(frozen=True)
class Problem:
    pid: str
    question: str
    gold: str


def _last_boxed(text: str) -> str | None:
    """Conteudo do ultimo \\boxed{...} (ou \\fbox{...}) com chaves balanceadas."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def load_problems(name: str, n: int | None = None, seed: int = 0) -> list[Problem]:
    """Subamostra deterministica de `n` problemas (todos se n=None)."""
    df = pd.read_parquet(DATASETS / f"{name}.parquet")
    if name in ("math500", "aime24", "aime25"):
        key = "unique_id" if "unique_id" in df else "id"
        probs = [Problem(f"{name}/{r[key]}", r["problem"], str(r["answer"])) for _, r in df.iterrows()]
    elif name == "math_train":
        probs = [Problem(f"{name}/{i}", r["problem"], _last_boxed(r["solution"]) or "")
                 for i, r in df.iterrows()]
        probs = [p for p in probs if p.gold]
    elif name == "gpqa_diamond":
        probs = []
        for i, r in df.iterrows():
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = np.random.default_rng(1000 + i).permutation(4)   # embaralhamento fixo por questao
            body = "\n".join(f"{L}) {str(opts[j]).strip()}" for L, j in zip(LETTERS, order))
            gold = LETTERS[int(np.where(order == 0)[0][0])]
            probs.append(Problem(f"{name}/{i}", f"{str(r['Question']).strip()}\n\n{body}\n\n"
                                 "Answer with the letter of the correct option.", gold))
    elif name == "bbh":
        df = df[~df.task.isin(BBH_EXCLUDE)].reset_index(drop=True)
        probs = [Problem(f"{name}/{r['task']}/{r['idx']}", r["input"], STR_GOLD + str(r["target"]))
                 for _, r in df.iterrows()]
    elif name == "gsm8k":
        probs = [Problem(f"{name}/{i}", r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
                 for i, r in df.iterrows()]
    else:
        raise KeyError(name)
    if n is not None and n < len(probs):
        idx = np.random.default_rng(seed).choice(len(probs), size=n, replace=False)
        probs = [probs[i] for i in sorted(idx)]
    return probs


def chat_prompt(tokenizer, question: str, thinking: bool = True) -> str:
    msgs = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)


def final_answer(text: str) -> str | None:
    """Resposta final: ultimo \\boxed depois de </think>. Sem </think> -> None (cortado)."""
    if THINK_END not in text:
        return None
    return _last_boxed(text.split(THINK_END, 1)[1])


def _norm_str(x: str) -> str:
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = x.strip().strip("$").strip().strip(".").strip()
    if m := re.fullmatch(r"\(?([A-R])\)?", x):
        return m.group(1)
    return x.lower()


def is_correct(answer: str | None, gold: str) -> bool:
    if answer is None:
        return False
    if gold.startswith(STR_GOLD):              # BBH: "(B)" == "B", yes/no/true/false sem caixa
        return _norm_str(answer) == _norm_str(gold[len(STR_GOLD):])
    if gold in LETTERS:                        # multipla escolha: compara a letra
        m = re.search(r"\b([A-D])\b", answer.replace("\\text{", " ").replace("}", " "))
        return bool(m) and m.group(1) == gold
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${answer}$")))
    except Exception:
        return answer.strip() == gold.strip()
