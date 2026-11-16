# Pré-registro — limiar de troca de sinal "pensar protege / amplifica" sob compressão

Registrado em 2026-10-14, ANTES de gerar ou medir qualquer dado das configurações de teste abaixo.

## Definições (fixas)
- acc = fração de amostras corretas por problema, média sobre problemas (orçamento 16384, T=0.6, top-p 0.95, top-k 20).
- retenção_pensando = acc_Q(pensando, 16k) / acc_D(pensando, 16k).
- retenção_sem_pensar = acc_Q(sem raciocínio, resposta ≤ 2048 tokens) / acc_D(idem) — `scripts/swap_answer.py`, b=0.
- dif = retenção_pensando − retenção_sem_pensar (> 0: pensar protege).
- TV/token = TV média entre as distribuições de amostragem de D e Q, por teacher forcing sobre os traces pensando de D
  (`scripts/teacher_force.py`), soma de TV / soma de tokens.

## Regra preditiva
- TV/token < 0.114  →  prever dif > 0
- TV/token > 0.114  →  prever dif < 0
- Exclusão (no chão): retenção_sem_pensar < 0.2 OU retenção_pensando < 0.2.

## Conjunto de desenvolvimento (já visto): 16 pares de `results/swap_all.csv` (1.7B/4B/8B MATH-500, 4B GPQA).

## Conjunto de teste (não visto)
1. Qwen3-14B, MATH-500 (300 problemas × 4): rtn-w4g128, gptq-w4g128, wanda-0.3, gptq-w3g128, wanda-0.5.
2. Compressores novos em modelos já vistos (4B e 8B, MATH-500): gptq-w3g64, rtn-w4g0 (por canal), wanda-0.4.
3. Configs vistas no pensar mas sem medição sem pensar (15): W8, AWQ, GPTQ-W4, Wanda-30% etc. — teste mais fraco
   (TV já conhecido), reportado separadamente.

## Critério de sucesso (conjuntos 1 e 2, pontos não excluídos)
- Sinal correto em ≥ 80% dos pontos, E
- Spearman(TV/token, dif) ≤ −0.6.
Se falhar, a hipótese do limiar universal é considerada refutada.
