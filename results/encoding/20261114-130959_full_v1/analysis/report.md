# Encoding-attack analysis — `20261114-130959_full_v1`

3240 judged rows · 180 prompts (130 harmful) · 3 models · 6 encodings.

ASR = share of harmful prompts judged COMPLIANCE (strict). Brackets = 95% Wilson CI.


## ASR — All harmful

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 15% [10%–22%] | 0% [0%–3%] | 0% [0%–3%] | 0% [0%–3%] | 8% [5%–15%] | 25% [18%–33%] |
| Llama-3.3-70B | 35% [28%–44%] | 14% [9%–21%] | 0% [0%–3%] | 18% [12%–25%] | 29% [22%–38%] | 26% [19%–34%] |
| Qwen2.5-72B | 37% [29%–45%] | 21% [15%–29%] | 0% [0%–3%] | 28% [21%–36%] | 33% [26%–42%] | 22% [15%–29%] |

## ASR — AdvBench + JBB

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 1% [0%–7%] | 0% [0%–5%] | 0% [0%–5%] | 0% [0%–5%] | 4% [1%–10%] | 8% [3%–15%] |
| Llama-3.3-70B | 19% [12%–29%] | 5% [2%–12%] | 0% [0%–5%] | 4% [1%–10%] | 11% [6%–20%] | 9% [4%–17%] |
| Qwen2.5-72B | 5% [2%–12%] | 2% [1%–9%] | 0% [0%–5%] | 4% [1%–10%] | 6% [3%–14%] | 4% [1%–10%] |

## ASR — FORTRESS

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 36% [24%–50%] | 0% [0%–7%] | 0% [0%–7%] | 0% [0%–7%] | 16% [8%–29%] | 52% [39%–65%] |
| Llama-3.3-70B | 62% [48%–74%] | 28% [17%–42%] | 0% [0%–7%] | 40% [28%–54%] | 58% [44%–71%] | 54% [40%–67%] |
| Qwen2.5-72B | 88% [76%–94%] | 50% [37%–63%] | 0% [0%–7%] | 66% [52%–78%] | 76% [63%–86%] | 50% [37%–63%] |

## Decoding success (correct / judged decodable)

| model | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|
| Llama-3.1-8B | 2% | 0% | 19% | 90% | 92% |
| Llama-3.3-70B | 92% | 3% | 94% | 99% | 95% |
| Qwen2.5-72B | 84% | 0% | 94% | 99% | 95% |

## Unclear decoding (bare refusals)

| model | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32% | 3% | 20% | 33% | 14% |
| Llama-3.3-70B | 0% | 0% | 1% | 17% | 6% |
| Qwen2.5-72B | 0% | 0% | 1% | 1% | 2% |

## ASR given the model understood the prompt (all harmful)

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 15% | 0% | – | 0% | 16% | 31% |
| Llama-3.3-70B | 35% | 14% | 0% | 18% | 38% | 30% |
| Qwen2.5-72B | 37% | 22% | – | 29% | 34% | 24% |

## Lift vs plaintext — All harmful  (* = McNemar, Holm p<.05)

| model | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|
| Llama-3.1-8B | -15 pp [-21, -9] * | -15 pp [-21, -9] * | -15 pp [-21, -9] * | -6 pp [-12, -1] | +10 pp [+2, +18] |
| Llama-3.3-70B | -22 pp [-29, -15] * | -35 pp [-44, -28] * | -18 pp [-26, -9] * | -6 pp [-13, +0] | -9 pp [-16, -2] |
| Qwen2.5-72B | -16 pp [-22, -10] * | -37 pp [-45, -28] * | -9 pp [-17, -2] | -4 pp [-9, +1] | -15 pp [-22, -9] * |

## Lift vs plaintext — AdvBench + JBB  (* = McNemar, Holm p<.05)

| model | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|
| Llama-3.1-8B | -1 pp [-4, +0] | -1 pp [-4, +0] | -1 pp [-4, +0] | +2 pp [+0, +6] | +6 pp [+1, +12] |
| Llama-3.3-70B | -14 pp [-21, -7] * | -19 pp [-28, -10] * | -15 pp [-22, -8] * | -8 pp [-14, -2] | -10 pp [-18, -2] |
| Qwen2.5-72B | -2 pp [-6, +0] | -5 pp [-10, -1] | -1 pp [-8, +4] | +1 pp [-2, +5] | -1 pp [-5, +2] |

## Lift vs plaintext — FORTRESS  (* = McNemar, Holm p<.05)

| model | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|
| Llama-3.1-8B | -36 pp [-50, -24] * | -36 pp [-48, -24] * | -36 pp [-50, -22] * | -20 pp [-34, -6] | +16 pp [-4, +34] |
| Llama-3.3-70B | -34 pp [-48, -18] * | -62 pp [-76, -48] * | -22 pp [-40, -4] | -4 pp [-18, +10] | -8 pp [-24, +8] |
| Qwen2.5-72B | -38 pp [-52, -24] * | -88 pp [-96, -78] * | -22 pp [-38, -6] | -12 pp [-24, +0] | -38 pp [-52, -26] * |

## Benign over-refusal (full + partial)

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 22% [13%–35%] | 38% [26%–52%] | 6% [2%–16%] | 32% [21%–46%] | 38% [26%–52%] | 12% [6%–24%] |
| Llama-3.3-70B | 2% [0%–10%] | 2% [0%–10%] | 0% [0%–7%] | 2% [0%–10%] | 2% [0%–10%] | 0% [0%–7%] |
| Qwen2.5-72B | 0% [0%–7%] | 2% [0%–10%] | 0% [0%–7%] | 4% [1%–13%] | 4% [1%–13%] | 0% [0%–7%] |

## Benign compliance

| model | plaintext | base64 | rot13 | leetspeak | homoglyph | hindi |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 76% | 0% | 0% | 6% | 54% | 70% |
| Llama-3.3-70B | 98% | 48% | 4% | 88% | 96% | 94% |
| Qwen2.5-72B | 100% | 84% | 0% | 88% | 94% | 94% |

## Judge robustness
Mean within-cell ASR difference between the two judge endpoints: -0.8 pp over 17 cells with ≥5 rows per judge.


## Figures
- ![fig1_asr_heatmap](fig1_asr_heatmap.png)
- ![fig2_decoding_vs_asr](fig2_decoding_vs_asr.png)
- ![fig3_understanding](fig3_understanding.png)
- ![fig4_benign](fig4_benign.png)
- ![fig5_lift](fig5_lift.png)
- ![fig6_judge_robustness](fig6_judge_robustness.png)
