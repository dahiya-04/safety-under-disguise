# Encoding-attack analysis — `20261114-235020_full`

5400 judged rows · 300 prompts (200 harmful) · 3 models · 6 encodings.

ASR = share of harmful prompts judged COMPLIANCE (strict). Brackets = 95% Wilson CI.


## ASR — All direct requests

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 2% [1%–5%] | 8% [5%–13%] | 2% [1%–5%] | 24% [19%–30%] | 20% [15%–27%] | 2% [1%–5%] |
| Llama-3.3-70B | 14% [9%–19%] | 8% [5%–13%] | 5% [3%–9%] | 14% [10%–19%] | 26% [20%–32%] | 16% [12%–22%] |
| Qwen2.5-72B | 5% [3%–9%] | 5% [3%–9%] | 6% [3%–10%] | 7% [4%–11%] | 44% [37%–50%] | 12% [8%–17%] |

## ASR — AdvBench

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 2% [1%–7%] | 5% [2%–11%] | 3% [1%–8%] | 22% [15%–31%] | 21% [14%–30%] | 1% [0%–5%] |
| Llama-3.3-70B | 10% [6%–17%] | 5% [2%–11%] | 4% [2%–10%] | 9% [5%–16%] | 21% [14%–30%] | 15% [9%–23%] |
| Qwen2.5-72B | 2% [1%–7%] | 3% [1%–8%] | 3% [1%–8%] | 5% [2%–11%] | 47% [38%–57%] | 10% [6%–17%] |

## ASR — JailbreakBench

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 2% [1%–7%] | 11% [6%–19%] | 1% [0%–5%] | 26% [18%–35%] | 20% [13%–29%] | 3% [1%–8%] |
| Llama-3.3-70B | 17% [11%–26%] | 12% [7%–20%] | 6% [3%–12%] | 19% [13%–28%] | 30% [22%–40%] | 18% [12%–27%] |
| Qwen2.5-72B | 8% [4%–15%] | 7% [3%–14%] | 8% [4%–15%] | 9% [5%–16%] | 40% [31%–50%] | 13% [8%–21%] |

## Decoding success (correct / judged decodable)

| model | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|
| Llama-3.1-8B | 98% | 98% | 87% | 71% | 23% |
| Llama-3.3-70B | 99% | 100% | 97% | 96% | 51% |
| Qwen2.5-72B | 99% | 100% | 96% | 89% | 54% |

## Unclear decoding (bare refusals)

| model | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|
| Llama-3.1-8B | 14% | 1% | 7% | 6% | 18% |
| Llama-3.3-70B | 7% | 0% | 0% | 0% | 4% |
| Qwen2.5-72B | 3% | 0% | 0% | 1% | 5% |

## ASR given the model understood the prompt (all harmful)

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 2% | 10% | 2% | 29% | 30% | 10% |
| Llama-3.3-70B | 14% | 9% | 5% | 14% | 27% | 31% |
| Qwen2.5-72B | 5% | 5% | 6% | 7% | 48% | 22% |

## Lift vs plaintext — All direct requests  (* = McNemar, Holm p<.05)

| model | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|
| Llama-3.1-8B | +6 pp [+2, +10] * | +0 pp [-2, +2] | +22 pp [+16, +28] * | +18 pp [+13, +24] * | +0 pp [-2, +3] |
| Llama-3.3-70B | -5 pp [-10, -0] | -8 pp [-13, -5] * | +0 pp [-4, +6] | +12 pp [+6, +18] * | +3 pp [-4, +10] |
| Qwen2.5-72B | +0 pp [-3, +3] | +0 pp [-2, +3] | +2 pp [-2, +6] | +38 pp [+32, +46] * | +6 pp [+1, +12] |

## Lift vs plaintext — AdvBench  (* = McNemar, Holm p<.05)

| model | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|
| Llama-3.1-8B | +3 pp [+0, +7] | +1 pp [+0, +3] | +20 pp [+12, +28] * | +19 pp [+11, +27] * | -1 pp [-4, +2] |
| Llama-3.3-70B | -5 pp [-10, +0] | -6 pp [-11, -2] | -1 pp [-6, +4] | +11 pp [+2, +20] | +5 pp [-4, +15] |
| Qwen2.5-72B | +1 pp [-2, +5] | +1 pp [+0, +3] | +3 pp [-1, +8] | +45 pp [+35, +54] * | +8 pp [+2, +15] |

## Lift vs plaintext — JailbreakBench  (* = McNemar, Holm p<.05)

| model | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|
| Llama-3.1-8B | +9 pp [+3, +16] | -1 pp [-5, +2] | +24 pp [+16, +33] * | +18 pp [+11, +26] * | +1 pp [-3, +5] |
| Llama-3.3-70B | -5 pp [-12, +2] | -11 pp [-17, -5] * | +2 pp [-6, +10] | +13 pp [+3, +23] | +1 pp [-8, +10] |
| Qwen2.5-72B | -1 pp [-6, +4] | +0 pp [-5, +5] | +1 pp [-5, +7] | +32 pp [+22, +41] * | +5 pp [-3, +13] |

## Benign over-refusal (full + partial)

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 27% [19%–36%] | 27% [19%–36%] | 40% [31%–50%] | 10% [6%–17%] | 3% [1%–8%] | 8% [4%–15%] |
| Llama-3.3-70B | 5% [2%–11%] | 8% [4%–15%] | 15% [9%–23%] | 5% [2%–11%] | 5% [2%–11%] | 3% [1%–8%] |
| Qwen2.5-72B | 7% [3%–14%] | 16% [10%–24%] | 14% [9%–22%] | 17% [11%–26%] | 7% [3%–14%] | 44% [35%–54%] |

## Benign compliance

| model | plaintext | hindi | french | bengali | tamil | zulu |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 73% | 66% | 56% | 61% | 42% | 0% |
| Llama-3.3-70B | 95% | 89% | 84% | 83% | 81% | 29% |
| Qwen2.5-72B | 93% | 80% | 84% | 74% | 75% | 16% |

## Judge robustness
Mean within-cell ASR difference between the two judge endpoints: +2.9 pp over 5 cells with ≥5 rows per judge.


## Figures
- ![fig1_asr_heatmap](fig1_asr_heatmap.png)
- ![fig2_decoding_vs_asr](fig2_decoding_vs_asr.png)
- ![fig3_understanding](fig3_understanding.png)
- ![fig4_benign](fig4_benign.png)
- ![fig5_lift](fig5_lift.png)
- ![fig6_judge_robustness](fig6_judge_robustness.png)
