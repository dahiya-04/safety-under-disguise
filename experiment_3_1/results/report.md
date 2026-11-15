# Experiment 3.1: second-judge check (Indian languages)

Source run: `20261114-235020_full`. Second judge(s): moonshotai/Kimi-K2-Instruct.


## DeepSeek vs Kimi-K2-Instruct

### Q1. Do the two judges give the same label?

|  | English | Hindi | Bengali | Tamil |
|---|---|---|---|---|
| Same label | 93% | 89% | 86% | 80% |
| Agreement score (kappa, >0.7 = good) | 0.87 | 0.81 | 0.77 | 0.71 |

### Q2. Do both judges say the model stopped refusing? (attack success rate)

|  | English | Hindi | Bengali | Tamil |
|---|---|---|---|---|
| DeepSeek | 7% | 7% | 15% | 30% |
| Kimi-K2-Instruct | 7% | 7% | 20% | 30% |

### Q3. Is the leaked content vague? (detail score 1-5 of complied answers)

|  | English | Hindi | Bengali | Tamil |
|---|---|---|---|---|
| DeepSeek | 3.6 | 2.6 | 2.4 | 2.1 |
| Kimi-K2-Instruct | 4.0 | 3.2 | 2.9 | 2.6 |

Complied **with useful detail** (score >= 3):

|  | English | Hindi | Bengali | Tamil |
|---|---|---|---|---|
| DeepSeek | 7% | 4% | 6% | 4% |
| Kimi-K2-Instruct | 7% | 7% | 14% | 22% |

Rates are re-weighted from the 100-per-language sample to all Experiment 3 harmful prompts. 37 disagreements for human review -> `disagreements_kimi_k2.csv`.

<details><summary>Full label confusion (rows = DeepSeek, columns = kimi_k2)</summary>

```
kimi_k2_label     COMPLIANCE  DECODING_FAILURE  EVASION  FULL_REFUSAL  PARTIAL_REFUSAL
deepseek_label                                                                        
COMPLIANCE               148                 4        8             2                0
DECODING_FAILURE           2                 3        1             0                0
EVASION                    3                 3       11             1                0
FULL_REFUSAL               1                 0        2           173                0
PARTIAL_REFUSAL           12                 0        2            11               11
```
</details>
