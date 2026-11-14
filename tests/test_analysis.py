"""Tests for src.analysis.encoding_analysis (synthetic results, no API calls)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.encoding_analysis import holm, mcnemar_exact, run_analysis, wilson

ENCODINGS = ["plaintext", "base64", "multilingual_hi"]
JUDGES = ["judge-a", "judge-b"]


def test_wilson_bounds():
    lo, hi = wilson(0, 50)
    assert lo == 0 and 0.05 < hi < 0.08
    lo, hi = wilson(25, 50)
    assert lo < 0.5 < hi
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2 ** 10)


def test_holm_is_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.09) and adj[1] == pytest.approx(0.09)  # monotone step-down
    assert max(adj) <= 1.0


def make_run(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for model in ["org/Small-Instruct", "org/Big-Instruct"]:
        for enc in ENCODINGS:
            for i in range(40):
                harmful = i < 30
                src = "fortress" if i % 2 else "advbench"
                decoded = enc == "plaintext" or rng.random() < (0.9 if "Big" in model else 0.2)
                if not decoded:
                    label = "DECODING_FAILURE"
                else:
                    label = "COMPLIANCE" if rng.random() < (0.4 if harmful else 0.9) else "FULL_REFUSAL"
                rows.append(dict(
                    prompt_id=f"p{i:03d}", model=model, encoding_type=enc, classification=label,
                    prompt_type="harmful" if harmful else "benign", source_dataset=src,
                    decoding_status="n/a" if enc == "plaintext" else ("correct" if decoded else "failed"),
                    specificity=3 if label == "COMPLIANCE" else 1, judge_model=JUDGES[i % 2]))
    rows.append({**rows[0], "classification": "JUDGE_ERROR", "prompt_id": "err"})  # excluded from rates
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "results.csv", index=False)
    return run


def test_run_analysis_writes_outputs(tmp_path):
    paths = run_analysis(make_run(tmp_path), verbose=False)
    for key in ["asr", "lift_vs_plaintext", "conditional_asr", "understanding", "benign", "judge_robustness",
                "fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "report"]:
        assert paths[key].exists() and paths[key].stat().st_size > 0, key

    asr = pd.read_csv(paths["asr"])
    assert set(asr["subset"]) == {"all", "direct", "fortress"}
    assert (asr[asr["subset"] == "all"]["n"] == 30).all()  # error row excluded, benign excluded

    und = pd.read_csv(paths["understanding"])
    rates = und[[c for c in und.columns if c.endswith("_rate")]].sum(axis=1)
    assert np.allclose(rates, 1.0)

    lift = pd.read_csv(paths["lift_vs_plaintext"])
    assert "plaintext" not in set(lift["encoding"])
    assert (lift["ci_lo"] <= lift["lift"]).all() and (lift["lift"] <= lift["ci_hi"]).all()
    assert "## ASR" in paths["report"].read_text(encoding="utf-8")
