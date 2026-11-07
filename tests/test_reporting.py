from ironlaw_classifier.pipeline import Sample
from ironlaw_classifier.report import CASES, STAGES, summarize


def test_amdahl_and_summary():
    # Controlled toy example: 8 ms pre + 1 ms infer + 1 ms post baseline.
    stages = {
        "baseline": (8.0, 1.0, 1.0),
        "inference_only": (8.0, 0.5, 1.0),
        "preprocess_only": (2.0, 1.0, 1.0),
        "both": (2.0, 0.5, 1.0),
    }
    records = [Sample(case, *values, sum(values)) for case, values in stages.items() for _ in range(3)]
    report = summarize(records)
    assert report["baseline_dominant_stage"] == "preprocessing_ms"
    assert abs(report["baseline"]["mean_ms"]["end_to_end_ms"] - 10.0) < 1e-10
    assert abs(report["improvement_percent_vs_baseline"]["inference_only"] - 5.0) < 1e-10
    assert abs(report["improvement_percent_vs_baseline"]["preprocess_only"] - 60.0) < 1e-10
    assert abs(report["amdahl_projections"]["inference_only"]["predicted_end_to_end_speedup"] - (10 / 9.5)) < 1e-10
