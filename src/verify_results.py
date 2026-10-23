"""Check split integrity, raw provenance and independently recomputed metrics."""

import hashlib, json
import numpy as np
import pandas as pd
from run_analysis import ROOT, load_data, split_data, metrics


def main():
    df, _ = load_data()
    splits = split_data(df)
    manifest = pd.read_csv(
        ROOT / "reports/split_manifest.csv", index_col="clean_row_id"
    )
    assert len(manifest) == len(df)
    assert manifest.groupby("group_id")["split"].nunique().max() == 1
    for name, x in splits.items():
        assert set(manifest[manifest.split == name].index) == set(x.index)
    assert splits["train"].population.eq("US-born").all()
    assert splits["target_cal"].population.eq("Foreign-born").all()
    assert splits["target_test"].population.eq("Foreign-born").all()
    metadata = json.loads((ROOT / "reports/run_metadata.json").read_text())
    for f, h in metadata["raw_sha256"].items():
        assert hashlib.sha256((ROOT / "data/raw" / f).read_bytes()).hexdigest() == h
    predictions = pd.read_csv(ROOT / "reports/predictions.csv")
    assert np.isfinite(predictions.p).all() and predictions.p.between(0, 1).all()
    summary = pd.read_csv(ROOT / "reports/metrics.csv")
    for row in summary.itertuples():
        p = predictions[
            (predictions.model == row.model)
            & (predictions.domain == row.domain)
            & (predictions.variant == row.variant)
        ]
        assert not p.row_id.duplicated().any()
        assert set(p.row_id) == set(splits[row.domain].index)
        assert np.array_equal(p.y.to_numpy(), df.loc[p.row_id, "y"].to_numpy())
        for k, v in metrics(p.y.to_numpy(), p.p.to_numpy()).items():
            assert np.isclose(v, getattr(row, k), atol=1e-12), (row.model, k)
        for k in ["roc_auc", "pr_auc_ap", "brier"]:
            assert getattr(row, k + "_lo") <= getattr(row, k + "_hi")
    print(
        "PASS: disjoint predictor groups, complete partitions, source-only training, provenance hashes, labels, finite probabilities and all 12 metric rows."
    )


if __name__ == "__main__":
    main()
