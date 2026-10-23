"""Reproducible population-shift experiment. Run from any directory."""

from pathlib import Path
import argparse, hashlib, json, platform, urllib.request, zipfile, io
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
FEATURES = NUM + CAT
COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_data():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "adult.data").exists() or not (raw / "adult.test").exists():
        payload = urllib.request.urlopen(
            "https://archive.ics.uci.edu/static/public/2/adult.zip", timeout=60
        ).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for name in ["adult.data", "adult.test", "adult.names"]:
                (raw / name).write_bytes(z.read(name))
    frames = [
        pd.read_csv(
            raw / f, names=COLS, skipinitialspace=True, na_values="?", comment="|"
        )
        for f in ["adult.data", "adult.test"]
    ]
    df = pd.concat(frames, ignore_index=True)
    df["income"] = df.income.str.rstrip(".")
    df["y"] = (df.income == ">50K").astype(int)
    initial = len(df)
    df = df.dropna(subset=["native-country"]).copy()
    known = len(df)
    # Deduplicate full source rows; keep conflicting outcomes, group them together.
    df = df.drop_duplicates(COLS).reset_index(drop=True)
    df["population"] = np.where(
        df["native-country"].eq("United-States"), "US-born", "Foreign-born"
    )
    keys = FEATURES + ["population"]
    df["group_id"] = df.groupby(keys, dropna=False, sort=True).ngroup()
    conflicting = int((df.groupby("group_id").y.transform("nunique") > 1).sum())
    df["population"] = np.where(
        df["native-country"].eq("United-States"), "US-born", "Foreign-born"
    )
    audit = dict(
        raw_rows=initial,
        unknown_birthplace_removed=initial - known,
        exact_duplicates_removed=known - len(df),
        conflicting_label_rows_retained=conflicting,
        retained_rows=len(df),
        unique_predictor_groups=int(df.group_id.nunique()),
    )
    return df, audit


def split_data(df):
    groups = df.groupby("group_id").agg(
        population=("population", "first"), rate=("y", "mean")
    )
    groups["stratum"] = (groups.rate >= 0.5).astype(int)
    source = groups[groups.population == "US-born"]
    target = groups[groups.population == "Foreign-born"]
    train, rest = train_test_split(
        source, test_size=0.4, stratify=source.stratum, random_state=SEED
    )
    source_cal, source_test = train_test_split(
        rest, test_size=0.5, stratify=rest.stratum, random_state=SEED + 1
    )
    target_cal, target_test = train_test_split(
        target, test_size=0.5, stratify=target.stratum, random_state=SEED + 2
    )
    partitions = dict(
        train=train,
        source_cal=source_cal,
        source_test=source_test,
        target_cal=target_cal,
        target_test=target_test,
    )
    splits = {k: df[df.group_id.isin(g.index)].copy() for k, g in partitions.items()}
    ids = [set(x.group_id) for x in splits.values()]
    assert all(not a & b for i, a in enumerate(ids) for b in ids[i + 1 :])
    assert sum(len(x) for x in splits.values()) == len(df)
    assert all(x.y.nunique() == 2 for x in splits.values())
    return splits


def preprocessor():
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUM,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                CAT,
            ),
        ]
    )


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrator(y, p):
    # Approximately unpenalized affine logistic recalibration of log odds.
    return LogisticRegression(C=1e6, max_iter=2000).fit(logit(p), y)


def metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "mean_probability": float(np.mean(p)),
    }


def bootstrap(y, p, reps, seed, groups):
    rng = np.random.default_rng(seed)
    values = []
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(reps):
        ix = np.concatenate(
            [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
        )
        if np.unique(y[ix]).size == 2:
            values.append(metrics(y[ix], p[ix]))
    return {
        k: np.percentile([v[k] for v in values], [2.5, 97.5]).tolist()
        for k in ["roc_auc", "pr_auc_ap", "brier"]
    }


def main(reps=500):
    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df, audit = load_data()
    splits = split_data(df)
    manifest = pd.concat(
        [
            x.assign(split=k)[["split", "population", "group_id", "y"]]
            for k, x in splits.items()
        ]
    ).sort_index()
    manifest.to_csv(out / "split_manifest.csv", index_label="clean_row_id")
    eda = []
    for name, x in splits.items():
        eda.append(
            dict(
                split=name,
                n=len(x),
                prevalence=float(x.y.mean()),
                age_mean=float(x.age.mean()),
                education_mean=float(x["education-num"].mean()),
                hours_mean=float(x["hours-per-week"].mean()),
                female_share=float(x.sex.eq("Female").mean()),
            )
        )
    pd.DataFrame(eda).to_csv(out / "population_summary.csv", index=False)
    df.groupby("population")[FEATURES].apply(lambda x: x.isna().mean()).to_csv(
        out / "missingness.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ["age", "education-num", "hours-per-week"]):
        for pop, x in df.groupby("population"):
            ax.hist(x[col], bins=20, density=True, alpha=0.45, label=pop)
        ax.set(xlabel=col, ylabel="Density", title=f"Distribution of {col}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "figures/population_shift.png", dpi=180)
    plt.close(fig)
    models = {
        "Logistic regression": LogisticRegression(C=1, max_iter=2000),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1,
            early_stopping=False,
            random_state=SEED,
        ),
    }
    rows = []
    preds = []
    calparams = []
    deltas = []
    subgroups = []
    diagnostic = []
    shift_deltas = []
    for mi, (name, estimator) in enumerate(models.items()):
        model = make_pipeline(preprocessor(), estimator)
        model.fit(splits["train"][FEATURES], splits["train"].y)
        calibration = {}
        for cname in ["source_cal", "target_cal"]:
            x = splits[cname]
            c = calibrator(x.y, model.predict_proba(x[FEATURES])[:, 1])
            calibration[cname] = c
            calparams.append(
                dict(
                    model=name,
                    calibration_sample=cname,
                    intercept=float(c.intercept_[0]),
                    slope=float(c.coef_[0, 0]),
                )
            )
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        domain_predictions = {}
        for di, domain in enumerate(["source_test", "target_test"]):
            x = splits[domain]
            y = x.y.to_numpy()
            base = model.predict_proba(x[FEATURES])[:, 1]
            domain_predictions[domain] = (y, base)
            diag = calibrator(y, base)
            diagnostic.append(
                dict(
                    model=name,
                    domain=domain,
                    intercept=float(diag.intercept_[0]),
                    slope=float(diag.coef_[0, 0]),
                )
            )
            for sex in ["Female", "Male"]:
                mask = x.sex.eq(sex).to_numpy()
                if y[mask].sum() >= 20 and (1 - y[mask]).sum() >= 20:
                    subgroups.append(
                        dict(
                            model=name,
                            domain=domain,
                            sex=sex,
                            n=int(mask.sum()),
                            prevalence=float(y[mask].mean()),
                            **metrics(y[mask], base[mask]),
                        )
                    )
            variants = {
                "Original": base,
                "Source recalibrated": calibration["source_cal"].predict_proba(
                    logit(base)
                )[:, 1],
                "Target recalibrated": calibration["target_cal"].predict_proba(
                    logit(base)
                )[:, 1],
            }
            for vi, (variant, p) in enumerate(variants.items()):
                m = metrics(y, p)
                ci = bootstrap(
                    y, p, reps, SEED + 100 * mi + 10 * di, x.group_id.to_numpy()
                )  # same resamples within domain
                rows.append(
                    dict(
                        model=name,
                        domain=domain,
                        variant=variant,
                        n=len(y),
                        prevalence=float(y.mean()),
                        **m,
                        **{
                            f"{k}_{side}": v[j]
                            for k, v in ci.items()
                            for j, side in enumerate(["lo", "hi"])
                        },
                    )
                )
                preds.extend(
                    dict(
                        row_id=int(i),
                        model=name,
                        domain=domain,
                        variant=variant,
                        y=int(yy),
                        p=float(pp),
                    )
                    for i, yy, pp in zip(x.index, y, p)
                )
                if variant == "Original":
                    fpr, tpr, _ = roc_curve(y, p)
                    axes[di, 0].plot(fpr, tpr, label=f'AUC {m["roc_auc"]:.3f}')
                    precision, recall, _ = precision_recall_curve(y, p)
                    axes[di, 1].plot(
                        recall, precision, label=f'AP {m["pr_auc_ap"]:.3f}'
                    )
                observed, predicted = calibration_curve(
                    y, p, n_bins=10, strategy="quantile"
                )
                axes[di, 2].plot(predicted, observed, "o-", label=variant)
            # Paired target recalibration Brier difference on identical test rows.
            rng = np.random.default_rng(SEED + di)
            loss = (variants["Target recalibrated"] - y) ** 2 - (base - y) ** 2
            blocks = [
                np.flatnonzero(x.group_id.to_numpy() == g) for g in x.group_id.unique()
            ]
            bs = [
                float(
                    loss[
                        np.concatenate(
                            [
                                blocks[j]
                                for j in rng.integers(0, len(blocks), len(blocks))
                            ]
                        )
                    ].mean()
                )
                for _ in range(reps)
            ]
            deltas.append(
                dict(
                    model=name,
                    domain=domain,
                    target_recalibration_brier_delta=float(loss.mean()),
                    lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)),
                )
            )
            for j, (xl, yl, title) in enumerate(
                [
                    ("False positive rate", "True positive rate", "ROC"),
                    ("Recall", "Precision", "Precision–recall"),
                    (
                        "Mean predicted probability",
                        "Observed event frequency",
                        "Calibration: quantile bins",
                    ),
                ]
            ):
                ax = axes[di, j]
                ax.set(
                    xlabel=xl,
                    ylabel=yl,
                    title=f"{domain}: {title}",
                    xlim=(0, 1),
                    ylim=(0, 1),
                )
                ax.legend(fontsize=8)
                if j != 1:
                    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
                else:
                    ax.axhline(
                        y.mean(), color="gray", linestyle="--", label="Prevalence"
                    )
        ys, ps = domain_predictions["source_test"]
        yt, pt = domain_predictions["target_test"]
        rng = np.random.default_rng(SEED + mi)
        differences = []
        sb = [
            np.flatnonzero(splits["source_test"].group_id.to_numpy() == g)
            for g in splits["source_test"].group_id.unique()
        ]
        tb = [
            np.flatnonzero(splits["target_test"].group_id.to_numpy() == g)
            for g in splits["target_test"].group_id.unique()
        ]
        for _ in range(reps):
            si = np.concatenate([sb[j] for j in rng.integers(0, len(sb), len(sb))])
            ti = np.concatenate([tb[j] for j in rng.integers(0, len(tb), len(tb))])
            sm = metrics(ys[si], ps[si])
            tm = metrics(yt[ti], pt[ti])
            differences.append(
                {k: tm[k] - sm[k] for k in ["roc_auc", "pr_auc_ap", "brier"]}
            )
        for k in ["roc_auc", "pr_auc_ap", "brier"]:
            vals = [d[k] for d in differences]
            shift_deltas.append(
                dict(
                    model=name,
                    metric=k,
                    target_minus_source=metrics(yt, pt)[k] - metrics(ys, ps)[k],
                    lo=float(np.percentile(vals, 2.5)),
                    hi=float(np.percentile(vals, 97.5)),
                )
            )
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(
            out / f'figures/{"logistic" if mi==0 else "boosting"}_evaluation.png',
            dpi=180,
        )
        plt.close(fig)
        if mi == 0:
            coef = pd.DataFrame(
                {
                    "feature": model[0].get_feature_names_out(),
                    "log_odds_coefficient": model[1].coef_[0],
                }
            )
            coef.to_csv(out / "logistic_coefficients.csv", index=False)
    pd.DataFrame(subgroups).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(diagnostic).to_csv(out / "calibration_diagnostics.csv", index=False)
    pd.DataFrame(shift_deltas).to_csv(out / "shift_deltas.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(calparams).to_csv(out / "recalibration_parameters.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "paired_brier_deltas.csv", index=False)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "data/raw").glob("adult.*")
        if p.is_file()
    }
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                seed=SEED,
                bootstrap_replicates=reps,
                audit=audit,
                python=platform.python_version(),
                sklearn=sklearn.__version__,
                numpy=np.__version__,
                pandas=pd.__version__,
                matplotlib=matplotlib.__version__,
                raw_sha256=hashes,
            ),
            indent=2,
        )
    )
    print(
        pd.DataFrame(rows)[
            ["model", "domain", "variant", "roc_auc", "pr_auc_ap", "brier"]
        ].to_string(index=False)
    )
    print("Audit:", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap replicates")
    main(args.bootstrap)
