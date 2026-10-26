from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thresholding import apply_thresholds, load_thresholds, read_csv, read_model_scores


def detection_metrics(judged: pd.DataFrame) -> dict:
    labels = pd.to_numeric(judged.y)
    positive = labels.eq(1)
    predicted = judged.prediction
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    fp = int((~positive & predicted).sum())
    tn = int((~positive & ~predicted).sum())

    def ratio(a, b):
        return a / b if b else None

    ranks = judged.R.round(12).rank(method="average")
    n_pos, n_neg = tp + fn, tn + fp
    auc = (float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
           if n_pos and n_neg else None)
    return {
        "n": len(judged), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "TPR": ratio(tp, n_pos), "FPR": ratio(fp, n_neg), "AUC": auc,
        "accuracy": ratio(tp + tn, len(judged)),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, n_pos),
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def hit_metrics(
    judged: pd.DataFrame,
    tokens: pd.DataFrame,
    truth: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5),
) -> tuple[pd.DataFrame, dict]:
    ks = tuple(sorted(set(ks)))
    attacks = judged[pd.to_numeric(judged.y).eq(1)]
    grouped = {model: rows for model, rows in tokens.groupby("model", sort=False)}
    truth = truth.set_index("model")
    records = []
    for item in attacks.itertuples(index=False):
        ranking = grouped[item.model]
        rank_column = f"rank_{item.ranking_component}"
        ranks = pd.to_numeric(ranking[rank_column])
        targets = set(truth.loc[item.model, "trigger_token_ids"])
        matched = ranks[ranking.token_id.isin(targets)]
        rank = None if matched.empty else int(matched.min())
        record = {
            "model": item.model,
            "ranking_component": item.ranking_component,
            "accepted_token_ids": json.dumps(sorted(targets)),
            "trigger_rank": rank,
        }
        record.update({f"hit_at_{k}": rank is not None and rank <= k for k in ks})
        records.append(record)
    columns = ["model", "ranking_component", "accepted_token_ids", "trigger_rank",
               *(f"hit_at_{k}" for k in ks)]
    per_model = pd.DataFrame(records, columns=columns)
    summary = {"n": len(per_model)}
    for k in ks:
        count = int(per_model[f"hit_at_{k}"].sum())
        summary[f"hit_at_{k}_count"] = count
        summary[f"Hit@{k}"] = count / len(per_model) if len(per_model) else None
    return per_model, summary


def evaluate(
    input: Path | str,
    thresholds: Path | str,
    output_dir: Path | str,
    *,
    ks: tuple[int, ...] = (1, 5),
) -> dict:
    directory = Path(input)
    metadata = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*/metadata.json"))
    ])
    scores = read_model_scores(directory)
    labels = metadata.set_index("model").label.map({"benign": 0, "backdoored": 1})
    scores["y"] = scores.model.map(labels)
    fitted = load_thresholds(thresholds)
    judged = apply_thresholds(scores, fitted)
    metrics = {"detection": detection_metrics(judged.dropna(subset=["y"]))}
    hits = None
    if ks:
        tokens = pd.concat([
            read_csv(path) for path in sorted(directory.glob("*/token_scores.csv"))
        ], ignore_index=True)
        hits, metrics["trigger_discovery"] = hit_metrics(
            judged, tokens, metadata, tuple(ks)
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judged.to_csv(output_dir / "decisions.csv", index=False)
    if hits is not None:
        hits.to_csv(output_dir / "hits.csv", index=False)
    else:
        (output_dir / "hits.csv").unlink(missing_ok=True)
    summary = json.dumps(metrics, indent=2)
    (output_dir / "metrics.json").write_text(summary + "\n", encoding="utf-8")
    return metrics
