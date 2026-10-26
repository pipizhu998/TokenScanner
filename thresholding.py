from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = {"UNet": "has_unet", "TE": "has_te"}


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype={"model": str})


def read_model_scores(directory: Path | str) -> pd.DataFrame:
    return pd.concat(
        [read_csv(path) for path in sorted(Path(directory).glob("*/model_scores.csv"))],
        ignore_index=True,
    )


def prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    result = scores.copy().reset_index(drop=True)
    for component, flag in COMPONENTS.items():
        parsed = result[flag].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        result[flag] = parsed.astype(bool)
        column = f"A_{component}"
        result[column] = pd.to_numeric(result[column].mask(result[column].eq("")))
    return result


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, values, side="right") / len(reference)


def fit_thresholds(reference_scores: pd.DataFrame, quantile: float = 0.9) -> dict:
    reference = prepare_scores(reference_scores)
    groups = []
    for (has_unet, has_te), rows in reference.groupby(["has_unet", "has_te"], sort=True):
        group = {"has_unet": bool(has_unet), "has_te": bool(has_te), "n_reference": len(rows)}
        joint = np.zeros(len(rows), dtype=float)
        for component, flag in COMPONENTS.items():
            values = rows[f"A_{component}"].to_numpy(float)
            pool = np.sort(values) if group[flag] else None
            group[f"{component}_reference"] = None if pool is None else pool.tolist()
            if pool is not None:
                joint = np.maximum(joint, ecdf(pool, values))
        group["tau_c"] = float(np.quantile(joint, quantile, method="higher"))
        groups.append(group)
    return {"quantile": quantile, "groups": groups}


def apply_thresholds(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    result = prepare_scores(scores)
    groups = {(g["has_unet"], g["has_te"]): g for g in thresholds["groups"]}
    for column in ("r_UNet", "r_TE", "r_joint", "tau_c"):
        result[column] = np.nan
    for flags, rows in result.groupby(["has_unet", "has_te"], sort=False):
        group = groups[flags]
        joint = np.zeros(len(rows), dtype=float)
        for component, flag in COMPONENTS.items():
            if group[flag]:
                percentile = ecdf(np.asarray(group[f"{component}_reference"]),
                                  rows[f"A_{component}"].to_numpy(float))
                result.loc[rows.index, f"r_{component}"] = percentile
                joint = np.maximum(joint, percentile)
        result.loc[rows.index, "r_joint"] = joint
        result.loc[rows.index, "tau_c"] = group["tau_c"]
    result["R"] = result.r_joint / result.tau_c
    result["prediction"] = result.r_joint.ge(result.tau_c)
    use_te = ~result.has_unet | (result.has_te & result.r_TE.gt(result.r_UNet))
    result["ranking_component"] = np.where(use_te, "TE", "UNet")
    return result


def save_thresholds(thresholds: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")


def load_thresholds(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
