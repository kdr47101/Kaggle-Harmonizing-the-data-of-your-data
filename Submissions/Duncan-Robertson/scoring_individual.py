from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_DIR = SCRIPT_DIR / "data" / "Gemini-Extract"
DEFAULT_TRUE_DIR = SCRIPT_DIR / "data" / "Training-SDRF"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "detailed_evaluation_metrics.csv"
MISSING_VALUE = "Not Applicable"


class ParticipantVisibleError(Exception):
    pass


def normalize_scalar(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    text = str(value).strip()
    if not text:
        return MISSING_VALUE
    if text.lower() in {
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "not applicable",
        "not available",
    }:
        return MISSING_VALUE
    return text


def flatten_json_value(value: Any) -> List[str]:
    if isinstance(value, list):
        flattened: List[str] = []
        for item in value:
            flattened.extend(flatten_json_value(item))
        return flattened
    return [normalize_scalar(value)]


def load_metadata_json(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # Case 1: row-wise JSON
    if isinstance(data, list):
        rows: List[Dict[str, Any]] = []
        for i, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                raise ParticipantVisibleError(
                    f"Expected each row in {json_path} to be a JSON object, but row {i} is {type(row).__name__}."
                )
            rows.append(row)
        return rows

    # Case 2: dict keyed by raw data file
    if isinstance(data, dict):
        rows: List[Dict[str, Any]] = []
        for raw_file, metadata in data.items():
            if not isinstance(metadata, dict):
                raise ParticipantVisibleError(
                    f"Expected each top-level entry in {json_path} to map to a JSON object, "
                    f"but key {raw_file!r} maps to {type(metadata).__name__}."
                )

            row: Dict[str, Any] = {"Raw Data File": raw_file}
            for key, value in metadata.items():
                row[key] = value
            rows.append(row)
        return rows

    raise ParticipantVisibleError(
        f"Expected a top-level list or dict in {json_path}, got {type(data).__name__}."
    )


def metadata_json_to_dataframe(json_path: Path) -> pd.DataFrame:
    rows = load_metadata_json(json_path)
    flattened_rows: List[Dict[str, Any]] = []

    for row in rows:
        out: Dict[str, Any] = {}

        for key, value in row.items():
            values = [v for v in flatten_json_value(value) if v != MISSING_VALUE]
            if not values:
                continue
            if len(values) == 1:
                out[key] = values[0]
            else:
                out[key] = values

        if "PXD" not in out:
            pxd_from_name = json_path.stem.split("_")[0].strip()
            if pxd_from_name:
                out["PXD"] = pxd_from_name

        flattened_rows.append(out)

    df = pd.DataFrame(flattened_rows)
    if df.empty:
        raise ParticipantVisibleError(f"No usable rows were found in {json_path}.")
    return df


def load_sdrf(sdrf_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    if "PXD" not in sdrf_df.columns:
        raise ParticipantVisibleError("Both solution and submission must include a 'PXD' column.")

    sdrf_dict: Dict[str, Dict[str, List[str]]] = {}

    for pxd, pxd_df in sdrf_df.groupby("PXD"):
        sdrf_dict[pxd] = {}

        for col in pxd_df.columns:
            collected: List[str] = []

            for value in pxd_df[col].tolist():
                if isinstance(value, list):
                    flattened = [normalize_scalar(v) for v in value]
                else:
                    flattened = flatten_json_value(value)

                for v in flattened:
                    if v == MISSING_VALUE:
                        continue

                    if "NT=" in v:
                        parts = [r.strip() for r in v.split(";") if "NT=" in r]
                        if parts:
                            nt_value = parts[0].replace("NT=", "").strip()
                            collected.append(nt_value if nt_value else v.strip())
                        else:
                            collected.append(v.strip())
                    else:
                        collected.append(v.strip())

            uniq = list(dict.fromkeys(collected))
            if not uniq:
                continue
            if len(uniq) == 1 and uniq[0] == MISSING_VALUE:
                continue

            sdrf_dict[pxd][col] = uniq

    return sdrf_dict


def _string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def harmonize_and_evaluate_datasets(
    truth: Dict[str, Dict[str, List[str]]],
    pred: Dict[str, Dict[str, List[str]]],
    threshold: float = 0.80,
    complete_absence: float = float("nan"),
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]], pd.DataFrame]:
    eval_metrics = {
        "pxd": [],
        "AnnotationType": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "jacc": [],
    }
    harmonized_truth: Dict[str, Dict[str, List[int]]] = {}
    harmonized_pred: Dict[str, Dict[str, List[int]]] = {}

    common_pubs = set(truth) & set(pred)
    for pub in common_pubs:
        harmonized_truth[pub], harmonized_pred[pub] = {}, {}

        for category in (set(truth[pub]) & set(pred[pub])):
            vals_truth = list(dict.fromkeys(truth[pub][category]))
            vals_pred = list(dict.fromkeys(pred[pub][category]))
            all_vals = vals_truth + [v for v in vals_pred if v not in vals_truth]

            if len(vals_truth) == 0 and len(vals_pred) == 0:
                harm_truth = []
                harm_pred = []
            elif len(all_vals) == 1:
                str2cid = {all_vals[0]: 0}
                harm_truth = [str2cid[s] for s in vals_truth]
                harm_pred = [str2cid[s] for s in vals_pred]
            else:
                n_vals = len(all_vals)
                dist = np.zeros((n_vals, n_vals), dtype=float)

                for i in range(n_vals):
                    for j in range(i + 1, n_vals):
                        sim = _string_similarity(all_vals[i], all_vals[j])
                        d = 1.0 - sim
                        dist[i, j] = d
                        dist[j, i] = d

                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    metric="precomputed",
                    linkage="average",
                    distance_threshold=1.0 - threshold,
                )
                labels = clusterer.fit_predict(dist)
                str2cid = {s: int(labels[i]) for i, s in enumerate(all_vals)}
                harm_truth = [str2cid[s] for s in vals_truth]
                harm_pred = [str2cid[s] for s in vals_pred]

            harmonized_truth[pub][category] = harm_truth
            harmonized_pred[pub][category] = harm_pred

            uniq = sorted(set(harm_truth) | set(harm_pred))
            if not uniq:
                p = r = f = complete_absence
                j = 1.0
            else:
                y_true = [1 if u in harm_truth else 0 for u in uniq]
                y_pred = [1 if u in harm_pred else 0 for u in uniq]

                p = precision_score(y_true, y_pred, average="macro", zero_division=0)
                r = recall_score(y_true, y_pred, average="macro", zero_division=0)
                f = f1_score(y_true, y_pred, average="macro", zero_division=0)

                set_truth, set_pred = set(harm_truth), set(harm_pred)
                j = 1.0 if (not set_truth and not set_pred) else len(set_truth & set_pred) / len(set_truth | set_pred)

            eval_metrics["pxd"].append(pub)
            eval_metrics["AnnotationType"].append(category)
            eval_metrics["precision"].append(p)
            eval_metrics["recall"].append(r)
            eval_metrics["f1"].append(f)
            eval_metrics["jacc"].append(j)

    return harmonized_truth, harmonized_pred, pd.DataFrame(eval_metrics)


def score_dataframes(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "ID",
) -> Tuple[pd.DataFrame, float]:
    if row_id_column_name and row_id_column_name in solution.columns:
        solution = solution.drop(columns=[row_id_column_name])
    if row_id_column_name and row_id_column_name in submission.columns:
        submission = submission.drop(columns=[row_id_column_name])

    if "PXD" not in solution.columns or "PXD" not in submission.columns:
        raise ParticipantVisibleError("Both solution and submission must include a 'PXD' column.")

    sol = load_sdrf(solution)
    sub = load_sdrf(submission)
    _, _, eval_df = harmonize_and_evaluate_datasets(sol, sub, threshold=0.80)

    vals = eval_df["f1"].dropna()
    return eval_df, float(vals.mean()) if not vals.empty else 0.0


def score_json_files(
    solution_json: Path,
    submission_json: Path,
    row_id_column_name: str = "ID",
) -> Tuple[pd.DataFrame, float]:
    solution_df = metadata_json_to_dataframe(solution_json)
    submission_df = metadata_json_to_dataframe(submission_json)
    return score_dataframes(solution_df, submission_df, row_id_column_name=row_id_column_name)


def resolve_pair_from_pxd(pxd: str, truth_dir: Path, pred_dir: Path) -> Tuple[Path, Path]:
    truth_path = truth_dir / f"{pxd}_metadata.json"
    pred_path = pred_dir / f"{pxd}_metadata.json"

    if not truth_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {truth_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

    return truth_path, pred_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a predicted PXD metadata JSON against the ground-truth PXD metadata JSON."
    )
    parser.add_argument(
        "--pxd",
        type=str,
        help="PXD accession, e.g. PXD000070. If provided, --truth-json and --pred-json are optional.",
    )
    parser.add_argument("--truth-json", type=Path, help="Path to the ground-truth JSON file.")
    parser.add_argument("--pred-json", type=Path, help="Path to the predicted JSON file.")
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUE_DIR,
        help=f"Directory containing ground-truth JSON files. Default: {DEFAULT_TRUE_DIR}",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help=f"Directory containing predicted JSON files. Default: {DEFAULT_PRED_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the detailed evaluation metrics CSV. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    if args.truth_json and args.pred_json:
        truth_json = args.truth_json
        pred_json = args.pred_json
    elif args.pxd:
        truth_json, pred_json = resolve_pair_from_pxd(args.pxd, args.truth_dir, args.pred_dir)
    else:
        raise ParticipantVisibleError("Provide either --pxd or both --truth-json and --pred-json.")

    eval_df, final_score = score_json_files(truth_json, pred_json, row_id_column_name="ID")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(args.output, index=False)

    print(eval_df)
    print(f"\nFinal SDRF Average F1 Score: {final_score:.6f}")
    print(f"Detailed metrics saved to: {args.output}")


if __name__ == "__main__":
    main()