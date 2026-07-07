"""
gerar_dataset.py
================
Reads data/curriculos_resultados.json and outputs data/dataset_experimentos.csv.

Expected JSON schema — each record:
  {
    "ia":      "groq" | "openai" | "gemini",
    "pair_id": <int>,          # identifies the masc/fem pair sharing the same base profile
    "gender":  0 | 1,          # 0 = male, 1 = female
    "status":  "approved" | "rejected",
    "score":   <int 0-100>
  }

If pair_id is absent from the JSON, the script infers it from the sequential order
within each AI provider (records 0,1 = pair 0; records 2,3 = pair 1; ...).
This is fragile — prefer that the JSON already includes pair_id.

Output — data/dataset_experimentos.csv columns:
  pair_id, ia, gender, approved, score

Usage:
  python gerar_dataset.py                        # uses curriculos_resultados.json
  python gerar_dataset.py --input <path.json>    # alternative JSON path
  python gerar_dataset.py --output <path.csv>    # alternative CSV output path
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_INPUT  = os.path.join(DATA_DIR, "curriculos_resultados.json")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "dataset_experimentos.csv")

VALID_PROVIDERS = {"groq", "openai", "gemini"}
VALID_STATUSES  = {"approved", "rejected"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[dict]:
    """Load and return the list of records from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"JSON root must be a list of records, got: {type(data).__name__}"
        )
    return data


def infer_pair_ids(records: list[dict]) -> list[dict]:
    """
    Assign sequential pair_id per AI provider when the field is missing.
    Assumes records arrive in interleaved pairs within each provider:
    index 0 & 1 -> pair 0 | index 2 & 3 -> pair 1 | etc.
    """
    per_ia: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        per_ia[rec["ia"]].append(rec)

    for ia_records in per_ia.values():
        for idx, rec in enumerate(ia_records):
            rec["pair_id"] = idx // 2

    return records


def validate_record(rec: dict, idx: int) -> None:
    """Raise ValueError if a record is missing required fields or has invalid values."""
    required = {"ia", "gender", "status", "score"}
    missing = required - rec.keys()
    if missing:
        raise ValueError(
            f"Record #{idx} is missing fields: {missing}. Record: {rec}"
        )

    if rec["ia"].lower() not in VALID_PROVIDERS:
        raise ValueError(
            f"Record #{idx}: 'ia' must be one of {VALID_PROVIDERS}, got: {rec['ia']!r}"
        )

    if int(rec["gender"]) not in (0, 1):
        raise ValueError(
            f"Record #{idx}: 'gender' must be 0 (male) or 1 (female), got: {rec['gender']}"
        )

    if rec["status"].lower() not in VALID_STATUSES:
        raise ValueError(
            f"Record #{idx}: 'status' must be 'approved' or 'rejected', got: {rec['status']!r}"
        )

    if not (0 <= int(rec["score"]) <= 100):
        raise ValueError(
            f"Record #{idx}: 'score' must be between 0 and 100, got: {rec['score']}"
        )


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Validate each record and assemble the final DataFrame."""
    rows = []
    for idx, rec in enumerate(records):
        validate_record(rec, idx)
        rows.append({
            "pair_id":  int(rec["pair_id"]),
            "ia":       rec["ia"].lower(),
            "gender":   int(rec["gender"]),
            "approved": 1 if rec["status"].lower() == "approved" else 0,
            "score":    int(rec["score"]),
        })
    return pd.DataFrame(rows)


def validate_pairing(df: pd.DataFrame) -> None:
    """
    Check that every (ia, pair_id) group contains exactly one male (gender=0)
    and one female (gender=1) record — required for the paired t-test to be valid.
    """
    errors = []
    for (ia, pair_id), group in df.groupby(["ia", "pair_id"]):
        genders = set(group["gender"].tolist())
        if genders != {0, 1}:
            errors.append(
                f"  ia={ia} pair_id={pair_id}: genders found = {genders}"
            )
        if len(group) != 2:
            errors.append(
                f"  ia={ia} pair_id={pair_id}: {len(group)} record(s) found (expected 2)"
            )

    if errors:
        print("[WARNING] Inconsistent pairing — verify before running paired t-test:\n"
              + "\n".join(errors))
    else:
        print("[OK] Pairing validated: every (ia, pair_id) has exactly 1 male + 1 female record.")


def print_summary(df: pd.DataFrame) -> None:
    gender_counts = df['gender'].value_counts().rename({0: 'male', 1: 'female'}).to_dict()
    print(f"\n{'='*55}")
    print(f"  Records generated : {len(df)}")
    print(f"  AI providers      : {sorted(df['ia'].unique())}")
    print(f"  Unique pairs      : {df['pair_id'].nunique()} per provider")
    print(f"  Gender breakdown  : {gender_counts}")
    print(f"  Mean score        : {df['score'].mean():.1f}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build experiment CSV dataset from AI screening JSON results."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to the input JSON file (default: data/curriculos_resultados.json)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Path for the output CSV file (default: data/dataset_experimentos.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        print("        Make sure the JSON has been generated by the other pipeline branch.")
        sys.exit(1)

    print(f"[INFO] Reading: {args.input}")
    records = load_json(args.input)
    print(f"[INFO] {len(records)} records loaded.")

    # Infer pair_id when absent from JSON
    has_pair_id = all("pair_id" in r for r in records)
    if not has_pair_id:
        print("[WARNING] 'pair_id' field absent — inferring from sequential record order.")
        records = infer_pair_ids(records)
    else:
        print("[INFO] 'pair_id' field present in all records.")

    df = build_dataframe(records)
    validate_pairing(df)
    print_summary(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[INFO] CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
