from __future__ import annotations

import pandas as pd


def load_dataset(path: str, text_column: str, label_column: str) -> pd.DataFrame:
    from pathlib import Path
    data_path = Path(path)
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent / path
    df = pd.read_csv(data_path)
    missing = [c for c in [text_column, label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[[text_column, label_column]].copy()
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].astype(str)
    return df


def normalize_labels(df: pd.DataFrame, label_column: str, positive_labels: list) -> pd.DataFrame:
    positive = {str(v).lower() for v in positive_labels}
    out = df.copy()
    out[label_column] = out[label_column].apply(lambda x: 1 if str(x).lower() in positive else 0)
    return out
