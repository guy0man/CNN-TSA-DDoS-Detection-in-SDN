import os, json, argparse
import pandas as pd
from typing import Dict, Any, Optional

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pick_metrics(d: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    if not isinstance(d, dict):
        for k in METRIC_KEYS:
            out[k] = float("nan")
        return out
    for k in METRIC_KEYS:
        v = d.get(k)
        out[k] = float(v) if v is not None else float("nan")
    return out

def _format_split(prefix: str, m: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix} Accuracy":   m["accuracy"],
        f"{prefix} Precision":  m["precision"],
        f"{prefix} Recall":     m["recall"],
        f"{prefix} F1-Score":   m["f1"],
        f"{prefix} ROC-AUC":    m["roc_auc"],
    }

def _parse_run_dir(run_dir: str, root: str):
    """
    Expecting structure:
      {root}/{variant}/{hp}/{timestamp}/
    Returns (variant, hp, timestamp) best-effort even if structure differs.
    """
    rel = os.path.relpath(run_dir, root).replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    variant = parts[0] if len(parts) >= 1 else ""
    hp      = parts[1] if len(parts) >= 2 else ""
    ts      = parts[2] if len(parts) >= 3 else ""
    return variant, hp, ts

def _collect_run_dirs(root: str):
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metrics_test.json" in filenames or "metrics_test_thresholded.json" in filenames:
            runs.append(dirpath)
    # stable order: by variant, hp, timestamp (lexicographic)
    runs.sort()
    return runs

def _assign_preproc_method_ids(variants):
    return {v: i + 1 for i, v in enumerate(variants)}

def _assign_training_variation(df: pd.DataFrame) -> pd.DataFrame:
    # Number runs 1..N within each variant in the order they appear
    df = df.copy()
    df["Training Variation"] = None
    for v, g in df.groupby("variant", sort=False):
        order = list(g.index)
        mapping = {idx: i + 1 for i, idx in enumerate(order)}
        df.loc[order, "Training Variation"] = [mapping[idx] for idx in order]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="models/cnn_tsa", help="Root folder containing run subdirectories.")
    ap.add_argument("--out",  default="results_table.csv", help="Output CSV path.")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Root folder not found: {args.root}")

    run_dirs = _collect_run_dirs(args.root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs with metrics found under: {args.root}")

    rows = []
    for rd in run_dirs:
        variant, hp, ts = _parse_run_dir(rd, args.root)

        train_json = _safe_load_json(os.path.join(rd, "metrics_train.json"))
        val_json   = _safe_load_json(os.path.join(rd, "metrics_val.json"))
        test_json  = _safe_load_json(os.path.join(rd, "metrics_test.json")) \
                     or _safe_load_json(os.path.join(rd, "metrics_test_thresholded.json"))

        m_train = _pick_metrics(train_json)
        m_val   = _pick_metrics(val_json)
        m_test  = _pick_metrics(test_json)

        flat = {
            "variant": variant,
            "hp_label": hp,
            "timestamp": ts,
            "run_dir": rd,
        }
        flat.update(_format_split("Train", m_train))
        flat.update(_format_split("Test",  m_test))
        flat.update(_format_split("Val",   m_val))
        rows.append(flat)

    df = pd.DataFrame(rows)

    # Map variants to numeric "Preprocessing Method" ids and assign Training Variation per variant
    variants_order = list(pd.unique(df["variant"]))
    vmap = _assign_preproc_method_ids(variants_order)
    df["Preprocessing Method"] = df["variant"].map(vmap)
    df = _assign_training_variation(df)

    # Final column order
    ordered_cols = [
        "Preprocessing Method", "Training Variation",
        "Train Accuracy", "Train Precision", "Train Recall", "Train F1-Score", "Train ROC-AUC",
        "Test Accuracy",  "Test Precision",  "Test Recall",  "Test F1-Score",  "Test ROC-AUC",
        "Val Accuracy",   "Val Precision",   "Val Recall",   "Val F1-Score",   "Val ROC-AUC",
        # keep identifiers at the end for traceability
        "variant", "hp_label", "timestamp", "run_dir",
    ]
    df = df.reindex(columns=[c for c in ordered_cols if c in df.columns])

    # Round: ROC-AUC to 6 decimals, others to 4
    for col in df.select_dtypes(include="number").columns:
        if "ROC-AUC" in col:
            df[col] = df[col].round(6)
        else:
            df[col] = df[col].round(4)

    df.to_csv(args.out, index=False)
    print(f"✅ Wrote results (rounded) → {args.out}")
    print("Variants → Method numbers:", vmap)

if __name__ == "__main__":
    main()
