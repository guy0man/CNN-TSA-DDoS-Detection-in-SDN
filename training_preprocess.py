# training_preprocess.py
import os, json, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import pyarrow.parquet as pq

# -----------------------
# Helpers
# -----------------------
def _abs(p):
    try:
        return os.path.abspath(p)
    except Exception:
        return p

def _save_manifest(output_dir, entries: dict):
    mf = {"output_dir": _abs(output_dir), "files": {}}
    for k, info in entries.items():
        if info is None: continue
        path, arr = info
        mf["files"][k] = {"path": _abs(path), "exists": bool(os.path.exists(path)),
                          "shape": tuple(getattr(arr, "shape", ())), "dtype": str(getattr(arr, "dtype", ""))}
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=2)
    print(f"[INFO] Wrote manifest → {os.path.join(output_dir, 'manifest.json')}")

def _counts(y):
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}

def sanitize_array(X):
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X

def build_preprocessor(num_cols, cat_cols, scaler_kind='minmax'):
    scaler = MinMaxScaler() if scaler_kind == 'minmax' else StandardScaler(with_mean=False)
    try:
        ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01,
                            max_categories=50, sparse_output=True, dtype=np.float32)
    except TypeError:
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=True, dtype=np.float32)

    sanitize = FunctionTransformer(sanitize_array, feature_names_out='one-to-one', validate=False)
    num_pipeline = Pipeline([('sanitize', sanitize), ('imputer', SimpleImputer(strategy='median')), ('scaler', scaler)])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)])
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

def ohe_feature_names(pre):
    try:
        return pre.get_feature_names_out().tolist()
    except Exception:
        num_names = list(pre.transformers_[0][2])
        cat_names, cat_cols = [], list(pre.transformers_[1][2])
        if len(cat_cols) > 0:
            ohe = pre.transformers_[1][1].named_steps['ohe']
            try: cats = ohe.categories_
            except Exception: cats = []
            for col, cats_i in zip(cat_cols, cats):
                cat_names.extend([f"{col}={c}" for c in cats_i])
        return num_names + cat_names

def correlation_filter_df(df_in, threshold=0.95, cap_after=None):
    if df_in.shape[1] == 0: return df_in, []
    num_only = df_in.select_dtypes(include=[np.number])
    if num_only.shape[1] == 0: return df_in, []
    std = num_only.std(numeric_only=True)
    nonconst_cols = std.index[std > 0]
    if len(nonconst_cols) <= 1: return df_in, []
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = num_only[nonconst_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if upper[c].gt(threshold).any()]
    X_red = df_in.drop(columns=to_drop, errors='ignore')
    if cap_after and X_red.shape[1] > cap_after:
        X_red = X_red.iloc[:, :cap_after]
    return X_red, to_drop

def _read_parquet_stream(path, columns=None, batch_rows=10000, inner_rows=2000, sample_frac=None, seed=42):
    rng = np.random.default_rng(seed)
    frames = []
    pf = pq.ParquetFile(path, memory_map=False)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        n = batch.num_rows
        for start in range(0, n, inner_rows):
            sub = batch.slice(start, min(inner_rows, n - start))
            pdf = sub.to_pandas(split_blocks=True, self_destruct=True)
            if sample_frac and 0 < sample_frac < 1.0 and len(pdf) > 0:
                take = int(len(pdf) * sample_frac)
                if 0 < take < len(pdf):
                    idxs = rng.choice(len(pdf), size=take, replace=False)
                    pdf = pdf.iloc[idxs]
            if not pdf.empty: frames.append(pdf)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or [])

def _resolve_col(df, wanted):
    cols = df.columns.astype(str).str.replace('\ufeff','', regex=False).str.strip()
    df.columns = cols
    if wanted in df.columns: return wanted
    lowmap = {c.lower(): c for c in df.columns}
    key = wanted.lower()
    if key in lowmap: return lowmap[key]
    for c in df.columns:
        if c.strip() == wanted: return c
    raise KeyError(f"Required column '{wanted}' not found. First 25 columns: {list(df.columns)[:25]}")

# --- NEW: canonicalize original label text (before collapsing to 0/1)
_DEF_REPL = {'UDP-LAG':'UDPLAG','BENIGN ':'BENIGN'}
_DDOS_CANON = {
    "SYN","SYN2","UDP","UDPLAG","UDPLAG2","PORTMAP","MSSQL","NETBIOS","NTP","SNMP",
    "LDAP","SSDP","TFTP","TFTP2","WEBDDOS",
    "DRDOS_DNS","DRDOS_LDAP","DRDOS_MSSQL","DRDOS_NTP","DRDOS_SNMP","DRDOS_SSDP","DRDOS_UDP","DRDOS_NETBIOS"
}
def _canonical_label_text(series: pd.Series) -> pd.Series:
    s = (series.astype(str)
         .str.replace('\ufeff','', regex=False)
         .str.strip().str.upper().replace(_DEF_REPL))
    # keep BENIGN as is; map anything in known DDoS set to its canonical token
    return s.where(~s.isin(_DDOS_CANON) & (s != "BENIGN"), s)

def _normalize_labels(df, label_col):
    """Collapse to binary 0/1, but keep df['LabelText'] with original subtype."""
    # Create LabelText first
    df['LabelText'] = _canonical_label_text(df[label_col])

    # If numeric 0/1 already, just ensure dtype and keep LabelText as set above (may be NaN)
    if pd.api.types.is_numeric_dtype(df[label_col]):
        s = pd.to_numeric(df[label_col], errors='coerce')
        uniq = pd.Series(s.dropna().unique())
        if uniq.isin([0, 1, 0.0, 1.0]).all():
            df[label_col] = s.astype('int8'); return

    # Collapse to binary
    ytxt = df['LabelText']
    y = ytxt.map({"BENIGN":0}).fillna(1).astype('Int8')  # any non-BENIGN becomes 1
    df[label_col] = y

# -----------------------
# Scenario volume-aware splitter
# -----------------------
def split_scenarios_by_volume(
    df, scenario_col,
    test_frac=0.2, val_frac=0.1, random_state=42, n_restarts=4000,
    min_scen_train=5, min_scen_val=4, min_scen_test=4,
    diversity_bonus=0.002
):
    rng = np.random.default_rng(random_state)
    sizes = df.groupby(scenario_col).size()
    scen = sizes.sort_values(ascending=False).index.tolist()
    vol  = sizes.loc[scen].to_numpy(float)
    total = float(vol.sum())
    tgt = {"train": total * (1.0 - test_frac - val_frac),
           "val":   total * val_frac,
           "test":  total * test_frac}

    best = None

    def score_fn(used, bins):
        vol_err = sum(((used[k]-tgt[k])/total)**2 for k in ("train","val","test"))
        div = -(diversity_bonus * (len(bins["train"]) + len(bins["val"]) + len(bins["test"])))
        return vol_err + div

    for _ in range(n_restarts):
        eq = {}
        for s in scen:
            eq.setdefault(sizes[s], []).append(s)
        order = []
        for sz in sorted(eq.keys(), reverse=True):
            block = eq[sz]
            rng.shuffle(block)
            order.extend(block)

        used = {"train":0.0, "val":0.0, "test":0.0}
        bins = {"train":[], "val":[], "test":[]}

        for s in order:
            s_vol = float(sizes[s])
            rem = {k: tgt[k]-used[k] for k in used}
            cand = sorted(rem.keys(), key=lambda k: rem[k], reverse=True)
            if rem[cand[0]] <= 0:
                cand = sorted(used.keys(), key=lambda k: abs((used[k]+s_vol)-tgt[k]))
            chosen = None
            for k in cand:
                if (used[k] + s_vol) <= (tgt[k] + min(vol)):
                    chosen = k; break
            if chosen is None:
                chosen = cand[0]
            used[chosen] += s_vol
            bins[chosen].append(s)

        def promote(src, dst, need):
            while len(bins[dst]) < need and len(bins[src]) > 1:
                mv = min(bins[src], key=lambda x: sizes[x])
                bins[src].remove(mv); bins[dst].append(mv)
        promote("train","val", min_scen_val)
        promote("train","test",min_scen_test)
        if len(bins["train"]) < min_scen_train:
            pools = sorted([(sizes[s], "val", s) for s in bins["val"]] +
                           [(sizes[s], "test", s) for s in bins["test"]])
            i = 0
            while len(bins["train"]) < min_scen_train and i < len(pools):
                _, src, s = pools[i]; i += 1
                bins[src].remove(s); bins["train"].append(s)

        used = {k: float(sizes.loc[bins[k]].sum()) if len(bins[k]) else 0.0 for k in bins}
        sc = score_fn(used, bins)
        if (best is None) or (sc < best[0]):
            best = (sc, set(bins["train"]), set(bins["val"]), set(bins["test"]))

    _, train_s, val_s, test_s = best
    return train_s, val_s, test_s


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description='Preprocess CICDDoS2019 unified dataset.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--label_col', default='Label')
    ap.add_argument('--timestamp_col', default='Timestamp')
    ap.add_argument('--test_frac', type=float, default=None)
    ap.add_argument('--val_frac', type=float, default=None)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--k_features', type=int, default=45)
    ap.add_argument('--corr_thresh', type=float, default=0.9)
    ap.add_argument('--cap_after_corr', type=int, default=None)
    ap.add_argument('--scaler', choices=['minmax','standard'], default='standard')
    ap.add_argument('--split_mode', choices=['random','temporal','group'], default='temporal',
                    help="temporal: earlier→train, later→val/test; random: stratified random; group: split by Scenario")
    ap.add_argument('--sample_frac', type=float, default=None, help="Optionally subsample rows (0-1) before FS.")
    ap.add_argument('--parquet_batch_rows', type=int, default=10_000)
    ap.add_argument('--parquet_inner_rows', type=int, default=2_000)
    ap.add_argument('--max_dense_features', type=int, default=15000,
                    help="Do not densify if transformed feature count exceeds this.")
    ap.add_argument('--windowing', choices=['on','off'], default='off')
    ap.add_argument('--window_size', type=int, default=48)
    ap.add_argument('--window_stride', type=int, default=24)
    ap.add_argument('--compat_names', action='store_true',
                    help="Also save duplicate files under the alternative naming (X_* and Xw_*) for loader compatibility.")
    ap.add_argument('--cap_per_class', type=int, default=None,
                    help="If set, cap rows per class (0 and 1) BEFORE splitting.")
    ap.add_argument('--drop_label_corr_thresh', type=float, default=0.8,
                    help="Drop any feature whose abs Pearson correlation with label on TRAIN exceeds this.")

    args = ap.parse_args()

    # Fractions
    test_frac = args.test_frac
    val_frac  = args.val_frac
    if (test_frac is not None) or (val_frac is not None):
        if (test_frac is None) or (val_frac is None):
            raise ValueError("Provide BOTH --test_frac and --val_frac, or neither.")
        if test_frac < 0 or val_frac < 0 or (test_frac + val_frac) >= 1.0:
            raise ValueError("--test_frac + --val_frac must be < 1.0 and each >= 0.")
        train_frac = 1.0 - test_frac - val_frac
        args.test_size = float(test_frac)
        args.val_size  = float(val_frac / (1.0 - test_frac))
        print(f"[INFO] Using fractions → train={train_frac:.2f} val={val_frac:.2f} test={test_frac:.2f} "
              f"(internal: test_size={args.test_size:.3f}, val_size={args.val_size:.3f})")
    else:
        args.test_size = 0.20
        args.val_size  = 0.10 / (1.0 - args.test_size)
        print(f"[INFO] Defaulting to train=0.70 val=0.10 test=0.20 "
              f"(internal: test_size={args.test_size:.3f}, val_size={args.val_size:.3f})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"[INFO] Streaming parquet: {args.input}")
    pf = pq.ParquetFile(args.input, memory_map=False)
    use_cols = [name for name in pf.schema.names]
    df = _read_parquet_stream(args.input, columns=use_cols,
                              batch_rows=args.parquet_batch_rows,
                              inner_rows=args.parquet_inner_rows,
                              sample_frac=args.sample_frac, seed=args.random_state)
    if df.empty:
        raise RuntimeError("Input produced empty dataframe after streaming.")

    df.columns = df.columns.astype(str).str.replace('\ufeff','', regex=False).str.strip()
    args.label_col = _resolve_col(df, args.label_col)
    ts_col = None
    if args.timestamp_col:
        try: ts_col = _resolve_col(df, args.timestamp_col)
        except Exception: ts_col = None

    df = df.loc[:, ~df.columns.duplicated()]

    # --- build LabelText, then binary label
    _normalize_labels(df, args.label_col)
    df = df.dropna(subset=[args.label_col])
    df[args.label_col] = df[args.label_col].astype('int8')

    # Optional pre-split capping by class
    if args.cap_per_class is not None:
        rng = np.random.default_rng(args.random_state)
        capped = []
        for cls in (0, 1):
            part = df[df[args.label_col] == cls]
            if len(part) > args.cap_per_class:
                idx = rng.choice(len(part), size=args.cap_per_class, replace=False)
                part = part.iloc[idx]
            capped.append(part)
        df = pd.concat(capped, ignore_index=True)
        df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
        print(f"[INFO] Applied --cap_per_class={args.cap_per_class} (pre-split). "
              f"New size={len(df):,} | class counts={_counts(df[args.label_col].to_numpy())}")

    # Timestamp handling
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        print(f"[INFO] Rows total={len(df):,} | rows with valid Timestamp={df[ts_col].notna().sum():,}")
        df = df[~df[ts_col].isna()].sort_values(ts_col).reset_index(drop=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Feature list — remove 'Inbound', 'Flow ID', ANY '*Port*'
    feature_cols = [c for c in df.columns if c not in [args.label_col, ts_col]]
    drop_explicit = {"Inbound", "Flow ID", "Flow ID ", "FlowID"}
    feature_cols = [c for c in feature_cols
                    if (c not in drop_explicit) and ("Port" not in c)]
    # Never use Scenario as a feature
    feature_cols = [c for c in feature_cols if c != 'Scenario']

    # === Split
    scenario_distribution_meta = None
    label_breakdown_meta = None

    if args.split_mode == 'group' and "Scenario" in df.columns:
        scenario_col = "Scenario"
        train_s, val_s, test_s = split_scenarios_by_volume(
            df, scenario_col, test_frac=args.test_size, val_frac=args.val_size, random_state=args.random_state
        )
        df_train = df[df[scenario_col].isin(train_s)]
        df_val   = df[df[scenario_col].isin(val_s)]
        df_test  = df[df[scenario_col].isin(test_s)]

        def _pct(n): return f"{100*n/len(df):.2f}%"
        print(f"[INFO] Scenario-level split (by volume): "
              f"train={len(df_train):,} ({_pct(len(df_train))}) "
              f"val={len(df_val):,} ({_pct(len(df_val))}) "
              f"test={len(df_test):,} ({_pct(len(df_test))})")
        print(f"[INFO] #scenarios → train={len(train_s)} val={len(val_s)} test={len(test_s)}")

        # Per-split scenario distribution (print + metadata)
        def _scen_table(dfp: pd.DataFrame) -> pd.DataFrame:
            if len(dfp) == 0:
                return pd.DataFrame(columns=["rows","pct"])
            s = dfp.groupby(scenario_col).size().sort_values(ascending=False)
            tbl = s.to_frame(name="rows")
            tbl["pct"] = (tbl["rows"] / max(len(dfp), 1) * 100).round(2)
            return tbl

        tbl_tr = _scen_table(df_train); tbl_va = _scen_table(df_val); tbl_te = _scen_table(df_test)
        print("\n[INFO] Scenario distribution (TRAIN):"); print(tbl_tr.to_string())
        print("\n[INFO] Scenario distribution (VAL):");   print(tbl_va.to_string())
        print("\n[INFO] Scenario distribution (TEST):");  print(tbl_te.to_string()); print("")

        scenario_distribution_meta = {
            "train": {"rows": tbl_tr["rows"].astype(int).to_dict(), "pct": tbl_tr["pct"].astype(float).to_dict()},
            "val":   {"rows": tbl_va["rows"].astype(int).to_dict(), "pct": tbl_va["pct"].astype(float).to_dict()},
            "test":  {"rows": tbl_te["rows"].astype(int).to_dict(), "pct": tbl_te["pct"].astype(float).to_dict()}
        }
    else:
        print("[WARN] No Scenario column; using temporal/random split.")
        if args.split_mode == 'temporal' and ts_col is not None:
            df_sorted = df.sort_values(ts_col).reset_index(drop=True)
            n = len(df_sorted)
            n_test = int(round(args.test_size * n))
            n_trainval = n - n_test
            df_trainval = df_sorted.iloc[:n_trainval]
            df_test     = df_sorted.iloc[n_trainval:]
            n_tv = len(df_trainval)
            n_val = int(round(args.val_size * n_tv))
            n_train = n_tv - n_val
            df_train = df_trainval.iloc[:n_train]
            df_val   = df_trainval.iloc[n_train:]
        else:
            df_trainval, df_test = train_test_split(
                df, test_size=args.test_size, random_state=args.random_state,
                stratify=df[args.label_col] if args.split_mode == 'random' else None
            )
            df_train, df_val = train_test_split(
                df_trainval, test_size=args.val_size, random_state=args.random_state,
                stratify=df_trainval[args.label_col] if args.split_mode == 'random' else None
            )

    # --- NEW: Per-split label subtype breakdown
    def _label_breakdown(dfp: pd.DataFrame) -> dict:
        total = max(len(dfp), 1)
        # counts over LabelText (BENIGN and subtypes)
        vc = dfp['LabelText'].value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in vc.items()}
        pct_of_split = {str(k): float(v)*100.0/total for k, v in vc.items()}
        # Among DDoS only (exclude BENIGN)
        ddos_mask = dfp['LabelText'].ne("BENIGN")
        ddos_total = int(ddos_mask.sum())
        if ddos_total > 0:
            vc_ddos = dfp.loc[ddos_mask, 'LabelText'].value_counts()
            pct_within_ddos = {str(k): float(v)*100.0/ddos_total for k, v in vc_ddos.items()}
        else:
            pct_within_ddos = {}
        return {"counts": counts, "pct_of_split": pct_of_split, "pct_within_ddos": pct_within_ddos}

    lb_train = _label_breakdown(df_train)
    lb_val   = _label_breakdown(df_val)
    lb_test  = _label_breakdown(df_test)
    label_breakdown_meta = {"train": lb_train, "val": lb_val, "test": lb_test}

    # Pretty print compact label subtype tables
    def _print_label_table(name, lb):
        print(f"\n[INFO] Label subtype breakdown ({name})")
        rows = []
        for k, v in lb["counts"].items():
            p_all = lb["pct_of_split"].get(k, 0.0)
            p_dd  = lb["pct_within_ddos"].get(k, np.nan)
            rows.append((k, v, p_all, p_dd))
        dfp = pd.DataFrame(rows, columns=["LabelText","count","% of split","% within DDoS"])
        # Sort: BENIGN first, then by count desc
        dfp["_ord"] = dfp["LabelText"].apply(lambda x: 0 if x=="BENIGN" else 1)
        dfp = dfp.sort_values(["_ord","count"], ascending=[True, False]).drop(columns="_ord")
        print(dfp.to_string(index=False))
    _print_label_table("TRAIN", lb_train)
    _print_label_table("VAL",   lb_val)
    _print_label_table("TEST",  lb_test)
    print("")

    X_train, y_train = df_train[[c for c in feature_cols if c in df_train.columns]].copy(), df_train[args.label_col].values
    X_val,   y_val   = df_val[[c for c in feature_cols if c in df_val.columns]].copy(),   df_val[args.label_col].values
    X_test,  y_test  = df_test[[c for c in feature_cols if c in df_test.columns]].copy(),  df_test[args.label_col].values
    ts_train, ts_val, ts_test = df_train.get(ts_col), df_val.get(ts_col), df_test.get(ts_col)

    print(f"[INFO] Split sizes → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    total = len(X_train) + len(X_val) + len(X_test)
    print("[INFO] Split ratios → "
          f"train={len(X_train)/total:.4f}  val={len(X_val)/total:.4f}  test={len(X_test)/total:.4f}")
    print(f"[INFO] Class counts → train={_counts(y_train)}  val={_counts(y_val)}  test={_counts(y_test)}")

    # --- De-duplicate WITHIN each split (keeps alignment)
    def _dedup_split(X_df: pd.DataFrame, y: np.ndarray, ts: pd.Series | None):
        mask = ~X_df.duplicated()
        removed = int((~mask).sum())
        if removed:
            print(f"[CLEAN] Split de-dup removed {removed:,} duplicate rows (kept {mask.sum():,}).")
        X_clean = X_df.loc[mask].reset_index(drop=True)
        y_clean = y[mask.to_numpy()]
        ts_clean = ts.loc[mask].reset_index(drop=True) if ts is not None else None
        return X_clean, y_clean, ts_clean

    X_train, y_train, ts_train = _dedup_split(X_train, y_train, ts_train)
    X_val,   y_val,   ts_val   = _dedup_split(X_val,   y_val,   ts_val)
    X_test,  y_test,  ts_test  = _dedup_split(X_test,  y_test,  ts_test)

    print(f"[INFO] Post-dedup split sizes → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"[INFO] Post-dedup class counts → train={_counts(y_train)}  val={_counts(y_val)}  test={_counts(y_test)}")

    # Drop all-NaN columns (based on TRAIN)
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"[INFO] Dropping all-NaN TRAIN columns: {len(all_nan_cols)}")
        X_train.drop(columns=all_nan_cols, inplace=True, errors='ignore')
        X_val.drop(columns=[c for c in all_nan_cols if c in X_val.columns], inplace=True, errors='ignore')
        X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns], inplace=True, errors='ignore')

    # Preprocess (fit on TRAIN only)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'string', 'category', 'boolean']).columns.tolist()
    pre = build_preprocessor(num_cols, cat_cols, scaler_kind=args.scaler)
    pre.fit(X_train)
    Xt_train = pre.transform(X_train); Xt_val = pre.transform(X_val); Xt_test = pre.transform(X_test)
    feat_names = ohe_feature_names(pre)

    # Remove constant features
    vt = VarianceThreshold(threshold=0.0)
    Xt_train_vt = vt.fit_transform(Xt_train)
    Xt_val_vt   = vt.transform(Xt_val)
    Xt_test_vt  = vt.transform(Xt_test)
    vt_mask = vt.get_support()
    feat_names = np.array(feat_names)[vt_mask].tolist()
    print(f"[INFO] VarianceThreshold removed {(~vt_mask).sum()} constant features.")

    def to_dense_safe(Xs):
        if hasattr(Xs, "toarray") and Xs.shape[1] <= int(args.max_dense_features):
            return Xs.toarray()
        return Xs

    Xt_train_d = to_dense_safe(Xt_train_vt)
    Xt_val_d   = to_dense_safe(Xt_val_vt)
    Xt_test_d  = to_dense_safe(Xt_test_vt)

    # Feature–feature correlation prune (on TRAIN)
    if not hasattr(Xt_train_d, "toarray"):
        train_df = pd.DataFrame(Xt_train_d, columns=feat_names, index=X_train.index)
        train_df_corr, corr_dropped = correlation_filter_df(train_df, threshold=args.corr_thresh,
                                                            cap_after=args.cap_after_corr)
        kept_names = list(train_df_corr.columns)
        val_df  = pd.DataFrame(Xt_val_d,  columns=feat_names, index=X_val.index).drop(columns=corr_dropped, errors='ignore')
        test_df = pd.DataFrame(Xt_test_d, columns=feat_names, index=X_test.index).drop(columns=corr_dropped, errors='ignore')
        df_train_sel, df_val_sel, df_test_sel = train_df_corr, val_df, test_df
    else:
        kept_names = feat_names; corr_dropped = []
        df_train_sel = pd.DataFrame(Xt_train_d, columns=kept_names, index=X_train.index)
        df_val_sel   = pd.DataFrame(Xt_val_d,   columns=kept_names, index=X_val.index)
        df_test_sel  = pd.DataFrame(Xt_test_d,  columns=kept_names, index=X_test.index)

    # Drop features highly correlated with label
    dropped_by_label_corr = []
    if df_train_sel.shape[1] > 0:
        num_cols_tr = df_train_sel.select_dtypes(include=[np.number]).columns
        if len(num_cols_tr) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_s = df_train_sel[num_cols_tr].corrwith(pd.Series(y_train, index=df_train_sel.index)).abs()
            high = corr_s[corr_s > float(args.drop_label_corr_thresh)].index.tolist()
            if high:
                print(f"[CLEAN] Dropping {len(high)} features with |corr(label)| > {args.drop_label_corr_thresh}")
                df_train_sel = df_train_sel.drop(columns=high, errors='ignore')
                df_val_sel   = df_val_sel.drop(columns=high, errors='ignore')
                df_test_sel  = df_test_sel.drop(columns=high, errors='ignore')
                dropped_by_label_corr = high
        kept_names = list(df_train_sel.columns)

    # SelectKBest
    Xuni_train = df_train_sel.to_numpy()
    Xuni_val   = df_val_sel.to_numpy()
    Xuni_test  = df_test_sel.to_numpy()
    k = min(args.k_features, Xuni_train.shape[1]) if Xuni_train.shape[1] > 0 else 0
    if k == 0:
        raise ValueError("No features available after preprocessing.")
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(Xuni_train, y_train)
    X_val_sel   = selector.transform(Xuni_val)
    X_test_sel  = selector.transform(Xuni_test)
    support = selector.get_support()
    selected_names = list(np.array(kept_names)[support])

    # De-dup across splits in model-input space
    from pandas.util import hash_pandas_object
    def _hash_df_selected(df_in: pd.DataFrame, cols):
        tmp = df_in[cols].copy()
        for c in tmp.columns:
            if pd.api.types.is_numeric_dtype(tmp[c]):
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("float32").round(8)
            else:
                tmp[c] = tmp[c].astype("string")
        tmp = tmp.fillna("<NA>")
        return hash_pandas_object(tmp, index=False).astype("uint64")

    h_tr = set(_hash_df_selected(df_train_sel, selected_names))
    h_va = _hash_df_selected(df_val_sel,   selected_names)
    h_te = _hash_df_selected(df_test_sel,  selected_names)
    keep_va = ~h_va.isin(h_tr); keep_te = ~h_te.isin(h_tr)
    n_drop_va = int((~keep_va).sum()); n_drop_te = int((~keep_te).sum())
    if n_drop_va or n_drop_te:
        print(f"[CLEAN] Removing post-transform duplicates vs TRAIN → val_drop={n_drop_va}  test_drop={n_drop_te}")

    df_val_sel = df_val_sel.loc[keep_va].copy(); y_val = y_val[keep_va.to_numpy()]
    if ts_val is not None: ts_val = ts_val.loc[keep_va].copy()
    df_test_sel = df_test_sel.loc[keep_te].copy(); y_test = y_test[keep_te.to_numpy()]
    if ts_test is not None: ts_test = ts_test.loc[keep_te].copy()

    # Rebuild matrices after dedup
    Xuni_val   = df_val_sel.to_numpy()
    Xuni_test  = df_test_sel.to_numpy()
    X_val_sel  = selector.transform(Xuni_val)
    X_test_sel = selector.transform(Xuni_test)

    # Ensure no overlap test vs val
    h_va_after = set(_hash_df_selected(df_val_sel, selected_names))
    h_te_after = _hash_df_selected(df_test_sel, selected_names)
    keep_te2 = ~h_te_after.isin(h_va_after)
    n_drop_te2 = int((~keep_te2).sum())
    if n_drop_te2:
        print(f"[CLEAN] Removing post-transform duplicates vs VAL → test_drop={n_drop_te2}")
        df_test_sel = df_test_sel.loc[keep_te2].copy(); y_test = y_test[keep_te2.to_numpy()]
        if ts_test is not None: ts_test = ts_test.loc[keep_te2].copy()
        Xuni_test  = df_test_sel.to_numpy()
        X_test_sel = selector.transform(Xuni_test)

    # === SAVE CORE (flat)
    p_Xtr = os.path.join(args.output_dir, 'X_train.npy')
    p_ytr = os.path.join(args.output_dir, 'y_train.npy')
    p_Xva = os.path.join(args.output_dir, 'X_val.npy')
    p_yva = os.path.join(args.output_dir, 'y_val.npy')
    p_Xte = os.path.join(args.output_dir, 'X_test.npy')
    p_yte = os.path.join(args.output_dir, 'y_test.npy')
    np.save(p_Xtr, X_train_sel); np.save(p_ytr, y_train)
    np.save(p_Xva, X_val_sel);   np.save(p_yva, y_val)
    np.save(p_Xte, X_test_sel);  np.save(p_yte, y_test)

    joblib.dump(pre,      os.path.join(args.output_dir, 'preprocessor.joblib'))
    joblib.dump(selector, os.path.join(args.output_dir, 'selector.joblib'))

    manifest_entries = {
        "X_train": (p_Xtr, X_train_sel), "y_train": (p_ytr, y_train),
        "X_val":   (p_Xva, X_val_sel),   "y_val":   (p_yva, y_val),
        "X_test":  (p_Xte, X_test_sel),  "y_test":  (p_yte, y_test),
    }

    # -----------------------
    # Leak diagnostics (flat)
    # -----------------------
    from pandas.util import hash_pandas_object
    def _row_hash_df(df_in: pd.DataFrame) -> pd.Series:
        tmp = df_in.copy()
        for c in tmp.columns:
            if pd.api.types.is_numeric_dtype(tmp[c]):
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("float32")
            else:
                tmp[c] = tmp[c].astype("string")
        tmp = tmp.fillna("<NA>")
        return hash_pandas_object(tmp, index=False).astype("uint64")
    train_hash = set(_row_hash_df(X_train)); val_hash = set(_row_hash_df(X_val)); test_hash = set(_row_hash_df(X_test))
    overlap_tr_te = len(train_hash & test_hash); overlap_tr_va = len(train_hash & val_hash); overlap_va_te = len(val_hash & test_hash)

    meta = {}
    meta["leak_diagnostics"] = {
        "row_overlap_pre_transform": {
            "train_test": int(overlap_tr_te), "train_val":  int(overlap_tr_va), "val_test":   int(overlap_va_te)
        },
        "expected_zero": True
    }

    # === WINDOWS (3-D)
    if args.windowing == 'on':
        df_tr = df_train_sel[selected_names]; df_va = df_val_sel[selected_names]; df_te = df_test_sel[selected_names]

        def _build_windows(X_df, y, ts_series, window_size=48, stride=24, label_policy="majority"):
            if ts_series is None or ts_series.isna().all() or len(X_df) == 0:
                order = np.arange(len(X_df))
            else:
                order = np.argsort(ts_series.values.astype('datetime64[ns]'))
            X_seq = X_df.iloc[order].to_numpy(); y_seq = np.array(y)[order]
            windows, labels = [], []
            for start in range(0, len(X_seq) - window_size + 1, stride):
                seg = X_seq[start:start+window_size]
                seg_y = y_seq[start:start+window_size]
                vals, counts = np.unique(seg_y, return_counts=True)
                lbl = vals[np.argmax(counts)] if label_policy == "majority" else (1 if np.any(seg_y != 0) else 0)
                windows.append(seg); labels.append(lbl)
            if not windows: return np.empty((0, window_size, X_seq.shape[1])), np.empty((0,))
            return np.stack(windows), np.array(labels)

        Xw_tr, yw_tr = _build_windows(df_tr, y_train, ts_train, window_size=args.window_size, stride=args.window_stride)
        Xw_va, yw_va = _build_windows(df_va, y_val,   ts_val,   window_size=args.window_size, stride=args.window_stride)
        Xw_te, yw_te = _build_windows(df_te, y_test,  ts_test,  window_size=args.window_size, stride=args.window_stride)

        def _hash_windows(Xw: np.ndarray) -> pd.Series:
            if Xw.size == 0: return pd.Series([], dtype="uint64")
            flat = Xw.reshape((Xw.shape[0], -1)); dfw = pd.DataFrame(flat)
            return hash_pandas_object(dfw, index=False).astype("uint64")

        hw_tr = set(_hash_windows(Xw_tr)); hw_va = _hash_windows(Xw_va); hw_te = _hash_windows(Xw_te)
        keep_va_w = ~hw_va.isin(hw_tr); keep_te_w = ~hw_te.isin(hw_tr)
        drop_va_w = int((~keep_va_w).sum()); drop_te_w = int((~keep_te_w).sum())
        if drop_va_w or drop_te_w:
            print(f"[CLEAN] Removing window overlap vs TRAIN → val_drop={drop_va_w}  test_drop={drop_te_w}")
            Xw_va = Xw_va[keep_va_w.to_numpy()]; yw_va = yw_va[keep_va_w.to_numpy()]
            Xw_te = Xw_te[keep_te_w.to_numpy()]; yw_te = yw_te[keep_te_w.to_numpy()]

        pw_Xwtr = os.path.join(args.output_dir, 'Xw_train.npy')
        pw_ywtr = os.path.join(args.output_dir, 'yw_train.npy')
        pw_Xwva = os.path.join(args.output_dir, 'Xw_val.npy')
        pw_ywva = os.path.join(args.output_dir, 'yw_val.npy')
        pw_Xwte = os.path.join(args.output_dir, 'Xw_test.npy')
        pw_ywte = os.path.join(args.output_dir, 'yw_test.npy')
        np.save(pw_Xwtr, Xw_tr); np.save(pw_ywtr, yw_tr)
        np.save(pw_Xwva, Xw_va); np.save(pw_ywva, yw_va)
        np.save(pw_Xwte, Xw_te); np.save(pw_ywte, yw_te)
        print(f"[INFO] Windowed shapes: train={getattr(Xw_tr,'shape',None)}, val={getattr(Xw_va,'shape',None)}, test={getattr(Xw_te,'shape',None)}")

        manifest_entries.update({"Xw_train": (pw_Xwtr, Xw_tr), "yw_train": (pw_ywtr, yw_tr),
                                 "Xw_val": (pw_Xwva, Xw_va), "yw_val": (pw_ywva, yw_va),
                                 "Xw_test": (pw_Xwte, Xw_te), "yw_test": (pw_ywte, yw_te)})

        hw_va2 = set(_hash_windows(Xw_va)); hw_te2 = set(_hash_windows(Xw_te))
        meta["leak_diagnostics"]["window_overlap"] = {
            "train_val_after_clean": int(len(hw_tr & hw_va2)),
            "train_test_after_clean": int(len(hw_tr & hw_te2))
        }

    # Extra correlation sanity (top 10) on TRAIN
    try:
        num_only = X_train.select_dtypes(include=[np.number])
        if num_only.shape[1] > 0:
            std = num_only.std(numeric_only=True); num_nonconst = num_only.loc[:, std > 0]
            if num_nonconst.shape[1] > 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_s = num_nonconst.corrwith(pd.Series(y_train, index=num_nonconst.index)).abs()
                meta_top = corr_s.dropna().sort_values(ascending=False).head(10).to_dict()
            else:
                meta_top = {}
        else:
            meta_top = {}
    except Exception as e:
        meta_top = {"error": str(e)}

    wtxt = f"{args.scaler}_corr{args.corr_thresh:g}_k{args.k_features}_{'w'+str(args.window_size)+'s'+str(args.window_stride) if args.windowing=='on' else 'wOFF'}"
    meta.update({
        "preproc_tag": wtxt,
        "selected_features": selected_names,
        "corr_dropped": corr_dropped,
        "dropped_by_label_corr": dropped_by_label_corr,
        "num_features_final": len(selected_names),
        "split_mode": args.split_mode,
        "split_sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "class_counts": {"train": _counts(y_train), "val": _counts(y_val), "test": _counts(y_test)},
        "removed_columns_explicit": sorted(list(drop_explicit)) + ["*Port*"],
        "windowing": args.windowing,
        "window": {"size": args.window_size, "stride": args.window_stride} if args.windowing=='on' else None,
        "params": vars(args),
        "top_label_correlations_train_preview": meta_top,
        # NEW: include per-split label subtype breakdown
        "label_breakdown": label_breakdown_meta
    })
    if scenario_distribution_meta is not None:
        meta["scenario_distribution"] = scenario_distribution_meta

    with open(os.path.join(args.output_dir, 'preprocess_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    _save_manifest(args.output_dir, manifest_entries)
    print(f"\n✅ Done. Selected {len(selected_names)} features.")
    print(f"   Saved arrays & artifacts → {args.output_dir}")

if __name__ == '__main__':
    main()
