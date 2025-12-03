# merge_dataset.py
import os, glob, re, json, time, gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta

SRC_DIR = "CICDDos2019"
CSV_GLOB = os.path.join(SRC_DIR, "*.csv")
OUT_DIR = "merged_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "combined.parquet")
OUT_LABELMAP_JSON = os.path.join(OUT_DIR, "LabelEncoding.json")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "merge_summary.json")

CHUNK_SIZE = 200_000
ROW_GROUP_SIZE = 400_000
SEED = 42
np.random.seed(SEED)

# -------------------------
# Balancing / caps
# -------------------------
# Target: per-scenario approx 1:2 (BENIGN:DDoS). You can change to 1.5, 3.0, etc.
PER_SCENARIO_ATTACK_RATIO = 1.0

# Optional GLOBAL caps (None = no cap)
GLOBAL_MAX_BENIGN = None   # e.g., 100_000
GLOBAL_MAX_DDOS   = None   # e.g., 200_000

STRING_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "SimillarHTTP", "Label"}
DROP_COLS = {"Flow ID", "Source IP", "Destination IP", "SimillarHTTP"}

LABEL_MAP = {
    "SYN": "DDoS", "SYN2": "DDoS",
    "UDP": "DDoS", "UDP-LAG": "DDoS", "UDPLAG": "DDoS", "UDPLAG2": "DDoS",
    "PORTMAP": "DDoS", "MSSQL": "DDoS", "NETBIOS": "DDoS",
    "NTP": "DDoS", "SNMP": "DDoS", "LDAP": "DDoS", "SSDP": "DDoS",
    "TFTP": "DDoS", "TFTP2": "DDoS",
    "DRDOS_DNS": "DDoS", "DRDOS_LDAP": "DDoS", "DRDOS_MSSQL": "DDoS",
    "DRDOS_NTP": "DDoS", "DRDOS_SNMP": "DDoS", "DRDOS_SSDP": "DDoS",
    "DRDOS_UDP": "DDoS", "DRDOS_NETBIOS": "DDoS", "WEBDDOS": "DDoS",
    "BENIGN": "BENIGN", "BENIGN ": "BENIGN",
}
LABEL_TO_INT = {"BENIGN": 0, "DDoS": 1}

def norm_cols(cols):
    return [str(c).strip().replace("\ufeff", "") for c in cols]

def canonical_scenario(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0].upper().replace(" ", "")
    name = name.replace("UDP-LAG", "UDPLAG")
    if not name.startswith("DRDOS_"):
        name = re.sub(r"(\d+)$", "", name)
    return name

def normalize_labels_inplace(df: pd.DataFrame):
    label_like = [c for c in df.columns if str(c).strip().replace("\ufeff", "").lower() == "label"]
    if not label_like:
        return
    lab = df[label_like].bfill(axis=1).iloc[:, 0] if len(label_like) > 1 else df[label_like[0]]
    df.drop(columns=label_like, inplace=True, errors="ignore")
    lab = (
        lab.astype(str)
           .str.replace("\ufeff", "", regex=False)
           .str.strip()
           .str.upper()
           .replace({"UDP-LAG": "UDPLAG", "BENIGN ": "BENIGN"})
           .map(LABEL_MAP)
           .fillna(lab)
           .map(LABEL_TO_INT)
    )
    df["Label"] = pd.to_numeric(lab, errors="coerce").astype("Int8")

def coerce_dtypes(df: pd.DataFrame, canonical_cols, string_cols):
    """Force canonical column order and numeric consistency."""
    for c in canonical_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.reindex(columns=canonical_cols)
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    for c in string_cols:
        if c in df.columns and c != "Label":
            df[c] = df[c].astype("string[pyarrow]")
    for c in df.columns:
        if c not in string_cols and c != "Label":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32", copy=False)
    return df

def build_schema(canonical_cols, string_cols):
    """Create Arrow schema for writing Parquet."""
    fields = []
    for c in canonical_cols:
        if c == "Label":
            fields.append(pa.field(c, pa.int8()))
        elif c in string_cols:
            fields.append(pa.field(c, pa.string()))
        else:
            fields.append(pa.field(c, pa.float32()))
    fields.append(pa.field("Scenario", pa.string()))
    return pa.schema(fields)

# ---- de-dup helper (now INCLUDING Scenario → only de-dup within each scenario)
SEEN_ROW_HASHES = {}  # dict: scen_key -> set()

def hash_rows(df: pd.DataFrame) -> pd.Series:
    # include Scenario in the hash so “same row in different scenarios” is NOT a dup
    cols_for_hash = [*CANONICAL_COLS, "Scenario"]
    tmp = df[cols_for_hash].copy()
    for c in tmp.columns:
        if pd.api.types.is_string_dtype(tmp[c]):
            tmp[c] = tmp[c].astype(str)
        elif c != "Label":
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("float32")
    return pd.util.hash_pandas_object(tmp, index=False).astype("uint64")

def all_global_caps_met():
    # If neither cap is set, never stop early
    if GLOBAL_MAX_BENIGN is None and GLOBAL_MAX_DDOS is None:
        return False

    if GLOBAL_MAX_BENIGN is None:
        return global_counts[1] >= GLOBAL_MAX_DDOS

    if GLOBAL_MAX_DDOS is None:
        return global_counts[0] >= GLOBAL_MAX_BENIGN

    return (global_counts[0] >= GLOBAL_MAX_BENIGN) and (global_counts[1] >= GLOBAL_MAX_DDOS)


start_time = time.time()
csv_files = sorted(glob.glob(CSV_GLOB))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {SRC_DIR}")

# Canonical column set
all_cols = set()
for f in csv_files:
    try:
        head = pd.read_csv(f, nrows=0, encoding="utf-8-sig", encoding_errors="ignore")
    except Exception:
        head = pd.read_csv(f, nrows=0, encoding="utf-8-sig", encoding_errors="ignore", engine="python")
    all_cols.update(norm_cols(head.columns))

string_cols_present = [c for c in STRING_COLS if c in all_cols and c not in DROP_COLS and c != "Label"]
other_cols = sorted([c for c in all_cols if c not in STRING_COLS])

CANONICAL_COLS = ["Label"] + string_cols_present + other_cols
ARROW_SCHEMA = build_schema(CANONICAL_COLS, set(string_cols_present))

# Counters
global_counts = {0: 0, 1: 0}
scenario_rows = {}
scenario_counts = {}  # scen_key -> {"benign": int, "ddos": int}

if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)
writer = pq.ParquetWriter(OUT_PATH, ARROW_SCHEMA, compression="snappy", use_dictionary=True)

total_rows = 0
scenario_keys = sorted({canonical_scenario(f) for f in csv_files})
for sk in scenario_keys:
    scenario_counts[sk] = {"benign": 0, "ddos": 0}

for f in csv_files:
    scen_key = canonical_scenario(f)
    print(f"\n📂 Processing {os.path.basename(f)}  →  scenario={scen_key}")

    for chunk in pd.read_csv(
        f, chunksize=CHUNK_SIZE, low_memory=False, encoding="utf-8-sig", encoding_errors="ignore"
    ):
        # Basic cleaning
        chunk.columns = norm_cols(chunk.columns)
        chunk = chunk.loc[:, ~pd.Index(chunk.columns).duplicated(keep="first")]
        drop_like = [c for c in chunk.columns if c.lower().startswith("unnamed")]
        if drop_like:
            chunk.drop(columns=drop_like, inplace=True, errors="ignore")

        keep = [c for c in CANONICAL_COLS if c in chunk.columns or c == "Label"]
        chunk = chunk[keep]

        normalize_labels_inplace(chunk)
        chunk.dropna(subset=["Label"], inplace=True)
        if len(chunk) == 0:
            continue

        chunk = coerce_dtypes(chunk, CANONICAL_COLS, set(string_cols_present))
        chunk["Scenario"] = scen_key

        labels_np = chunk["Label"].to_numpy(copy=False)
        benign_mask = (labels_np == 0)
        ddos_mask   = (labels_np == 1)

        # -------------------------
        # Per-scenario balancing
        # -------------------------
        # Accept ALL benign for this scenario (subject to optional global cap).
        keep_benign = chunk.loc[benign_mask]

        # Global BENIGN cap, if any
        if GLOBAL_MAX_BENIGN is not None:
            remaining_b = max(GLOBAL_MAX_BENIGN - global_counts[0], 0)
            if len(keep_benign) > remaining_b:
                keep_benign = keep_benign.sample(n=remaining_b, random_state=SEED)

        # Update benign count tentatively to compute DDoS quota
        b_new = len(keep_benign)
        scen_b_before = scenario_counts[scen_key]["benign"]
        scen_d_before = scenario_counts[scen_key]["ddos"]

        # Quota: at most ratio * benign_so_far (including new benign in this chunk) minus ddos_so_far
        benign_total_after = scen_b_before + b_new
        ddos_quota = int(np.floor(PER_SCENARIO_ATTACK_RATIO * benign_total_after)) - scen_d_before
        ddos_quota = max(ddos_quota, 0)

        # Candidate DDoS rows in this chunk
        ddos_part = chunk.loc[ddos_mask]

        # Apply per-scenario quota
        if len(ddos_part) > ddos_quota:
            ddos_part = ddos_part.sample(n=ddos_quota, random_state=SEED)

        # Also respect optional GLOBAL DDoS cap
        if GLOBAL_MAX_DDOS is not None:
            remaining_d = max(GLOBAL_MAX_DDOS - global_counts[1], 0)
            if len(ddos_part) > remaining_d:
                ddos_part = ddos_part.sample(n=remaining_d, random_state=SEED)

        keep_ddos = ddos_part

        # Merge + scenario-local de-dup
        merged = pd.concat([keep_benign, keep_ddos], ignore_index=True)
        if merged.empty:
            continue

        merged = merged.drop_duplicates()
        merged = merged.loc[:, ~pd.Index(merged.columns).duplicated(keep="first")]

        if scen_key not in SEEN_ROW_HASHES:
            SEEN_ROW_HASHES[scen_key] = set()

        h = hash_rows(merged)
        mask_new = ~h.isin(SEEN_ROW_HASHES[scen_key])
        if not mask_new.any():
            continue

        merged = merged.loc[mask_new].copy()
        SEEN_ROW_HASHES[scen_key].update(h.loc[mask_new].tolist())

        # Write
        tbl = pa.Table.from_pandas(merged, preserve_index=False)
        if tbl.schema != ARROW_SCHEMA:
            tbl = tbl.cast(ARROW_SCHEMA, safe=False)
        writer.write_table(tbl)

        # Update counters
        wrote_labels = merged["Label"].to_numpy(copy=False)
        wrote_b = int((wrote_labels == 0).sum())
        wrote_d = int((wrote_labels == 1).sum())

        scenario_counts[scen_key]["benign"] += wrote_b
        scenario_counts[scen_key]["ddos"]   += wrote_d
        global_counts[0] += wrote_b
        global_counts[1] += wrote_d

        total_rows += len(merged)
        scenario_rows[scen_key] = scenario_rows.get(scen_key, 0) + len(merged)

        if total_rows % 50_000 == 0:
            print(f"  → written {total_rows:,} rows | benign={global_counts[0]:,}, ddos={global_counts[1]:,} | seen_hashes={len(SEEN_ROW_HASHES):,}")

        if all_global_caps_met():
            print("✅ Global caps met. Stopping early.")
            break

    if all_global_caps_met():
        break
    gc.collect()

writer.close()

# Build summary (with per-scenario benign/ddos & ratios)
per_scenario_counts = {
    sk: {
        "benign": int(scenario_counts[sk]["benign"]),
        "ddos":   int(scenario_counts[sk]["ddos"]),
        "ratio_ddos_per_benign": (
            float(scenario_counts[sk]["ddos"]) / scenario_counts[sk]["benign"]
            if scenario_counts[sk]["benign"] > 0 else None
        )
    }
    for sk in sorted(scenario_counts.keys())
}

summary = {
    "target_ratio_ddos_per_benign": PER_SCENARIO_ATTACK_RATIO,
    "global_caps": {"benign": GLOBAL_MAX_BENIGN, "ddos": GLOBAL_MAX_DDOS},
    "final_counts_global": {"benign": int(global_counts[0]), "ddos": int(global_counts[1]), "total_rows_written": int(total_rows)},
    "per_scenario_counts": per_scenario_counts,
    "per_scenario_rows_written": scenario_rows,
}

with open(OUT_SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)
with open(OUT_LABELMAP_JSON, "w") as f:
    json.dump(LABEL_TO_INT, f, indent=2)

elapsed = timedelta(seconds=round(time.time() - start_time))
print(f"\n Merge complete → {OUT_PATH}")
print(f"BENIGN={global_counts[0]:,}  |  DDoS={global_counts[1]:,}  |  TOTAL={total_rows:,}")
print(f"Duration: {elapsed}")
