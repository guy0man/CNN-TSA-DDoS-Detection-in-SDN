import os, glob, time, re
from collections import defaultdict
import pandas as pd

CSV_GLOB = "CICDDos2019/*.csv"
CHUNK_SIZE = 500_000

CANON_MAP = {
    "UDP-LAG": "UDPLAG",
    "UDP_LAG": "UDPLAG",
}
def canonical_scenario(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    u = stem.upper()
    u = CANON_MAP.get(u, u)
    if not u.startswith("DRDOS_"):
        m = re.match(r"^(.*?)(\d+)$", u) 
        if m:
            u = m.group(1)
    if u.startswith("DRDOS_"):
        key = "DrDoS_" + u.split("DRDOS_", 1)[1]
    else:
        pretty = {"UDPLAG": "UDPLag", "NETBIOS": "NetBIOS", "MSSQL": "MSSQL", "TFTP": "TFTP", "UDP": "UDP", "SYN": "Syn", "LDAP": "LDAP", "PORTMAP": "Portmap"}
        key = pretty.get(u, u.title())
    return key

def is_label(colname: str) -> bool:
    return colname.strip().replace("\ufeff", "").lower() == "label"

ATTACK_STRINGS = {
    "SYN","UDP","UDP-LAG","UDPLAG","PORTMAP","MSSQL","NETBIOS","NTP","SNMP",
    "LDAP","SSDP","TFTP","WEBDDOS","DRDOS_DNS","DRDOS_LDAP","DRDOS_MSSQL",
    "DRDOS_NTP","DRDOS_SNMP","DRDOS_SSDP","DRDOS_UDP","DRDOS_NETBIOS"
}
def normalize_label_series(s: pd.Series) -> pd.Series:
    u = s.astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    u = u.replace({"UDP-LAG": "UDPLAG", "BENIGN ": "BENIGN"})
    is_benign = u.eq("BENIGN")
    is_attack = u.isin(ATTACK_STRINGS)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out.loc[is_benign] = "BENIGN"
    out.loc[is_attack] = "DDoS"
    return out

total_benign = 0
total_ddos   = 0
total_rows_scanned = 0

per_file = {}
per_scenario = defaultdict(lambda: {"BENIGN": 0, "DDoS": 0, "LABELED": 0})

start = time.time()

for f in glob.glob(CSV_GLOB):
    print(f"📂 Scanning {f} ...")
    benign_f = 0
    ddos_f   = 0
    labeled_f = 0
    scen_key = canonical_scenario(f)

    try:
        found_any = False
        chunk_iter = pd.read_csv(
            f,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            on_bad_lines="skip",
            engine="python",
            encoding="utf-8-sig",
            usecols=lambda c: is_label(c)
        )
        for ch in chunk_iter:
            ch.columns = [c.strip().replace("\ufeff", "") for c in ch.columns]
            if "Label" not in ch.columns:
                continue
            lab = normalize_label_series(ch["Label"])
            benign_c = int((lab == "BENIGN").sum())
            ddos_c   = int((lab == "DDoS").sum())
            labeled_c = benign_c + ddos_c

            total_benign += benign_c
            total_ddos   += ddos_c
            total_rows_scanned += len(ch)

            benign_f += benign_c
            ddos_f   += ddos_c
            labeled_f += labeled_c

            per_scenario[scen_key]["BENIGN"] += benign_c
            per_scenario[scen_key]["DDoS"]   += ddos_c
            per_scenario[scen_key]["LABELED"]+= labeled_c

            found_any = True

        if not found_any:
            print(f"⚠️ No 'Label' column detected in {f} (after cleaning); skipping.")

    except Exception as e:
        print(f"❌ Error reading {f}: {e}")

    per_file[f] = (benign_f, ddos_f, labeled_f)

elapsed = time.time() - start

print("\n==== SUMMARY (binary normalized) ====")
print(f"✅ Total BENIGN rows: {total_benign:,}")
print(f"✅ Total DDoS rows  : {total_ddos:,}")
print(f"✅ Rows scanned (label col only): {total_rows_scanned:,}")

print("\n---- Per-scenario (after canonicalization) ----")
rows = []
for scen, d in per_scenario.items():
    rows.append((scen, d["BENIGN"], d["DDoS"], d["LABELED"]))
rows.sort(key=lambda t: t[3], reverse=True)
for scen, b, d, l in rows:
    print(f"{scen:15s}  BENIGN={b:8,d}  DDoS={d:8,d}  LABELED={l:8,d}")

print("\n---- Top files by labeled rows ----")
top_files = sorted(per_file.items(), key=lambda kv: kv[1][2], reverse=True)[:10]
for path, (b, d, l) in top_files:
    print(f"{os.path.basename(path):25s}  BENIGN={b:8,d}  DDoS={d:8,d}  LABELED={l:8,d}")
