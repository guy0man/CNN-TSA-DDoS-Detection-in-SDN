# aggregate_plots.py
import os, json, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

# Try to enable DET with normal deviate axes if SciPy is available
try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------------------------
# utils
# ---------------------------
def _is_dir(p): return os.path.isdir(p)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def load_arrays(run_dir):
    y_true = np.load(os.path.join(run_dir, "y_true_test.npy"))
    y_score = np.load(os.path.join(run_dir, "y_score_test.npy"))
    # prefer stored y_pred if present; else use selected_threshold.txt; else 0.5
    pred_path = os.path.join(run_dir, "y_pred_test.npy")
    if os.path.exists(pred_path):
        y_pred = np.load(pred_path)
    else:
        thr_path = os.path.join(run_dir, "selected_threshold.txt")
        thr = 0.5
        if os.path.exists(thr_path):
            try:
                with open(thr_path) as f:
                    thr = float(f.readline().strip())
            except Exception:
                pass
        y_pred = (y_score >= thr).astype(int)
    return y_true.astype(int), y_score.astype(float), y_pred.astype(int)

def safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def latest_subdir(p):
    subs = [d for d in os.listdir(p) if _is_dir(os.path.join(p, d))]
    if not subs: return None
    subs.sort()
    return os.path.join(p, subs[-1])

def select_run_dir(hp_dir, mode="latest"):
    """
    hp_dir = models/cnn_tsa/<variant>/<hp_label>
    mode: latest | best_pr | best_roc
    """
    if not _is_dir(hp_dir): return None
    cands = []
    for sub in os.listdir(hp_dir):
        rd = os.path.join(hp_dir, sub)
        if not _is_dir(rd): continue
        m = safe_read_json(os.path.join(rd, "metrics_test_thresholded.json"))
        cands.append((sub, rd, m))
    if not cands:
        return latest_subdir(hp_dir)

    if mode == "latest":
        cands.sort(key=lambda x: x[0])
        return cands[-1][1]

    key = "pr_auc" if mode == "best_pr" else "roc_auc"
    ranked = [(rd, m.get(key, float("-inf")), sub) for sub, rd, m in cands]
    ranked.sort(key=lambda x: x[1])
    return ranked[-1][0]

def try_load_history(run_dir):
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        for col in ["train_loss","val_loss","val_accuracy","val_precision","val_recall","val_f1","lr"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None

# ---------------------------
# plotting helpers (multi-lines)
# ---------------------------
def plot_roc_multi(series, outpath_base, title, zoom_fpr_max=0.02, zoom_tpr_min=0.98):
    # full
    plt.figure()
    for label, y_true, y_score, _ in series:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC={auc(fpr,tpr):.4f})")
    plt.plot([0,1],[0,1],'--', lw=1, color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {title}"); plt.legend()
    savefig(outpath_base.replace(".png","_full.png"))

    # zoom
    plt.figure()
    for label, y_true, y_score, _ in series:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=1.5, label=label)
    plt.plot([0,1],[0,1],'--', lw=1, color='gray')
    plt.xlim(0.0, zoom_fpr_max); plt.ylim(zoom_tpr_min, 1.0)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (zoom) — {title}"); plt.legend()
    savefig(outpath_base.replace(".png","_zoom.png"))

def plot_pr_multi(series, outpath_base, title, zoom_min=0.90):
    # full
    plt.figure()
    for label, y_true, y_score, _ in series:
        p, r, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(r, p, lw=1.5, label=f"{label} (AP={ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {title}"); plt.legend()
    savefig(outpath_base.replace(".png","_full.png"))

    # zoom
    plt.figure()
    for label, y_true, y_score, _ in series:
        p, r, _ = precision_recall_curve(y_true, y_score)
        plt.plot(r, p, lw=1.5, label=label)
    plt.xlim(zoom_min, 1.0); plt.ylim(zoom_min, 1.0)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall (zoom) — {title}"); plt.legend()
    savefig(outpath_base.replace(".png","_zoom.png"))

def plot_calibration_multi(series, outpath, title, bins=10):
    plt.figure()
    plt.plot([0,1], [0,1], '--', color='gray', label="Perfect")
    for label, y_true, y_score, _ in series:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_score)
        plt.plot(prob_pred, prob_true, marker='o', lw=1.5, label=f"{label} (Brier={brier:.4f})")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration — {title}"); plt.legend()
    savefig(outpath)

def plot_det_multi(series, outpath, title):
    """
    DET curve: FNR vs FPR. If SciPy present, use normal deviate axes.
    """
    plt.figure()
    for label, y_true, y_score, _ in series:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        if _HAS_SCIPY:
            # probit axes
            x = norm.ppf(fpr.clip(1e-6, 1-1e-6))
            y = norm.ppf(fnr.clip(1e-6, 1-1e-6))
            plt.plot(x, y, lw=1.5, label=label)
        else:
            plt.plot(fpr, fnr, lw=1.5, label=label)
    if _HAS_SCIPY:
        plt.xlabel("FPR (probit)"); plt.ylabel("FNR (probit)")
        plt.title(f"DET (normal deviate) — {title}")
    else:
        plt.xlabel("False Positive Rate"); plt.ylabel("False Negative Rate")
        plt.title(f"DET (linear axes) — {title}")
    plt.legend()
    savefig(outpath)

def plot_threshold_sweep_f1_multi(series, outpath, title):
    plt.figure()
    thr = np.linspace(0.01, 0.99, 99)
    for label, y_true, y_score, _ in series:
        f1s = []
        for t in thr:
            yp = (y_score >= t).astype(int)
            tp = np.sum((y_true==1)&(yp==1)); fp = np.sum((y_true==0)&(yp==1))
            fn = np.sum((y_true==1)&(yp==0))
            prec = tp/(tp+fp) if (tp+fp) else 0.0
            rec  = tp/(tp+fn) if (tp+fn) else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
            f1s.append(f1)
        plt.plot(thr, f1s, lw=1.5, label=label)
    plt.ylim(0, 1.0)
    plt.xlabel("Threshold"); plt.ylabel("F1")
    plt.title(f"Threshold Sweep (F1) — {title}")
    plt.legend()
    savefig(outpath)

def plot_losses_multi(histories, outpath_base, title, show_train=False):
    # histories: list of (label, df or None)
    # val_loss
    plt.figure()
    for label, df in histories:
        if df is None or "val_loss" not in df.columns: continue
        plt.plot(np.arange(1, len(df)+1), df["val_loss"], lw=1.5, label=label)
    plt.xlabel("Epoch"); plt.ylabel("Val BCE loss")
    plt.title(f"Validation loss — {title}"); plt.legend()
    savefig(outpath_base.replace(".png","_val_loss.png"))
    if show_train:
        plt.figure()
        for label, df in histories:
            if df is None or "train_loss" not in df.columns: continue
            plt.plot(np.arange(1, len(df)+1), df["train_loss"], lw=1.5, label=label)
        plt.xlabel("Epoch"); plt.ylabel("Train BCE loss")
        plt.title(f"Train loss — {title}"); plt.legend()
        savefig(outpath_base.replace(".png","_train_loss.png"))

def save_confusion(y_true, y_pred, outpath, labels=("Benign (0)","Attack (1)")):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.xticks([0,1], labels); plt.yticks([0,1], labels)
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.colorbar()
    savefig(outpath)

# ---------------------------
# driver
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Iterate models/cnn_tsa and make plots per preprocessing method with lines per HP variant.")
    ap.add_argument("--models_root", default="models/cnn_tsa",
                    help="Root containing <variant>/<hp>/<timestamp>.")
    ap.add_argument("--selection", choices=["latest","best_pr","best_roc"], default="latest",
                    help="Within each HP label, which timestamp to use.")
    ap.add_argument("--zoom_pr_min", type=float, default=0.90)
    ap.add_argument("--zoom_roc_fpr_max", type=float, default=0.02)
    ap.add_argument("--zoom_roc_tpr_min", type=float, default=0.98)
    ap.add_argument("--calibration_bins", type=int, default=10)
    ap.add_argument("--show_train_loss", action="store_true",
                    help="Also plot train_loss curves (in addition to val_loss).")
    ap.add_argument("--make_superplot", action="store_true",
                    help="Also make ROC/PR super-plots comparing best line from each variant (by PR-AUC).")
    args = ap.parse_args()

    root = args.models_root
    if not _is_dir(root):
        print(f"[ERR] Not a directory: {root}")
        return

    variants = [d for d in os.listdir(root) if _is_dir(os.path.join(root, d))]
    variants.sort()
    overall_rows = []
    super_curves = []

    for variant in variants:
        variant_dir = os.path.join(root, variant)
        outdir = os.path.join(variant_dir, "aggregate_plots")
        confdir = os.path.join(outdir, "confusions")
        ensure_dir(outdir); ensure_dir(confdir)

        series = []           # (label, y_true, y_score, y_pred)
        histories = []        # (label, df)
        per_variant_rows = [] # metrics table

        for hp_label in sorted(os.listdir(variant_dir)):
            hp_dir = os.path.join(variant_dir, hp_label)
            if not _is_dir(hp_dir): continue
            run_dir = select_run_dir(hp_dir, mode=args.selection)
            if not run_dir: continue

            try:
                y_true, y_score, y_pred = load_arrays(run_dir)
            except Exception as e:
                print(f"[WARN] skipping {run_dir}: {e}")
                continue

            series.append((hp_label, y_true, y_score, y_pred))
            hist = try_load_history(run_dir)
            histories.append((hp_label, hist))

            # confusion matrix per HP (saved individually)
            save_confusion(y_true, y_pred, os.path.join(confdir, f"{hp_label}.png"))

            # metrics for CSV
            per_variant_rows.append({
                "variant": variant,
                "hp_label": hp_label,
                "selection": args.selection,
                "run_dir": run_dir,
                "roc_auc": float(roc_auc_score(y_true, y_score)),
                "pr_auc":  float(average_precision_score(y_true, y_score)),
                "brier":   float(brier_score_loss(y_true, y_score)),
                "tp": int(np.sum((y_true==1)&(y_pred==1))),
                "fp": int(np.sum((y_true==0)&(y_pred==1))),
                "fn": int(np.sum((y_true==1)&(y_pred==0))),
                "tn": int(np.sum((y_true==0)&(y_pred==0))),
            })

        if not series:
            print(f"[WARN] No runs found under {variant_dir}")
            continue

        # Multi-line plots for this preprocessing method
        plot_roc_multi(series,
                       os.path.join(outdir, f"{variant}_roc.png"),
                       title=variant,
                       zoom_fpr_max=args.zoom_roc_fpr_max,
                       zoom_tpr_min=args.zoom_roc_tpr_min)

        plot_pr_multi(series,
                      os.path.join(outdir, f"{variant}_pr.png"),
                      title=variant,
                      zoom_min=args.zoom_pr_min)

        plot_calibration_multi(series,
                               os.path.join(outdir, f"{variant}_calibration.png"),
                               title=variant,
                               bins=args.calibration_bins)

        plot_det_multi(series,
                       os.path.join(outdir, f"{variant}_det.png"),
                       title=variant)

        plot_threshold_sweep_f1_multi(series,
                                      os.path.join(outdir, f"{variant}_threshold_f1.png"),
                                      title=variant)

        plot_losses_multi(histories,
                          os.path.join(outdir, f"{variant}_losses.png"),
                          title=variant,
                          show_train=args.show_train_loss)

        # CSV summary per variant + accumulate for global table
        dfv = pd.DataFrame(per_variant_rows)
        dfv.to_csv(os.path.join(outdir, "aggregate_summary.csv"), index=False)
        overall_rows.extend(per_variant_rows)

        # pick best by PR-AUC for optional superplot
        if args.make_superplot:
            best = max(per_variant_rows, key=lambda r: r["pr_auc"])
            y_true_b, y_score_b, _ = load_arrays(best["run_dir"])
            super_curves.append((variant, y_true_b, y_score_b, None))

        print(f"[OK] Wrote plots for {variant} → {outdir}")

    # global summary
    if overall_rows:
        df_all = pd.DataFrame(overall_rows)
        df_all.to_csv(os.path.join(root, "summary_all_variants.csv"), index=False)
        print(f"[OK] summary_all_variants.csv → {root}")

    # super-plots comparing best per variant
    if args.make_superplot and super_curves:
        out_all = os.path.join(root, "aggregate_plots"); ensure_dir(out_all)

        # ROC super
        plt.figure()
        for label, y_true, y_score, _ in super_curves:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC={auc(fpr,tpr):.4f})")
        plt.plot([0,1],[0,1],'--', lw=1, color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC — Best per Preprocessing Variant"); plt.legend()
        savefig(os.path.join(out_all, "variants_best_roc_full.png"))

        # PR super
        plt.figure()
        for label, y_true, y_score, _ in super_curves:
            p, r, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            plt.plot(r, p, lw=1.5, label=f"{label} (AP={ap:.4f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall — Best per Preprocessing Variant"); plt.legend()
        savefig(os.path.join(out_all, "variants_best_pr_full.png"))

        print(f"[OK] Super-plots → {out_all}")

if __name__ == "__main__":
    main()
