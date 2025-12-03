# model_developement.py
import os, time, json, random, argparse, platform
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from contextlib import nullcontext
import pandas as pd

# ==========================
# Repro & CUDA setup
# ==========================
def set_global_seed(seed: int, deterministic: bool):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic

# ==========================
# Model
# ==========================
class CNN_TSA(nn.Module):
    def __init__(
        self,
        num_features=45,
        num_heads=2,
        hidden_dim=64,
        ff_hidden=128,
        dropout=0.1,
        pool2=True,
        cnn_dropout=0.0,
        ffn_dropout=0.0
    ):
        super().__init__()
        # CNN tower
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(2)               # 48 -> 24
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.cnn_drop = nn.Dropout(cnn_dropout)
        self.pool2 = nn.MaxPool1d(2) if pool2 else nn.Identity()  # 24 -> 12 (or keep 24)

        # Transformer block
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mhsa  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff_drop = nn.Dropout(ffn_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            self.ff_drop,
            nn.Linear(ff_hidden, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

        # Head
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.clf_drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)      # (B, F, T) for Conv1d
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.cnn_drop(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1)      # (B, T', E)

        # Transformer
        x_norm = self.norm1(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.ffn(x_norm))

        # Global avg pool + head
        x = x.mean(dim=1)           # (B, E)
        x = self.clf_drop(F.relu(self.fc1(x)))
        logits = self.fc2(x).squeeze(1)
        return logits               # raw logits

# ==========================
# Helpers
# ==========================
def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.,=" else "_" for c in str(s))

def _variant_label_from_data_dir(data_dir: str) -> str:
    meta_path = os.path.join(data_dir, "preprocess_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tag = meta.get("preproc_tag")
            if tag:
                return _safe_slug(tag)
            params = meta.get("params", {})
            scaler = params.get("scaler", "standard")
            corr   = params.get("corr_thresh", 0.9)
            kfeat  = params.get("k_features", 45)
            windowing = params.get("windowing", "off")
            wtxt = f"w{params.get('window_size',48)}s{params.get('window_stride',24)}" if windowing=='on' else "wOFF"
            return _safe_slug(f"{scaler}_corr{corr:g}_k{kfeat}_{wtxt}")
        except Exception:
            pass
    return _safe_slug(os.path.basename(os.path.normpath(data_dir)))

def _hp_label(lr: float, wd: float, pat: int) -> str:
    def fnum(x):
        s = f"{x:.0e}" if x < 1e-2 or x >= 10 else f"{x:g}"
        return s.replace("+", "")
    return f"lr{fnum(lr)}_wd{fnum(wd)}_pat{pat}"

def _maybe_load(path, fname):
    f = os.path.join(path, fname)
    return np.load(f) if os.path.exists(f) else None

def load_numpy_dataset(path):
    Xtr, ytr = np.load(os.path.join(path,'Xw_train.npy')), np.load(os.path.join(path,'yw_train.npy'))
    Xva, yva = np.load(os.path.join(path,'Xw_val.npy')),   np.load(os.path.join(path,'yw_val.npy'))
    Xte, yte = np.load(os.path.join(path,'Xw_test.npy')),  np.load(os.path.join(path,'yw_test.npy'))
    Str = _maybe_load(path, "Sw_train.npy")
    Sva = _maybe_load(path, "Sw_val.npy")
    Ste = _maybe_load(path, "Sw_test.npy")
    Ttr = _maybe_load(path, "Tw_train.npy")
    Tva = _maybe_load(path, "Tw_val.npy")
    Tte = _maybe_load(path, "Tw_test.npy")
    return (
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32),
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.float32),
        torch.tensor(Xte, dtype=torch.float32),
        torch.tensor(yte, dtype=torch.float32),
        Str, Sva, Ste, Ttr, Tva, Tte
    )

def predict_logits_and_proba(model, loader, device):
    """Evaluation on clean data — no augmentation, no calibration."""
    model.eval()
    y_true, y_score, y_logit = [], [], []
    use_cuda = torch.cuda.is_available()
    amp_ctx = (lambda: torch.amp.autocast(device_type="cuda")) if use_cuda else (lambda: nullcontext())
    with torch.no_grad(), amp_ctx():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy().astype(int))
            y_score.extend(probs.cpu().numpy())
            y_logit.extend(logits.cpu().numpy())
    return np.array(y_true), np.array(y_score), np.array(y_logit)

def _metrics_from_scores(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),  # AUPRC (primary)
        "brier": float(brier_score_loss(y_true, y_score)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }, y_pred

def evaluate_split(model, X, y, device, batch_size=512, thr=0.5):
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    y_true, y_score, y_logit = predict_logits_and_proba(model, loader, device)
    metrics, _ = _metrics_from_scores(y_true, y_score, thr=thr)
    return metrics, (y_true, y_score, y_logit)

def evaluate_by_group(model, X, y, groups, device, name="test", thr=0.5, batch_size=512):
    if groups is None:
        return None
    df_g = pd.DataFrame({"g": groups})
    uniq = pd.unique(df_g["g"].astype(str))
    rows = []
    for g in uniq:
        mask = (df_g["g"].astype(str).values == str(g))
        if mask.sum() == 0:
            continue
        m, _ = evaluate_split(model, X[mask], y[mask], device, batch_size=batch_size, thr=thr)
        m["group"] = str(g); m["n"] = int(mask.sum()); m["split"] = name
        rows.append(m)
    return pd.DataFrame(rows) if rows else None

def evaluate_over_time(model, X, y, ts, device, name="test", thr=0.5, nbins=5, batch_size=512):
    if ts is None:
        return None
    q = pd.qcut(pd.Series(ts), q=nbins, duplicates="drop")
    return evaluate_by_group(model, X, y, q.astype(str).values, device, name, thr, batch_size)

def adversarial_val_auc(Xtr, Xte):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def featurize(X):
        return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)

    X0 = featurize(Xtr); X1 = featurize(Xte)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])

    X = StandardScaler().fit_transform(X)  # helps optimizer
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-4)
    clf.fit(X, y)
    s = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, s))


def bootstrap_ci(y_true, y_score, thr, metric="f1", n=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        m, _ = _metrics_from_scores(y_true[b], y_score[b], thr)
        vals.append(m[metric])
    lo, hi = np.quantile(vals, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def _make_run_dir(root="models/cnn_tsa", subpath: str = ""):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, subpath, ts) if subpath else os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path

# ==========================
# Loss options
# ==========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        pt = torch.where(targets.bool(), p, 1 - p)
        loss = (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class BCEWithLogitsLossSmooth(nn.Module):
    def __init__(self, eps=0.05, pos_weight=None, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        t = targets.clamp(0,1) * (1 - self.eps) + 0.5 * self.eps
        loss = F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight, reduction='none')
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

# ==========================
# Training
# ==========================
def build_scheduler(optimizer, total_epochs, mode):
    mode = (mode or "plateau").lower()
    if mode == "cosine_warmup":
        warmup = max(1, int(0.1 * total_epochs))
        main = total_epochs - warmup
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, main))
            ],
            milestones=[warmup]
        )
        return sched, "cosine_warmup"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    return sched, "plateau"

def pick_threshold(y_true, y_score, objective="f1", cost_fp=1.0, cost_fn=5.0, cost_tp=1.0):
    grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_val = 0.5, -np.inf
    for t in grid:
        y_pred = (y_score >= t).astype(int)
        if objective == "utility":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            val = cost_tp*tp - cost_fp*fp - cost_fn*fn
        else:
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, t
    return float(best_thr)

def train_model(
    data_dir="PreprocessedOut/ts-split",
    out_dir="models/cnn_tsa",
    num_features=45, num_heads=2, hidden_dim=64,
    lr=1e-3, weight_decay=1e-4,
    batch_size=512, epochs=50,
    patience=10, pool2=True,
    device=None, subdir="",
    loss_type="bce", focal_gamma=2.0,
    scheduler_mode="plateau",
    deterministic=False, seed=42,
    thr_objective="f1", cost_fp=1.0, cost_fn=5.0, cost_tp=1.0,
    # regularization
    cnn_dropout=0.0, ffn_dropout=0.0,
    # threshold mode
    thr_mode="fixed",
    fixed_threshold=0.5,
):
    set_global_seed(seed, deterministic)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device} | deterministic={deterministic}")

    run_dir = _make_run_dir(out_dir, subdir)
    print(f"[INFO] Saving artifacts to: {run_dir}")

    (Xtr, ytr, Xva, yva, Xte, yte,
     S_tr, S_va, S_te, T_tr, T_va, T_te) = load_numpy_dataset(data_dir)

    inferred_features = int(Xtr.shape[2]) if Xtr.ndim == 3 else num_features
    if inferred_features != num_features:
        print(f"[INFO] Overriding num_features: arg={num_features} -> inferred={inferred_features}")
        num_features = inferred_features

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    print(f"[INFO] Train pos={int((ytr==1).sum())} neg={int((ytr==0).sum())}  "
          f"Val pos={int((yva==1).sum())} neg={int((yva==0).sum())}  "
          f"Test pos={int((yte==1).sum())} neg={int((yte==0).sum())}")

    model = CNN_TSA(
        num_features=num_features,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        pool2=pool2,
        cnn_dropout=cnn_dropout,
        ffn_dropout=ffn_dropout
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched, sched_name = build_scheduler(opt, epochs, scheduler_mode)

    # Loss
    pos = (ytr == 1).sum().item()
    neg = (ytr == 0).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)
    if loss_type.lower() == "focal":
        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        loss_label = f"focal(gamma={focal_gamma})"
    elif loss_type.lower() == "smooth":
        criterion = BCEWithLogitsLossSmooth(eps=0.05, pos_weight=pos_weight)
        loss_label = "bce_smooth(0.05)"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_label = "bce"
    print(f"[INFO] Loss={loss_label} pos_weight={pos_weight.item():.4f} | Scheduler={sched_name}")

    # AMP
    use_cuda = torch.cuda.is_available()
    amp_ctx = (lambda: torch.amp.autocast(device_type="cuda")) if use_cuda else (lambda: nullcontext())
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    # Config & env
    config = {
        "data_dir": data_dir,
        "subdir": subdir,
        "num_features": num_features, "num_heads": num_heads, "hidden_dim": hidden_dim,
        "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs, "patience": patience, "pool2": pool2,
        "device": device, "deterministic": deterministic, "seed": seed,
        "loss": loss_label, "scheduler": sched_name,
        "thr_objective": thr_objective, "cost_fp": cost_fp, "cost_fn": cost_fn, "cost_tp": cost_tp,
        "regularization": {"cnn_dropout": cnn_dropout, "ffn_dropout": ffn_dropout},
        "thresholding": {"mode": thr_mode, "fixed_threshold": fixed_threshold},
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_val_loss, patience_counter = float('inf'), 0
    save_path = os.path.join(run_dir, "best_weights.pt")
    t_train_start = time.time()

    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "lr": [], "epoch_time_s": []
    }

    # ---- Training loop
    for epoch in range(1, epochs+1):
        t_ep = time.time()
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if use_cuda:
                with amp_ctx():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xtr)

        # Validation
        model.eval()
        val_running = 0.0
        y_true_va, y_pred_va = [], []
        with torch.no_grad(), amp_ctx():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                y_true_va.extend(yb.cpu().numpy().astype(int))
                y_pred_va.extend(preds.cpu().numpy().astype(int))
        val_loss = val_running / len(Xva)

        # Step scheduler
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(val_loss)
        else:
            sched.step()
        lr_now = opt.param_groups[0]["lr"]

        val_acc = accuracy_score(y_true_va, y_pred_va)
        val_prc = precision_score(y_true_va, y_pred_va, zero_division=0)
        val_rec = recall_score(y_true_va, y_pred_va, zero_division=0)
        val_f1  = f1_score(y_true_va, y_pred_va, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prc)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["lr"].append(lr_now)
        history["epoch_time_s"].append(time.time() - t_ep)

        print(f"Epoch {epoch:03d} | Train {train_loss:.5f} | Val {val_loss:.5f} | "
              f"Acc {val_acc:.4f} P {val_prc:.4f} R {val_rec:.4f} F1 {val_f1:.4f} | LR {lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), save_path)
            print("  ↳ Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stop] epoch={epoch} no improvement for {patience} epochs.")
                break

    # === Evaluation Phase ===
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Threshold selection
    y_true_va, y_score_va, y_logit_va = predict_logits_and_proba(
        model, DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False), device
    )
    if thr_mode.lower() == "fixed":
        best_thr = float(fixed_threshold)
        print(f"[INFO] Using FIXED threshold: {best_thr:.3f}")
    else:
        if thr_objective.lower() == "utility":
            best_thr = pick_threshold(y_true_va, y_score_va, objective="utility",
                                      cost_fp=cost_fp, cost_fn=cost_fn, cost_tp=cost_tp)
            print(f"[INFO] Selected threshold on VAL (utility): {best_thr:.3f}")
        else:
            best_thr = pick_threshold(y_true_va, y_score_va, objective="f1")
            print(f"[INFO] Selected threshold on VAL (F1): {best_thr:.3f}")

    # Save calibration curve on VAL (for diagnostics only; no calibration is applied)
    prob_true, prob_pred = calibration_curve(y_true_va, y_score_va, n_bins=10, strategy="quantile")
    pd.DataFrame({"prob_pred":prob_pred, "prob_true":prob_true}).to_csv(
        os.path.join(run_dir, "calibration_val_raw.csv"), index=False)

    # Compute metrics for all splits @best_thr (NO calibration)
    train_metrics, (y_true_tr, y_score_tr, y_logit_tr) = evaluate_split(
        model, Xtr, ytr, device, batch_size, thr=best_thr
    )
    val_metrics,   (y_true_va2, y_score_va2, _) = evaluate_split(
        model, Xva, yva, device, batch_size, thr=best_thr
    )
    test_metrics,  (y_true_te, y_score_te, y_logit_te) = evaluate_split(
        model, Xte, yte, device, batch_size, thr=best_thr
    )

    # Bootstrap CI for F1 on TEST
    try:
        lo, hi = bootstrap_ci(y_true_te, y_score_te, best_thr, metric="f1", n=1000, alpha=0.05, seed=seed)
        test_metrics["f1_ci95_lo"] = lo
        test_metrics["f1_ci95_hi"] = hi
    except Exception as e:
        test_metrics["f1_ci95_error"] = str(e)

    # Adversarial validation (train vs test separability)
    try:
        adv_auc = adversarial_val_auc(Xtr.numpy(), Xte.numpy())
        print(f"[CHECK] Adversarial validation AUC (train vs test separability): {adv_auc:.3f}")
        test_metrics["adv_val_auc_train_vs_test"] = adv_auc
    except Exception as e:
        test_metrics["adv_val_auc_error"] = str(e)

    # Per-scenario and per-time evaluations (if arrays provided)
    try:
        dfs = []
        if 'S_te' in locals() and S_te is not None:
            df_by_scen = evaluate_by_group(model, Xte, yte, S_te, device, name="test", thr=best_thr, batch_size=batch_size)
            if df_by_scen is not None:
                df_by_scen["group_type"] = "scenario"
                dfs.append(df_by_scen)
        if 'T_te' in locals() and T_te is not None:
            df_by_time = evaluate_over_time(model, Xte, yte, T_te, device, name="test", thr=best_thr, nbins=5, batch_size=batch_size)
            if df_by_time is not None:
                df_by_time["group_type"] = "time_bin"
                dfs.append(df_by_time)
        if dfs:
            per_slice = pd.concat(dfs, ignore_index=True)
            per_slice.to_csv(os.path.join(run_dir, "metrics_test_slices.csv"),
                             index=False, float_format="%.6f")
    except Exception as e:
        with open(os.path.join(run_dir, "slice_eval_error.txt"), "w") as f:
            f.write(str(e))

    # PR curve (TEST) for reference
    pr_prec, pr_recall, pr_thresh = precision_recall_curve(y_true_te, y_score_te)
    pd.DataFrame({"precision":pr_prec, "recall":pr_recall}).to_csv(os.path.join(run_dir, "pr_curve_test.csv"), index=False)

    # Confusion matrix details
    tn, fp, fn, tp = confusion_matrix(y_true_te, (y_score_te>=best_thr).astype(int), labels=[0,1]).ravel()
    test_metrics.update({"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)})

    # ---- Add runtime info
    train_time_s = time.time() - t_train_start
    epochs_ran   = len(history["train_loss"])
    for m in (train_metrics, val_metrics, test_metrics):
        m["train_time_sec"] = float(train_time_s)
        m["train_time_min"] = float(train_time_s / 60.0)
        m["epochs_ran"] = int(epochs_ran)

    # Save metrics and history
    with open(os.path.join(run_dir, "metrics_train.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    pd.DataFrame([{"split": "train", **train_metrics},
                  {"split": "val", **val_metrics},
                  {"split": "test", **test_metrics}]) \
      .to_csv(os.path.join(run_dir, "metrics_all_splits.csv"), index=False, float_format="%.6f")

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    # Final print — headline PR-AUC first (AUPRC), then others
    print("\n=== FINAL PERFORMANCE (@thr from VAL) ===")
    for name, metrics in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"[{name} @thr={metrics['threshold']:.3f}] "
              f"AUPRC={metrics['pr_auc']:.6f}  ROC-AUC={metrics['roc_auc']:.6f}  "
              f"F1={metrics['f1']*100:.2f}%  P={metrics['precision']*100:.2f}%  "
              f"R={metrics['recall']*100:.2f}%  Acc={metrics['accuracy']*100:.2f}%  "
              f"Brier={metrics['brier']:.6f}")
    print(f"\n✅ All artifacts saved in: {run_dir}")

    return test_metrics, run_dir

# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", default=["PreprocessedOut/ts-split"],
                    help="One or more preprocessed data directories (must contain Xw_*.npy/yw_*.npy; "
                         "optionally Sw_*.npy/Tw_*.npy for per-slice eval).")
    ap.add_argument("--out_dir", type=str, default="models/cnn_tsa")
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_features", type=int, default=45, help="Fallback; auto-inferred from data windows.")
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="One or more learning rates to try.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--pool2", action="store_true")

    # Loss / scheduler / reproducibility
    ap.add_argument("--loss", choices=["bce","focal","smooth"], default="bce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--scheduler", choices=["plateau","cosine_warmup"], default="plateau")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Thresholding
    ap.add_argument("--thr_objective", choices=["f1","utility"], default="f1")
    ap.add_argument("--cost_fp", type=float, default=1.0)
    ap.add_argument("--cost_fn", type=float, default=5.0)
    ap.add_argument("--cost_tp", type=float, default=1.0)
    ap.add_argument("--thr_mode", choices=["f1","fixed"], default="fixed",
                    help="Use 'fixed' to apply a constant threshold; 'f1' to pick best on VAL.")
    ap.add_argument("--fixed_threshold", type=float, default=0.5,
                    help="Decision threshold used when --thr_mode fixed.")

    # Regularization knobs
    ap.add_argument("--cnn_dropout", type=float, default=0.0)
    ap.add_argument("--ffn_dropout", type=float, default=0.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []
    t0_all = time.time()

    for ddir in args.data_dir:
        variant = _variant_label_from_data_dir(ddir)
        for lrate in args.lr:
            hp = _hp_label(lrate, args.weight_decay, args.patience)
            subdir = os.path.join(variant, hp)

            t0 = time.time()
            print(f"\n=== RUN: variant={variant} | {hp} | data_dir={ddir} ===")
            metrics, run_dir = train_model(
                data_dir=ddir,
                out_dir=args.out_dir,
                num_features=args.num_features,
                num_heads=args.num_heads,
                hidden_dim=args.hidden_dim,
                lr=lrate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                pool2=args.pool2,
                subdir=subdir,
                loss_type=args.loss,
                focal_gamma=args.focal_gamma,
                scheduler_mode=args.scheduler,
                deterministic=args.deterministic,
                seed=args.seed,
                thr_objective=args.thr_objective,
                cost_fp=args.cost_fp, cost_fn=args.cost_fn, cost_tp=args.cost_tp,
                cnn_dropout=args.cnn_dropout,
                ffn_dropout=args.ffn_dropout,
                thr_mode=args.thr_mode,
                fixed_threshold=args.fixed_threshold
            )
            dt = (time.time() - t0) / 60.0
            print(f"✅ Completed in {dt:.2f} min → {run_dir}")

            summary_rows.append({
                "run_dir": run_dir,
                "variant": variant,
                "hp_label": hp,
                "data_dir": ddir,
                "lr": lrate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                # Headline PR-AUC (AUPRC) first
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "brier": metrics["brier"],
                "threshold": metrics["threshold"],
                "train_time_min": metrics["train_time_min"],
                "epochs_ran": metrics["epochs_ran"],
                "adv_val_auc_train_vs_test": metrics.get("adv_val_auc_train_vs_test", np.nan),
                "f1_ci95_lo": metrics.get("f1_ci95_lo", np.nan),
                "f1_ci95_hi": metrics.get("f1_ci95_hi", np.nan),
                "tp": metrics.get("tp", np.nan),
                "fp": metrics.get("fp", np.nan),
                "tn": metrics.get("tn", np.nan),
                "fn": metrics.get("fn", np.nan),
                "thr_objective": args.thr_objective
            })

    summary_path = os.path.join(args.out_dir, "summary_runs.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    dt_all = (time.time() - t0_all) / 60.0
    print(f"\n📄 Wrote summary: {summary_path}")
    print(f"⏱️ Total wall time: {dt_all:.2f} min")
