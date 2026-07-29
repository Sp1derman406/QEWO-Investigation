"""
scalability_fraud.py

Investigates QEWO scalability on the credit-card fraud dataset.
Architecture: 30 -> H -> H -> 2 (two hidden layers, tanh, softmax, no biases).
H swept over {8, 16, 32, 64, 128, 256}.

QEWO is implemented as classical coordinate descent — numpy argmin replaces
the Grover circuit. The evaluation count per epoch (N x R) is identical to
what a quantum implementation would count; on quantum hardware Grover's
algorithm reduces this to N x sqrt(R) oracle calls.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import kagglehub
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ========================= CONFIG =========================
SEEDS         = [42, 0, 1, 2, 3]
HIDDEN_SIZES  = [8, 16, 32, 64, 128, 256]
LR_CANDIDATES = [1e-3, 1e-2, 1e-1]
RESOLUTION    = 3
WEIGHT_DECAY  = 1e-3
MAX_EPOCHS    = 20
PATIENCE      = 3          # early stopping patience (val F1)
SEARCH_STD    = 0.5        # initial search window scale
MIN_STD       = 0.1
MAX_STD       = 2.0
LR_SWEEP_EPOCHS = 5        # quick epochs used for LR selection only


# ========================= LOGGING =========================
os.makedirs("logs", exist_ok=True)
_logfile = open("logs/scalability_fraud.log", "w", buffering=1)

def log(msg=""):
    print(msg, flush=True)
    _logfile.write(msg + "\n")


# ========================= DATA =========================
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df   = pd.read_csv(os.path.join(path, "creditcard.csv"))

fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0].sample(len(fraud), random_state=42)
df_bal = (pd.concat([fraud, legit])
            .sample(frac=1, random_state=42)
            .drop_duplicates(subset=df.columns.difference(["Class"])))

X = df_bal.drop(columns=["Class"]).values
y = df_bal["Class"].values

# 65 / 15 / 20 split  (0.1875 × 80% ≈ 15% of total)
x_tv,    x_test,  y_tv,    y_test  = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
x_train, x_val,   y_train, y_val   = train_test_split(
    x_tv, y_tv, test_size=0.1875, random_state=42, stratify=y_tv)

mu    = x_train.mean(axis=0)
sigma = x_train.std(axis=0) + 1e-8
x_train = (x_train - mu) / sigma
x_val   = (x_val   - mu) / sigma
x_test  = (x_test  - mu) / sigma

enc        = OneHotEncoder(sparse_output=False)
y_train_1h = enc.fit_transform(y_train.reshape(-1, 1))
y_val_1h   = enc.transform(y_val.reshape(-1, 1))
y_test_1h  = enc.transform(y_test.reshape(-1, 1))

N_IN  = x_train.shape[1]   # 30
N_OUT = 2

log("=" * 62)
log("Experiment: scalability_fraud")
log(f"  Dataset:  {len(X)} samples (balanced, deduped)")
log(f"  Split:    {len(x_train)} train / {len(x_val)} val / {len(x_test)} test")
log(f"  H sweep:  {HIDDEN_SIZES}")
log(f"  Seeds:    {SEEDS}")
log(f"  R={RESOLUTION}  λ={WEIGHT_DECAY}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")
log("=" * 62)


# ========================= HELPERS =========================
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _forward(x, W1, W2, W3):
    h1 = np.tanh(x @ W1)
    h2 = np.tanh(h1 @ W2)
    p  = _softmax(h2 @ W3)
    return h1, h2, p


def _loss(p, y_1h, W1, W2, W3):
    ce  = -np.mean(np.sum(y_1h * np.log(p + 1e-9), axis=1))
    reg = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    return float(ce + reg)


def _metrics(p, y):
    yh = np.argmax(p, axis=1)
    return dict(
        f1        = f1_score(y, yh, zero_division=0),
        recall    = recall_score(y, yh, zero_division=0),
        precision = precision_score(y, yh, zero_division=0),
        auc       = roc_auc_score(y, p[:, 1]),
    )


def _val_f1(W1, W2, W3):
    _, _, p = _forward(x_val, W1, W2, W3)
    return f1_score(y_val, np.argmax(p, axis=1), zero_division=0)


def _adam_update(W1, W2, W3, states, lr, t):
    """One ADAM step. states = [mW1,vW1, mW2,vW2, mW3,vW3]."""
    mW1, vW1, mW2, vW2, mW3, vW3 = states
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(x_train)

    h1, h2, p = _forward(x_train, W1, W2, W3)
    dZ3 = (p - y_train_1h) / n
    gW3 = h2.T @ dZ3 + 2*WEIGHT_DECAY*W3
    dH2 = dZ3 @ W3.T
    dZ2 = dH2 * (1 - h2**2)
    gW2 = h1.T @ dZ2 + 2*WEIGHT_DECAY*W2
    dH1 = dZ2 @ W2.T
    dZ1 = dH1 * (1 - h1**2)
    gW1 = x_train.T @ dZ1 + 2*WEIGHT_DECAY*W1

    for W, g, m, v in [(W1,gW1,mW1,vW1),(W2,gW2,mW2,vW2),(W3,gW3,mW3,vW3)]:
        m[:] = b1*m + (1-b1)*g
        v[:] = b2*v + (1-b2)*g**2
        W -= lr * (m / (1-b1**t)) / (np.sqrt(v / (1-b2**t)) + eps)


# ========================= LR SWEEP =========================
def select_lr(W1_init, W2_init, W3_init):
    best_lr, best_vf1 = None, -1.0
    for lr in LR_CANDIDATES:
        W1, W2, W3 = W1_init.copy(), W2_init.copy(), W3_init.copy()
        states = [np.zeros_like(W1), np.zeros_like(W1),
                  np.zeros_like(W2), np.zeros_like(W2),
                  np.zeros_like(W3), np.zeros_like(W3)]
        for t in range(1, LR_SWEEP_EPOCHS + 1):
            _adam_update(W1, W2, W3, states, lr, t)
        vf1 = _val_f1(W1, W2, W3)
        marker = " <-- best" if vf1 > best_vf1 else ""
        log(f"    lr={lr:.0e}: val F1={vf1:.4f}{marker}")
        if vf1 > best_vf1:
            best_vf1, best_lr = vf1, lr
    return best_lr


# ========================= ADAM =========================
def run_adam(W1_init, W2_init, W3_init, lr):
    W1, W2, W3 = W1_init.copy(), W2_init.copy(), W3_init.copy()
    states = [np.zeros_like(W1), np.zeros_like(W1),
              np.zeros_like(W2), np.zeros_like(W2),
              np.zeros_like(W3), np.zeros_like(W3)]

    best_vf1, pat = -1.0, 0
    bW1, bW2, bW3 = W1.copy(), W2.copy(), W3.copy()
    tr_losses, te_losses = [], []
    epochs_run = 0

    for t in range(1, MAX_EPOCHS + 1):
        _adam_update(W1, W2, W3, states, lr, t)

        _, _, p_tr = _forward(x_train, W1, W2, W3)
        _, _, p_te = _forward(x_test,  W1, W2, W3)
        tr_losses.append(_loss(p_tr, y_train_1h, W1, W2, W3))
        te_losses.append(_loss(p_te, y_test_1h,  W1, W2, W3))
        epochs_run = t

        vf1 = _val_f1(W1, W2, W3)
        if vf1 > best_vf1:
            best_vf1, pat = vf1, 0
            bW1, bW2, bW3 = W1.copy(), W2.copy(), W3.copy()
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    _, _, p_te = _forward(x_test, bW1, bW2, bW3)
    return _metrics(p_te, y_test), tr_losses, te_losses, epochs_run


# ========================= QEWO (classical coordinate descent) =========================
def run_qewo(W1_init, W2_init, W3_init):
    W1, W2, W3 = W1_init.copy(), W2_init.copy(), W3_init.copy()
    s1, s2, s3 = SEARCH_STD, SEARCH_STD, SEARCH_STD
    cumul = 0
    MID = RESOLUTION // 2   # index of the middle (current-value) candidate

    best_vf1, pat = -1.0, 0
    bW1, bW2, bW3 = W1.copy(), W2.copy(), W3.copy()
    tr_losses, te_losses = [], []
    epoch_times = []
    epochs_run = 0

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()

        # ── W1 ─────────────────────────────────────────────────────────────
        std1 = max(np.std(W1), 1e-6)
        _, _, p_pre = _forward(x_train, W1, W2, W3)
        loss_pre = _loss(p_pre, y_train_1h, W1, W2, W3)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                orig  = W1[i, j]
                cands = np.linspace(orig - s1*std1, orig + s1*std1, RESOLUTION)
                ls    = []
                for c in cands:
                    W1[i, j] = c
                    h1t = np.tanh(x_train @ W1)
                    h2t = np.tanh(h1t @ W2)
                    pt  = _softmax(h2t @ W3)
                    ls.append(_loss(pt, y_train_1h, W1, W2, W3))
                    cumul += 1
                W1[i, j] = orig
                best_idx = int(np.argmin(ls))
                if ls[best_idx] < ls[MID]:   # MID candidate = orig value
                    W1[i, j] = cands[best_idx]

        _, _, p_post = _forward(x_train, W1, W2, W3)
        loss_post = _loss(p_post, y_train_1h, W1, W2, W3)
        s1 = max(s1*0.95, MIN_STD) if loss_post < loss_pre else min(s1*1.05, MAX_STD)

        # ── W2 (precompute h1 — fixed while W1 unchanged) ──────────────────
        h1    = np.tanh(x_train @ W1)
        std2  = max(np.std(W2), 1e-6)
        loss_pre = loss_post

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                orig  = W2[i, j]
                cands = np.linspace(orig - s2*std2, orig + s2*std2, RESOLUTION)
                ls    = []
                for c in cands:
                    W2[i, j] = c
                    h2t = np.tanh(h1 @ W2)
                    pt  = _softmax(h2t @ W3)
                    ls.append(_loss(pt, y_train_1h, W1, W2, W3))
                    cumul += 1
                W2[i, j] = orig
                best_idx = int(np.argmin(ls))
                if ls[best_idx] < ls[MID]:
                    W2[i, j] = cands[best_idx]

        _, _, p_post = _forward(x_train, W1, W2, W3)
        loss_post = _loss(p_post, y_train_1h, W1, W2, W3)
        s2 = max(s2*0.95, MIN_STD) if loss_post < loss_pre else min(s2*1.05, MAX_STD)

        # ── W3 (precompute h1, h2 — both fixed while W1,W2 unchanged) ──────
        h1    = np.tanh(x_train @ W1)
        h2    = np.tanh(h1 @ W2)
        std3  = max(np.std(W3), 1e-6)
        loss_pre = loss_post

        for i in range(W3.shape[0]):
            for j in range(W3.shape[1]):
                orig  = W3[i, j]
                cands = np.linspace(orig - s3*std3, orig + s3*std3, RESOLUTION)
                ls    = []
                for c in cands:
                    W3[i, j] = c
                    pt = _softmax(h2 @ W3)
                    ls.append(_loss(pt, y_train_1h, W1, W2, W3))
                    cumul += 1
                W3[i, j] = orig
                best_idx = int(np.argmin(ls))
                if ls[best_idx] < ls[MID]:
                    W3[i, j] = cands[best_idx]

        _, _, p_post = _forward(x_train, W1, W2, W3)
        loss_post = _loss(p_post, y_train_1h, W1, W2, W3)
        s3 = max(s3*0.95, MIN_STD) if loss_post < loss_pre else min(s3*1.05, MAX_STD)

        # ── record ──────────────────────────────────────────────────────────
        _, _, p_tr = _forward(x_train, W1, W2, W3)
        _, _, p_te = _forward(x_test,  W1, W2, W3)
        tr_losses.append(_loss(p_tr, y_train_1h, W1, W2, W3))
        te_losses.append(_loss(p_te, y_test_1h,  W1, W2, W3))
        epoch_times.append(time.time() - t0)
        epochs_run = epoch + 1

        vf1 = _val_f1(W1, W2, W3)
        if vf1 > best_vf1:
            best_vf1, pat = vf1, 0
            bW1, bW2, bW3 = W1.copy(), W2.copy(), W3.copy()
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    _, _, p_te = _forward(x_test, bW1, bW2, bW3)
    evals_per_epoch = cumul / max(epochs_run, 1)
    return _metrics(p_te, y_test), tr_losses, te_losses, evals_per_epoch, epoch_times, epochs_run


# ========================= MAIN LOOP =========================
results = {}   # H -> {N, adam: {...}, qewo: {...}}

for H in HIDDEN_SIZES:
    N = N_IN*H + H*H + H*N_OUT

    log()
    log("=" * 62)
    log(f"  H = {H}  |  N = {N} weights  |  classical evals/epoch ≈ {N*RESOLUTION}")
    log("=" * 62)

    # LR sweep on validation set (seed 42 init)
    np.random.seed(42)
    W1_sw = np.random.uniform(-1, 1, (N_IN, H))
    W2_sw = np.random.uniform(-1, 1, (H, H))
    W3_sw = np.random.uniform(-1, 1, (H, N_OUT))
    log("  ADAM LR sweep (seed=42, 5 epochs, val F1):")
    best_lr = select_lr(W1_sw, W2_sw, W3_sw)
    log(f"  => selected lr = {best_lr}")

    a_f1, a_rec, a_prec, a_auc = [], [], [], []
    q_f1, q_rec, q_prec, q_auc = [], [], [], []
    q_evals, q_times = [], []

    for seed in SEEDS:
        log(f"\n  Seed {seed}")
        np.random.seed(seed)
        W1_i = np.random.uniform(-1, 1, (N_IN, H))
        W2_i = np.random.uniform(-1, 1, (H, H))
        W3_i = np.random.uniform(-1, 1, (H, N_OUT))

        am, _, _, a_ep = run_adam(W1_i, W2_i, W3_i, best_lr)
        log(f"    ADAM ({a_ep} epochs)  F1={am['f1']:.3f}  recall={am['recall']:.3f}  "
            f"prec={am['precision']:.3f}  AUC={am['auc']:.3f}")
        a_f1.append(am['f1']); a_rec.append(am['recall'])
        a_prec.append(am['precision']); a_auc.append(am['auc'])

        qm, _, _, q_ev, q_t, q_ep = run_qewo(W1_i, W2_i, W3_i)
        log(f"    QEWO ({q_ep} epochs)  F1={qm['f1']:.3f}  recall={qm['recall']:.3f}  "
            f"prec={qm['precision']:.3f}  AUC={qm['auc']:.3f}")
        log(f"          evals/epoch={q_ev:.0f}  time/epoch={np.mean(q_t):.1f}s")
        q_f1.append(qm['f1']); q_rec.append(qm['recall'])
        q_prec.append(qm['precision']); q_auc.append(qm['auc'])
        q_evals.append(q_ev); q_times.append(np.mean(q_t))

    results[H] = dict(
        N    = N,
        adam = dict(f1=np.mean(a_f1), f1_std=np.std(a_f1),
                    recall=np.mean(a_rec), recall_std=np.std(a_rec),
                    precision=np.mean(a_prec), auc=np.mean(a_auc)),
        qewo = dict(f1=np.mean(q_f1), f1_std=np.std(q_f1),
                    recall=np.mean(q_rec), recall_std=np.std(q_rec),
                    precision=np.mean(q_prec), auc=np.mean(q_auc),
                    evals=np.mean(q_evals), time=np.mean(q_times)),
    )

    r = results[H]
    log(f"\n  === H={H} summary ===")
    log(f"  ADAM  F1={r['adam']['f1']:.3f}±{r['adam']['f1_std']:.3f}  "
        f"recall={r['adam']['recall']:.3f}  AUC={r['adam']['auc']:.3f}")
    log(f"  QEWO  F1={r['qewo']['f1']:.3f}±{r['qewo']['f1_std']:.3f}  "
        f"recall={r['qewo']['recall']:.3f}  AUC={r['qewo']['auc']:.3f}")
    log(f"  QEWO  evals/epoch={r['qewo']['evals']:.0f}  time/epoch={r['qewo']['time']:.1f}s")


# ========================= FINAL TABLE =========================
log()
log("=" * 62)
log("SCALABILITY SUMMARY")
log(f"{'H':>4}  {'N':>7}  {'ADAM F1':>8}  {'QEWO F1':>8}  "
    f"{'ADAM rec':>9}  {'QEWO rec':>9}  {'evals/ep':>10}  {'s/ep':>6}")
log("-" * 62)
for H in HIDDEN_SIZES:
    r = results[H]
    log(f"{H:>4}  {r['N']:>7}  "
        f"{r['adam']['f1']:.3f}±{r['adam']['f1_std']:.3f}  "
        f"{r['qewo']['f1']:.3f}±{r['qewo']['f1_std']:.3f}  "
        f"{r['adam']['recall']:.3f}     {r['qewo']['recall']:.3f}     "
        f"{r['qewo']['evals']:>10.0f}  {r['qewo']['time']:>6.1f}")
log("=" * 62)


# ========================= FIGURES =========================
Hs = HIDDEN_SIZES
Ns = [results[H]['N'] for H in Hs]

# ── Figure 1: F1 vs N ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(Ns, [results[H]['adam']['f1'] for H in Hs],
            yerr=[results[H]['adam']['f1_std'] for H in Hs],
            marker='o', color='tab:blue', label='ADAM', capsize=4, linewidth=2)
ax.errorbar(Ns, [results[H]['qewo']['f1'] for H in Hs],
            yerr=[results[H]['qewo']['f1_std'] for H in Hs],
            marker='s', color='tab:orange', label='QEWO', capsize=4, linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('N (total weights, log scale)')
ax.set_ylabel('Test F1 (mean ± std, 5 seeds)')
ax.set_title('Scalability: F1 vs network size  (30→H→H→2, fraud dataset)')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('scalability_fraud_f1_vs_n.pdf')
plt.close()

# ── Figure 2: Recall vs N ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(Ns, [results[H]['adam']['recall'] for H in Hs],
            yerr=[results[H]['adam']['recall_std'] for H in Hs],
            marker='o', color='tab:blue', label='ADAM', capsize=4, linewidth=2)
ax.errorbar(Ns, [results[H]['qewo']['recall'] for H in Hs],
            yerr=[results[H]['qewo']['recall_std'] for H in Hs],
            marker='s', color='tab:orange', label='QEWO', capsize=4, linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('N (total weights, log scale)')
ax.set_ylabel('Test Recall (mean ± std, 5 seeds)')
ax.set_title('Scalability: Recall vs network size  (30→H→H→2, fraud dataset)')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('scalability_fraud_recall_vs_n.pdf')
plt.close()

# ── Figure 3: Evaluation cost vs N (log-log) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
classical_evals = [results[H]['qewo']['evals'] for H in Hs]
quantum_evals   = [results[H]['N'] * RESOLUTION**0.5 for H in Hs]
ax.plot(Ns, classical_evals, marker='s', color='tab:orange', linewidth=2,
        label=f'QEWO classical (N×{RESOLUTION})')
ax.plot(Ns, quantum_evals, marker='s', color='tab:orange', linewidth=2,
        linestyle='--', label=f'QEWO quantum (N×√{RESOLUTION})')
ax.axhline(1, color='tab:blue', linewidth=2, linestyle='-', label='ADAM (1 gradient step)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N (total weights, log scale)')
ax.set_ylabel('Loss-function evaluations per epoch (log scale)')
ax.set_title('Evaluation cost scaling')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('scalability_fraud_evals_vs_n.pdf')
plt.close()

# ── Figure 4: Wall-clock time per epoch vs N ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ns, [results[H]['qewo']['time'] for H in Hs],
        marker='s', color='tab:orange', linewidth=2, label='QEWO')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N (total weights, log scale)')
ax.set_ylabel('Wall-clock time per epoch (s, log scale)')
ax.set_title('QEWO wall-clock time scaling (classical coordinate descent)')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('scalability_fraud_time_vs_n.pdf')
plt.close()

log("\nDone. Figures saved:")
log("  scalability_fraud_f1_vs_n.pdf")
log("  scalability_fraud_recall_vs_n.pdf")
log("  scalability_fraud_evals_vs_n.pdf")
log("  scalability_fraud_time_vs_n.pdf")
log("  logs/scalability_fraud.log")

_logfile.close()
