"""
build_notebook_2.py
===================
Generates Notebook_2_DeepFM.ipynb using the nbformat v4 API.
Run:  python build_notebook_2.py

Output: Notebook_2_DeepFM.ipynb  (ready for Google Colab)
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import textwrap


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(src).strip())


def md(src: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(src).strip())


cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE / INTRO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""
# Notebook 2: DeepFM
### iCog Labs Intern Training: Predicting Risky Drug Combinations

**Your task:** build a **DeepFM** model for the same problem as Notebook 1: predict whether
a drug interaction is **Severe** (`Label = 1`) or **Not severe** (`Label = 0`) from
`Drug 1` and `Drug 2`. Then **compare it with your FM**.

---
### Rules for this notebook
- **Libraries are allowed.** Recommended: **PyTorch** (`nn.Embedding`, `nn.Linear`,
  `nn.BCEWithLogitsLoss`, `torch.optim.Adam`).
  Also allowed: TensorFlow/Keras, or a ready-made library such as DeepCTR-Torch.
  If you use a ready-made DeepFM, you must still explain every component in the Q&A.
- You **must** use the **same split files** you saved in Notebook 1
  (`split_train.csv`, `split_val.csv`, `split_test.csv`). Otherwise the FM vs DeepFM
  comparison is not fair.
- Inputs are `Drug 1` and `Drug 2` **only**. No `Interaction Description`, no `Severity`.
- Use `SEED = 42` everywhere.
- Questions about your choices and results will be asked **live during the evaluation**,
  so make sure you understand every step you implement.

### How to work
Follow the `# TODO` comments. Run top to bottom before submitting so all outputs are visible.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 1. Setup"))

cells.append(code("""
# Install PyTorch if not already present (Colab usually has it)
# !pip install -q torch

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 11})

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version :", torch.__version__)
print("Device          :", DEVICE)
print("NumPy version   :", np.__version__)
print("Pandas version  :", pd.__version__)
print("SEED            :", SEED)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD SAME SPLITS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 2. Load the SAME splits as the FM notebook"))

cells.append(code("""
# TODO: Upload split_train.csv, split_val.csv, split_test.csv (and fm_results.csv) from Notebook 1.
# TODO: Load them into train_df, val_df, test_df.
# TODO: Print the size and Severe percentage of each split. They must match Notebook 1 exactly.

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")
test_df  = pd.read_csv("split_test.csv")
fm_results_df = pd.read_csv("fm_results.csv")

print("Split sizes and class balance:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pct = split["Label"].mean() * 100
    print(f"  {name:6s}: {len(split):,} rows  |  Severe = {pct:.2f}%")

print()
print("FM baseline results (from Notebook 1):")
display(fm_results_df)

# Sanity: confirm the splits look right
assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
print("\\n[OK] Splits loaded and non-empty.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENCODE DRUGS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 3. Encode the drugs"))

cells.append(code("""
# -------------------------------------------------------------------------
# CRITICAL: IDENTICAL offset vocabulary to Notebook 1
# -------------------------------------------------------------------------
# Drug 1 values -> IDs  0      .. n1-1
# Drug 2 values -> IDs  n1     .. n1+n2-1   (offset by n1)
# n_features = n1 + n2  (total embedding table size)
#
# This is exactly the same encoding used in Notebook 1, making the FM and
# DeepFM embeddings conceptually aligned on the same feature space.
# -------------------------------------------------------------------------

# Build vocabulary from TRAIN split ONLY (no val/test leakage)
train_d1_drugs = sorted(train_df["Drug 1"].unique())
train_d2_drugs = sorted(train_df["Drug 2"].unique())

n1         = len(train_d1_drugs)
n2         = len(train_d2_drugs)
n_features = n1 + n2

d1_to_id = {drug: i       for i, drug in enumerate(train_d1_drugs)}
d2_to_id = {drug: i + n1  for i, drug in enumerate(train_d2_drugs)}

id_to_d1 = {v: k for k, v in d1_to_id.items()}
id_to_d2 = {v: k for k, v in d2_to_id.items()}

print(f"n1 (unique Drug 1 in train) : {n1}")
print(f"n2 (unique Drug 2 in train) : {n2}")
print(f"n_features = n1 + n2        : {n_features}")
print(f"Drug-1 ID range             : [0, {n1-1}]")
print(f"Drug-2 ID range (offset)    : [{n1}, {n1+n2-1}]")


# TODO: Build tensors and DataLoaders
def encode_split(split_df, device=DEVICE):
    \"\"\"Map drug names -> offset integer IDs; return (id_d1, id_d2, y) tensors.\"\"\"
    id_d1  = torch.tensor(split_df["Drug 1"].map(d1_to_id).to_numpy(),
                          dtype=torch.long,    device=device)
    id_d2  = torch.tensor(split_df["Drug 2"].map(d2_to_id).to_numpy(),
                          dtype=torch.long,    device=device)
    labels = torch.tensor(split_df["Label"].to_numpy(),
                          dtype=torch.float32, device=device)
    return id_d1, id_d2, labels


train_d1t, train_d2t, y_train_t = encode_split(train_df)
val_d1t,   val_d2t,   y_val_t   = encode_split(val_df)
test_d1t,  test_d2t,  y_test_t  = encode_split(test_df)

# TODO: Wrap in DataLoaders (batch_size=256, shuffle=True for train only)
BATCH_SIZE = 256

train_loader = DataLoader(
    TensorDataset(train_d1t, train_d2t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)
val_loader = DataLoader(
    TensorDataset(val_d1t, val_d2t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(test_d1t, test_d2t, y_test_t),
    batch_size=BATCH_SIZE, shuffle=False
)

# Class-imbalance weighting: pos_weight = N0 / N1 (from TRAIN only)
N1 = int((y_train_t == 1).sum().item())
N0 = int((y_train_t == 0).sum().item())
pos_weight = torch.tensor([N0 / N1], dtype=torch.float32, device=DEVICE)

print(f"\\nClass counts  N0={N0:,}  N1={N1:,}  pos_weight={pos_weight.item():.4f}")
print(f"\\nDataLoader batches -- train: {len(train_loader)}  "
      f"val: {len(val_loader)}  test: {len(test_loader)}")

# Sanity: no NaN IDs
assert not torch.isnan(train_d1t.float()).any(), "NaN in train Drug 1 IDs"
assert not torch.isnan(train_d2t.float()).any(), "NaN in train Drug 2 IDs"
print("[OK] All drug IDs resolved without NaN.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DeepFM MODEL
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""
## 4. The DeepFM model

```
        drug_1 ID        drug_2 ID
            |                |
      [shared embeddings, size k]         <- ONE embedding table per field, used by BOTH parts
            |                |
   FM part:  bias + w[a] + w[b] + <v_a, v_b>   -+
                                                +-> sum -> sigmoid -> P(Severe)
   Deep part: concat(v_a, v_b) -> MLP -> 1 value -+
```

**Shared embeddings:** `W` (n_features x 1) for first-order weights and `V` (n_features x k)
for latent vectors — both indexed into the same offset vocabulary.

**FM branch:** $z_{FM} = w_0 + W[a] + W[b] + \\langle V[a], V[b] \\rangle$

**Deep branch:** $z_{Deep} = \\text{MLP}([V[a] \\| V[b]])$ where $\\|$ denotes concatenation.

**Combined:** $z = z_{FM} + z_{Deep}$, no sigmoid (BCEWithLogitsLoss handles it).
"""))

cells.append(code("""
class DeepFM(nn.Module):
    \"\"\"
    DeepFM for binary drug-interaction severity prediction.

    Parameters
    ----------
    n_features : int
        Total feature IDs = n1 + n2 (offset vocabulary size).
    k : int
        Latent embedding dimension (shared by FM and Deep branches).
    hidden_layers : tuple[int, ...]
        Width of each MLP hidden layer (e.g. (64, 32) or (128, 64, 32)).
    dropout : float
        Dropout probability applied after each hidden activation.
    mode : str
        'full'      -> z_FM + z_Deep
        'fm_only'   -> z_FM  only
        'deep_only' -> z_Deep only

    Architecture details
    --------------------
    Shared embedding tables (indexed by offset vocab IDs):
        W : nn.Embedding(n_features, 1)    first-order drug weights
        V : nn.Embedding(n_features, k)    latent interaction embeddings

    FM branch (eq. for two active features a=id_d1, b=id_d2):
        z_FM = w0 + W[a] + W[b] + sum_f V[a,f]*V[b,f]

    Deep branch (shallow MLP):
        h0     = concat([V[a], V[b]])      shape (N, 2k)
        h_i    = Dropout(ReLU(Linear(h_{i-1})))   for each hidden layer
        z_Deep = Linear(h_last, 1).squeeze(-1)    shape (N,)

    Output: raw logit z (no sigmoid) -- used with nn.BCEWithLogitsLoss.
    \"\"\"

    def __init__(
        self,
        n_features:    int,
        k:             int,
        hidden_layers: tuple = (128, 64),
        dropout:       float = 0.3,
        mode:          str   = "full",
    ) -> None:
        super().__init__()

        assert mode in ("full", "fm_only", "deep_only"), f"Invalid mode: {mode}"
        self.mode = mode
        self.k    = k

        # ── Shared embedding tables ───────────────────────────────────────
        self.w0 = nn.Parameter(torch.zeros(1))
        self.W  = nn.Embedding(n_features, 1)
        self.V  = nn.Embedding(n_features, k)

        # ── Deep branch MLP ───────────────────────────────────────────────
        mlp_layers = []
        in_dim = 2 * k
        for h_dim in hidden_layers:
            mlp_layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # ── Weight initialisation ─────────────────────────────────────────
        nn.init.normal_(self.W.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.V.weight, mean=0.0, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        self.n_features    = n_features
        self.hidden_layers = hidden_layers
        self.dropout       = dropout

    def forward(self, id_d1: torch.Tensor, id_d2: torch.Tensor) -> torch.Tensor:
        \"\"\"Returns raw combined logit z of shape (N,). No sigmoid applied.\"\"\"
        # Embedding lookups
        v_a = self.V(id_d1)               # (N, k)
        v_b = self.V(id_d2)               # (N, k)
        W_a = self.W(id_d1).squeeze(-1)   # (N,)
        W_b = self.W(id_d2).squeeze(-1)   # (N,)

        # FM branch
        interaction = (v_a * v_b).sum(dim=-1)          # (N,)
        z_fm  = self.w0 + W_a + W_b + interaction      # (N,)

        # Deep branch
        h0     = torch.cat([v_a, v_b], dim=-1)          # (N, 2k)
        z_deep = self.mlp(h0).squeeze(-1)               # (N,)

        # Branch selection
        if self.mode == "full":
            return z_fm + z_deep
        elif self.mode == "fm_only":
            return z_fm
        else:   # deep_only
            return z_deep

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────
_m = DeepFM(n_features=n_features, k=16, hidden_layers=(64, 32), dropout=0.3).to(DEVICE)
_d1 = torch.zeros(8, dtype=torch.long, device=DEVICE)
_d2 = torch.ones(8,  dtype=torch.long, device=DEVICE) + n1
_z  = _m(_d1, _d2)
assert _z.shape == (8,), f"Expected (8,), got {_z.shape}"
assert _z.dtype == torch.float32
print(f"DeepFM forward-pass sanity PASSED.")
print(f"  n_features={n_features}  k=16  hidden=(64,32)  dropout=0.3")
print(f"  Output shape: {tuple(_z.shape)}  dtype: {_z.dtype}")
print(f"  Total params: {_m.num_parameters:,}")
del _m, _d1, _d2, _z
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LOSS, OPTIMIZER, TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 5. Loss, optimizer and training loop"))

cells.append(code("""
# Dynamic threshold sweep range (same as Notebook 1)
THRESHOLDS = np.round(np.arange(0.20, 0.61, 0.05), 2)


def evaluate_loader(model, loader, criterion, device=DEVICE):
    \"\"\"
    Run inference on a DataLoader.
    Returns (all_logits, all_targets, avg_loss) as numpy arrays and float.
    \"\"\"
    model.eval()
    logits_list, targets_list = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for b_d1, b_d2, b_y in loader:
            b_d1, b_d2, b_y = b_d1.to(device), b_d2.to(device), b_y.to(device)
            logits = model(b_d1, b_d2)
            loss   = criterion(logits, b_y)
            total_loss += loss.item() * len(b_y)
            total_n    += len(b_y)
            logits_list.append(logits.cpu().numpy())
            targets_list.append(b_y.cpu().numpy())

    all_logits  = np.concatenate(logits_list)
    all_targets = np.concatenate(targets_list)
    return all_logits, all_targets, total_loss / total_n


def best_threshold_f1(logits, targets, thresholds=THRESHOLDS):
    \"\"\"Sweep thresholds and return (best_f1, best_threshold).\"\"\"
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_f1, best_thr = -1.0, 0.5
    for t in thresholds:
        f1 = float(f1_score(targets, (probs >= t).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_f1, best_thr


def train_model(
    model,
    train_loader,
    val_loader,
    pos_weight,
    epochs:    int   = 30,
    lr:        float = 0.001,
    device:    torch.device = DEVICE,
    verbose:   bool  = True,
) -> tuple:
    \"\"\"
    Train a DeepFM model with Adam + BCEWithLogitsLoss(pos_weight).
    Saves best model weights based on highest validation F1 (dynamic threshold).

    Returns
    -------
    (trained_model, history, best_val_f1, best_threshold)
    \"\"\"
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = {"train_loss": [], "val_loss": [], "val_f1": []}

    best_val_f1   = -1.0
    best_threshold = 0.5
    best_weights  = copy.deepcopy(model.state_dict())

    if verbose:
        print(f"  {'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | "
              f"{'Best Val F1':^12} | {'Thresh':^7}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_samples = 0.0, 0

        for b_d1, b_d2, b_y in train_loader:
            b_d1, b_d2, b_y = b_d1.to(device), b_d2.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_d1, b_d2)
            loss   = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_y)
            n_samples    += len(b_y)

        train_loss = running_loss / n_samples
        val_logits, val_targets, val_loss = evaluate_loader(model, val_loader, criterion, device)
        epoch_f1, epoch_thr = best_threshold_f1(val_logits, val_targets)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(epoch_f1)

        if epoch_f1 > best_val_f1:
            best_val_f1   = epoch_f1
            best_threshold = epoch_thr
            best_weights  = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"  {epoch:^6d} | {train_loss:^10.4f} | {val_loss:^10.4f} | "
                  f"{best_val_f1:^12.4f} | {best_threshold:^7.2f}")

    model.load_state_dict(best_weights)
    return model, history, best_val_f1, best_threshold


# Demonstration run (k=16, 3 epochs) to verify loop works
print("Demo training run (k=16, 3 epochs):")
torch.manual_seed(SEED)
_demo_model = DeepFM(n_features=n_features, k=16, hidden_layers=(64, 32),
                      dropout=0.3, mode="full").to(DEVICE)
_demo_model, _demo_h, _demo_f1, _demo_thr = train_model(
    _demo_model, train_loader, val_loader,
    pos_weight, epochs=3, lr=0.001, verbose=True
)
print(f"\\n[OK] Training loop verified. Demo val F1={_demo_f1:.4f} @ threshold={_demo_thr:.2f}")
del _demo_model, _demo_h, _demo_f1, _demo_thr
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 6. Evaluate on the validation set"))

cells.append(code("""
def report_metrics(logits, targets, split_name: str = "", threshold: float = 0.5) -> dict:
    \"\"\"
    Compute and print Accuracy, Precision, Recall, F1, Confusion Matrix.
    Returns a dict of all metrics.

    Note: logits are raw (no sigmoid) — we apply sigmoid here for probabilities.
    \"\"\"
    probs  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (probs >= threshold).astype(int)
    m = {
        "accuracy":  float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall":    float(recall_score(targets, preds, zero_division=0)),
        "f1":        float(f1_score(targets, preds, zero_division=0)),
        "cm":        confusion_matrix(targets, preds),
    }
    cm = m["cm"]
    title = f"{split_name} Evaluation (threshold={threshold:.2f})"
    print(f"\\n{'─'*54}")
    print(f"  {title}")
    print(f"{'─'*54}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                  Pred=0   Pred=1")
    print(f"    Actual=0 :  {cm[0,0]:7d}  {cm[0,1]:7d}   (TN / FP)")
    print(f"    Actual=1 :  {cm[1,0]:7d}  {cm[1,1]:7d}   (FN / TP)")
    print(f"{'─'*54}")
    return m


# TODO: Demonstrate with a quick k=8 model to show threshold tuning in action.
print("Demo validation evaluation (k=8, 5 epochs -- illustration only):")
torch.manual_seed(SEED)
_val_demo = DeepFM(n_features=n_features, k=8, hidden_layers=(64, 32),
                    dropout=0.3, mode="full").to(DEVICE)
_val_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
_val_demo, _, _, _ = train_model(_val_demo, train_loader, val_loader,
                                  pos_weight, epochs=5, lr=0.001, verbose=False)
_val_logits, _val_targets, _ = evaluate_loader(_val_demo, val_loader, _val_crit)

# TODO: Probabilities = sigmoid(logits). Dynamic threshold sweep.
print("\\nThreshold sweep on VALIDATION set (demo k=8, 5 epochs):")
print(f"  {'Threshold':>10}  {'Val F1':>8}")
print(f"  {'-'*10}  {'-'*8}")
_probs = 1.0 / (1.0 + np.exp(-_val_logits))
for t in THRESHOLDS:
    vf1 = f1_score(_val_targets, (_probs >= t).astype(int), zero_division=0)
    print(f"  {t:>10.2f}  {vf1:>8.4f}")

_best_demo_f1, _best_demo_thr = best_threshold_f1(_val_logits, _val_targets)
report_metrics(_val_logits, _val_targets, split_name="Val (demo k=8)", threshold=_best_demo_thr)
del _val_demo, _val_logits, _val_targets, _probs
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPERIMENTS (GRID SEARCH)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 7. Experiments"))

cells.append(code("""
# -------------------------------------------------------------------------
# TODO: Grid search over:
#   k            in [4, 8, 16, 32]
#   hidden_layers in [(64, 32), (128, 64, 32)]
#   dropout       in [0.0, 0.3]
#
# For each combination: train a fresh full DeepFM (mode='full'), record best Val F1.
# Select the configuration with the highest Val F1 -- never look at the test set.
# -------------------------------------------------------------------------

K_VALUES      = [4, 8, 16, 32]
HIDDEN_OPTS   = [(64, 32), (128, 64, 32)]
DROPOUT_OPTS  = [0.0, 0.3]
EPOCHS        = 30
LR            = 0.001

grid_results    = []
grid_models     = {}
grid_histories  = {}

for k in K_VALUES:
    for hidden in HIDDEN_OPTS:
        for drop in DROPOUT_OPTS:
            cfg_name = f"k={k} | h={hidden} | dr={drop}"
            print(f"\\n{'='*60}")
            print(f"  Config: {cfg_name}")
            print(f"{'='*60}")

            torch.manual_seed(SEED)
            np.random.seed(SEED)

            model = DeepFM(
                n_features=n_features, k=k,
                hidden_layers=hidden, dropout=drop,
                mode="full"
            ).to(DEVICE)

            model, history, best_f1, best_thr = train_model(
                model, train_loader, val_loader,
                pos_weight, epochs=EPOCHS, lr=LR, verbose=True
            )

            grid_results.append({
                "k":             k,
                "hidden_layers": str(hidden),
                "dropout":       drop,
                "val_f1":        round(best_f1, 4),
                "threshold":     best_thr,
                "config_name":   cfg_name,
            })
            grid_models[cfg_name]    = model
            grid_histories[cfg_name] = history

            print(f"  -> best_val_f1={best_f1:.4f}  threshold={best_thr:.2f}")

# TODO: Build results table
grid_df = pd.DataFrame(grid_results).sort_values("val_f1", ascending=False)
print("\\nGrid Search Results (sorted by Val F1):")
display(grid_df[["k", "hidden_layers", "dropout", "val_f1", "threshold"]].reset_index(drop=True))
"""))

cells.append(code("""
# TODO: Plot Val F1 comparison across configurations

# ── Plot 1: Val F1 by k (averaged over hidden/dropout) ───────────────────
k_avg = grid_df.groupby("k")["val_f1"].max().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(k_avg["k"].astype(str), k_avg["val_f1"],
            color=["#4C9BE8", "#6BD490", "#F28C38", "#E86A6A"],
            edgecolor="white", linewidth=1.2)
for i, (_, row) in enumerate(k_avg.iterrows()):
    axes[0].text(i, row["val_f1"] + 0.004, f"{row['val_f1']:.4f}",
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_title("Best Val F1 by Embedding Dim k", fontweight="bold")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Val F1 (best threshold)")
axes[0].set_ylim(0, max(k_avg["val_f1"]) + 0.06)
axes[0].grid(axis="y", alpha=0.3)

# ── Plot 2: Full heatmap-style grid -- k vs hidden, coloured by dropout ──
for drop, color, lbl in [(0.0, "#4C9BE8", "dropout=0.0"),
                          (0.3, "#F28C38", "dropout=0.3")]:
    sub = grid_df[grid_df["dropout"] == drop]
    axes[1].scatter(sub["k"], sub["val_f1"], s=120, color=color,
                    label=lbl, zorder=3, edgecolors="white", linewidths=0.8)
    for _, row in sub.iterrows():
        axes[1].annotate(row["hidden_layers"],
                         (row["k"], row["val_f1"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

axes[1].set_title("Val F1: k vs Config (colored by dropout)", fontweight="bold")
axes[1].set_xlabel("Embedding Dim k"); axes[1].set_ylabel("Val F1")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.suptitle("DeepFM Hyperparameter Grid Search", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# TODO: Choose final configuration using VALIDATION F1 only
best_cfg_row    = grid_df.iloc[0]
BEST_CFG_NAME   = best_cfg_row["config_name"]
BEST_K          = int(best_cfg_row["k"])
BEST_HIDDEN     = eval(best_cfg_row["hidden_layers"])   # convert str back to tuple
BEST_DROPOUT    = float(best_cfg_row["dropout"])
BEST_THRESHOLD  = float(best_cfg_row["threshold"])
BEST_VAL_F1     = float(best_cfg_row["val_f1"])
best_full_model = grid_models[BEST_CFG_NAME]

print(f"\\n[OK] Best configuration (by Val F1):")
print(f"  k={BEST_K}  hidden={BEST_HIDDEN}  dropout={BEST_DROPOUT}")
print(f"  Val F1={BEST_VAL_F1:.4f}  threshold={BEST_THRESHOLD:.2f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ABLATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 8. Ablation"))

cells.append(code("""
# -------------------------------------------------------------------------
# TODO: Using the BEST hyperparameters from grid search, train 3 variants:
#   (a) fm_only   -- FM branch only,  Deep branch zeroed
#   (b) deep_only -- Deep branch only, FM branch zeroed
#   (c) full      -- Both branches combined
# Compare Validation F1 for each variant to understand each branch's contribution.
# -------------------------------------------------------------------------

ABLATION_MODES  = ["full", "fm_only", "deep_only"]
ablation_results = {}
ablation_models  = {}
ablation_hists   = {}

for mode in ABLATION_MODES:
    print(f"\\n{'='*55}")
    print(f"  Ablation: mode='{mode}'  k={BEST_K}  hidden={BEST_HIDDEN}  dropout={BEST_DROPOUT}")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DeepFM(
        n_features=n_features, k=BEST_K,
        hidden_layers=BEST_HIDDEN, dropout=BEST_DROPOUT,
        mode=mode
    ).to(DEVICE)

    model, history, best_f1, best_thr = train_model(
        model, train_loader, val_loader,
        pos_weight, epochs=EPOCHS, lr=LR, verbose=True
    )

    ablation_results[mode] = {"val_f1": best_f1, "threshold": best_thr}
    ablation_models[mode]  = model
    ablation_hists[mode]   = history

    print(f"  -> mode='{mode}': best_val_f1={best_f1:.4f} @ threshold={best_thr:.2f}")

# ── Ablation summary table ────────────────────────────────────────────────
abl_df = pd.DataFrame([
    {"mode": m, "val_f1": ablation_results[m]["val_f1"],
     "threshold": ablation_results[m]["threshold"]}
    for m in ABLATION_MODES
]).sort_values("val_f1", ascending=False)

print("\\nAblation Study Results:")
display(abl_df.reset_index(drop=True))

# ── Ablation bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
colors  = {"full": "#4C9BE8", "fm_only": "#6BD490", "deep_only": "#F28C38"}
for i, row in abl_df.iterrows():
    bar = ax.bar(row["mode"], row["val_f1"],
                 color=colors[row["mode"]], edgecolor="white", linewidth=1.2)
    ax.text(list(abl_df["mode"]).index(row["mode"]),
            row["val_f1"] + 0.005,
            f"{row['val_f1']:.4f}", ha="center", fontsize=10, fontweight="bold")

ax.set_title(f"Ablation Study: Val F1 by Mode\\n(k={BEST_K}, hidden={BEST_HIDDEN}, dropout={BEST_DROPOUT})",
             fontweight="bold")
ax.set_xlabel("Mode"); ax.set_ylabel("Validation F1 (best threshold)")
ax.set_ylim(0, max(abl_df["val_f1"]) + 0.07)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# The best ablation mode for final testing (should be 'full')
best_ablation_mode  = abl_df.iloc[0]["mode"]
FINAL_MODEL         = ablation_models[best_ablation_mode]
FINAL_THRESHOLD     = ablation_results[best_ablation_mode]["threshold"]
print(f"\\nBest ablation mode: '{best_ablation_mode}' (Val F1={ablation_results[best_ablation_mode]['val_f1']:.4f})")
print(f"This model will be used for the final test evaluation.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 9. Final test evaluation"))

cells.append(code("""
# TODO: Evaluate the final DeepFM on the TEST set ONCE.
# IMPORTANT: Do NOT change anything after seeing these numbers.
#            That would turn your test set into a validation set.

print(f"Final model: mode='{FINAL_MODEL.mode}'")
print(f"  k={BEST_K}  hidden={BEST_HIDDEN}  dropout={BEST_DROPOUT}")
print(f"  threshold={FINAL_THRESHOLD:.2f}")
print("\\nRunning TEST set evaluation -- THIS CELL MUST BE RUN ONLY ONCE.\\n")

_final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
test_logits, test_targets, test_loss = evaluate_loader(
    FINAL_MODEL, test_loader, _final_criterion
)

test_metrics = report_metrics(
    test_logits, test_targets,
    split_name="TEST (DeepFM)", threshold=FINAL_THRESHOLD
)

DEEPFM_TEST_ACC  = test_metrics["accuracy"]
DEEPFM_TEST_PREC = test_metrics["precision"]
DEEPFM_TEST_REC  = test_metrics["recall"]
DEEPFM_TEST_F1   = test_metrics["f1"]
DEEPFM_VAL_F1    = ablation_results[best_ablation_mode]["val_f1"]
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FM vs DeepFM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 10. FM vs DeepFM comparison"))

cells.append(code("""
# TODO: Load fm_results.csv from Notebook 1 and build ONE comparison table:
#       model | embedding_dim_k | test_accuracy | test_precision | test_recall | test_f1

fm_row = fm_results_df.iloc[0].to_dict()

deepfm_row = {
    "model":          "DeepFM",
    "embedding_dim_k": BEST_K,
    "test_accuracy":   round(DEEPFM_TEST_ACC,  4),
    "test_precision":  round(DEEPFM_TEST_PREC, 4),
    "test_recall":     round(DEEPFM_TEST_REC,  4),
    "test_f1":         round(DEEPFM_TEST_F1,   4),
}

fm_compare_row = {
    "model":          "FM (from scratch)",
    "embedding_dim_k": fm_row.get("embedding_dim_k", "?"),
    "test_accuracy":   fm_row.get("test_accuracy",   "?"),
    "test_precision":  fm_row.get("test_precision",  "?"),
    "test_recall":     fm_row.get("test_recall",     "?"),
    "test_f1":         fm_row.get("test_f1",         "?"),
}

comparison_df = pd.DataFrame([fm_compare_row, deepfm_row])
comparison_df = comparison_df.set_index("model")
print("FM vs DeepFM Comparison on TEST set:")
display(comparison_df)

# TODO: Plot test F1 for both models side by side.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Plot 1: Test F1 comparison bar chart ──────────────────────────────────
models       = list(comparison_df.index)
test_f1_vals = [float(comparison_df.loc[m, "test_f1"]) for m in models]
bar_colors   = ["#6BD490", "#4C9BE8"]

bars = axes[0].bar(models, test_f1_vals, color=bar_colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, test_f1_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.006,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Test F1: FM vs DeepFM", fontweight="bold")
axes[0].set_ylabel("Test F1 Score"); axes[0].set_ylim(0, max(test_f1_vals) + 0.12)
axes[0].grid(axis="y", alpha=0.3)

# ── Plot 2: Full metrics comparison (grouped bar) ─────────────────────────
metrics      = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
metric_names = ["Accuracy", "Precision", "Recall", "F1"]
x            = np.arange(len(metric_names))
w            = 0.35

for i, (model_name, color) in enumerate(zip(models, bar_colors)):
    vals = [float(comparison_df.loc[model_name, m]) for m in metrics]
    bars2 = axes[1].bar(x + i * w - w/2, vals, w,
                         label=model_name, color=color,
                         edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars2, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")

axes[1].set_title("Full Metric Comparison: FM vs DeepFM", fontweight="bold")
axes[1].set_xticks(x); axes[1].set_xticklabels(metric_names)
axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1.15)
axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("FM vs DeepFM -- Test Set Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ── Improvement summary ───────────────────────────────────────────────────
fm_f1     = float(fm_compare_row["test_f1"])
deepfm_f1 = DEEPFM_TEST_F1
delta_f1  = deepfm_f1 - fm_f1
print(f"\\nTest F1 improvement: {fm_f1:.4f} (FM) -> {deepfm_f1:.4f} (DeepFM)")
print(f"Delta: {delta_f1:+.4f} ({'improvement' if delta_f1 > 0 else 'regression'})")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — REASONING TASK
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""
## 11. Reasoning task: the same drug pair as in Notebook 1

We decompose the DeepFM prediction for **Anagrelide + Droperidol** — the same pair
explained in Notebook 1 — into its FM branch contribution and Deep branch contribution.

This lets us directly compare the reasoning pathways of both models on an identical sample.
"""))

cells.append(code("""
# TODO: Decompose the SAME drug pair as Notebook 1 using the final DeepFM model.
# Pair: Anagrelide (Drug 1) + Droperidol (Drug 2)
# TODO: Print FM-part contribution, Deep-part contribution, and final probability.
# TODO: Compare with FM notebook's prediction and explain agreement or disagreement.

PAIR_D1 = "Anagrelide"
PAIR_D2 = "Droperidol"

# Verify the pair exists in the vocabulary
if PAIR_D1 not in d1_to_id:
    print(f"WARNING: '{PAIR_D1}' not found in Drug-1 vocabulary.")
    print(f"  Using a fallback TP pair from test set instead.")
    # Find a true positive in test set as fallback
    with torch.no_grad():
        _logits_all, _targets_all, _ = evaluate_loader(
            FINAL_MODEL, test_loader, nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        )
    _probs_all = 1.0 / (1.0 + np.exp(-_logits_all))
    _tp_mask   = (_probs_all >= FINAL_THRESHOLD) & (_targets_all == 1.0)
    if _tp_mask.any():
        _idx   = int(np.where(_tp_mask)[0][np.argmax(_probs_all[_tp_mask])])
        PAIR_D1 = id_to_d1[int(test_d1t[_idx].item())]
        PAIR_D2 = id_to_d2[int(test_d2t[_idx].item())]
        print(f"  Fallback pair: '{PAIR_D1}' + '{PAIR_D2}'")

a = d1_to_id[PAIR_D1]   # Drug-1 feature ID
b = d2_to_id[PAIR_D2]   # Drug-2 feature ID (offset)

a_t = torch.tensor([a], dtype=torch.long, device=DEVICE)
b_t = torch.tensor([b], dtype=torch.long, device=DEVICE)

FINAL_MODEL.eval()
with torch.no_grad():
    v_a = FINAL_MODEL.V(a_t)                        # (1, k)
    v_b = FINAL_MODEL.V(b_t)                        # (1, k)
    W_a = FINAL_MODEL.W(a_t).squeeze(-1).item()
    W_b = FINAL_MODEL.W(b_t).squeeze(-1).item()
    w0  = FINAL_MODEL.w0.item()

    interaction = (v_a * v_b).sum(dim=-1).item()
    z_fm        = w0 + W_a + W_b + interaction

    h0     = torch.cat([v_a, v_b], dim=-1)           # (1, 2k)
    z_deep = FINAL_MODEL.mlp(h0).squeeze(-1).item()

    z_total = z_fm + z_deep
    prob    = 1.0 / (1.0 + np.exp(-z_total))
    pred    = int(prob >= FINAL_THRESHOLD)

print(f"Drug pair decomposition: '{PAIR_D1}' (ID={a}) + '{PAIR_D2}' (ID={b})")
print(f"  Feature IDs: Drug-1 space [0,{n1-1}], Drug-2 space [{n1},{n1+n2-1}]")
print()
print(f"  FM Branch Breakdown")
print(f"  {'-'*52}")
print(f"  w0  (global bias)                   : {w0:+.8f}")
print(f"  W[{a}]  ({PAIR_D1} linear weight) : {W_a:+.8f}")
print(f"  W[{b}]  ({PAIR_D2} linear weight)  : {W_b:+.8f}")
print(f"  <V[{a}], V[{b}]> (pairwise dot)    : {interaction:+.8f}")
print(f"  {'-'*52}")
print(f"  z_FM   = w0 + W[a] + W[b] + <V[a],V[b]>  : {z_fm:+.8f}")
print()
print(f"  Deep Branch")
print(f"  {'-'*52}")
print(f"  z_Deep = MLP([V[{a}] || V[{b}]])           : {z_deep:+.8f}")
print()
print(f"  Combined Logit")
print(f"  {'-'*52}")
print(f"  z = z_FM + z_Deep                         : {z_total:+.8f}")
print(f"  p = sigma(z)                              : {prob:.8f}")
print(f"  Predicted (threshold={FINAL_THRESHOLD:.2f})              : {pred}  ({'Severe' if pred==1 else 'Not Severe'})")

# Check if the pair is in the test set and get ground truth
_pair_rows = test_df[
    ((test_df["Drug 1"] == PAIR_D1) & (test_df["Drug 2"] == PAIR_D2)) |
    ((test_df["Drug 1"] == PAIR_D2) & (test_df["Drug 2"] == PAIR_D1))
]
if not _pair_rows.empty:
    true_label = int(_pair_rows.iloc[0]["Label"])
    severity   = _pair_rows.iloc[0].get("Severity", "N/A")
    desc       = _pair_rows.iloc[0].get("Interaction Description", "N/A")
    print(f"\\n  Ground truth (from test_df):")
    print(f"    True Label  : {true_label} ({'Severe' if true_label==1 else 'Not Severe'})")
    print(f"    Severity    : {severity}")
    print(f"    Description : {desc}")
    agreement = pred == true_label
    print(f"\\n  DeepFM vs FM on this pair:")
    print(f"    DeepFM prediction: {pred} ({'correct' if agreement else 'incorrect'})")
else:
    print(f"\\n  Note: pair not found in test split (may be in train or val).")
"""))

cells.append(code("""
# FM vs DeepFM Reasoning Comparison — FM branch signal breakdown
print("Comparative reasoning: z_FM vs z_Deep contributions")
print(f"  z_FM   : {z_fm:+.6f}  ({abs(z_fm / (z_total + 1e-9))*100:.1f}% of |z_total|)")
print(f"  z_Deep : {z_deep:+.6f}  ({abs(z_deep / (z_total + 1e-9))*100:.1f}% of |z_total|)")
print()
if abs(z_fm) > abs(z_deep):
    print("  -> FM branch dominates: explicit pairwise interaction signal drives this prediction.")
    print("     The latent dot product <V[a], V[b]> is the main evidence.")
else:
    print("  -> Deep branch dominates: the MLP captures high-order non-linear interactions")
    print("     not representable by the degree-2 FM term alone.")
print()
print("  In Notebook 1, the NumPy FM relied entirely on z_FM (no Deep branch).")
print(f"  The DeepFM augments this with z_Deep = {z_deep:+.6f}, which")
print(f"  {'strengthens' if (z_fm > 0) == (z_deep > 0) else 'partially counteracts'} the FM signal.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 12. Results (standard format, required)"))

cells.append(code("""
# TODO: Fill in your final numbers. Keep the exact keys; used to compare all interns.

deepfm_results = {
    "model":           "DeepFM",
    "library":         "PyTorch",
    "split_ratio":     "80/10/10",
    "embedding_dim_k": BEST_K,
    "hidden_layers":   str(BEST_HIDDEN),
    "dropout":         BEST_DROPOUT,
    "learning_rate":   LR,
    "epochs":          EPOCHS,
    "threshold":       round(FINAL_THRESHOLD, 2),
    "val_f1":          round(DEEPFM_VAL_F1,      4),
    "test_accuracy":   round(DEEPFM_TEST_ACC,  4),
    "test_precision":  round(DEEPFM_TEST_PREC, 4),
    "test_recall":     round(DEEPFM_TEST_REC,  4),
    "test_f1":         round(DEEPFM_TEST_F1,   4),
}
print(deepfm_results)

# Save for external comparison
pd.DataFrame([deepfm_results]).to_csv("deepfm_results.csv", index=False)
print("\\n[OK] Results saved to deepfm_results.csv")

# Final comparison summary
print("\\n" + "="*60)
print("  FINAL COMPARISON: FM (Notebook 1) vs DeepFM (Notebook 2)")
print("="*60)
display(comparison_df)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write notebook
# ─────────────────────────────────────────────────────────────────────────────

nb = new_notebook()
nb.cells = cells
nb.metadata = {
    "colab": {"provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "name":         "python3",
    },
    "language_info": {"name": "python"},
}

output_path = "Notebook_2_DeepFM.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[OK]  Written: {output_path}")
print(f"      Cells:   {len(nb.cells)}")
print(f"      Sections: 12 (Setup through Results)")
print(f"      Upload to Google Colab and run top-to-bottom.")
print(f"      Required uploads: split_train.csv, split_val.csv, split_test.csv, fm_results.csv")
