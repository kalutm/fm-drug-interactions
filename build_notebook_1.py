"""
build_notebook_1.py
===================
Generates Notebook_1_FM.ipynb using the nbformat v4 API.
Run:  python build_notebook_1.py

Output: Notebook_1_FM.ipynb  (ready for Google Colab)
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import textwrap


# ─────────────────────────────────────────────────────────────────────────────
# Helper wrappers
# ─────────────────────────────────────────────────────────────────────────────

def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(src).strip())


def md(src: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(src).strip())


# ─────────────────────────────────────────────────────────────────────────────
# Build cell list
# ─────────────────────────────────────────────────────────────────────────────
cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE / INTRO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""
# Notebook 1: Factorization Machine (FM) from Scratch
### iCog Labs Intern Training: Predicting Risky Drug Combinations

**Your task:** build a Factorization Machine **from scratch** that predicts whether the
interaction between two drugs is **Severe** (`Label = 1`) or **Not severe** (`Label = 0`).

**Dataset:** `db_drug_interactions_severity.csv` (40,000 rows)

| Column | Use |
|---|---|
| `Drug 1`, `Drug 2` | The **only** model inputs |
| `Interaction Description` | Never an input (it reveals the answer). Read it for the reasoning task. |
| `Severity` | Never an input (the label is derived from it) |
| `Label` | The target: 1 = Severe, 0 = Mild or Moderate |

---
### Rules for this notebook
- **Allowed:** `pandas`, `numpy`, `matplotlib`, and from `scikit-learn` **only** `train_test_split` and the metrics functions.
- **Not allowed:** PyTorch, TensorFlow, Keras, or any FM / recommender library (`pyfm`, `fastFM`, `xlearn`, `torchfm`, `DeepCTR`, ...).
- You implement the FM equation, the loss, the **gradients**, and the training loop yourself with NumPy.
- Use `SEED = 42` everywhere randomness is involved.
- Questions about your choices and results will be asked **live during the evaluation**, so make sure you understand every step you implement.

### How to work
Each `# TODO` comment tells you what is expected. Replace every `raise NotImplementedError` with your code.
Run the whole notebook top to bottom before submitting, so all outputs are visible.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 1. Setup"))

cells.append(code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 11})

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

SEED = 42
np.random.seed(SEED)

print("NumPy  :", np.__version__)
print("Pandas :", pd.__version__)
print("SEED   :", SEED)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD AND EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 2. Load and explore the data"))

cells.append(code("""
# TODO: Upload db_drug_interactions_severity.csv to Colab (Files panel on the left), then load it.
df = pd.read_csv("db_drug_interactions_severity.csv")

# TODO: Print the shape, the first 5 rows, and check for missing values and duplicate rows.
print("Shape:", df.shape)
print("\\nFirst 5 rows:")
display(df.head())

print("\\nMissing values per column:")
print(df.isnull().sum())

n_dups = df.duplicated().sum()
print(f"\\nDuplicate rows: {n_dups}")

# TODO: Print the Label distribution as counts AND percentages.
print("\\nLabel distribution:")
label_counts = df["Label"].value_counts().sort_index()
label_pct    = df["Label"].value_counts(normalize=True).sort_index() * 100
label_summary = pd.DataFrame({"Count": label_counts, "Percentage (%)": label_pct.round(2)})
label_summary.index = ["0 - Not Severe", "1 - Severe"]
display(label_summary)

alpha_ratio = label_counts[0] / label_counts[1]
print(f"\\nClass imbalance ratio  N0/N1 = {label_counts[0]}/{label_counts[1]} ~= {alpha_ratio:.2f}")
print("-> Minority class (Severe) is only ~{:.1f}% of data. Use weighted loss + F1 metric.".format(label_pct[1]))

# TODO: How many unique drugs are in Drug 1? In Drug 2? In total?
n_d1      = df["Drug 1"].nunique()
n_d2      = df["Drug 2"].nunique()
all_drugs = set(df["Drug 1"]) | set(df["Drug 2"])
print(f"\\nUnique drugs in Drug 1 column : {n_d1}")
print(f"Unique drugs in Drug 2 column : {n_d2}")
print(f"Total unique drugs (union)    : {len(all_drugs)}")
"""))

cells.append(code("""
# TODO (optional): Plot how often each drug appears (histogram of appearance frequency).
drug1_counts = df["Drug 1"].value_counts()
drug2_counts = df["Drug 2"].value_counts()
all_counts   = (drug1_counts.add(drug2_counts, fill_value=0)).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(range(len(drug1_counts)),
            sorted(drug1_counts.values, reverse=True),
            color="#4C9BE8", width=1.0)
axes[0].set_title("Drug 1 - appearance frequency (sorted)", fontweight="bold")
axes[0].set_xlabel("Drug rank (by frequency)")
axes[0].set_ylabel("Count")

axes[1].hist(all_counts.values, bins=40, color="#F28C38", edgecolor="white")
axes[1].set_title("Total appearances per drug (histogram)", fontweight="bold")
axes[1].set_xlabel("Times a drug appears across both columns")
axes[1].set_ylabel("Number of drugs")

plt.tight_layout()
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INPUTS & TARGET
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 3. Choose inputs and target"))

cells.append(code("""
# TODO: Build X from the input columns and y from the target column.
#       Inputs: 'Drug 1' and 'Drug 2' ONLY.
#       Target: 'Label'.
#  Using 'Interaction Description' or 'Severity' as inputs is data leakage.
#    It will give near-perfect scores and an automatic fail on this criterion.

X = df[["Drug 1", "Drug 2"]].copy()   # shape: (N, 2) - only drug name strings
y = df["Label"].to_numpy(dtype=np.float64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\\nX sample (first 5 rows):")
display(X.head())
print("\\ny sample (first 5 values):", y[:5])
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 4. Train / validation / test split"))

cells.append(code("""
# -------------------------------------------------------------------------
# Canonical grouping split - keeps reversed pairs (A,B) & (B,A) together
# to prevent data leakage between splits.
#
# Strategy:
#   1. Build a canonical pair key: tuple(sorted([d1, d2]))
#   2. Group all row indices by that key.
#   3. Shuffle groups, then assign 80 / 10 / 10 % of GROUPS to train/val/test.
#   4. Ensure every drug in val/test appears in train (no cold-start).
#
# Ratio justification:
#   80% train  - FM needs sufficient positive examples (only ~16% of data)
#   10% val    - large enough for stable F1 estimation
#   10% test   - held-out, evaluated exactly once
# -------------------------------------------------------------------------

# Step 1 - canonical pair key
pair_key = df.apply(
    lambda r: tuple(sorted([r["Drug 1"], r["Drug 2"]])), axis=1
)

# Step 2 - group row indices by canonical key
group_to_rows = defaultdict(list)
for row_idx, key in enumerate(pair_key):
    group_to_rows[key].append(row_idx)

groups = list(group_to_rows.keys())
rng    = np.random.default_rng(SEED)
rng.shuffle(groups)

n_groups = len(groups)
n_val    = int(round(0.10 * n_groups))
n_test   = int(round(0.10 * n_groups))
n_train  = n_groups - n_val - n_test

train_groups = groups[:n_train]
val_groups   = groups[n_train : n_train + n_val]
test_groups  = groups[n_train + n_val :]


def rows_for_groups(grp_list):
    rows = []
    for g in grp_list:
        rows.extend(group_to_rows[g])
    return sorted(rows)


train_idx = rows_for_groups(train_groups)
val_idx   = rows_for_groups(val_groups)
test_idx  = rows_for_groups(test_groups)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# Step 4 - ensure every val/test drug appears in train (cold-start prevention)
train_drug_set = set(train_df["Drug 1"]) | set(train_df["Drug 2"])
val_only  = (set(val_df["Drug 1"])  | set(val_df["Drug 2"]))  - train_drug_set
test_only = (set(test_df["Drug 1"]) | set(test_df["Drug 2"])) - train_drug_set
print(f"Drugs in val  but NOT in train : {len(val_only)}")
print(f"Drugs in test but NOT in train : {len(test_only)}")
assert len(val_only) == 0 and len(test_only) == 0, (
    "Cold-start drugs detected - recheck split logic."
)

# TODO: Print the size and the Severe percentage of each split.
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pct = split["Label"].mean() * 100
    print(f"{name:6s}: {len(split):,} rows  |  Severe = {pct:.2f}%")

# TODO: Save the three splits so the DeepFM notebook uses EXACTLY the same data.
train_df.to_csv("split_train.csv", index=False)
val_df.to_csv("split_val.csv",     index=False)
test_df.to_csv("split_test.csv",   index=False)
print("\\n[OK] Splits saved: split_train.csv, split_val.csv, split_test.csv")
print("     Download these three files and upload them to the DeepFM notebook.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENCODE DRUGS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 5. Encode the drugs as feature IDs"))

cells.append(code("""
# -------------------------------------------------------------------------
# CRITICAL: OFFSET VOCABULARY
# -------------------------------------------------------------------------
# Drug 1 and Drug 2 are treated as TWO SEPARATE FEATURE FIELDS.
# Direction matters: "Aspirin inhibits Drug 2" != "Drug 1 inhibits Aspirin".
#
# Encoding:
#   drug_1 values -> IDs  0       .. n1-1
#   drug_2 values -> IDs  n1      .. n1+n2-1   (OFFSET by n1)
#
# This is equivalent to concatenating two separate one-hot vectors:
#   x = [0...1...0 | 0...0...1...0]
#       first field  second field
#   Total length = n1 + n2 = n_features
#
# Consequence for FM:
#   W and V must be sized (n_features,) and (n_features, k) respectively.
#   "Aspirin as Drug 1" (index a)  and  "Aspirin as Drug 2" (index b=a+n1)
#   learn DIFFERENT weights - capturing directional drug roles.
# -------------------------------------------------------------------------

# Build vocabulary from TRAIN ONLY (no leakage from val/test)
train_d1_drugs = sorted(train_df["Drug 1"].unique())
train_d2_drugs = sorted(train_df["Drug 2"].unique())

n1         = len(train_d1_drugs)   # unique Drug 1 values in train
n2         = len(train_d2_drugs)   # unique Drug 2 values in train
n_features = n1 + n2               # total feature IDs

# Mappings: Drug 1 -> [0, n1), Drug 2 -> [n1, n1+n2)
d1_to_id = {drug: i       for i, drug in enumerate(train_d1_drugs)}
d2_to_id = {drug: i + n1  for i, drug in enumerate(train_d2_drugs)}

# Reverse maps for interpretability
id_to_d1 = {v: k for k, v in d1_to_id.items()}
id_to_d2 = {v: k for k, v in d2_to_id.items()}

print(f"n1 (unique Drug 1 in train) : {n1}")
print(f"n2 (unique Drug 2 in train) : {n2}")
print(f"n_features = n1 + n2        : {n_features}")
print(f"Drug-1 ID range             : [0, {n1 - 1}]")
print(f"Drug-2 ID range (offset)    : [{n1}, {n1 + n2 - 1}]")


# TODO: Create integer arrays X_train_ids, X_val_ids, X_test_ids  (shape [n_rows, 2])
#       and label arrays y_train, y_val, y_test.
def encode_split(split_df):
    \"\"\"Map Drug 1 -> [0..n1-1], Drug 2 -> [n1..n1+n2-1]. Returns (id_d1, id_d2, y).\"\"\"
    id_d1  = split_df["Drug 1"].map(d1_to_id).to_numpy(dtype=np.int32)
    id_d2  = split_df["Drug 2"].map(d2_to_id).to_numpy(dtype=np.int32)
    labels = split_df["Label"].to_numpy(dtype=np.float64)
    return id_d1, id_d2, labels


train_d1, train_d2, y_train = encode_split(train_df)
val_d1,   val_d2,   y_val   = encode_split(val_df)
test_d1,  test_d2,  y_test  = encode_split(test_df)

X_train_ids = np.stack([train_d1, train_d2], axis=1)   # (N_train, 2)
X_val_ids   = np.stack([val_d1,   val_d2],   axis=1)
X_test_ids  = np.stack([test_d1,  test_d2],  axis=1)

# TODO: Print n_features
print(f"\\nX_train_ids shape : {X_train_ids.shape}  (each row = [id_drug1, id_drug2])")
print(f"X_val_ids shape   : {X_val_ids.shape}")
print(f"X_test_ids shape  : {X_test_ids.shape}")
print(f"\\nSample encodings (first 5 training rows):")
for i in range(5):
    d1n = train_df.iloc[i]["Drug 1"]
    d2n = train_df.iloc[i]["Drug 2"]
    print(f"  '{d1n}' -> {train_d1[i]}   |   '{d2n}' -> {train_d2[i]}")

# Sanity: no NaN mappings
assert not np.any(np.isnan(X_train_ids.astype(float))), "NaN in train IDs"
assert not np.any(np.isnan(X_val_ids.astype(float))),   "NaN in val IDs"
assert not np.any(np.isnan(X_test_ids.astype(float))),  "NaN in test IDs"
print("\\n[OK] All drug IDs resolved without NaN.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FM MODEL
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""
## 6. The FM model

Recall the FM equation:

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle v_i, v_j \rangle \, x_i x_j \qquad p = \sigma(\hat{y})$$

With one-hot inputs and exactly two active features (drug 1 = $a$, drug 2 = $b$) it simplifies to:

$$\hat{y} = w_0 + w_a + w_b + \langle v_a, v_b \rangle$$

Here $a$ = `id_drug1` $\in [0, n_1)$ and $b$ = `id_drug2` $\in [n_1, n_1+n_2)$.
Both index into the **same** weight vector $W$ and embedding matrix $V$, each of size $n_{\text{features}} = n_1 + n_2$.
"""))

cells.append(code("""
class FactorizationMachine:
    \"\"\"
    Factorization Machine for binary drug-interaction severity prediction.
    Built strictly from scratch using NumPy.

    Parameters
    ----------
    n_features : int
        Total number of feature IDs = n1 + n2  (offset vocabulary size).
    k : int
        Latent embedding dimension.
    seed : int
        RNG seed for reproducibility.

    Learnable parameters
    --------------------
    w0 : ndarray  shape (1,)              - global bias
    W  : ndarray  shape (n_features,)     - first-order weights (one per feature ID)
    V  : ndarray  shape (n_features, k)   - latent interaction embeddings

    Forward pass (two active features: id_drug1=a, id_drug2=b)
    -----------------------------------------------------------
        z = w0 + W[a] + W[b] + sum_f V[a,f] * V[b,f]
        p = sigmoid(z)

    Note: W and V are sized n_features = n1 + n2 because the offset vocabulary
    gives Drug 1 and Drug 2 separate index ranges. "Aspirin as Drug 1" and
    "Aspirin as Drug 2" learn different weights, capturing directionality.
    \"\"\"

    def __init__(self, n_features: int, k: int, seed: int = SEED) -> None:
        rng             = np.random.default_rng(seed)
        self.w0         = np.zeros(1, dtype=np.float64)
        self.W          = rng.normal(0.0, 0.01, size=(n_features,)).astype(np.float64)
        self.V          = rng.normal(0.0, 0.01, size=(n_features, k)).astype(np.float64)
        self.k          = k
        self.n_features = n_features

    # --- Forward pass -------------------------------------------------------

    def forward(self, id_d1: np.ndarray, id_d2: np.ndarray):
        \"\"\"
        FM forward pass. Returns (z, p) each of shape (N,).

        FM equation (simplified for two active features a=id_d1, b=id_d2):
            z = w0 + W[a] + W[b] + <V[a], V[b]>
            p = sigma(z) = 1 / (1 + exp(-z))
        \"\"\"
        lin1 = self.W[id_d1]                                    # (N,)
        lin2 = self.W[id_d2]                                    # (N,)
        intr = np.sum(self.V[id_d1] * self.V[id_d2], axis=-1)  # (N,)
        z    = self.w0[0] + lin1 + lin2 + intr                 # (N,)
        p    = 1.0 / (1.0 + np.exp(-z))                        # sigmoid
        return z, p

    # --- Backward pass (weighted BCE) ----------------------------------------

    def backward(self, id_d1, id_d2, p, y, alpha: float = 1.0):
        \"\"\"
        Analytical gradients for weighted BCE.

        Error signal (eq. 6w):
            delta = -alpha * y * (1 - p) + (1 - y) * p

        Gradients (chain rule):
            dL/dw0          = delta
            dL/dW[id_d1]    = delta
            dL/dW[id_d2]    = delta
            dL/dV[id_d1, f] = delta * V[id_d2, f]
            dL/dV[id_d2, f] = delta * V[id_d1, f]
        \"\"\"
        id_d1 = np.atleast_1d(np.asarray(id_d1, dtype=np.int32))
        id_d2 = np.atleast_1d(np.asarray(id_d2, dtype=np.int32))
        p     = np.atleast_1d(np.asarray(p,     dtype=np.float64))
        y     = np.atleast_1d(np.asarray(y,     dtype=np.float64))

        delta = -alpha * y * (1.0 - p) + (1.0 - y) * p   # (N,)

        return {
            "w0":   float(np.sum(delta)),
            "W_d1": delta.copy(),
            "W_d2": delta.copy(),
            "V_d1": delta[:, np.newaxis] * self.V[id_d2],  # (N, k)
            "V_d2": delta[:, np.newaxis] * self.V[id_d1],  # (N, k)
        }

    # --- Mini-batch training step --------------------------------------------

    def train_batch(self, id_d1, id_d2, y, lr: float, alpha: float = 1.0) -> float:
        \"\"\"
        One SGD update for a mini-batch.

        np.add.at is used instead of += to correctly accumulate gradients
        when the same drug index appears more than once in a batch.
        (Fancy-index += is buffered and silently drops duplicate updates.)
        \"\"\"
        _, p  = self.forward(id_d1, id_d2)
        loss  = float(-np.mean(
            alpha * y * np.log(p + 1e-15) + (1.0 - y) * np.log(1.0 - p + 1e-15)
        ))
        grads = self.backward(id_d1, id_d2, p, y, alpha)
        N     = len(id_d1)

        self.w0[0] -= lr * grads["w0"] / N
        np.add.at(self.W, id_d1, -lr * grads["W_d1"] / N)
        np.add.at(self.W, id_d2, -lr * grads["W_d2"] / N)
        np.add.at(self.V, id_d1, -lr * grads["V_d1"] / N)
        np.add.at(self.V, id_d2, -lr * grads["V_d2"] / N)
        return loss


# Quick sanity: forward-pass shapes and probability range
_fm_test = FactorizationMachine(n_features=n_features, k=4)
_z, _p   = _fm_test.forward(train_d1[:5], train_d2[:5])
assert _z.shape == (5,) and _p.shape == (5,), "Shape mismatch"
assert np.all((_p >= 0) & (_p <= 1)), "Probabilities out of [0, 1]"
print(f"FM forward-pass sanity check PASSED.")
print(f"  n_features = {n_features}   k = 4 (test)")
print(f"  z sample: {np.round(_z, 4)}")
print(f"  p sample: {np.round(_p, 4)}")
del _fm_test, _z, _p
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 7. Loss function"))

cells.append(code("""
# -------------------------------------------------------------------------
# Weighted Binary Cross-Entropy
#
# Standard BCE:
#   L = -[ y * log(p) + (1-y) * log(1-p) ]
#
# Problem: the model minimises BCE by predicting the majority class (Not Severe,
# ~84%) for every sample. That gives zero TP and an F1 of 0.
#
# Fix: scale the positive-class term by alpha = N0 / N1:
#   L_weighted = -[ alpha * y * log(p) + (1-y) * log(1-p) ]
#
# Effect: a Severe example's gradient contribution is alpha-times larger than
# a Not-Severe example's, forcing the model to take minority-class errors seriously.
# -------------------------------------------------------------------------

# Compute alpha from training labels ONLY (no leakage)
N1    = int(np.sum(y_train == 1))
N0    = int(np.sum(y_train == 0))
ALPHA = N0 / N1

print(f"Training set: N0 (Not Severe) = {N0:,}   N1 (Severe) = {N1:,}")
print(f"alpha = N0 / N1 = {N0} / {N1} = {ALPHA:.6f}")
print(f"\\nInterpretation: each Severe training example contributes {ALPHA:.1f}x more")
print(f"to the gradient magnitude than a Not-Severe example.")


def bce_loss_single(p: float, y: float, alpha: float = 1.0) -> float:
    \"\"\"Weighted BCE for a single sample (used by gradient checker).\"\"\"
    eps = 1e-15
    return -(alpha * y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))


def bce_loss_batch(p: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    \"\"\"Mean weighted BCE over a batch (used for logging).\"\"\"
    eps = 1e-15
    return float(-np.mean(
        alpha * y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)
    ))


# Sanity check
_p_test = np.array([0.9, 0.1, 0.8])
_y_test = np.array([1.0, 0.0, 1.0])
_loss_w  = bce_loss_batch(_p_test, _y_test, alpha=ALPHA)
_loss_uw = bce_loss_batch(_p_test, _y_test, alpha=1.0)
print(f"\\nLoss sanity on p={_p_test}, y={_y_test}:")
print(f"  Unweighted BCE           : {_loss_uw:.4f}")
print(f"  Weighted BCE (a={ALPHA:.2f})  : {_loss_w:.4f}  (higher - amplified positive term)")
del _p_test, _y_test, _loss_w, _loss_uw
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GRADIENTS DERIVATION (MARKDOWN) + GRADIENT CHECKER CODE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""
## 8. Gradients (the heart of "from scratch")

### Required step: derive the gradients before coding
Write your derivation in this cell (text or LaTeX). You will be asked to explain it during the evaluation.

**Hint:** for sigmoid + binary cross-entropy, the gradient of the loss with respect to the raw score is simply

$$\frac{\partial L}{\partial \hat{y}} = p - y$$

Now use the chain rule to find:
- $\partial\hat{y}/\partial w_0$ = ?
- $\partial\hat{y}/\partial w_a$ = ? and $\partial\hat{y}/\partial w_b$ = ?
- $\partial\hat{y}/\partial v_a$ = ? and $\partial\hat{y}/\partial v_b$ = ?

---

### Full Derivation — Weighted BCE Gradients

**Setup:** FM logit and probability

$$\hat{y} = w_0 + W[a] + W[b] + \langle V[a], V[b] \rangle, \qquad p = \sigma(\hat{y})$$

**Weighted Loss** (single sample, $\alpha = N_0 / N_1$):

$$L = -\bigl[\alpha \cdot y \cdot \log p \;+\; (1-y) \cdot \log(1-p)\bigr]$$

---

#### Step 1 — Gradient of $L$ with respect to $\hat{y}$

$$\frac{\partial L}{\partial p} = -\frac{\alpha y}{p} + \frac{1-y}{1-p}$$

Using the sigmoid derivative $\frac{\partial p}{\partial \hat{y}} = p(1-p)$:

$$\delta \;=\; \frac{\partial L}{\partial \hat{y}} \;=\; \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial \hat{y}}
= \left(-\frac{\alpha y}{p} + \frac{1-y}{1-p}\right) p(1-p)
= -\alpha y(1-p) + (1-y)p \quad \text{(eq. 6w)}$$

**Sanity checks:**
- $y = 0$: $\delta = p$ — identical to standard unweighted BCE
- $y = 1$: $\delta = -\alpha(1-p)$ — $\alpha\times$ stronger push for minority class

---

#### Step 2 — Gradients for each FM parameter (chain rule)

Since $\hat{y} = w_0 + W[a] + W[b] + \sum_f V[a,f] \cdot V[b,f]$:

$$\frac{\partial \hat{y}}{\partial w_0} = 1 \implies \boxed{\frac{\partial L}{\partial w_0} = \delta}$$

$$\frac{\partial \hat{y}}{\partial W[a]} = 1 \implies \boxed{\frac{\partial L}{\partial W[a]} = \delta}$$

$$\frac{\partial \hat{y}}{\partial W[b]} = 1 \implies \boxed{\frac{\partial L}{\partial W[b]} = \delta}$$

$$\frac{\partial \hat{y}}{\partial V[a,f]} = V[b,f] \implies \boxed{\frac{\partial L}{\partial V[a,f]} = \delta \cdot V[b,f]}$$

$$\frac{\partial \hat{y}}{\partial V[b,f]} = V[a,f] \implies \boxed{\frac{\partial L}{\partial V[b,f]} = \delta \cdot V[a,f]}$$

**Key insight:** the chain-rule expressions ($\delta \cdot 1$ for first-order, $\delta \cdot V[\cdot]$ for latent factors) are **identical** to the unweighted case. Only $\delta$ itself changes — the class-imbalance weighting is absorbed into the error signal.
"""))

cells.append(code("""
# -------------------------------------------------------------------------
# Gradient Checker -- central finite difference verification
#
# For parameter theta at flat index i:
#   numerical approx = [ L(theta_i + eps) - L(theta_i - eps) ] / (2*eps)
#
# We verify w0, W[id_d1], and V[id_d1, 0] for both label-0 and label-1 samples.
# -------------------------------------------------------------------------

def check_gradients(fm, id_d1: int, id_d2: int, y: float,
                    alpha: float = 1.0, eps: float = 1e-5) -> None:
    \"\"\"Central finite-difference gradient checker for weighted BCE.\"\"\"
    d1a = np.array([id_d1], dtype=np.int32)
    d2a = np.array([id_d2], dtype=np.int32)
    ya  = np.array([y])

    _, p_a = fm.forward(d1a, d2a)
    grads  = fm.backward(d1a, d2a, p_a, ya, alpha)

    def loss_now():
        _, p_ = fm.forward(d1a, d2a)
        return bce_loss_single(float(p_[0]), y, alpha)

    def num_grad(arr: np.ndarray, flat_idx: int) -> float:
        arr.flat[flat_idx] += eps;  lp = loss_now()
        arr.flat[flat_idx] -= 2*eps; lm = loss_now()
        arr.flat[flat_idx] += eps   # restore
        return (lp - lm) / (2.0 * eps)

    a_w0  = grads["w0"]
    a_Wd1 = float(grads["W_d1"][0])
    a_Vd1 = float(grads["V_d1"][0, 0])

    n_w0  = num_grad(fm.w0, 0)
    n_Wd1 = num_grad(fm.W,  id_d1)                  # W is 1-D
    n_Vd1 = num_grad(fm.V,  id_d1 * fm.V.shape[1])  # V[id_d1, 0]

    ok_w0  = np.isclose(a_w0,  n_w0)
    ok_Wd1 = np.isclose(a_Wd1, n_Wd1)
    ok_Vd1 = np.isclose(a_Vd1, n_Vd1)

    print(f"  Check -- id_d1={id_d1}, id_d2={id_d2}, y={y}, alpha={alpha:.4f}, eps={eps}")
    print(f"  {'Param':<14}  {'Analytical':>14}  {'Numerical':>14}  Match")
    print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*5}")
    for label, ana, num, ok in [("w0", a_w0, n_w0, ok_w0),
                                  (f"W[{id_d1}]", a_Wd1, n_Wd1, ok_Wd1),
                                  (f"V[{id_d1},0]", a_Vd1, n_Vd1, ok_Vd1)]:
        tick = "[OK]" if ok else "[FAIL]"
        print(f"  {label:<14}  {ana:>+14.8f}  {num:>+14.8f}  {tick}")
    assert ok_w0 and ok_Wd1 and ok_Vd1, "Gradient check FAILED -- fix backward()"
    print()


# Run gradient checks on a label-0 sample and a label-1 sample
_fm_chk = FactorizationMachine(n_features=n_features, k=8, seed=SEED)

print("Checking label-0 sample (delta = p, same as unweighted):")
check_gradients(_fm_chk, int(train_d1[0]), int(train_d2[0]), float(y_train[0]), alpha=ALPHA)

print("Checking label-1 sample (delta = -alpha*(1-p), amplified push):")
pos_idx = int(np.argmax(y_train == 1.0))
check_gradients(_fm_chk, int(train_d1[pos_idx]), int(train_d2[pos_idx]),
                float(y_train[pos_idx]), alpha=ALPHA)

print("[OK] All gradient assertions passed -- weighted derivation is mathematically sound.")
del _fm_chk
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 9. Training loop"))

cells.append(code("""
def fit(fm, id_d1_tr, id_d2_tr, y_tr,
        id_d1_val, id_d2_val, y_val,
        epochs: int = 30, batch_size: int = 256,
        lr: float = 0.01, alpha: float = 1.0,
        verbose: bool = True) -> dict:
    \"\"\"
    Mini-batch SGD with best-val-loss checkpointing.

    Checkpoints on validation loss (not F1 at a fixed threshold) because --
    especially early in training with class imbalance -- F1 at 0.5 may stay 0
    for many epochs even as loss decreases meaningfully.
    The optimal threshold is found post-training via the dynamic sweep (Section 11).

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'val_f1'
    \"\"\"
    N       = len(id_d1_tr)
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = np.inf
    best_params   = None
    rng_fit       = np.random.default_rng(SEED)

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        perm               = rng_fit.permutation(N)
        d1_s, d2_s, y_s   = id_d1_tr[perm], id_d2_tr[perm], y_tr[perm]

        # Mini-batch loop
        batch_losses = []
        for start in range(0, N, batch_size):
            sl   = slice(start, start + batch_size)
            loss = fm.train_batch(d1_s[sl], d2_s[sl], y_s[sl], lr, alpha)
            batch_losses.append(loss)
        train_loss = float(np.mean(batch_losses))

        # Validation metrics
        _, p_val = fm.forward(id_d1_val, id_d2_val)
        val_loss = bce_loss_batch(p_val, y_val, alpha)
        val_f1   = float(f1_score(y_val, (p_val >= 0.5).astype(int), zero_division=0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Checkpoint on lowest validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_params   = {
                "w0": fm.w0.copy(),
                "W":  fm.W.copy(),
                "V":  fm.V.copy(),
            }

        if verbose:
            marker = "  <- best" if is_best else ""
            print(f"  Epoch {epoch:3d}/{epochs}"
                  f"  train={train_loss:.4f}"
                  f"  val={val_loss:.4f}"
                  f"  val_f1@0.5={val_f1:.4f}"
                  f"{marker}")

    # Restore best checkpoint
    if best_params is not None:
        fm.w0[:] = best_params["w0"]
        fm.W[:]  = best_params["W"]
        fm.V[:]  = best_params["V"]
        if verbose:
            print(f"  [restored best checkpoint, val_loss={best_val_loss:.4f}]")

    return history


# Quick demo run (k=4, 3 epochs) to confirm the loop works
print("Demo training run (k=4, 3 epochs) -- confirming loop is correct:")
_fm_demo = FactorizationMachine(n_features=n_features, k=4, seed=SEED)
_h_demo  = fit(_fm_demo, train_d1, train_d2, y_train,
               val_d1, val_d2, y_val,
               epochs=3, batch_size=256, lr=0.01, alpha=ALPHA, verbose=True)
assert len(_h_demo["train_loss"]) == 3
print("[OK] Training loop runs correctly.")
del _fm_demo, _h_demo
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 10. Evaluate on the validation set"))

cells.append(code("""
def evaluate(fm, id_d1, id_d2, y_true, split_name: str = "", threshold: float = 0.5) -> dict:
    \"\"\"Compute Accuracy, Precision, Recall, F1, and Confusion Matrix.\"\"\"
    _, p   = fm.forward(id_d1, id_d2)
    y_pred = (p >= threshold).astype(int)
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "cm":        confusion_matrix(y_true, y_pred),
    }
    cm    = m["cm"]
    title = f"{split_name} Evaluation (threshold={threshold:.2f})"
    print(f"\\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                  Pred=0   Pred=1")
    print(f"    Actual=0 :  {cm[0,0]:7d}  {cm[0,1]:7d}   (TN / FP)")
    print(f"    Actual=1 :  {cm[1,0]:7d}  {cm[1,1]:7d}   (FN / TP)")
    print(f"{'─'*52}")
    return m


# TODO: Dynamic threshold sweep -- try several thresholds, pick best Val F1.
# (Full sweep happens in Section 11; here we illustrate the concept.)
THRESHOLDS = np.round(np.arange(0.20, 0.61, 0.05), 2)

print("Demo validation evaluation (k=16, 3-epoch warmup -- for illustration only)")
_fm_val = FactorizationMachine(n_features=n_features, k=16, seed=SEED)
fit(_fm_val, train_d1, train_d2, y_train, val_d1, val_d2, y_val,
    epochs=3, batch_size=256, lr=0.01, alpha=ALPHA, verbose=False)

_, _p_val = _fm_val.forward(val_d1, val_d2)
print("\\nThreshold sweep on VALIDATION set (demo):")
print(f"  {'Threshold':>10}  {'Val F1':>8}")
print(f"  {'-'*10}  {'-'*8}")
for t in THRESHOLDS:
    vf1 = f1_score(y_val, (_p_val >= t).astype(int), zero_division=0)
    print(f"  {t:>10.2f}  {vf1:>8.4f}")

evaluate(_fm_val, val_d1, val_d2, y_val, split_name="Val (demo)", threshold=0.5)
del _fm_val, _p_val
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EXPERIMENT: k
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 11. Experiment: embedding dimension $k$"))

cells.append(code("""
# TODO: Train one model for each k in [4, 8, 16, 32] (same lr, epochs, split).
# TODO: Build a table with columns: k, train loss, val loss, val accuracy, val F1.
# TODO: Plot val F1 against k.
# TODO: Choose your final k based on VALIDATION F1 only.

K_VALUES   = [4, 8, 16, 32]
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 0.01

experiment_results = []
trained_models     = {}
all_histories      = {}

for k in K_VALUES:
    print(f"\\n{'='*55}")
    print(f"  Training FM   k={k}   epochs={EPOCHS}   lr={LR}   alpha={ALPHA:.4f}")
    print(f"{'='*55}")

    np.random.seed(SEED)
    fm      = FactorizationMachine(n_features=n_features, k=k, seed=SEED)
    history = fit(fm, train_d1, train_d2, y_train,
                  val_d1, val_d2, y_val,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                  alpha=ALPHA, verbose=True)
    all_histories[k] = history

    # Dynamic threshold sweep on validation set
    _, p_val = fm.forward(val_d1, val_d2)
    best_vf1 = -1.0
    best_thr = 0.5
    for thr in THRESHOLDS:
        vf1 = float(f1_score(y_val, (p_val >= thr).astype(int), zero_division=0))
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_thr = float(thr)

    # Collect summary metrics
    train_loss_final = history["train_loss"][-1]
    val_loss_final   = history["val_loss"][-1]
    _, p_val2 = fm.forward(val_d1, val_d2)
    val_acc   = float(accuracy_score(y_val, (p_val2 >= best_thr).astype(int)))

    experiment_results.append({
        "k":          k,
        "train_loss": round(train_loss_final, 4),
        "val_loss":   round(val_loss_final,   4),
        "val_acc":    round(val_acc,           4),
        "val_f1":     round(best_vf1,          4),
        "threshold":  best_thr,
    })
    trained_models[k] = fm
    print(f"  -> k={k}: best_val_f1={best_vf1:.4f} at threshold={best_thr:.2f}")

# TODO: Build results table
results_df = pd.DataFrame(experiment_results)
print("\\n")
display(results_df.set_index("k"))
"""))

cells.append(code("""
# TODO: Plot val F1 against k AND train vs val loss curves for each k.

# --- Plot 1: Train vs Validation Loss for each k -------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.flatten()

for ax, k in zip(axes, K_VALUES):
    h         = all_histories[k]
    epoch_ax  = range(1, len(h["train_loss"]) + 1)
    ax.plot(epoch_ax, h["train_loss"], label="Train Loss", color="#4C9BE8", linewidth=1.8)
    ax.plot(epoch_ax, h["val_loss"],   label="Val Loss",   color="#F28C38",
            linewidth=1.8, linestyle="--")
    ax.set_title(f"k = {k}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Train vs Validation Loss by Embedding Dimension k",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# --- Plot 2: Validation F1 vs k ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
k_vals  = results_df["k"].tolist()
f1_vals = results_df["val_f1"].tolist()
colors  = ["#4C9BE8", "#6BD490", "#F28C38", "#E86A6A"]

bars = ax.bar(k_vals, f1_vals, color=colors, width=2.5,
              edgecolor="white", linewidth=1.2)
for bar, f1 in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{f1:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

best_k_plot = results_df.loc[results_df["val_f1"].idxmax(), "k"]
ax.set_title(f"Validation F1 vs Embedding Dimension k\\n(Best: k={best_k_plot})",
             fontweight="bold")
ax.set_xlabel("Embedding Dimension k")
ax.set_ylabel("Validation F1 (best threshold)")
ax.set_xticks(k_vals)
ax.set_ylim(0, max(f1_vals) + 0.08)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --- Select best k --------------------------------------------------------
best_row       = results_df.loc[results_df["val_f1"].idxmax()]
BEST_K         = int(best_row["k"])
BEST_THRESHOLD = float(best_row["threshold"])
best_fm        = trained_models[BEST_K]

print(f"\\n[OK] Best k = {BEST_K}  (Val F1 = {best_row['val_f1']:.4f}  @  threshold = {BEST_THRESHOLD:.2f})")
print(f"     This model will be used for the final test evaluation.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 12. Final test evaluation (run once)"))

cells.append(code("""
# TODO: Evaluate on the TEST set ONCE.
# IMPORTANT: Do not change anything after seeing these numbers.
#            That would turn your test set into a validation set.

print(f"Final model: k={BEST_K}   threshold={BEST_THRESHOLD:.2f}   alpha={ALPHA:.4f}")
print("Running test set evaluation -- THIS CELL MUST BE RUN ONLY ONCE.\\n")

test_metrics = evaluate(best_fm, test_d1, test_d2, y_test,
                        split_name="TEST", threshold=BEST_THRESHOLD)

TEST_ACCURACY  = test_metrics["accuracy"]
TEST_PRECISION = test_metrics["precision"]
TEST_RECALL    = test_metrics["recall"]
TEST_F1        = test_metrics["f1"]
BEST_VAL_F1    = float(best_row["val_f1"])
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — REASONING TASK
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 13. Reasoning task: explain one prediction"))

cells.append(code("""
# TODO: Pick ONE drug pair from the test set that the model predicts as Severe.
# TODO: Print w0, W[a], W[b], dot(V[a], V[b]), total score, final probability.

_, p_test = best_fm.forward(test_d1, test_d2)

# Prefer a true positive (predicted Severe AND truly Severe)
tp_mask = (p_test >= BEST_THRESHOLD) & (y_test == 1.0)
if tp_mask.any():
    tp_indices  = np.where(tp_mask)[0]
    chosen_idx  = int(tp_indices[np.argmax(p_test[tp_indices])])
    sample_type = "True Positive (predicted Severe, actually Severe)"
else:
    chosen_idx  = int(np.argmax(p_test))
    sample_type = "Highest predicted probability (no TP at this threshold)"

a      = int(test_d1[chosen_idx])
b      = int(test_d2[chosen_idx])
d1_name = id_to_d1[a]
d2_name = id_to_d2[b]

# Decompose FM score into its four additive terms
w0_val   = float(best_fm.w0[0])
W_a      = float(best_fm.W[a])
W_b      = float(best_fm.W[b])
dot_VaVb = float(np.dot(best_fm.V[a], best_fm.V[b]))
z_val    = w0_val + W_a + W_b + dot_VaVb
p_val    = 1.0 / (1.0 + np.exp(-z_val))

print(f"Selected: {sample_type}")
print(f"  Drug 1 : '{d1_name}'  (feature ID {a})")
print(f"  Drug 2 : '{d2_name}'  (feature ID {b})")
print(f"  True label: {int(y_test[chosen_idx])} ({'Severe' if y_test[chosen_idx]==1 else 'Not Severe'})")
print()
print(f"  FM Score Decomposition")
print(f"  {'-'*52}")
print(f"  w0  (global bias)               : {w0_val:+.8f}")
print(f"  W[{a}]  (Drug 1 linear weight)  : {W_a:+.8f}")
print(f"  W[{b}]  (Drug 2 linear weight)  : {W_b:+.8f}")
print(f"  <V[{a}], V[{b}]>  (pairwise)   : {dot_VaVb:+.8f}")
print(f"  {'-'*52}")
print(f"  z = w0 + W[a] + W[b] + <V[a],V[b]>  : {z_val:+.8f}")
print(f"  p = sigma(z)                          : {p_val:.8f}")
print(f"  Prediction (threshold={BEST_THRESHOLD:.2f})           : {int(p_val >= BEST_THRESHOLD)}")

# Read interaction description from the dataset
try:
    row_data = test_df.iloc[chosen_idx]
    print(f"\\n  From dataset:")
    print(f"    Severity    : {row_data.get('Severity', 'N/A')}")
    print(f"    Description : {row_data.get('Interaction Description', 'N/A')}")
except Exception:
    pass
"""))

cells.append(code("""
# TODO (optional): Find 5 drugs whose Drug-1 vectors are most similar (cosine similarity) to drug a.
from numpy.linalg import norm

d1_embeddings = best_fm.V[:n1]      # Drug-1 embeddings only (IDs 0..n1-1)
query_vec     = best_fm.V[a]

cosine_sims = d1_embeddings @ query_vec / (
    norm(d1_embeddings, axis=1) * norm(query_vec) + 1e-15
)
cosine_sims[a] = -2.0   # exclude self

top5_ids   = np.argsort(-cosine_sims)[:5]
top5_names = [id_to_d1[i] for i in top5_ids]
top5_sims  = cosine_sims[top5_ids]

print(f"Top-5 Drug-1 drugs most similar to '{d1_name}' (cosine sim of latent V):")
print(f"  k={BEST_K}, Drug-1 embedding space only (IDs 0..n1-1)")
print(f"  {'#':<4} {'Drug':<30} {'Cosine Sim':>12}")
print(f"  {'-'*4} {'-'*30} {'-'*12}")
for rank, (name, sim) in enumerate(zip(top5_names, top5_sims), 1):
    print(f"  {rank:<4} {name:<30} {sim:>12.4f}")

print("\\nDo these make clinical sense? Check their Interaction Descriptions in the dataset.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 14. Results (standard format, required)"))

cells.append(code("""
# TODO: Fill in your final numbers.
# This cell is used to compare all interns, so keep the exact keys.
fm_results = {
    "model":           "FM (from scratch)",
    "split_ratio":     "80/10/10",
    "embedding_dim_k": BEST_K,
    "learning_rate":   LR,
    "epochs":          EPOCHS,
    "threshold":       BEST_THRESHOLD,
    "val_f1":          round(BEST_VAL_F1,    4),
    "test_accuracy":   round(TEST_ACCURACY,  4),
    "test_precision":  round(TEST_PRECISION, 4),
    "test_recall":     round(TEST_RECALL,    4),
    "test_f1":         round(TEST_F1,        4),
}
print(fm_results)

# TODO: Save the results for the comparison in the DeepFM notebook.
pd.DataFrame([fm_results]).to_csv("fm_results.csv", index=False)
print("\\n[OK] Results saved to fm_results.csv")
print("     Download this file and upload it to Notebook 2 (DeepFM) for comparison.")
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

output_path = "Notebook_1_FM.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[OK]  Written: {output_path}")
print(f"      Cells:   {len(nb.cells)}")
print(f"      Sections: 14 (Setup through Results)")
print(f"      Upload to Google Colab and run top-to-bottom.")
