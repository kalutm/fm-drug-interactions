"""
02_fm_numpy.py — Phase 4, 5 & 6 (Weighted BCE + Dynamic Thresholding)
=========================================================================
Factorization Machine (FM) built strictly from scratch with NumPy.
Only Drug 1 and Drug 2 are used as features (one-hot, exactly two active).

FM equation (general, n features):
    ŷ = w₀ + Σᵢ wᵢxᵢ + Σᵢ Σⱼ>ᵢ <vᵢ, vⱼ> xᵢxⱼ

With exactly two active features (x_{d1}=1, x_{d2}=1, all others 0):
    z = w₀ + W[d1] + W[d2] + <V[d1], V[d2]>
    p = σ(z)

Change log
----------
Phase 4 : Architecture + forward pass.
Phase 5 : Train-only vocabulary; analytical backward; gradient checker (✅).
Phase 6 : (a) train_batch, fit, evaluate, decompose_score, run_experiments.
           (b) FIX — Weighted BCE (α = N₀/N₁) to counter majority-class collapse.
           (c) FIX — Dynamic threshold sweep [0.20 … 0.60] on validation set.
           (d) Gradient checker updated and re-verified for weighted BCE.

Weighted BCE derivation
-----------------------
Loss (single sample):
    L = −[ α·y·log p + (1−y)·log(1−p) ]          α = N₀/N₁ ≈ 5.16

∂L/∂p = −αy/p + (1−y)/(1−p)
∂p/∂z = p(1−p)                               [sigmoid derivative]

New error signal:
    δ = ∂L/∂z = ∂L/∂p · ∂p/∂z
              = [−αy/p + (1−y)/(1−p)] · p(1−p)
              = −αy(1−p) + (1−y)p               (eq.6w)

Sanity check:
    y = 0  →  δ = p                   (identical to Phase 5: unweighted)
    y = 1  →  δ = −α(1−p)             (α times stronger push for positives)

All chain-rule expressions for w₀, W, V remain identical — only δ changes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

SEED = 42

# ============================================================================
# 1.  Data loading
# ============================================================================

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")
test_df  = pd.read_csv("split_test.csv")

print(f"Loaded — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ============================================================================
# 2.  Vocabulary — built from train ONLY (no leakage)
# ============================================================================

train_drugs: set[str] = set(train_df["Drug 1"]) | set(train_df["Drug 2"])
vocab:        dict[str, int] = {drug: idx for idx, drug in enumerate(sorted(train_drugs))}
idx_to_drug:  dict[int, str] = {idx: drug for drug, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

assert VOCAB_SIZE == 420, f"Expected 420, got {VOCAB_SIZE}."
print(f"Vocabulary: {VOCAB_SIZE} unique drugs (from train only)")

# ============================================================================
# 3.  Encode splits
# ============================================================================

def encode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Drug 1 / Drug 2 names to integer indices; return (d1, d2, labels)."""
    d1 = df["Drug 1"].map(vocab).to_numpy(dtype=np.int32)
    d2 = df["Drug 2"].map(vocab).to_numpy(dtype=np.int32)
    y  = df["Label"].to_numpy(dtype=np.float64)
    return d1, d2, y

train_d1, train_d2, train_y = encode(train_df)
val_d1,   val_d2,   val_y   = encode(val_df)
test_d1,  test_d2,  test_y  = encode(test_df)

# ============================================================================
# 4.  Class-imbalance weight  α = N₀ / N₁
#
#     Without weighting, the model minimises BCE by predicting the majority
#     class (Non-severe, ~84%) for every sample.  Scaling the positive-class
#     term by α forces the gradient on Severe examples to be α× larger,
#     making it impossible to ignore them while still minimising total loss.
# ============================================================================

N1    = int(np.sum(train_y == 1))
N0    = int(np.sum(train_y == 0))
ALPHA = N0 / N1

print(f"Class counts — Negative: {N0:,}  Positive: {N1:,}")
print(f"Positive weight  α = N₀/N₁ = {N0}/{N1} = {ALPHA:.6f}")

# ============================================================================
# 5.  Loss helpers  (weighted BCE, α=1 recovers standard BCE)
# ============================================================================

def bce_loss(p: float, y: float, alpha: float = 1.0) -> float:
    """
    Weighted binary cross-entropy for a *single* sample.
        L = −[ α·y·log(p) + (1−y)·log(1−p) ]
    Used by the gradient checker for finite-difference probes.
    """
    eps = 1e-15
    return -(alpha * y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))


def bce_batch(p: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """Mean weighted BCE over a batch (used for training and validation logs)."""
    eps = 1e-15
    return float(-np.mean(alpha * y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))


# ============================================================================
# 6.  Factorization Machine  (NumPy, strictly from scratch)
# ============================================================================

class FactorizationMachine:
    """
    Factorization Machine for binary drug-interaction severity prediction.

    Learnable parameters
    --------------------
    w0 : np.ndarray  shape (1,)              global bias
    W  : np.ndarray  shape (vocab_size, 1)   first-order drug weights
    V  : np.ndarray  shape (vocab_size, k)   latent interaction embeddings

    Forward pass  (two active features d1, d2 — eqs. 4–5)
    ------------------------------------------------------
        z = w₀ + W[d1] + W[d2] + Σ_f V[d1,f]·V[d2,f]
        p = σ(z) = 1 / (1 + e^{−z})

    Backward pass  (weighted BCE; eq. 6w)
    --------------------------------------
    New error signal:
        δ = −α·y·(1−p) + (1−y)·p

    Parameter gradients (chain rule — expressions unchanged from Phase 5,
    only δ is substituted):
        ∂L/∂w₀      = δ
        ∂L/∂W[d1]   = δ
        ∂L/∂W[d2]   = δ
        ∂L/∂V[d1,f] = δ · V[d2,f]
        ∂L/∂V[d2,f] = δ · V[d1,f]

    Why np.add.at in train_batch?
    ─────────────────────────────
    NumPy's fancy-index += is buffered: duplicate indices lose all but the
    last write.  np.add.at is unbuffered: every gradient contribution for
    the same drug is accumulated correctly — essential for correctness.
    """

    def __init__(self, vocab_size: int, k: int, seed: int = SEED) -> None:
        rng       = np.random.default_rng(seed)
        self.w0   = np.zeros(1, dtype=np.float64)
        self.W    = rng.normal(0.0, 0.01, size=(vocab_size, 1)).astype(np.float64)
        self.V    = rng.normal(0.0, 0.01, size=(vocab_size, k)).astype(np.float64)
        self.k    = k
        self.vocab_size = vocab_size

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        d1: "np.ndarray | int",
        d2: "np.ndarray | int",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Equations (4) and (5). Returns (z, p) each shape (N,)."""
        lin1 = self.W[d1].squeeze(-1)
        lin2 = self.W[d2].squeeze(-1)
        intr = np.sum(self.V[d1] * self.V[d2], axis=-1)
        z    = self.w0[0] + lin1 + lin2 + intr
        p    = 1.0 / (1.0 + np.exp(-z))
        return z, p

    # ── Backward (weighted BCE) ───────────────────────────────────────────────

    def backward(
        self,
        d1:    "np.ndarray | int",
        d2:    "np.ndarray | int",
        p:     "np.ndarray | float",
        y:     "np.ndarray | float",
        alpha: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Weighted-BCE analytical gradients — equation (6w).

        δ = −α·y·(1−p) + (1−y)·p

        Returns per-sample gradients (sums returned for w0 for easy /N later).
        """
        d1    = np.atleast_1d(np.asarray(d1, dtype=np.int32))
        d2    = np.atleast_1d(np.asarray(d2, dtype=np.int32))
        p     = np.atleast_1d(np.asarray(p,  dtype=np.float64))
        y     = np.atleast_1d(np.asarray(y,  dtype=np.float64))

        # Weighted error signal (eq. 6w)
        delta = -alpha * y * (1.0 - p) + (1.0 - y) * p   # shape (N,)

        return {
            "w0":   float(np.sum(delta)),                       # scalar
            "W_d1": delta.copy(),                               # (N,)
            "W_d2": delta.copy(),                               # (N,)
            "V_d1": delta[:, np.newaxis] * self.V[d2],         # (N, k)
            "V_d2": delta[:, np.newaxis] * self.V[d1],         # (N, k)
        }

    # ── Single mini-batch SGD step ────────────────────────────────────────────

    def train_batch(
        self,
        d1:    np.ndarray,
        d2:    np.ndarray,
        y:     np.ndarray,
        lr:    float,
        alpha: float = 1.0,
    ) -> float:
        """
        One SGD update for a mini-batch.

        Gradient normalisation: backward() returns ∑δ (summed over batch).
        Dividing each update by N converts to mean-gradient, matching
        the mean-over-batch convention of bce_batch().

        Returns mean weighted-BCE loss (float, for logging only).
        """
        _, p  = self.forward(d1, d2)
        loss  = bce_batch(p, y, alpha)
        grads = self.backward(d1, d2, p, y, alpha)
        N     = len(d1)

        # w0 — simple scalar update
        self.w0[0] -= lr * grads["w0"] / N

        # W — unbuffered accumulation along W[:,0] (a 1-D view)
        np.add.at(self.W[:, 0], d1, -lr * grads["W_d1"] / N)
        np.add.at(self.W[:, 0], d2, -lr * grads["W_d2"] / N)

        # V — unbuffered row-wise accumulation
        np.add.at(self.V, d1, -lr * grads["V_d1"] / N)
        np.add.at(self.V, d2, -lr * grads["V_d2"] / N)

        return loss

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(
        self,
        d1_tr:  np.ndarray,
        d2_tr:  np.ndarray,
        y_tr:   np.ndarray,
        d1_val: np.ndarray,
        d2_val: np.ndarray,
        y_val:  np.ndarray,
        epochs:     int   = 30,
        batch_size: int   = 256,
        lr:         float = 0.01,
        alpha:      float = 1.0,
        verbose:    bool  = True,
    ) -> dict:
        """
        Mini-batch SGD with validation-loss checkpointing.

        Checkpointing uses validation *loss* (not F1 at a fixed threshold)
        because — especially early in training with class imbalance — F1 at
        0.5 may remain zero for many epochs even as loss meaningfully
        decreases.  Validation F1 at threshold=0.5 is still logged each
        epoch for reference; the threshold itself is tuned post-training.

        Returns
        -------
        history : dict  {'train_loss', 'val_loss', 'val_f1'}
                  one float per epoch each.
        """
        N = len(d1_tr)
        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_loss = np.inf
        best_params:  dict | None = None
        rng = np.random.default_rng(SEED)

        for epoch in range(1, epochs + 1):

            # ── Shuffle ──────────────────────────────────────────────
            perm          = rng.permutation(N)
            d1_s, d2_s, y_s = d1_tr[perm], d2_tr[perm], y_tr[perm]

            # ── Mini-batch loop ───────────────────────────────────────
            batch_losses: list[float] = []
            for start in range(0, N, batch_size):
                sl = slice(start, start + batch_size)
                batch_losses.append(
                    self.train_batch(d1_s[sl], d2_s[sl], y_s[sl], lr, alpha)
                )
            train_loss = float(np.mean(batch_losses))

            # ── Validation (weighted loss + F1 at 0.5 for logging) ───
            _, p_val   = self.forward(d1_val, d2_val)
            val_loss   = bce_batch(p_val, y_val, alpha)
            vf1_at_05  = float(f1_score(y_val, (p_val >= 0.5).astype(int),
                                        zero_division=0))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(vf1_at_05)

            # ── Checkpoint on lowest val loss ─────────────────────────
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_params   = {
                    "w0": self.w0.copy(),
                    "W":  self.W.copy(),
                    "V":  self.V.copy(),
                }

            if verbose:
                marker = "  ◀ best" if is_best else ""
                print(
                    f"  Epoch {epoch:3d}/{epochs}"
                    f"  train_loss={train_loss:.4f}"
                    f"  val_loss={val_loss:.4f}"
                    f"  val_f1@0.5={vf1_at_05:.4f}"
                    f"{marker}"
                )

        # ── Restore best checkpoint ───────────────────────────────────
        if best_params is not None:
            self.w0 = best_params["w0"]
            self.W  = best_params["W"]
            self.V  = best_params["V"]
            if verbose:
                print(f"  ↩  Restored checkpoint  (best val_loss = {best_val_loss:.4f})")

        return history


# ============================================================================
# 7.  Evaluation
# ============================================================================

def evaluate(
    fm:         FactorizationMachine,
    d1:         np.ndarray,
    d2:         np.ndarray,
    y_true:     np.ndarray,
    split_name: str   = "",
    threshold:  float = 0.5,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1, and Confusion Matrix.
    Returns a dict of all five for programmatic access.
    """
    _, p   = fm.forward(d1, d2)
    y_pred = (p >= threshold).astype(int)

    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "cm":        confusion_matrix(y_true, y_pred),
    }
    cm    = m["cm"]
    title = f"  {split_name} Evaluation  (threshold={threshold:.2f})" if split_name \
            else f"  Evaluation  (threshold={threshold:.2f})"

    print(f"\n{'─'*56}")
    print(title)
    print(f"{'─'*56}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}   (predicted Severe → truly Severe)")
    print(f"  Recall    : {m['recall']:.4f}   (actual Severe → correctly caught)")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                   Pred 0     Pred 1")
    print(f"    Actual 0  :  {cm[0, 0]:8d}   {cm[0, 1]:8d}   (TN / FP)")
    print(f"    Actual 1  :  {cm[1, 0]:8d}   {cm[1, 1]:8d}   (FN / TP)")
    print(f"{'─'*56}")
    return m


# ============================================================================
# 8.  Score decomposition
# ============================================================================

def decompose_score(
    fm:      FactorizationMachine,
    d1_idx:  int,
    d2_idx:  int,
    y_true:  float,
    d1_name: str = "",
    d2_name: str = "",
) -> None:
    """
    Print every additive FM term for one drug pair:
        z = w₀ + W[d1] + W[d2] + <V[d1], V[d2]>
        p = σ(z)
    """
    w0_t   = float(fm.w0[0])
    W_d1_t = float(fm.W[d1_idx, 0])
    W_d2_t = float(fm.W[d2_idx, 0])
    int_t  = float(np.dot(fm.V[d1_idx], fm.V[d2_idx]))
    z      = w0_t + W_d1_t + W_d2_t + int_t
    p      = 1.0 / (1.0 + np.exp(-z))

    d1_lbl = f"{d1_name}  (idx {d1_idx})" if d1_name else f"idx {d1_idx}"
    d2_lbl = f"{d2_name}  (idx {d2_idx})" if d2_name else f"idx {d2_idx}"

    print(f"\n{'═'*64}")
    print(f"  Score Decomposition")
    print(f"  Formula : z = w₀ + W[d1] + W[d2] + <V[d1], V[d2]>")
    print(f"{'═'*64}")
    print(f"  Drug 1     : {d1_lbl}")
    print(f"  Drug 2     : {d2_lbl}")
    print(f"  True Label : {int(y_true)}  ({'Severe' if y_true == 1 else 'Non-severe'})")
    print(f"{'─'*64}")
    print(f"  w₀            (global bias)             : {w0_t:+.8f}")
    print(f"  W[d1]         (Drug 1 linear weight)    : {W_d1_t:+.8f}")
    print(f"  W[d2]         (Drug 2 linear weight)    : {W_d2_t:+.8f}")
    print(f"  <V[d1],V[d2]> (pairwise interaction)    : {int_t:+.8f}")
    print(f"{'─'*64}")
    print(f"  z  = Σ above                            : {z:+.8f}")
    print(f"  p  = σ(z) = 1 / (1 + exp(−z))          : {p:.8f}")
    print(f"  Prediction                              : {int(p >= 0.5)}"
          f"  ({'Severe' if p >= 0.5 else 'Non-severe'})")
    print(f"{'═'*64}\n")


# ============================================================================
# 9.  Gradient checker  (updated for weighted BCE)
# ============================================================================

def check_gradients(
    fm:    FactorizationMachine,
    d1:    int,
    d2:    int,
    y:     float,
    alpha: float = 1.0,
    eps:   float = 1e-5,
) -> None:
    """
    Central finite-difference verification — weighted BCE formulation.

    Checks w₀, W[d1, 0], V[d1, 0] against analytical gradients.
    Uses bce_loss(..., alpha) for the numerical probes so the finite
    differences are computed on the same loss function as backward().

    For parameter θ at flat index i:
        numerical ≈ [ L(θ_i + ε) − L(θ_i − ε) ] / (2ε)
    """
    d1_a = np.array([d1])
    d2_a = np.array([d2])
    y_a  = np.array([y])

    _, p_a = fm.forward(d1_a, d2_a)
    grads  = fm.backward(d1_a, d2_a, p_a, y_a, alpha)

    def loss_now() -> float:
        _, p_ = fm.forward(d1_a, d2_a)
        return bce_loss(float(p_[0]), y, alpha)

    def num_grad(param_arr: np.ndarray, flat_i: int) -> float:
        param_arr.flat[flat_i] += eps;   lp = loss_now()
        param_arr.flat[flat_i] -= 2*eps; lm = loss_now()
        param_arr.flat[flat_i] += eps                       # restore
        return (lp - lm) / (2.0 * eps)

    a_w0  = grads["w0"]
    a_Wd1 = float(grads["W_d1"][0])
    a_Vd1 = float(grads["V_d1"][0, 0])

    # Flat indices: W has shape (vocab_size,1), V has shape (vocab_size,k)
    n_w0  = num_grad(fm.w0, 0)
    n_Wd1 = num_grad(fm.W,  d1 * fm.W.shape[1])      # d1 * 1 = d1
    n_Vd1 = num_grad(fm.V,  d1 * fm.V.shape[1])      # d1 * k  (column 0)

    print(f"\n{'─'*68}")
    print(f"  Gradient Check  d1={d1}  d2={d2}  y={y}  α={alpha:.4f}  ε={eps}")
    print(f"{'─'*68}")
    fmt = "  {:<14s}  analytical={:+.10f}   numerical={:+.10f}   ✓={}"
    print(fmt.format("w0",         a_w0,  n_w0,  np.isclose(a_w0,  n_w0)))
    print(fmt.format(f"W[{d1},0]", a_Wd1, n_Wd1, np.isclose(a_Wd1, n_Wd1)))
    print(fmt.format(f"V[{d1},0]", a_Vd1, n_Vd1, np.isclose(a_Vd1, n_Vd1)))

    assert np.isclose(a_w0,  n_w0),  f"w0  mismatch  ({a_w0:.8f} vs {n_w0:.8f})"
    assert np.isclose(a_Wd1, n_Wd1), f"W[d1] mismatch ({a_Wd1:.8f} vs {n_Wd1:.8f})"
    assert np.isclose(a_Vd1, n_Vd1), f"V[d1,0] mismatch ({a_Vd1:.8f} vs {n_Vd1:.8f})"
    print("  ✅  All gradient assertions passed — weighted derivation is sound.")
    print(f"{'─'*68}\n")


# ============================================================================
# 10. Hyperparameter search + dynamic thresholding + final test evaluation
# ============================================================================

THRESHOLDS = np.round(np.arange(0.20, 0.61, 0.05), 2)


def run_experiments(
    k_values:   list  = [4, 8, 16, 32],
    epochs:     int   = 30,
    batch_size: int   = 256,
    lr:         float = 0.01,
    alpha:      float = ALPHA,
) -> tuple:
    """
    For each k  ∈ k_values:
        1. Train a fresh FM with weighted BCE (parameter α).
        2. Sweep thresholds on the validation set to find the best Val F1.

    Select the (k, threshold) pair with the highest Val F1.
    Evaluate that model on the test set exactly ONCE.

    Returns  (best_fm, best_k, best_threshold)
    """
    results: list[dict] = []
    models:  dict       = {}

    for k in k_values:
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  k={k}   epochs={epochs}   batch={batch_size}   lr={lr}   α={alpha:.4f}")
        print(f"{sep}")

        fm = FactorizationMachine(vocab_size=VOCAB_SIZE, k=k, seed=SEED)
        fm.fit(
            train_d1, train_d2, train_y,
            val_d1,   val_d2,   val_y,
            epochs=epochs, batch_size=batch_size, lr=lr, alpha=alpha,
        )

        # ── Threshold sweep on validation set ────────────────────────
        _, p_val   = fm.forward(val_d1, val_d2)
        best_vf1   = -np.inf
        best_thr   = 0.5
        thr_rows: list[tuple] = []

        for thr in THRESHOLDS:
            y_pred = (p_val >= thr).astype(int)
            vf1    = float(f1_score(val_y, y_pred, zero_division=0))
            thr_rows.append((thr, vf1))
            if vf1 > best_vf1:
                best_vf1 = vf1
                best_thr = float(thr)

        # Print sweep table for this k
        print(f"\n  Threshold sweep (k={k}, α={alpha:.4f}):")
        print(f"  {'Threshold':>10}  {'Val F1':>8}")
        print(f"  {'─'*10}  {'─'*8}")
        for thr, vf1 in thr_rows:
            star = "  ← best" if abs(thr - best_thr) < 1e-9 else ""
            print(f"  {thr:>10.2f}  {vf1:>8.4f}{star}")

        results.append({"k": k, "val_f1": best_vf1, "threshold": best_thr})
        models[k] = fm

        print(f"\n  → k={k}: best_val_f1={best_vf1:.4f}  best_threshold={best_thr:.2f}")

    # ── Experiment summary table ──────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  Experiment Summary")
    print(f"{'─'*50}")
    print(f"  {'k':>4}  |  Best Val F1  |  Threshold")
    print(f"  {'─'*4}  +  {'─'*13}  +  {'─'*10}")
    for r in results:
        print(f"  {r['k']:>4d}  |  {r['val_f1']:>10.4f}   |  {r['threshold']:>6.2f}")
    print(f"{'═'*50}")

    best_r   = max(results, key=lambda r: r["val_f1"])
    best_k   = best_r["k"]
    best_thr = best_r["threshold"]
    best_fm  = models[best_k]

    print(f"\n  ★  Best config: k={best_k}  threshold={best_thr:.2f}"
          f"  (Val F1={best_r['val_f1']:.4f})")

    # ── Final test evaluation (exactly ONE run) ───────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL TEST EVALUATION")
    print(f"  Model: k={best_k}   threshold={best_thr:.2f}   α={alpha:.4f}")
    print(f"  (Test set touched for the first and only time below)")
    print(f"{'═'*60}")
    evaluate(best_fm, test_d1, test_d2, test_y,
             split_name="Test", threshold=best_thr)

    # ── Score decomposition ───────────────────────────────────────────
    # Prefer a true positive (predicted Severe AND truly Severe).
    # Fall back to the sample with the highest predicted probability.
    _, p_test = best_fm.forward(test_d1, test_d2)
    tp_mask   = (p_test >= best_thr) & (test_y == 1.0)

    if tp_mask.any():
        # Among true positives, pick the one with the highest confidence
        tp_indices  = np.where(tp_mask)[0]
        top_idx     = int(tp_indices[np.argmax(p_test[tp_indices])])
        sample_type = "true positive (predicted Severe AND truly Severe)"
    else:
        top_idx     = int(np.argmax(p_test))
        sample_type = "highest predicted probability (no TP found at this threshold)"

    print(f"\n  Decomposing: {sample_type}")
    decompose_score(
        best_fm,
        d1_idx  = int(test_d1[top_idx]),
        d2_idx  = int(test_d2[top_idx]),
        y_true  = float(test_y[top_idx]),
        d1_name = idx_to_drug[int(test_d1[top_idx])],
        d2_name = idx_to_drug[int(test_d2[top_idx])],
    )

    return best_fm, best_k, best_thr


# ============================================================================
# 11. Entry point
# ============================================================================

if __name__ == "__main__":

    # ── Step 1: Verify weighted-BCE gradients BEFORE training ────────────────
    print(f"\n{'═'*68}")
    print(f"  Phase 6 Gradient Verification — Weighted BCE  (α = {ALPHA:.6f})")
    print(f"{'═'*68}")

    _fm_chk = FactorizationMachine(vocab_size=VOCAB_SIZE, k=8, seed=SEED)

    # Label-0 sample: δ = p (α drops out; same as Phase 5)
    check_gradients(_fm_chk,
                    d1=int(train_d1[0]), d2=int(train_d2[0]),
                    y=float(train_y[0]), alpha=ALPHA)

    # Label-1 sample: δ = −α(1−p)  (the amplified positive case)
    pos_mask  = train_y == 1.0
    first_pos = int(np.argmax(pos_mask))
    check_gradients(_fm_chk,
                    d1=int(train_d1[first_pos]), d2=int(train_d2[first_pos]),
                    y=float(train_y[first_pos]), alpha=ALPHA)

    del _fm_chk
    print("  Math verified. Proceeding to training.\n")

    # ── Step 2: Hyperparameter search + test evaluation ──────────────────────
    best_fm, best_k, best_thr = run_experiments(
        k_values   = [4, 8, 16, 32],
        epochs     = 30,
        batch_size = 256,
        lr         = 0.01,
        alpha      = ALPHA,
    )

    print(f"\n🎉  Phase 6 complete.")
    print(f"    Best model: k={best_k}  threshold={best_thr:.2f}  α={ALPHA:.4f}")
    print(f"    Ready for Phase 7: DeepFM in PyTorch.\n")
