"""
02_fm_numpy.py — Phase 4 & 5: Architecture, Forward Pass & Backward Pass
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
Phase 5 : (a) Vocabulary built from train only (no leakage).
           (b) Analytical backward pass.
           (c) Central finite-difference gradient checker.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Load data splits
# ---------------------------------------------------------------------------

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")
test_df  = pd.read_csv("split_test.csv")

print(f"Loaded splits — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")


# ---------------------------------------------------------------------------
# 2.  Build vocabulary STRICTLY from train_df
#
#     Rationale: the vocabulary defines the embedding row indices.  Including
#     val/test drugs to avoid KeyErrors would constitute indirect data leakage
#     (those drugs' existence would influence the index mapping used during
#     training).  Since every drug in val/test is provably present in train
#     (same canonical-pair grouping), no KeyErrors will occur here.
# ---------------------------------------------------------------------------

train_drugs: set[str] = set(train_df["Drug 1"]) | set(train_df["Drug 2"])

# Deterministic sort → same index map every run
vocab: dict[str, int] = {drug: idx for idx, drug in enumerate(sorted(train_drugs))}
VOCAB_SIZE = len(vocab)   # expected: 420

print(f"Vocabulary size (unique drugs in train): {VOCAB_SIZE}")
assert VOCAB_SIZE == 420, f"Expected 420 unique drugs, got {VOCAB_SIZE}."


def encode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map Drug 1 / Drug 2 names to integer indices and return labels.

    Returns
    -------
    d1_idx : np.ndarray  shape (N,) int32
    d2_idx : np.ndarray  shape (N,) int32
    labels : np.ndarray  shape (N,) float64
    """
    d1_idx = df["Drug 1"].map(vocab).to_numpy(dtype=np.int32)
    d2_idx = df["Drug 2"].map(vocab).to_numpy(dtype=np.int32)
    labels = df["Label"].to_numpy(dtype=np.float64)
    return d1_idx, d2_idx, labels


train_d1, train_d2, train_y = encode(train_df)
val_d1,   val_d2,   val_y   = encode(val_df)
test_d1,  test_d2,  test_y  = encode(test_df)

print(f"Sample encoding — '{train_df['Drug 1'].iloc[0]}' → {train_d1[0]}, "
      f"'{train_df['Drug 2'].iloc[0]}' → {train_d2[0]}, label={train_y[0]}")


# ---------------------------------------------------------------------------
# 3.  Binary Cross-Entropy loss (scalar helper used by the gradient checker)
# ---------------------------------------------------------------------------

def bce_loss(p: float, y: float) -> float:
    """
    Binary cross-entropy for a *single* sample.

        L = −[ y·log(p) + (1−y)·log(1−p) ]

    A small epsilon guards against log(0).
    """
    eps = 1e-15
    return -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))


# ---------------------------------------------------------------------------
# 4.  Factorization Machine class
# ---------------------------------------------------------------------------

class FactorizationMachine:
    """
    Factorization Machine for binary drug-interaction prediction.

    Learnable parameters
    --------------------
    w0 : np.ndarray  shape (1,)              — global bias
    W  : np.ndarray  shape (vocab_size, 1)   — first-order (linear) weights
    V  : np.ndarray  shape (vocab_size, k)   — latent interaction embeddings

    Forward-pass (two active features only)
    ---------------------------------------
    Standard FM over n one-hot features reduces when only d1 and d2 are 1:

      First-order:    Σᵢ wᵢxᵢ = w_{d1} + w_{d2}                       (2)

      Interaction:    Σᵢ Σⱼ>ᵢ <vᵢ,vⱼ> xᵢxⱼ
                        = <v_{d1}, v_{d2}>                              (3)
                        = Σ_f  V[d1,f] · V[d2,f]

      Full logit:     z = w₀ + W[d1] + W[d2] + V[d1]·V[d2]            (4)
      Probability:    p = σ(z) = 1/(1 + e^{−z})                        (5)

    Backward-pass derivation
    ------------------------
    Loss (BCE):   L = −[ y·log(p) + (1−y)·log(1−p) ]

    Chain rule through sigmoid + BCE collapses neatly:

      dL/dp  = −y/p + (1−y)/(1−p)
      dp/dz  = p(1−p)                        [sigmoid derivative]

      δ ≡ dL/dz = dL/dp · dp/dz
               = [−y/p + (1−y)/(1−p)] · p(1−p)
               = −y(1−p) + (1−y)p
               = p − y                                                  (6)

    Parameter gradients via chain rule (dL/dθ = δ · dz/dθ):

      dL/dw₀       = δ · 1           = δ                               (7)
      dL/dW[d1]    = δ · 1           = δ                               (8)
      dL/dW[d2]    = δ · 1           = δ                               (9)
      dL/dV[d1, f] = δ · V[d2, f]   ∀ f ∈ {1…k}                      (10)
      dL/dV[d2, f] = δ · V[d1, f]   ∀ f ∈ {1…k}                      (11)

    Equations (10) & (11) follow because ∂z/∂V[d1,f] = V[d2,f]
    and symmetrically ∂z/∂V[d2,f] = V[d1,f].
    """

    def __init__(self, vocab_size: int, k: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        self.w0: np.ndarray = np.zeros(1, dtype=np.float64)
        self.W:  np.ndarray = rng.normal(0.0, 0.01, size=(vocab_size, 1)).astype(np.float64)
        self.V:  np.ndarray = rng.normal(0.0, 0.01, size=(vocab_size, k)).astype(np.float64)

        self.k = k
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    def forward(
        self,
        drug1_idx: "np.ndarray | int",
        drug2_idx: "np.ndarray | int",
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        FM forward pass — equations (4) and (5).

        Parameters
        ----------
        drug1_idx : int or np.ndarray shape (N,)
        drug2_idx : int or np.ndarray shape (N,)

        Returns
        -------
        z : np.ndarray shape (N,)   raw logit
        p : np.ndarray shape (N,)   sigmoid probability ∈ (0, 1)
        """
        # W shape: (vocab_size, 1) → squeeze trailing dim → (N,)
        linear_d1  = self.W[drug1_idx].squeeze(-1)
        linear_d2  = self.W[drug2_idx].squeeze(-1)

        v_d1 = self.V[drug1_idx]                          # (N, k)
        v_d2 = self.V[drug2_idx]                          # (N, k)

        interaction = np.sum(v_d1 * v_d2, axis=-1)        # (N,)  eq.(3)

        z = self.w0[0] + linear_d1 + linear_d2 + interaction  # eq.(4)
        p = 1.0 / (1.0 + np.exp(-z))                          # eq.(5)

        return z, p

    # ------------------------------------------------------------------
    def backward(
        self,
        drug1_idx: "np.ndarray | int",
        drug2_idx: "np.ndarray | int",
        p:         "np.ndarray | float",
        y:         "np.ndarray | float",
    ) -> dict[str, np.ndarray]:
        """
        Analytical backward pass — equations (6)–(11).

        Inputs are cast to 1-D arrays so the method works identically
        for single samples (gradient checker) and mini-batches (training).

        Parameters
        ----------
        drug1_idx : int or np.ndarray shape (N,)
        drug2_idx : int or np.ndarray shape (N,)
        p         : np.ndarray shape (N,)   — sigmoid output from forward()
        y         : np.ndarray shape (N,)   — binary ground-truth labels

        Returns   (gradients only — weight update happens in the training loop)
        -------
        dict with keys:
            'w0'   : float               — dL/dw₀  summed over batch       eq.(7)
            'W_d1' : np.ndarray (N,)     — dL/dW[d1] per sample            eq.(8)
            'W_d2' : np.ndarray (N,)     — dL/dW[d2] per sample            eq.(9)
            'V_d1' : np.ndarray (N, k)   — dL/dV[d1] per sample            eq.(10)
            'V_d2' : np.ndarray (N, k)   — dL/dV[d2] per sample            eq.(11)

        Note on accumulation
        --------------------
        W_d1[i] and V_d1[i] are the gradient contributions for the specific
        drug at drug1_idx[i].  When the same drug appears multiple times in a
        batch, the training loop must use np.add.at (or equivalent) to
        accumulate these gradients into the full W and V matrices.
        """
        # Guarantee 1-D arrays so broadcasting is unambiguous
        drug1_idx = np.atleast_1d(np.asarray(drug1_idx, dtype=np.int32))
        drug2_idx = np.atleast_1d(np.asarray(drug2_idx, dtype=np.int32))
        p = np.atleast_1d(np.asarray(p,         dtype=np.float64))
        y = np.atleast_1d(np.asarray(y,         dtype=np.float64))

        # ── Error signal ──────────────────────────────────────────────
        delta = p - y                               # shape (N,)   eq.(6)

        # ── Global bias ───────────────────────────────────────────────
        grad_w0 = float(np.sum(delta))             # scalar        eq.(7)

        # ── First-order weights (same gradient: δ for each drug) ─────
        grad_W_d1 = delta.copy()                   # shape (N,)    eq.(8)
        grad_W_d2 = delta.copy()                   # shape (N,)    eq.(9)

        # ── Latent embedding gradients (cross-interaction) ────────────
        # δ reshaped to (N, 1) to broadcast over k factors
        delta_col = delta[:, np.newaxis]           # shape (N, 1)

        grad_V_d1 = delta_col * self.V[drug2_idx]  # shape (N, k)  eq.(10)
        grad_V_d2 = delta_col * self.V[drug1_idx]  # shape (N, k)  eq.(11)

        return {
            "w0":   grad_w0,
            "W_d1": grad_W_d1,
            "W_d2": grad_W_d2,
            "V_d1": grad_V_d1,
            "V_d2": grad_V_d2,
        }


# ---------------------------------------------------------------------------
# 5.  Central finite-difference gradient checker
#
#     For any scalar parameter θ, the central difference approximation is:
#
#         ∂L/∂θ  ≈  [ L(θ + ε) − L(θ − ε) ] / (2ε)
#
#     This converges as O(ε²) whereas the forward difference is only O(ε).
#     We check three representative parameters: w0, W[d1, 0], V[d1, 0].
# ---------------------------------------------------------------------------

def check_gradients(
    fm:        FactorizationMachine,
    d1:        int,
    d2:        int,
    y:         float,
    eps:       float = 1e-5,
) -> None:
    """
    Verify analytical gradients against central finite differences.

    Checks w₀, W[d1, 0], and V[d1, 0] for a *single* training example.
    Asserts np.isclose (rtol=1e-5, atol=1e-8) on each pair.

    Parameters
    ----------
    fm  : FactorizationMachine — model (parameters are restored after each probe)
    d1  : int                  — Drug-1 index for the test sample
    d2  : int                  — Drug-2 index for the test sample
    y   : float                — ground-truth label (0.0 or 1.0)
    eps : float                — perturbation magnitude (default 1e-5)
    """
    d1_arr = np.array([d1])
    d2_arr = np.array([d2])
    y_arr  = np.array([y])

    # ── Analytical gradients ──────────────────────────────────────────
    _, p_arr = fm.forward(d1_arr, d2_arr)
    grads    = fm.backward(d1_arr, d2_arr, p_arr, y_arr)

    anal_w0      = grads["w0"]          # scalar
    anal_W_d1    = grads["W_d1"][0]     # scalar (gradient for W[d1, 0])
    anal_V_d1_f0 = grads["V_d1"][0, 0] # scalar (gradient for V[d1, 0])

    # ── Helper: compute scalar BCE loss at current parameter state ────
    def current_loss() -> float:
        _, p_ = fm.forward(d1_arr, d2_arr)
        return bce_loss(float(p_[0]), y)

    # ── Probe w₀ ─────────────────────────────────────────────────────
    fm.w0[0] += eps;  l_plus  = current_loss()
    fm.w0[0] -= 2*eps; l_minus = current_loss()
    fm.w0[0] += eps                            # restore
    num_w0 = (l_plus - l_minus) / (2.0 * eps)

    # ── Probe W[d1, 0] ────────────────────────────────────────────────
    fm.W[d1, 0] += eps;  l_plus  = current_loss()
    fm.W[d1, 0] -= 2*eps; l_minus = current_loss()
    fm.W[d1, 0] += eps                         # restore
    num_W_d1 = (l_plus - l_minus) / (2.0 * eps)

    # ── Probe V[d1, 0] ────────────────────────────────────────────────
    fm.V[d1, 0] += eps;  l_plus  = current_loss()
    fm.V[d1, 0] -= 2*eps; l_minus = current_loss()
    fm.V[d1, 0] += eps                         # restore
    num_V_d1_f0 = (l_plus - l_minus) / (2.0 * eps)

    # ── Report ────────────────────────────────────────────────────────
    header = f"\n{'─'*65}\nGradient check  d1={d1}  d2={d2}  y={y}  ε={eps}\n{'─'*65}"
    print(header)
    fmt = "  {:<14s}  analytical={:+.10f}   numerical={:+.10f}   match={}"
    print(fmt.format("w0",
          anal_w0,      num_w0,      np.isclose(anal_w0,      num_w0)))
    print(fmt.format(f"W[{d1}, 0]",
          anal_W_d1,    num_W_d1,    np.isclose(anal_W_d1,    num_W_d1)))
    print(fmt.format(f"V[{d1}, 0]",
          anal_V_d1_f0, num_V_d1_f0, np.isclose(anal_V_d1_f0, num_V_d1_f0)))

    # ── Assert ────────────────────────────────────────────────────────
    assert np.isclose(anal_w0,      num_w0),      \
        f"w0 gradient mismatch: analytical={anal_w0:.8f}, numerical={num_w0:.8f}"
    assert np.isclose(anal_W_d1,    num_W_d1),    \
        f"W[d1] gradient mismatch: analytical={anal_W_d1:.8f}, numerical={num_W_d1:.8f}"
    assert np.isclose(anal_V_d1_f0, num_V_d1_f0), \
        f"V[d1,0] gradient mismatch: analytical={anal_V_d1_f0:.8f}, numerical={num_V_d1_f0:.8f}"

    print(f"{'─'*65}")
    print("  ✅  All gradient assertions passed — derivation is flawless.")
    print(f"{'─'*65}\n")


# ---------------------------------------------------------------------------
# 6.  Entry point: forward → backward → gradient check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    K = 8
    fm = FactorizationMachine(vocab_size=VOCAB_SIZE, k=K, seed=42)

    # ── Parameter shapes ──────────────────────────────────────────────
    print("\n--- Parameter shapes ---")
    print(f"  w0 : {fm.w0.shape}        global bias")
    print(f"  W  : {fm.W.shape}  linear weights  (one row per drug)")
    print(f"  V  : {fm.V.shape}  latent embeddings  (one row per drug)")

    # ── Forward pass on first 5 training samples ──────────────────────
    batch_d1 = train_d1[:5]
    batch_d2 = train_d2[:5]
    batch_y  = train_y[:5]

    z, p = fm.forward(batch_d1, batch_d2)

    print("\n--- Forward pass (first 5 training samples) ---")
    print(f"  Drug-1 indices : {batch_d1}")
    print(f"  Drug-2 indices : {batch_d2}")
    print(f"  True labels    : {batch_y}")
    print(f"  Logits  z      : {np.round(z, 6)}")
    print(f"  Probas  p      : {np.round(p, 6)}")

    assert z.shape == (5,),           f"Logit shape mismatch: {z.shape}"
    assert p.shape == (5,),           f"Prob  shape mismatch: {p.shape}"
    assert np.all((p > 0) & (p < 1)), "Sigmoid output must be in (0, 1)"
    print("  ✅  Forward pass assertions passed.")

    # ── Backward pass on same batch ───────────────────────────────────
    grads = fm.backward(batch_d1, batch_d2, p, batch_y)

    print("\n--- Backward pass (first 5 training samples) ---")
    print(f"  delta   (p−y)  : {np.round(p - batch_y, 6)}")
    print(f"  grad_w0        : {grads['w0']:+.8f}   (sum of δ over batch)")
    print(f"  grad_W_d1[0]   : {grads['W_d1'][0]:+.8f}   (= δ₀)")
    print(f"  grad_W_d2[0]   : {grads['W_d2'][0]:+.8f}   (= δ₀)")
    print(f"  grad_V_d1[0,:] : {np.round(grads['V_d1'][0], 6)}  (= δ₀·V[d2_0])")
    print(f"  grad_V_d2[0,:] : {np.round(grads['V_d2'][0], 6)}  (= δ₀·V[d1_0])")

    assert grads["W_d1"].shape == (5,),   "grad_W_d1 shape error"
    assert grads["V_d1"].shape == (5, K), "grad_V_d1 shape error"
    print("  ✅  Backward pass shape assertions passed.")

    # ── Gradient checker (two samples: one from each label class) ─────
    # Label-0 example
    check_gradients(fm, d1=int(train_d1[0]), d2=int(train_d2[0]), y=float(train_y[0]))

    # Label-1 example — find first positive in training set
    pos_mask = train_y == 1.0
    first_pos = int(np.argmax(pos_mask))
    check_gradients(fm, d1=int(train_d1[first_pos]),
                        d2=int(train_d2[first_pos]),
                        y=float(train_y[first_pos]))

    print("🎉  Phase 5 complete.  Ready for Phase 6: training loop.")
