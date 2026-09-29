"""
03_deepfm_pytorch.py — Phase 7: DeepFM Architecture & Forward Pass
===================================================================
DeepFM combines a Factorization Machine branch (interpretable, captures
linear effects and explicit pairwise interactions) with a Deep Neural
Network branch (captures implicit non-linear, high-order interactions).

Reference
─────────
Guo et al., "DeepFM: A Factorization-Machine based Neural Network
for CTR Prediction", IJCAI 2017.

Architecture overview
─────────────────────
    Input:   (d1, d2) — LongTensor drug indices

    Shared embedding tables  (trained jointly by both branches):
        W  : nn.Embedding(420, 1)    first-order drug weights
        V  : nn.Embedding(420, k)    latent interaction embeddings

    ┌─────────────────────────────────────────────────┐
    │               FM Branch                         │
    │                                                 │
    │  z_FM = w₀ + W[d1] + W[d2] + <V[d1], V[d2]>   │
    │                                                 │
    │  (identical to the verified NumPy FM equation)  │
    └───────────────────┬─────────────────────────────┘
                        │
    ┌───────────────────┴─────────────────────────────┐
    │               Deep Branch                       │
    │                                                 │
    │  h₀     = [V[d1] ‖ V[d2]]   shape: (N, 2k)    │
    │  h₁     = Dropout(ReLU(Linear(2k → H)))        │
    │  z_Deep = Linear(H → 1)      shape: (N,)       │
    └───────────────────┬─────────────────────────────┘
                        │
                   z = z_FM + z_Deep       (raw logit)

    No sigmoid here — nn.BCEWithLogitsLoss is used during training
    for numerical stability (log-sum-exp trick applied internally).

Change log
──────────
Phase 7 : DeepFM architecture + forward-pass smoke test.
          Training loop deferred to Phase 8.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# Reproducibility
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================================
# 1.  Data loading
# ============================================================================

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")
test_df  = pd.read_csv("split_test.csv")

print(f"Loaded — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ============================================================================
# 2.  Vocabulary  (train only — identical rule as 02_fm_numpy.py)
#
#     Using sorted(train_drugs) guarantees the same deterministic index
#     mapping as the NumPy FM, so embeddings are conceptually comparable.
# ============================================================================

train_drugs: set[str] = set(train_df["Drug 1"]) | set(train_df["Drug 2"])
vocab:        dict[str, int] = {drug: idx for idx, drug in enumerate(sorted(train_drugs))}
idx_to_drug:  dict[int, str] = {idx: drug for drug, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

assert VOCAB_SIZE == 420, f"Expected 420, got {VOCAB_SIZE}."
print(f"Vocabulary: {VOCAB_SIZE} unique drugs (from train only)")

# ============================================================================
# 3.  Encode helper  — returns tensors on the target device
# ============================================================================

def encode(
    df:     pd.DataFrame,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map Drug 1 / Drug 2 names → integer indices; labels → float32.

    Returns
    -------
    d1 : LongTensor   shape (N,)
    d2 : LongTensor   shape (N,)
    y  : FloatTensor  shape (N,)
    """
    d1 = torch.tensor(df["Drug 1"].map(vocab).to_numpy(), dtype=torch.long,    device=device)
    d2 = torch.tensor(df["Drug 2"].map(vocab).to_numpy(), dtype=torch.long,    device=device)
    y  = torch.tensor(df["Label"].to_numpy(),              dtype=torch.float32, device=device)
    return d1, d2, y


train_d1, train_d2, train_y = encode(train_df)
val_d1,   val_d2,   val_y   = encode(val_df)
test_d1,  test_d2,  test_y  = encode(test_df)

print(f"Tensors — d1 dtype: {train_d1.dtype}  y dtype: {train_y.dtype}")

# ============================================================================
# 4.  DeepFM  (nn.Module)
# ============================================================================

class DeepFM(nn.Module):
    """
    DeepFM for binary drug-interaction severity prediction.

    Parameters
    ──────────
    vocab_size : int
        Number of unique drugs (420).
    k : int
        Latent embedding dimension (e.g. 32).
    hidden_dim : int | None
        Width of the single MLP hidden layer.
        Defaults to 4 * k when None.
    dropout : float
        Dropout probability applied after the hidden activation.
        Default 0.3.

    Shared embedding tables
    ───────────────────────
    w0 : nn.Parameter  shape (1,)
        Global bias — initialised to 0.
    W  : nn.Embedding  shape (vocab_size, 1)
        First-order drug weight.  W[d] captures the independent
        "severity propensity" of drug d.
        Initialised ~ N(0, 0.01) to match the NumPy FM baseline.
    V  : nn.Embedding  shape (vocab_size, k)
        Latent interaction embedding.  V[d] is the k-dimensional
        factor vector of drug d, shared between FM and Deep branches.
        Initialised ~ N(0, 0.01).

    FM Branch
    ─────────
    Applies the exact same two-feature simplification derived in Phase 4:

        z_FM = w₀  +  W[d1]  +  W[d2]  +  Σ_f V[d1,f]·V[d2,f]

    This is a degree-2 polynomial in the input features, capturing
    all explicit pairwise interactions via the dot product.

    Deep Branch
    ───────────
    The concatenated latent pair is passed through a shallow MLP:

        h₀     = [V[d1] ‖ V[d2]]              shape (N, 2k)
        h₁     = Dropout( ReLU( W₁·h₀ + b₁ ) )   shape (N, H)
        z_Deep = W₂·h₁ + b₂                   shape (N, 1) → (N,)

    Key design choice — shared V
    ─────────────────────────────
    V is shared between the FM and Deep branches (as in the original
    DeepFM paper).  This means both branches jointly train the same
    latent representation, regularising V and reducing total parameter
    count compared to two separate embedding tables.

    Output
    ──────
    Raw logit (no sigmoid):   z = z_FM + z_Deep   shape (N,)

    The training loop uses nn.BCEWithLogitsLoss, which internally
    applies sigmoid + BCE in a single numerically stable operation via
    the log-sum-exp identity:

        loss = max(z,0) − z·y + log(1 + exp(−|z|))
    """

    def __init__(
        self,
        vocab_size: int,
        k:          int,
        hidden_dim: int   | None = None,
        dropout:    float        = 0.3,
    ) -> None:
        super().__init__()

        H = hidden_dim if hidden_dim is not None else 4 * k   # MLP width

        # ── Shared embedding parameters ───────────────────────────────
        self.w0 = nn.Parameter(torch.zeros(1))       # scalar global bias
        self.W  = nn.Embedding(vocab_size, 1)         # first-order weights
        self.V  = nn.Embedding(vocab_size, k)         # latent embeddings

        # ── Deep branch MLP ───────────────────────────────────────────
        # Architecture: Linear → ReLU → Dropout → Linear
        # Input: 2k (concatenated V[d1] and V[d2])
        # Output: 1 scalar per sample (raw contribution to logit)
        self.mlp = nn.Sequential(
            nn.Linear(2 * k, H),     # project 2k-dim input → H-dim hidden
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(H, 1),         # compress H-dim → scalar
        )

        # ── Weight initialisation ─────────────────────────────────────
        # Embedding tables — small normal (mirrors NumPy FM Phase 4/5/6).
        # Prevents early saturation of the FM dot-product term.
        nn.init.normal_(self.W.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.V.weight, mean=0.0, std=0.01)
        # MLP linear layers — Kaiming uniform (PyTorch default for Linear),
        # which is the correct choice for ReLU activations.

        # Store hyper-parameters for inspection
        self.k          = k
        self.hidden_dim = H
        self.vocab_size = vocab_size

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        d1: torch.Tensor,   # LongTensor  shape (N,)
        d2: torch.Tensor,   # LongTensor  shape (N,)
    ) -> torch.Tensor:      # FloatTensor shape (N,)  — raw logit
        """
        DeepFM forward pass — returns z (no sigmoid).

        Step 1 — Embedding lookups
        ─────────────────────────
        v_d1, v_d2 : (N, k)    latent factors for Drug 1 and Drug 2
        W_d1, W_d2 : (N,)      first-order weights (squeezed from (N,1))

        Step 2 — FM branch
        ──────────────────
        interaction : (N,)   element-wise product summed over k factors
                             = Σ_f V[d1,f] · V[d2,f]
        z_FM        : (N,)   w₀ + W[d1] + W[d2] + interaction

        Step 3 — Deep branch
        ────────────────────
        h0     : (N, 2k)  — cat([v_d1, v_d2], dim=-1)
        z_Deep : (N,)     — MLP(h0).squeeze(-1)

        Step 4 — Combine
        ────────────────
        z : (N,)  — z_FM + z_Deep   raw logit
        """
        # ── Step 1: Embedding lookups ─────────────────────────────────
        v_d1 = self.V(d1)                              # (N, k)
        v_d2 = self.V(d2)                              # (N, k)
        W_d1 = self.W(d1).squeeze(-1)                  # (N, 1) → (N,)
        W_d2 = self.W(d2).squeeze(-1)                  # (N, 1) → (N,)

        # ── Step 2: FM branch ─────────────────────────────────────────
        interaction = (v_d1 * v_d2).sum(dim=-1)        # (N,)
        z_FM = self.w0 + W_d1 + W_d2 + interaction    # (N,)
        # self.w0 has shape (1,); broadcasts to (N,) automatically.

        # ── Step 3: Deep branch ───────────────────────────────────────
        h0     = torch.cat([v_d1, v_d2], dim=-1)       # (N, 2k)
        z_Deep = self.mlp(h0).squeeze(-1)               # (N, 1) → (N,)

        # ── Step 4: Combined raw logit ────────────────────────────────
        z = z_FM + z_Deep                              # (N,)
        return z

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 5.  Model inspection helper
# ============================================================================

def print_model_summary(model: DeepFM) -> None:
    """Print parameter table with name, shape, and count."""
    print(f"\n{'─'*60}")
    print(f"  {'Parameter':<32}  {'Shape':>14}   {'#Params':>8}")
    print(f"  {'─'*32}  {'─'*14}   {'─'*8}")
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            print(f"  {name:<32}  {str(tuple(p.shape)):>14}   {n:>8,}")
    print(f"  {'─'*32}  {'─'*14}   {'─'*8}")
    print(f"  {'Total trainable':<32}  {'':>14}   {total:>8,}")
    print(f"{'─'*60}")


# ============================================================================
# 6.  Smoke test — architecture validation
# ============================================================================

if __name__ == "__main__":

    K = 32   # best k from the NumPy FM experiment (Phase 6)

    # ── Instantiate ───────────────────────────────────────────────────
    model = DeepFM(
        vocab_size = VOCAB_SIZE,
        k          = K,
        hidden_dim = None,      # defaults to 4*K = 128
        dropout    = 0.3,
    ).to(DEVICE)

    print(f"\n{'═'*60}")
    print(f"  DeepFM  —  k={K}  hidden_dim={model.hidden_dim}  dropout=0.3")
    print(f"{'═'*60}")
    print(model)
    print_model_summary(model)

    # ── Forward pass on four synthetic drug pairs ─────────────────────
    # Deliberately pick spread-out indices to exercise the full vocab range
    d1_test = torch.tensor([  0,  50, 100, 419], dtype=torch.long, device=DEVICE)
    d2_test = torch.tensor([  1, 200, 300, 418], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        z_out = model(d1_test, d2_test)

    print(f"\n{'─'*60}")
    print(f"  Smoke-test  —  batch_size=4")
    print(f"{'─'*60}")
    print(f"  Drug-1 indices  : {d1_test.tolist()}")
    print(f"  Drug-2 indices  : {d2_test.tolist()}")
    print(f"  Output logits z : {[round(v, 6) for v in z_out.tolist()]}")
    print(f"  Sigmoid(z)      : {[round(v, 6) for v in torch.sigmoid(z_out).tolist()]}")
    print(f"  Output shape    : {tuple(z_out.shape)}")
    print(f"  Output dtype    : {z_out.dtype}")

    # ── Decompose one sample into FM and Deep contributions ───────────
    with torch.no_grad():
        # Manually trace sample index 0 to verify both branches work
        s_d1 = d1_test[:1]     # shape (1,)
        s_d2 = d2_test[:1]

        v1 = model.V(s_d1)                             # (1, k)
        v2 = model.V(s_d2)                             # (1, k)
        W1 = model.W(s_d1).squeeze(-1)                # (1,)
        W2 = model.W(s_d2).squeeze(-1)                # (1,)

        intr   = (v1 * v2).sum(dim=-1)                 # (1,)
        z_fm   = model.w0 + W1 + W2 + intr             # (1,)
        z_deep = model.mlp(torch.cat([v1, v2], dim=-1)).squeeze(-1)  # (1,)
        z_tot  = z_fm + z_deep

    print(f"\n  Branch decomposition (sample 0: d1={d1_test[0].item()}, d2={d2_test[0].item()}):")
    print(f"    w₀             : {model.w0.item():+.6f}")
    print(f"    W[d1]          : {W1.item():+.6f}")
    print(f"    W[d2]          : {W2.item():+.6f}")
    print(f"    <V[d1],V[d2]>  : {intr.item():+.6f}")
    print(f"    z_FM           : {z_fm.item():+.6f}")
    print(f"    z_Deep (MLP)   : {z_deep.item():+.6f}")
    print(f"    z = z_FM + z_Deep : {z_tot.item():+.6f}")

    # ── Assertions ────────────────────────────────────────────────────
    assert z_out.shape == (4,),              f"Shape error: {z_out.shape}"
    assert z_out.dtype == torch.float32,     f"dtype error: {z_out.dtype}"
    assert not torch.isnan(z_out).any(),     "NaN detected in output"
    assert not torch.isinf(z_out).any(),     "Inf detected in output"
    assert abs(z_tot.item() - z_out[0].item()) < 1e-5, "Branch sum mismatch"

    print(f"\n{'─'*60}")
    print(f"  ✅  All assertions passed.")
    print(f"  Architecture connects correctly: FM branch + Deep branch → z")
    print(f"  Total trainable parameters: {model.num_parameters:,}")
    print(f"{'─'*60}")
    print(f"\n  Ready for Phase 8: training loop + weighted BCEWithLogitsLoss.")
