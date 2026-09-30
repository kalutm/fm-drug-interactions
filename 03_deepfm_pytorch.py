"""
03_deepfm_pytorch.py — Phase 8: DeepFM Training, Ablation Studies & Evaluation
================================================================================
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

Phase 8 Pipeline
────────────────
1. Data Loaders: TensorDataset + DataLoader (batch_size=256)
2. Training Loop: Adam (lr=0.001), weighted BCEWithLogitsLoss, best model saving
3. Dynamic Thresholding: Sweep [0.20, 0.60, step=0.05] on validation set
4. Ablation Studies: Train 3 modes (full, fm_only, deep_only) @ k=32
5. Final Evaluation: Test set run once for best ablation mode
6. Pair Decomposition: Detailed logit breakdown for Suvorexant + Droperidol
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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
# ============================================================================

train_drugs: set[str] = set(train_df["Drug 1"]) | set(train_df["Drug 2"])
vocab:        dict[str, int] = {drug: idx for idx, drug in enumerate(sorted(train_drugs))}
idx_to_drug:  dict[int, str] = {idx: drug for drug, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

assert VOCAB_SIZE == 420, f"Expected 420, got {VOCAB_SIZE}."
print(f"Vocabulary: {VOCAB_SIZE} unique drugs (from train split only)")

# ============================================================================
# 3.  Encode helper & DataLoader preparation
# ============================================================================

def encode(
    df:     pd.DataFrame,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Drug 1 / Drug 2 names → integer indices; labels → float32."""
    d1 = torch.tensor(df["Drug 1"].map(vocab).to_numpy(), dtype=torch.long,    device=device)
    d2 = torch.tensor(df["Drug 2"].map(vocab).to_numpy(), dtype=torch.long,    device=device)
    y  = torch.tensor(df["Label"].to_numpy(),              dtype=torch.float32, device=device)
    return d1, d2, y


train_d1, train_d2, train_y = encode(train_df)
val_d1,   val_d2,   val_y   = encode(val_df)
test_d1,  test_d2,  test_y  = encode(test_df)

# Calculate class imbalance weighting factor: pos_weight = N_0 / N_1
N0 = (train_y == 0).sum().item()
N1 = (train_y == 1).sum().item()
pos_weight_val = N0 / N1
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
print(f"Class imbalance: N0={int(N0):,}, N1={int(N1):,}, pos_weight={pos_weight_val:.4f}")

BATCH_SIZE = 256
train_dataset = TensorDataset(train_d1, train_d2, train_y)
val_dataset   = TensorDataset(val_d1, val_d2, val_y)
test_dataset  = TensorDataset(test_d1, test_d2, test_y)

# DataLoader setup (shuffle=True for train, shuffle=False for val/test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# Threshold sweep grid for dynamic thresholding on validation set
THRESHOLDS = np.round(np.arange(0.20, 0.61, 0.05), 2)

# ============================================================================
# 4.  DeepFM Module (with Ablation mode support)
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
    mode : str
        Branch inclusion mode: 'full', 'fm_only', or 'deep_only'.
    """

    def __init__(
        self,
        vocab_size: int,
        k:          int,
        hidden_dim: int   | None = None,
        dropout:    float        = 0.3,
        mode:       str          = "full",
    ) -> None:
        super().__init__()

        assert mode in ("full", "fm_only", "deep_only"), f"Invalid mode: {mode}"
        self.mode = mode

        H = hidden_dim if hidden_dim is not None else 4 * k   # MLP width

        # ── Shared embedding parameters ───────────────────────────────
        self.w0 = nn.Parameter(torch.zeros(1))       # scalar global bias
        self.W  = nn.Embedding(vocab_size, 1)         # first-order weights
        self.V  = nn.Embedding(vocab_size, k)         # latent embeddings

        # ── Deep branch MLP ───────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(2 * k, H),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(H, 1),
        )

        # ── Weight initialisation ─────────────────────────────────────
        nn.init.normal_(self.W.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.V.weight, mean=0.0, std=0.01)

        self.k          = k
        self.hidden_dim = H
        self.vocab_size = vocab_size

    def forward(
        self,
        d1: torch.Tensor,
        d2: torch.Tensor,
    ) -> torch.Tensor:
        """
        DeepFM forward pass — returns raw logit z based on selected mode.
        """
        v_d1 = self.V(d1)                              # (N, k)
        v_d2 = self.V(d2)                              # (N, k)
        W_d1 = self.W(d1).squeeze(-1)                  # (N,)
        W_d2 = self.W(d2).squeeze(-1)                  # (N,)

        # ── FM branch ─────────────────────────────────────────────────
        interaction = (v_d1 * v_d2).sum(dim=-1)        # (N,)
        z_FM = self.w0 + W_d1 + W_d2 + interaction    # (N,)

        # ── Deep branch ───────────────────────────────────────────────
        h0     = torch.cat([v_d1, v_d2], dim=-1)       # (N, 2k)
        z_Deep = self.mlp(h0).squeeze(-1)               # (N,)

        # ── Branch selection based on ablation mode ───────────────────
        if self.mode == "full":
            return z_FM + z_Deep
        elif self.mode == "fm_only":
            return z_FM
        elif self.mode == "deep_only":
            return z_Deep

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# 5.  Training Pipeline & Validation Evaluation
# ============================================================================

def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference on a DataLoader.
    Returns (logits, targets, average_bce_loss).
    """
    model.eval()
    all_logits  = []
    all_targets = []
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for b_d1, b_d2, b_y in loader:
            b_d1, b_d2, b_y = b_d1.to(device), b_d2.to(device), b_y.to(device)
            logits = model(b_d1, b_d2)
            loss = criterion(logits, b_y)
            total_loss += loss.item() * len(b_y)
            total_samples += len(b_y)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(b_y.cpu().numpy())

    avg_loss = total_loss / total_samples
    return np.concatenate(all_logits), np.concatenate(all_targets), avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.001,
    device: torch.device = DEVICE,
) -> tuple[nn.Module, float, float]:
    """
    Train DeepFM model using Adam optimizer and BCEWithLogitsLoss.
    Evaluates dynamic thresholding on val set per epoch and saves best weights.

    Returns:
        (best_model, best_val_f1, best_threshold)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_threshold = 0.50
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\nBeginning training: Mode={model.mode} | Epochs={epochs} | LR={lr}")
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Best Val F1':^12} | {'Optimal Thresh':^14}")
    print(f"{'─'*7}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*14}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for b_d1, b_d2, b_y in train_loader:
            b_d1, b_d2, b_y = b_d1.to(device), b_d2.to(device), b_y.to(device)

            optimizer.zero_grad()
            logits = model(b_d1, b_d2)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(b_y)
            n_samples += len(b_y)

        train_loss = running_loss / n_samples

        # ── Validation Evaluation & Dynamic Thresholding ─────────────
        val_logits, val_targets, val_loss = evaluate_loader(model, val_loader, device=device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        epoch_best_f1 = -1.0
        epoch_best_thresh = 0.50

        for t in THRESHOLDS:
            preds = (val_probs >= t).astype(int)
            f1 = f1_score(val_targets, preds, zero_division=0)
            if f1 > epoch_best_f1:
                epoch_best_f1 = f1
                epoch_best_thresh = t

        # Track and save best model checkpoint based on Validation F1
        if epoch_best_f1 > best_val_f1:
            best_val_f1 = epoch_best_f1
            best_threshold = epoch_best_thresh
            best_weights = copy.deepcopy(model.state_dict())

        print(f"{epoch:^7d} | {train_loss:^10.4f} | {val_loss:^10.4f} | {best_val_f1:^12.4f} | {best_threshold:^14.2f}")

    # Load best weights before returning
    model.load_state_dict(best_weights)
    return model, best_val_f1, best_threshold

# ============================================================================
# 6.  Ablation Studies
# ============================================================================

def run_ablations(
    vocab_size: int = VOCAB_SIZE,
    k: int = 32,
    epochs: int = 30,
) -> tuple[dict[str, dict], str]:
    """
    Train three separate DeepFM models: mode="full", "fm_only", "deep_only".
    Returns dictionary of results and the best performing mode name.
    """
    modes = ["full", "fm_only", "deep_only"]
    results = {}

    print(f"\n{'═'*65}")
    print(f"  Ablation Studies (k={k}, epochs={epochs})")
    print(f"{'═'*65}")

    for mode in modes:
        # Set seeds before initializing each model to ensure consistent baseline setup
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = DeepFM(vocab_size=vocab_size, k=k, dropout=0.3, mode=mode).to(DEVICE)
        trained_model, best_f1, best_thresh = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight_tensor,
            epochs=epochs,
            lr=0.001,
            device=DEVICE,
        )

        results[mode] = {
            "model": trained_model,
            "val_f1": best_f1,
            "best_thresh": best_thresh,
        }

    # Summary table
    print(f"\n{'═'*60}")
    print(f"  Ablation Study Results Summary")
    print(f"{'═'*60}")
    print(f"  {'Mode':<15} | {'Val F1':>10} | {'Optimal Threshold':>18}")
    print(f"  {'─'*15}─┼─{'─'*10}─┼─{'─'*18}")
    for mode in modes:
        print(f"  {mode:<15} | {results[mode]['val_f1']:>10.4f} | {results[mode]['best_thresh']:>18.2f}")
    print(f"{'═'*60}")

    best_mode = max(results, key=lambda m: results[m]["val_f1"])
    print(f"  🏆 Best Performing Mode: '{best_mode}' (Val F1 = {results[best_mode]['val_f1']:.4f})")

    return results, best_mode

# ============================================================================
# 7.  Final Evaluation & Pair Decomposition
# ============================================================================

def final_test_evaluation(
    model: nn.Module,
    threshold: float,
) -> None:
    """Run Test set evaluation exactly once using the best model and threshold."""
    test_logits, test_targets, test_loss = evaluate_loader(model, test_loader, device=DEVICE)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds)

    print(f"\n{'═'*60}")
    print(f"  Final Test Evaluation (Mode: '{model.mode}', Thresh: {threshold:.2f})")
    print(f"{'═'*60}")
    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  Test Precision : {prec:.4f}")
    print(f"  Test Recall    : {rec:.4f}")
    print(f"  Test F1 Score  : {f1:.4f}")
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:<5} | FP: {cm[0,1]:<5}")
    print(f"    FN: {cm[1,0]:<5} | TP: {cm[1,1]:<5}")
    print(f"{'═'*60}")


def decompose_pair(
    model: DeepFM,
    drug1: str = "Suvorexant",
    drug2: str = "Droperidol",
    threshold: float = 0.50,
) -> None:
    """
    Decompose predicted logit for a target drug pair into individual components:
    w0, W[d1], W[d2], <V[d1], V[d2]>, z_FM, z_Deep, total z, and probability.
    """
    d1_idx = vocab[drug1]
    d2_idx = vocab[drug2]

    d1_t = torch.tensor([d1_idx], dtype=torch.long, device=DEVICE)
    d2_t = torch.tensor([d2_idx], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        v1 = model.V(d1_t)                              # (1, k)
        v2 = model.V(d2_t)                              # (1, k)
        W1 = model.W(d1_t).squeeze(-1)                 # (1,)
        W2 = model.W(d2_t).squeeze(-1)                 # (1,)
        w0 = model.w0.item()

        intr = (v1 * v2).sum(dim=-1).item()             # scalar
        z_fm = w0 + W1.item() + W2.item() + intr

        h0 = torch.cat([v1, v2], dim=-1)                # (1, 2k)
        z_deep = model.mlp(h0).squeeze(-1).item()       # scalar
        z_tot = model(d1_t, d2_t).item()                # scalar
        prob = 1.0 / (1.0 + np.exp(-z_tot))
        pred = 1 if prob >= threshold else 0

    print(f"\n{'═'*65}")
    print(f"  Pair Decomposition: {drug1} (idx={d1_idx}) + {drug2} (idx={d2_idx})")
    print(f"{'═'*65}")
    print(f"  w₀ (Global Bias)            : {w0:+.6f}")
    print(f"  W[{drug1}] (Linear weight)  : {W1.item():+.6f}")
    print(f"  W[{drug2}] (Linear weight)  : {W2.item():+.6f}")
    print(f"  <V[{drug1}], V[{drug2}]>     : {intr:+.6f}")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  z_FM (FM branch logit)       : {z_fm:+.6f}")
    print(f"  z_Deep (MLP branch logit)    : {z_deep:+.6f}")
    print(f"  z_Total (Raw combined logit) : {z_tot:+.6f}")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Predicted Probability p(y=1) : {prob:.4f}")
    print(f"  Classification Threshold     : {threshold:.2f}")
    print(f"  Predicted Class Label        : {pred}")
    print(f"  True Ground-Truth Label      : 1 (Severe)")
    print(f"{'═'*65}")

# ============================================================================
# 8. Main Execution Block
# ============================================================================

if __name__ == "__main__":
    K = 32
    EPOCHS = 30

    # 1. Run Ablation Studies
    ablation_results, best_mode = run_ablations(vocab_size=VOCAB_SIZE, k=K, epochs=EPOCHS)

    # 2. Extract best model and optimal threshold
    best_model = ablation_results[best_mode]["model"]
    best_threshold = ablation_results[best_mode]["best_thresh"]

    # 3. Perform final test evaluation (exactly once)
    final_test_evaluation(best_model, threshold=best_threshold)

    # 4. Decompose score for Suvorexant + Droperidol
    decompose_pair(best_model, drug1="Suvorexant", drug2="Droperidol", threshold=best_threshold)
