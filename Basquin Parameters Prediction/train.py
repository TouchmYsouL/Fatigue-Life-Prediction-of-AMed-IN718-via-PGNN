import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# =========================================================
# 0) Reproducibility
# =========================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =========================================================
# 1) Load data
# =========================================================
# DATA_CSV = "SN_train_augmented_localEC_bidirectional_v3_full.csv"
DATA_CSV = "SN_full_augmented_1.csv"
df = pd.read_csv(DATA_CSV)

# X: keep your original behavior (all columns except last 3)
X = df.iloc[:, :-3].values
# y: SN params
y = df[["log_a", "b"]].values

# label col (last col)
label_col = df.columns[-1]
labels = df[label_col].values  # numpy array

# =========================================================
# 2) Scaling
# =========================================================
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# =========================================================
# 3) 70 / 15 / 15 split (Train / Val / Test)
# =========================================================
n = X_tensor.shape[0]
perm = torch.randperm(n)

X_tensor = X_tensor[perm]
y_tensor = y_tensor[perm]
labels = labels[perm.numpy()]  # keep aligned

n_train = int(0.70 * n)
n_val = int(0.15 * n)

X_train = X_tensor[:n_train]
y_train = y_tensor[:n_train]
label_train = labels[:n_train]

X_val = X_tensor[n_train:n_train + n_val]
y_val = y_tensor[n_train:n_train + n_val]
label_val = labels[n_train:n_train + n_val]

X_test = X_tensor[n_train + n_val:]
y_test = y_tensor[n_train + n_val:]
label_test = labels[n_train + n_val:]

print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# =========================================================
# 4) Model
# =========================================================
class SN_ANN(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dims=[32, 64, 128, 64, 32]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def huber_loss(y_true, y_pred, delta=1.0):
    err = y_true - y_pred
    small = torch.abs(err) <= delta
    return torch.where(small, 0.5 * err**2, delta * (torch.abs(err) - 0.5 * delta)).mean()

def hybrid_loss(y_true, y_pred, alpha=0.7, delta=1.0):
    return alpha * mse_loss(y_true, y_pred) + (1 - alpha) * huber_loss(y_true, y_pred, delta=delta)

def data_loss(y_true, y_pred, loss_type="hybrid", alpha=0.7, delta=1.0):
    if loss_type == "mse":
        return mse_loss(y_true, y_pred)
    elif loss_type == "huber":
        return huber_loss(y_true, y_pred, delta=delta)
    elif loss_type == "hybrid":
        return hybrid_loss(y_true, y_pred, alpha=alpha, delta=delta)
    else:
        raise ValueError("loss_type must be 'mse', 'huber', or 'hybrid'")

def l2_regularization(model, lambda_reg=1e-4):
    return lambda_reg * sum(torch.sum(p**2) for p in model.parameters())

y_mean_t = torch.tensor(y_scaler.mean_, dtype=torch.float32)   # shape (2,)
y_scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float32) # shape (2,)

def inverse_scale_y(y_scaled_pred: torch.Tensor) -> torch.Tensor:
    """y_real = y_scaled * scale + mean"""
    device = y_scaled_pred.device
    return y_scaled_pred * y_scale_t.to(device) + y_mean_t.to(device)

def physics_loss_sn(y_scaled_pred,
                    loga_min=2.0, loga_max=4.0,
                    b_max=0.0,
                    penalty_power=2):
    """
    Soft constraints:
      - b <= 0  (penalize b > 0)
      - log_a in [loga_min, loga_max] (penalize out-of-range)
    """
    y_real = inverse_scale_y(y_scaled_pred)
    log_a = y_real[:, 0]
    b = y_real[:, 1]

    p_b = torch.relu(b - b_max)
    p_low = torch.relu(loga_min - log_a)
    p_high = torch.relu(log_a - loga_max) 

    if penalty_power == 1:
        return (p_b + p_low + p_high).mean()
    else:
        return (p_b**2 + p_low**2 + p_high**2).mean()

def total_loss(model, X, y,
               lambda_reg=1e-4,
               lambda_phys=1e-3,
               loss_type="hybrid",
               alpha=0.7,
               delta=1.0,
               loga_min=2.0,
               loga_max=4.0):
    """
    Returns:
      loss_total, loss_data(detached), loss_l2(detached), loss_phys(detached)
    """
    pred = model(X)

    # data loss (scaled space)
    loss_d = data_loss(y, pred, loss_type=loss_type, alpha=alpha, delta=delta)

    # L2 in loss
    loss_l2 = l2_regularization(model, lambda_reg=lambda_reg)

    # physics loss (REAL space constraints)
    loss_p = physics_loss_sn(pred, loga_min=loga_min, loga_max=loga_max, b_max=0.0, penalty_power=2)

    loss_total = loss_d + loss_l2 + lambda_phys * loss_p
    return loss_total, loss_d.detach(), loss_l2.detach(), loss_p.detach()

# =========================================================
# 5) Train
# =========================================================
def train_model(model,
                X_train, y_train,
                X_val, y_val,
                lr=5e-3,
                epochs=10000,
                lambda_reg=1e-4,
                lambda_phys=1e-3,
                loss_type="hybrid",
                alpha=0.7,
                delta=1.0,
                loga_min=2.0,
                loga_max=4.0,
                patience=300,
                min_delta=1e-6,
                print_every=200,
                save_path="best_sn_model.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    train_total_hist = []
    val_data_hist = []

    for epoch in range(1, epochs + 1):
        model.train()

        loss_total, loss_d, loss_l2, loss_p = total_loss(
            model, X_train, y_train,
            lambda_reg=lambda_reg,
            lambda_phys=lambda_phys,
            loss_type=loss_type,
            alpha=alpha,
            delta=delta,
            loga_min=loga_min,
            loga_max=loga_max
        )

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_d = data_loss(y_val, val_pred, loss_type=loss_type, alpha=alpha, delta=delta).item()

        train_total_hist.append(loss_total.item())
        val_data_hist.append(val_d)

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"[Epoch {epoch}] "
                f"TrainTotal={loss_total.item():.6f} (data={loss_d.item():.6f}, l2={loss_l2.item():.6f}, phys={loss_p.item():.6f}) "
                f"| ValData={val_d:.6f}"
            )

        # ---- early stopping on val_d ----
        if val_d < best_val - min_delta:
            best_val = val_d
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best ValData={best_val:.6f}")
                break

    # load best weights
    model.load_state_dict(best_state)
    print(f"Best model loaded. (Best ValData={best_val:.6f}, saved to {save_path})")
    return model, train_total_hist, val_data_hist

# =========================================================
# 6) Evaluation
# =========================================================
def evaluate(model, X, y, y_scaler, dataset_name=""):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X).cpu().numpy()
        y_true_scaled = y.cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {dataset_name} Set Evaluation:\nMSE = {mse:.6f} | R² = {r2:.6f}")
    return y_true, y_pred

def plot_loss_curves(train_total, val_data):
    plt.figure(figsize=(8, 5))
    plt.plot(train_total, label="Train Total Loss")
    plt.plot(val_data, label="Val Data Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pred_vs_true(y_true, y_pred, title_prefix=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].scatter(y_true[:, 0], y_pred[:, 0], edgecolor="k", alpha=0.8)
    mn0, mx0 = min(y_true[:, 0]), max(y_true[:, 0])
    axs[0].plot([mn0, mx0], [mn0, mx0], "r--")
    axs[0].set_title(f"{title_prefix} log(a)")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Predicted")
    axs[0].grid(True)

    axs[1].scatter(y_true[:, 1], y_pred[:, 1], edgecolor="k", alpha=0.8)
    mn1, mx1 = min(y_true[:, 1]), max(y_true[:, 1])
    axs[1].plot([mn1, mx1], [mn1, mx1], "r--")
    axs[1].set_title(f"{title_prefix} b")
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Predicted")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)

    model = SN_ANN(input_dim=X_train.shape[1])

    # ---- training config (you can tune these) ----
    cfg = dict(
        lr=5e-3,
        epochs=10000,
        lambda_reg=1e-4,
        lambda_phys=1e-3,
        loss_type="hybrid",
        alpha=0.7,
        delta=1.0,
        loga_min=2.0,
        loga_max=4.0,
        patience=300,
        min_delta=1e-6,
        print_every=200,
        save_path="best_sn_model.pt",
    )

    trained_model, train_hist, val_hist = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        **cfg
    )

    # ---- evaluation ----
    y_train_true, y_train_pred = evaluate(trained_model, X_train, y_train, y_scaler, "Train")
    y_val_true,   y_val_pred   = evaluate(trained_model, X_val,   y_val,   y_scaler, "Validation")
    y_test_true,  y_test_pred  = evaluate(trained_model, X_test,  y_test,  y_scaler, "Test")

    # ---- plots ----
    plot_loss_curves(train_hist, val_hist)
    plot_pred_vs_true(y_train_true, y_train_pred, "Train")
    plot_pred_vs_true(y_val_true,   y_val_pred,   "Val")
    plot_pred_vs_true(y_test_true,  y_test_pred,  "Test")

    # ---- save final weights (best already saved during training) ----
    torch.save(trained_model.state_dict(), "model.pth")
    print("Saved model.pth (final) and best_sn_model.pt (best val).")

    # ---- save test results ----
    results_df = pd.DataFrame({
        "Label": label_test,
        "log_a_true": y_test_true[:, 0],
        "b_true": y_test_true[:, 1],
        "log_a_pred": y_test_pred[:, 0],
        "b_pred": y_test_pred[:, 1],
    })
    results_df.to_csv("Results/Test_True_vs_Predicted.csv", index=False)
    print("Test results saved to Results/Test_True_vs_Predicted.csv")

    # ---- plot S-N curve for test samples ----
    N = np.logspace(3, 7, 200)
    for i in range(len(y_test_true)):
        log_a_true, b_true = y_test_true[i]
        log_a_pred, b_pred = y_test_pred[i]

        S_true = 10**log_a_true * N**b_true
        S_pred = 10**log_a_pred * N**b_pred

        plt.figure(figsize=(6, 4))
        plt.plot(N, S_true, label="True", linestyle="--")
        plt.plot(N, S_pred, label="Predicted", linestyle="-")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Cycles (N)")
        plt.ylabel("Stress Range (MPa)")
        plt.title(f"S-N Curve: {label_test[i]}")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Results/{label_test[i]}.png", dpi=300)
        plt.close()
