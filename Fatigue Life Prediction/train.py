import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
import joblib

# ------------------------
# Step 1: Load & feature engineering
# ------------------------
df = pd.read_csv("IN718_full_augmented.csv")

df["sigma_over_UTS"] = df["stress_range"] / df["UTS"]
df["sigma_over_YTS"] = df["stress_range"] / df["YTS"]
X_num = df[["sigma_over_UTS", "sigma_over_YTS", "YTS"]]
X_cat = pd.get_dummies(
    df[["geometry_type", "surface_condition", "post_process", "build_direction", "stress_ratio"]].astype(str),
    drop_first=False
)

X_df = pd.concat([X_num, X_cat], axis=1)
feature_names = list(X_df.columns)
X = X_df.values

y = np.log10(df["fatigue_life"].values.astype(float)).reshape(-1, 1)

# ------------------------
# Step 2: Scale
# ------------------------
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ------------------------
# Step 3: Split 70/15/15
# ------------------------
n = X_tensor.shape[0]
perm = torch.randperm(n)
X_tensor = X_tensor[perm]
y_tensor = y_tensor[perm]

n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train, y_train = X_tensor[:n_train], y_tensor[:n_train]
X_val,   y_val   = X_tensor[n_train:n_train + n_val], y_tensor[n_train:n_train + n_val]
X_test,  y_test  = X_tensor[n_train + n_val:], y_tensor[n_train + n_val:]

# ------------------------
# Step 4: Model
# ------------------------
class DA_PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 64, 128, 64, 32], output_dim=1):
        super(DA_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.SELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.SELU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]), nn.SELU(),
            nn.Linear(hidden_dims[2], hidden_dims[3]), nn.SELU(),
            nn.Linear(hidden_dims[3], hidden_dims[4]), nn.SELU(),
            nn.Linear(hidden_dims[4], output_dim)
        )
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

# ------------------------
# Step 5: Train
# ------------------------
def train_model(model, X_train, y_train, X_val, y_val,
                learning_rate=1e-3, max_epochs=2000,
                weight_decay=1e-5, patience=150,
                print_every=50, save_path="best_model_stress.pt",
                loss_type="hybrid", alpha=0.7, delta=1.0,
                mono_features=("sigma_over_UTS", "sigma_over_YTS"),
                mono_lambda=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    mono_idx = []
    for name in mono_features:
        if name in feature_names:
            mono_idx.append(feature_names.index(name))
    mono_idx = torch.tensor(mono_idx, dtype=torch.long) if len(mono_idx) > 0 else None

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    counter = 0
    train_loss_history, val_loss_history = [], []

    def data_loss(y_true, y_pred):
        if loss_type == "mse":
            return mse_loss(y_true, y_pred)
        elif loss_type == "huber":
            return huber_loss(y_true, y_pred, delta=delta)
        elif loss_type == "hybrid":
            return hybrid_loss(y_true, y_pred, alpha=alpha, delta=delta)
        else:
            raise ValueError("loss_type must be 'mse', 'huber', or 'hybrid'")

    for epoch in range(1, max_epochs + 1):
        model.train()

        X_req = X_train.clone().detach().requires_grad_(True)
        y_pred = model(X_req)
        loss_data = data_loss(y_train, y_pred)

        if mono_idx is not None and len(mono_idx) > 0:
            grads = torch.autograd.grad(y_pred.sum(), X_req, create_graph=True)[0]
            grad_sel = grads[:, mono_idx]
            mono_penalty = torch.relu(grad_sel).mean()
            loss = loss_data + mono_lambda * mono_penalty
        else:
            mono_penalty = torch.tensor(0.0, device=X_train.device)
            loss = loss_data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = data_loss(y_val, y_val_pred)

        train_loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"[Epoch {epoch}] TrainLoss={loss.item():.5f} | ValLoss={val_loss.item():.5f} "
                  f"(data={loss_data.item():.5f}, mono={mono_penalty.item():.5f}, "
                  f"type={loss_type}, alpha={alpha}, delta={delta}, mono_lambda={mono_lambda})")

        # early stopping
        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(best_model_state, save_path)
    model.load_state_dict(best_model_state)
    print("Best model loaded.")
    return model, train_loss_history, val_loss_history

# ------------------------
# Step 6: Evaluation
# ------------------------
def evaluate_model(model, X_set, y_set, set_name="Set", make_plot=False):
    model.eval()
    with torch.no_grad():
        y_pred_log = model(X_set).numpy()
        y_true_log = y_set.numpy()

    mse_log = mean_squared_error(y_true_log, y_pred_log)
    r2_log  = r2_score(y_true_log, y_pred_log)
    print(f"{set_name} -> MSE_log: {mse_log:.6f}, R2_log: {r2_log:.6f}")

    if make_plot:
        plt.figure(figsize=(6, 6))
        y_true = 10**y_true_log
        y_pred = 10**y_pred_log
        plt.loglog(y_true, y_pred, 'o', alpha=0.8, label=f"{set_name} Prediction", markeredgecolor="k")
        x_vals = np.logspace(3, 9, 100)
        plt.plot(x_vals, x_vals, 'r--', label="1:1 Line")
        plt.plot(x_vals, x_vals * 3, 'g--', label="+3-fold");  plt.plot(x_vals, x_vals / 3, 'g--')
        plt.plot(x_vals, x_vals * 2, 'b--', label="+2-fold");  plt.plot(x_vals, x_vals / 2, 'b--')
        plt.xlim(1e3, 1e9)
        plt.ylim(1e3, 1e9)
        plt.xlabel("Experimental $N_f$"); plt.ylabel("Predicted $N_f$")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))
        plt.legend(loc="upper right"); plt.title(f"{set_name}: Prediction vs Experimental (log-log)")
        plt.tight_layout(); plt.show()

    return mse_log, r2_log

# ------------------------
# Step 7: Run
# ------------------------
if __name__ == "__main__":
    input_dim = X_train.shape[1]
    model = DA_PINN(input_dim=input_dim)

    trained_model, train_loss, val_loss = train_model(
        model, X_train, y_train, X_val, y_val,
        learning_rate=1e-3, max_epochs=2000, weight_decay=1e-5, patience=200,
        loss_type="hybrid", alpha=0.7, delta=1.0,
        mono_features=("sigma_over_UTS", "sigma_over_YTS"),
        mono_lambda=1e-3
    )

    mse_tr, r2_tr = evaluate_model(trained_model, X_train, y_train, set_name="Train")
    mse_va, r2_va = evaluate_model(trained_model, X_val,   y_val,   set_name="Val")
    mse_te, r2_te = evaluate_model(trained_model, X_test,  y_test,  set_name="Test",  make_plot=True)

    print("\nFinal Summary (log-domain):")
    print(f"Train -> MSE_log: {mse_tr:.6f}, R2_log: {r2_tr:.6f}")
    print(f"Val   -> MSE_log: {mse_va:.6f}, R2_log: {r2_va:.6f}")
    print(f"Test  -> MSE_log: {mse_te:.6f}, R2_log: {r2_te:.6f}")

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss,   label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (log10 Nf, with monotonic penalty)")
    plt.title("Training and Validation Loss Curves")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Save X scaler
    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")

    ## save results
    # ------------------------
rows = []
for split_name, Xs, ys in [("Train", X_train, y_train),
                           ("Val",   X_val,   y_val),
                           ("Test",  X_test,  y_test)]:
    with torch.no_grad():
        y_pred_log = trained_model(Xs).cpu().numpy().ravel()
    y_true_log = ys.cpu().numpy().ravel()

    df_split = pd.DataFrame({
        "split":          split_name,
        "y_true_log10":   y_true_log,
        "y_pred_log10":   y_pred_log,
        "y_true":         10.0**y_true_log,
        "y_pred":         10.0**y_pred_log,
    })
    rows.append(df_split)

pred_all = pd.concat(rows, ignore_index=True)
pred_all.to_csv("predictions_KNN_aug.csv", index=False)

