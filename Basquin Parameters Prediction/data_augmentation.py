import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon

INPUT_CSV = "SN_train.csv"
OUTPUT_CSV = "SN_full_augmented_1.csv"
FIG_KNN = "SN_KNN_neighbors_visualization_v3.png"
FIG_EC_JS = "SN_EC_vs_JS_divergence_v3.png"
FIG_CONTOUR = "SN_loga_b_distribution_v3.png"

MIN_A = 1e-3
MAX_A = 1e6
MIN_B = -0.5
MAX_B = 0.5

N_NEIGHBORS = 3
N_BINS = 4
EC_CANDIDATES = np.linspace(0.1, 0.6, 6)


df = pd.read_csv(INPUT_CSV)
print(f"Loaded dataset: {df.shape[0]} samples, columns: {list(df.columns)}")

assert all(c in df.columns for c in ["log_a", "b"]), "Input CSV must contain 'log_a' and 'b'."

continuous_cols = ["log_a", "b"]
categorical_cols = [c for c in df.columns if c not in continuous_cols]

X_cont = df[continuous_cols].copy()
X_cat = df[categorical_cols].copy() if len(categorical_cols) else pd.DataFrame(index=df.index)

def visualize_knn(X_cont, n_neighbors=3, save_path=FIG_KNN):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_cont)
    _, indices = nbrs.kneighbors(X_cont)
    center_idx = len(X_cont) // 2
    neigh_idx = indices[center_idx]

    plt.figure(figsize=(7,6))
    plt.scatter(X_cont["log_a"], X_cont["b"], alpha=0.4, label="All Data")
    plt.scatter(X_cont.iloc[neigh_idx]["log_a"], X_cont.iloc[neigh_idx]["b"],
                color="orange", label=f"{n_neighbors}-Nearest Neighbors", s=60)
    plt.scatter(X_cont.iloc[center_idx]["log_a"], X_cont.iloc[center_idx]["b"],
                color="red", marker="x", s=80, label="Center Sample")
    plt.xlabel("log_a"); plt.ylabel("b")
    plt.title(f"KNN Visualization (n_neighbors={n_neighbors})")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved KNN visualization to {save_path}")

visualize_knn(X_cont, n_neighbors=N_NEIGHBORS)

def js_divergence(p, q, bins=30):
    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist = np.clip(p_hist, 1e-12, None)
    q_hist = np.clip(q_hist, 1e-12, None)
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return jensenshannon(p_hist, q_hist)

def augment_dataset_local_bidirectional_v2(X_cont, X_cat, EC=0.3, n_neighbors=3, n_bins=4):

    X = X_cont.copy()
    X["bin_a"] = pd.qcut(X["log_a"], n_bins, duplicates="drop")
    X["bin_b"] = pd.qcut(X["b"], n_bins, duplicates="drop")
    X["bin2d"] = list(zip(X["bin_a"], X["bin_b"]))

    Xcat = X_cat.reset_index(drop=True) if not X_cat.empty else pd.DataFrame(index=X.index)
    augmented_rows = []

    for bin2d in X["bin2d"].unique():
        sub_idx = X["bin2d"] == bin2d
        sub_cont = X_cont[sub_idx].copy()
        sub_cat = Xcat[sub_idx].copy()
        if len(sub_cont) < 2:
            continue

        k_eff = min(n_neighbors, len(sub_cont))
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(sub_cont)
        _, indices = nbrs.kneighbors(sub_cont)

        sub_vals = sub_cont.values
        local_sigma = np.zeros_like(sub_vals)
        for i, neigh_idx in enumerate(indices):
            local_sigma[i] = sub_cont.iloc[neigh_idx].std(ddof=0).values

        for i in range(len(sub_cont)):
            base = sub_cont.iloc[i]
            base_cat = sub_cat.iloc[i] if not sub_cat.empty else pd.Series(dtype=float)
            sigma_local = pd.Series(local_sigma[i], index=sub_cont.columns)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_cont = base.copy()
                    new_cont["log_a"] = base["log_a"] + dx * EC * sigma_local["log_a"]
                    new_cont["b"]      = base["b"] + dy * EC * sigma_local["b"]

                    new_cont["log_a"] = np.clip(new_cont["log_a"], np.log10(MIN_A), np.log10(MAX_A))
                    new_cont["b"] = np.clip(new_cont["b"], MIN_B, MAX_B)

                    new_row = pd.concat([new_cont, base_cat])
                    augmented_rows.append(new_row)

    if len(augmented_rows) == 0:
        return pd.DataFrame(columns=list(X_cont.columns) + list(X_cat.columns))
    return pd.DataFrame(augmented_rows)

results = []
for EC in EC_CANDIDATES:
    aug_df = augment_dataset_local_bidirectional_v2(X_cont, X_cat, EC=EC,
                                                    n_neighbors=N_NEIGHBORS, n_bins=N_BINS)
    combined = pd.concat([pd.concat([X_cont, X_cat], axis=1), aug_df], ignore_index=True)
    js_scores = []
    for col in ["log_a", "b"]:
        js = js_divergence(X_cont[col], combined[col])
        js_scores.append(js)
    mean_js = np.mean(js_scores)
    results.append((EC, mean_js))
    print(f"EC={EC:.2f}, Mean JS={mean_js:.6f}")

results_df = pd.DataFrame(results, columns=["EC", "Mean_JS"]).sort_values("Mean_JS")
best_EC = results_df.iloc[0]["EC"]
print(f"\n Best EC (lowest mean JS): {best_EC:.3f}")

plt.figure(figsize=(7,5))
plt.plot(results_df["EC"], results_df["Mean_JS"], marker="o")
plt.xlabel("Expansion Coefficient (EC)")
plt.ylabel("Mean JS Divergence (lower = better)")
plt.title("Optimal EC Selection for SN Parameters (2D Partition + KNN=3)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EC_JS, dpi=300)
plt.show()

aug_best = augment_dataset_local_bidirectional_v2(
    X_cont, X_cat, EC=best_EC, n_neighbors=N_NEIGHBORS, n_bins=N_BINS
)
augmented = pd.concat([pd.concat([X_cont, X_cat], axis=1), aug_best],
                      ignore_index=True).drop_duplicates().reset_index(drop=True)

print(f"Original: {len(df)}  |  Augmented: {len(augmented)}")

plt.figure(figsize=(7,6))
plt.scatter(df["log_a"], df["b"], alpha=0.7, label="Original", s=40)
plt.scatter(augmented["log_a"], augmented["b"], alpha=0.6, label=f"Augmented (EC={best_EC:.2f})", s=40, marker="x")
plt.xlabel("log_a"); plt.ylabel("b")
plt.title("Original vs Augmented S–N Parameters (2D Partition + KNN=3)")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout(); plt.savefig(FIG_CONTOUR, dpi=300)
plt.show()

augmented.to_csv(OUTPUT_CSV, index=False)
print(f" Saved augmented dataset: {OUTPUT_CSV}")
