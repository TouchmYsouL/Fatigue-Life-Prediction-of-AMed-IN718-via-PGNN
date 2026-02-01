# augment_in718_local_knn_builddir.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon

INPUT_CSV  = "IN718_full.csv"
OUTPUT_CSV = "IN718_full_augmented.csv"

FIG_KNN    = "KNN_neighbors_visualization.png"
FIG_EC_JS  = "EC_vs_JS_divergence.png"
FIG_SN     = "SN_loglog_KNN_local_bidirectional.png"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_LIFE = 1e3
MAX_LIFE = 1e9
LOG_MIN_LIFE = np.log10(MIN_LIFE)
LOG_MAX_LIFE = np.log10(MAX_LIFE)

N_NEIGHBORS   = 5
N_BINS        = 5
EC_CANDIDATES = np.linspace(0.1, 0.6, 6)

df = pd.read_csv(INPUT_CSV)

continuous_cols = ["fatigue_life", "stress_range", "YTS", "UTS"]

categorical_cols = [
    "geometry_type",
    "surface_condition",
    "post_process",
    "build_direction",
    "stress_ratio"
]

for col in ["fatigue_life", "stress_range"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

SR_MIN = max(1e-3, float(df["stress_range"].min()) * 0.5)
SR_MAX = float(df["stress_range"].max()) * 1.5

df["logNf"] = np.log10(df["fatigue_life"].clip(lower=MIN_LIFE))

X_cont = df[["logNf", "stress_range", "YTS", "UTS"]].copy()
X_cat = (
    df[categorical_cols].copy()
    if len(categorical_cols) and all(c in df.columns for c in categorical_cols)
    else pd.DataFrame(index=df.index)
)


def visualize_knn(X_cont, n_neighbors=5, save_path=FIG_KNN, center_strategy="median"):
    X2 = X_cont[["logNf", "stress_range"]].copy().reset_index(drop=True)

    if center_strategy == "median":
        center_idx = int((X2["logNf"] - X2["logNf"].median()).abs().idxmin())
    else:
        center_idx = len(X2) // 2

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X2)
    _, indices = nbrs.kneighbors(X2)
    neighbors_idx = indices[center_idx]

    plt.figure(figsize=(7, 6))
    plt.scatter(X2["logNf"], X2["stress_range"], alpha=0.45, label="All Data")
    plt.scatter(
        X2.iloc[neighbors_idx]["logNf"],
        X2.iloc[neighbors_idx]["stress_range"],
        label=f"{n_neighbors}-Nearest Neighbors",
        s=60,
    )
    plt.scatter(
        X2.loc[center_idx, "logNf"],
        X2.loc[center_idx, "stress_range"],
        label="Center Sample",
        s=80,
        marker="x",
    )
    plt.xlabel("log10(Fatigue Life)")
    plt.ylabel("Stress Range (MPa)")
    plt.title(f"KNN Visualization (n_neighbors={n_neighbors})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

visualize_knn(X_cont, n_neighbors=N_NEIGHBORS)

def js_divergence(p, q, bins=30):
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()

    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    p_hist = np.clip(p_hist, 1e-12, None)
    q_hist = np.clip(q_hist, 1e-12, None)

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    return float(jensenshannon(p_hist, q_hist))

def augment_dataset_local_bidirectional(
    X_cont,
    X_cat,
    EC=0.3,
    n_neighbors=5,
    n_bins=5,
    perturb_yts_uts=True,
):
    Xc = X_cont.copy().reset_index(drop=True)
    Xc["bin"] = pd.qcut(Xc["stress_range"], n_bins, duplicates="drop")

    Xcat = X_cat.reset_index(drop=True) if not X_cat.empty else pd.DataFrame(index=Xc.index)

    augmented_rows = []

    for b in Xc["bin"].unique():
        sub_idx = Xc["bin"] == b
        sub_cont = Xc.loc[sub_idx, ["logNf", "stress_range", "YTS", "UTS"]].copy()
        sub_cat = Xcat.loc[sub_idx].copy()

        if len(sub_cont) < n_neighbors:
            continue

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sub_cont)
        _, indices = nbrs.kneighbors(sub_cont)

        sub_vals = sub_cont.values
        local_sigma = np.zeros_like(sub_vals)

        for i, neigh_idx in enumerate(indices):
            local_sigma[i] = sub_cont.iloc[neigh_idx].std(ddof=0).values

        for i in range(len(sub_cont)):
            base = sub_cont.iloc[i]
            base_cat = sub_cat.iloc[i] if not sub_cat.empty else pd.Series(dtype=float)
            sigma_local = pd.Series(local_sigma[i], index=sub_cont.columns)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    new_cont = base.copy()

                    new_cont["logNf"] = base["logNf"] + dx * EC * sigma_local["logNf"]
                    new_cont["stress_range"] = base["stress_range"] + dy * EC * sigma_local["stress_range"]

                    if perturb_yts_uts:
                        for var in ["YTS", "UTS"]:
                            new_cont[var] = base[var] + EC * sigma_local[var] * (2 * np.random.rand() - 1)

                    new_cont["logNf"] = np.clip(new_cont["logNf"], LOG_MIN_LIFE, LOG_MAX_LIFE)
                    new_cont["stress_range"] = float(np.clip(new_cont["stress_range"], SR_MIN, SR_MAX))

                    new_row = pd.concat([new_cont, base_cat])
                    augmented_rows.append(new_row)

    if len(augmented_rows) == 0:
        return pd.DataFrame(columns=list(X_cont.columns) + list(X_cat.columns))

    return pd.DataFrame(augmented_rows)

results = []

for EC in EC_CANDIDATES:
    aug_df = augment_dataset_local_bidirectional(
        X_cont, X_cat, EC=EC,
        n_neighbors=N_NEIGHBORS,
        n_bins=N_BINS,
        perturb_yts_uts=True
    )

    combined = pd.concat(
        [pd.concat([X_cont, X_cat], axis=1), aug_df],
        ignore_index=True
    )

    js_scores = []
    for col in ["logNf", "stress_range", "YTS", "UTS"]:
        if col in X_cont.columns and col in combined.columns:
            js = js_divergence(X_cont[col].values, combined[col].values)
            js_scores.append(js)

    mean_js = float(np.mean(js_scores)) if js_scores else np.inf
    results.append((EC, mean_js))
    print(f"EC={EC:.2f} Mean JS Divergence={mean_js:.6f}")

results_df = pd.DataFrame(results, columns=["EC", "Mean_JS"]).sort_values("Mean_JS")
best_EC = float(results_df.iloc[0]["EC"])
print("\n Best EC:", best_EC)

plt.figure(figsize=(7, 5))
plt.plot(results_df["EC"], results_df["Mean_JS"], marker="o")
plt.xlabel("Expansion Coefficient (EC)")
plt.ylabel("Mean JS Divergence (lower = better)")
plt.title("Optimal EC Selection (Local KNN Bidirectional Augmentation)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EC_JS, dpi=300)
plt.show()


aug_best = augment_dataset_local_bidirectional(
    X_cont, X_cat, EC=best_EC,
    n_neighbors=N_NEIGHBORS,
    n_bins=N_BINS,
    perturb_yts_uts=True
)

base_full = pd.concat([X_cont, X_cat], axis=1).reset_index(drop=True)
augmented = pd.concat([base_full, aug_best], ignore_index=True)
augmented = augmented.drop_duplicates().reset_index(drop=True)

augmented["fatigue_life"] = np.power(10.0, augmented["logNf"])

min_life = float(augmented["fatigue_life"].min())
max_life = float(augmented["fatigue_life"].max())
min_sr   = float(augmented["stress_range"].min())
max_sr   = float(augmented["stress_range"].max())

print(f"\nMin fatigue_life (augmented): {min_life:.3e} (≥ {MIN_LIFE:.1e})")
print(f"Max fatigue_life (augmented): {max_life:.3e} (≤ {MAX_LIFE:.1e})")
print(f"Min stress_range (augmented): {min_sr:.3e} (≥ {SR_MIN:.3e})")
print(f"Max stress_range (augmented): {max_sr:.3e} (≤ {SR_MAX:.3e})")

augmented.to_csv(OUTPUT_CSV, index=False)
print(f" Save Data: {OUTPUT_CSV}")
print(f"Original samples: {len(df)}, Augmented samples: {len(augmented)}")

plt.figure(figsize=(7, 6))
plt.scatter(df["fatigue_life"], df["stress_range"], alpha=0.75, label="Original Data", s=36)
plt.scatter(
    augmented["fatigue_life"],
    augmented["stress_range"],
    alpha=0.6,
    label=f"Augmented Data (EC={best_EC:.2f})",
    s=36,
    marker="x",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Fatigue Life (cycles)")
plt.ylabel("Stress Range (MPa)")
plt.title("Original vs Augmented Data (log–log S–N Plot)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_SN, dpi=300)
plt.show()
