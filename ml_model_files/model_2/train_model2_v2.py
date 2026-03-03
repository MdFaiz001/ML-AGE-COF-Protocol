import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ---- Torch safe globals for PyG Data objects (only for loading .pt embeddings)
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
add_safe_globals([Data])

# =========================
# FILES (KEEP AS YOUR SETUP)
# =========================
META = "cof_meta_model2_flp_large_v2_slots.csv"
EMBED = "embed"

FEATURES = {
    "2c_fn": "features/2_con_fn_linker_feat_ms.csv",
    "2c_unfn": "features/2_con_unfn_linker_feat_ms.csv",
    "3c": "features/3_con_linker_feat_ms.csv",
    "4c": "features/4_con_linker_feat_ms.csv",
    "base": "features/base_feat_ms.csv",
}

# =========================
# TRAINING SETTINGS
# =========================
BATCH = 32
LR = 1e-3
MAX_EPOCHS = 3000
PATIENCE = 80
SEED = 42

# loss weights between the two heads
W_COUNT = 1.0
W_QAVG = 1.0

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def spearman_corr_numpy(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)

        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    rt = rankdata(y_true)
    rp = rankdata(y_pred)

    rt = rt - rt.mean()
    rp = rp - rp.mean()

    denom = np.sqrt((rt ** 2).sum()) * np.sqrt((rp ** 2).sum())
    if denom == 0:
        return 0.0
    return float((rt * rp).sum() / denom)


class COFModel2v3FactorizedDataset(Dataset):
    """
    Factorized targets:
      y_count = log1p(N_base_FLP_sim)
      y_qavg  = log1p(C_FLP_sim / (N_base_FLP_sim + eps))
    Reconstruct later: C_pred = expm1(y_count_pred) * expm1(y_qavg_pred)
    """

    def __init__(self, meta_csv, feature_paths, embed_dir, norm_stats=None, eps=1e-6):
        self.meta = pd.read_csv(meta_csv)
        self.embed_dir = embed_dir
        self.norm_stats = norm_stats
        self.eps = eps

        # feature tables
        self.fn2c = pd.read_csv(feature_paths["2c_fn"]).set_index("linker_id")
        self.unfn2c = pd.read_csv(feature_paths["2c_unfn"]).set_index("linker_id")
        self.l3c = pd.read_csv(feature_paths["3c"]).set_index("linker_id")
        self.l4c = pd.read_csv(feature_paths["4c"]).set_index("linker_id")
        self.base = pd.read_csv(feature_paths["base"]).set_index("linker_id")

        self.fn2c_cols = self.fn2c.columns.drop("name_x", errors="ignore")
        self.unfn2c_cols = self.unfn2c.columns.drop("name_x", errors="ignore")
        self.l3c_cols = self.l3c.columns.drop("name_x", errors="ignore")
        self.l4c_cols = self.l4c.columns.drop("name_x", errors="ignore")
        self.base_cols = self.base.columns.drop("name_x", errors="ignore")

        self.bridge_types = ["none", "ch2", "ph", "dir"]
        self.bridge_map = {b: i for i, b in enumerate(self.bridge_types)}

        required = [
            "topology_id",
            "coverage_fraction",
            "bridge_type",
            "edge_slots",
            "node_slots",
            "total_slots",
            "n_fn_edges_expected",
            "n_unfn_edges_expected",
            "node1_connectivity",
            "node2_connectivity",
            "node1_linker_id",
            "node2_linker_id",
            "parent_2c_id",
            "edge_fn_id",
            "base_id",
            "C_FLP_sim",
            "N_base_FLP_sim",
        ]
        missing = [c for c in required if c not in self.meta.columns]
        if missing:
            raise ValueError(f"Missing required columns in meta: {missing}")

    def __len__(self):
        return len(self.meta)

    def _get_vec(self, df, idx, cols):
        if int(idx) == -1:
            return torch.zeros(len(cols), dtype=torch.float32)
        series = df.loc[int(idx), cols]
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return torch.tensor(numeric.values, dtype=torch.float32)

    def _norm(self, x, key):
        if self.norm_stats is None:
            return x
        return (x - self.norm_stats[key]["mean"]) / self.norm_stats[key]["std"]

    def _load_topo(self, row):
        # Prefer explicit path if exists
        if "topology_pt_path" in self.meta.columns:
            p = str(row["topology_pt_path"])
            if p and os.path.exists(p):
                return torch.load(p, weights_only=False)

        topo_id = str(row["topology_id"])
        cand = os.path.join(self.embed_dir, f"{topo_id}.pt")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"Topology embedding not found: {cand}")
        return torch.load(cand, weights_only=False)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        topo_obj = self._load_topo(row)
        topo_vec = topo_obj.x.mean(dim=0).float() if hasattr(topo_obj, "x") else topo_obj.float()
        topo_vec = self._norm(topo_vec, "topo")

        # node average
        n1 = self._get_vec(
            self.l3c if row["node1_connectivity"] == 3 else self.l4c,
            row["node1_linker_id"],
            self.l3c_cols if row["node1_connectivity"] == 3 else self.l4c_cols,
        )
        n2 = self._get_vec(
            self.l3c if row["node2_connectivity"] == 3 else self.l4c,
            row["node2_linker_id"],
            self.l3c_cols if row["node2_connectivity"] == 3 else self.l4c_cols,
        )
        node_vec = 0.5 * (n1 + n2)
        node_vec = self._norm(node_vec, "node")

        # linkers concat
        parent_2c = self._get_vec(self.unfn2c, row["parent_2c_id"], self.unfn2c_cols)
        edge_fn = self._get_vec(self.fn2c, row["edge_fn_id"], self.fn2c_cols)
        linker_vec = torch.cat([parent_2c, edge_fn])
        linker_vec = self._norm(linker_vec, "linker")

        # base
        base_vec = self._get_vec(self.base, row["base_id"], self.base_cols)
        base_vec = self._norm(base_vec, "base")

        # misc numeric normalized + bridge onehot
        bridge = torch.zeros(len(self.bridge_types), dtype=torch.float32)
        bt = str(row["bridge_type"])
        if bt not in self.bridge_map:
            raise ValueError(f"Unknown bridge_type '{bt}'. Allowed: {self.bridge_types}")
        bridge[self.bridge_map[bt]] = 1.0

        misc_nums = torch.tensor(
            [
                float(row["coverage_fraction"]),
                float(row["edge_slots"]),
                float(row["n_fn_edges_expected"]),
                float(row["n_unfn_edges_expected"]),
                float(row["node_slots"]),
                float(row["total_slots"]),
            ],
            dtype=torch.float32,
        )
        misc_nums = self._norm(misc_nums, "misc_nums")
        misc_vec = torch.cat([misc_nums, bridge], dim=0)

        # factorized targets
        C = float(row["C_FLP_sim"])
        N = float(row["N_base_FLP_sim"])
        if C < 0: C = 0.0
        if N < 0: N = 0.0

        qavg = C / (N + self.eps)

        y_count = torch.tensor([np.log1p(N)], dtype=torch.float32)
        y_qavg = torch.tensor([np.log1p(qavg)], dtype=torch.float32)
        y_C = torch.tensor([np.log1p(C)], dtype=torch.float32)

        return {
            "topo": topo_vec,
            "node": node_vec,
            "linker": linker_vec,
            "base": base_vec,
            "misc": misc_vec,
            "y_count": y_count,
            "y_qavg": y_qavg,
            "y_C": y_C,
        }


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class COFModel2v3Factorized(nn.Module):
    def __init__(self, dims, hidden=160):
        super().__init__()

        self.topo_enc = MLP(dims["topo"], hidden)
        self.node_enc = MLP(dims["node"], hidden)
        self.linker_enc = MLP(dims["linker"], hidden)
        self.base_enc = MLP(dims["base"], hidden)
        self.misc_enc = MLP(dims["misc"], hidden // 2)

        self.topo_ln = nn.LayerNorm(hidden)
        self.node_ln = nn.LayerNorm(hidden)
        self.linker_ln = nn.LayerNorm(hidden)
        self.base_ln = nn.LayerNorm(hidden)
        self.misc_ln = nn.LayerNorm(hidden // 2)

        fused_dim = hidden * 4 + hidden // 2

        def head():
            return nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, 1),
                nn.Softplus(),  # ensures >=0 for log1p targets
            )

        self.head_count = head()
        self.head_qavg = head()

    def forward(self, b):
        topo = self.topo_ln(self.topo_enc(b["topo"]))
        node = self.node_ln(self.node_enc(b["node"]))
        linker = self.linker_ln(self.linker_enc(b["linker"]))
        base = self.base_ln(self.base_enc(b["base"]))
        misc = self.misc_ln(self.misc_enc(b["misc"]))

        x = torch.cat([topo, node, linker, base, misc], dim=1)
        return self.head_count(x), self.head_qavg(x)


def move_batch(b, device):
    out = {}
    for k, v in b.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def compute_norm_stats(dataset, train_indices):
    keys = ["topo", "node", "linker", "base"]
    sums = {k: None for k in keys}
    sqs = {k: None for k in keys}
    misc_sum = None
    misc_sq = None
    n = 0

    for i in train_indices:
        s = dataset[i]
        for k in keys:
            if sums[k] is None:
                sums[k] = torch.zeros_like(s[k])
                sqs[k] = torch.zeros_like(s[k])
            sums[k] += s[k]
            sqs[k] += s[k] ** 2

        misc_nums = s["misc"][:6]
        if misc_sum is None:
            misc_sum = torch.zeros_like(misc_nums)
            misc_sq = torch.zeros_like(misc_nums)
        misc_sum += misc_nums
        misc_sq += misc_nums ** 2
        n += 1

    out = {}
    for k in keys:
        mean = sums[k] / n
        var = sqs[k] / n - mean ** 2
        out[k] = {"mean": mean, "std": torch.sqrt(torch.clamp(var, 1e-12))}

    misc_mean = misc_sum / n
    misc_var = misc_sq / n - misc_mean ** 2
    out["misc_nums"] = {"mean": misc_mean, "std": torch.sqrt(torch.clamp(misc_var, 1e-12))}
    return out


# ==========================================================
# Build raw dataset (for split + stats)
# ==========================================================
raw = COFModel2v3FactorizedDataset(META, FEATURES, EMBED)

# GA-correct random split
idx_all = np.arange(len(raw))
rng = np.random.default_rng(SEED)
rng.shuffle(idx_all)

n = len(idx_all)
n_train = int(0.70 * n)
n_val = int(0.15 * n)

train_idx = idx_all[:n_train]
val_idx = idx_all[n_train:n_train + n_val]
test_idx = idx_all[n_train + n_val:]

print(f"Split sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

norm_stats = compute_norm_stats(raw, train_idx)

dataset = COFModel2v3FactorizedDataset(META, FEATURES, EMBED, norm_stats=norm_stats)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH)

sample = dataset[0]
dims = {k: sample[k].shape[0] for k in ["topo", "node", "linker", "base", "misc"]}

model = COFModel2v3Factorized(dims).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12, min_lr=1e-5)

loss_fn = torch.nn.SmoothL1Loss(beta=0.5)

best_val = float("inf")
best_state = None
pat = 0

print("\n=== Training start (factorized GA model: count + qavg) ===")
for ep in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_losses = []

    it = train_loader
    if TQDM:
        it = tqdm(train_loader, desc=f"Epoch {ep:04d} [train]", leave=False)

    for b in it:
        b = move_batch(b, DEVICE)
        opt.zero_grad()

        y_count_pred, y_qavg_pred = model(b)
        L = W_COUNT * loss_fn(y_count_pred, b["y_count"]) + W_QAVG * loss_fn(y_qavg_pred, b["y_qavg"])
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        tr_losses.append(float(L.detach().cpu()))
        if TQDM:
            it.set_postfix(loss=float(np.mean(tr_losses)))

    model.eval()
    va_losses = []
    with torch.no_grad():
        itv = val_loader
        if TQDM:
            itv = tqdm(val_loader, desc=f"Epoch {ep:04d} [val]", leave=False)
        for b in itv:
            b = move_batch(b, DEVICE)
            y_count_pred, y_qavg_pred = model(b)
            L = W_COUNT * loss_fn(y_count_pred, b["y_count"]) + W_QAVG * loss_fn(y_qavg_pred, b["y_qavg"])
            va_losses.append(float(L.detach().cpu()))
            if TQDM:
                itv.set_postfix(loss=float(np.mean(va_losses)))

    tr = float(np.mean(tr_losses))
    va = float(np.mean(va_losses))
    lr_now = opt.param_groups[0]["lr"]
    print(f"Epoch {ep:04d} | train {tr:.6f} | val {va:.6f} | lr {lr_now:.2e} | pat {pat}/{PATIENCE}")

    old_lr = lr_now
    scheduler.step(va)
    new_lr = opt.param_groups[0]["lr"]
    if new_lr != old_lr:
        print(f"  ↳ LR reduced to {new_lr:.2e}")

    if va < best_val:
        best_val = va
        pat = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping triggered")
            break

model.load_state_dict(best_state)
model.eval()

# ==========================================================
# TEST: reconstruct C_sim on RAW scale
# ==========================================================
y_true_raw, y_pred_raw = [], []
with torch.no_grad():
    for b in test_loader:
        b = move_batch(b, DEVICE)
        y_count_pred, y_qavg_pred = model(b)

        N_pred = np.expm1(y_count_pred.detach().cpu().numpy().ravel())
        q_pred = np.expm1(y_qavg_pred.detach().cpu().numpy().ravel())
        C_pred = N_pred * q_pred

        C_true = np.expm1(b["y_C"].detach().cpu().numpy().ravel())

        y_true_raw.append(C_true)
        y_pred_raw.append(C_pred)

y_true_raw = np.concatenate(y_true_raw)
y_pred_raw = np.concatenate(y_pred_raw)

mae = float(mean_absolute_error(y_true_raw, y_pred_raw))
r2 = float(r2_score(y_true_raw, y_pred_raw))
rho = float(spearman_corr_numpy(y_true_raw, y_pred_raw))

print("\n===== TEST METRICS (Model-2 v3 Factorized | GA split) =====")
print("MAE        :", mae)
print("R2         :", r2)
print("Spearman ρ :", rho)

np.save("model2_v3_ga_flp_large_y_true.npy", y_true_raw)
np.save("model2_v3_ga_flp_large_y_pred.npy", y_pred_raw)

torch.save(
    {
        "model_state_dict": best_state,
        "norm_stats": norm_stats,
        "dims": dims,
        "meta": META,
        "features": FEATURES,
        "metrics": {"MAE_raw": mae, "R2_raw": r2, "Spearman_rho": rho},
        "split": {
            "seed": SEED,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "split_type": "random_ga",
        },
        "notes": "Factorized model: predicts log1p(N_sim) and log1p(Q_avg_sim). Reconstructs C_sim=N*q.",
    },
    "model2_v3_ga_flp_large_final.pt",
)

print("\n✅ Saved model2_v3_ga_flp_large_final.pt")
print("Saved: model2_v3_ga_flp_large_y_true.npy, model2_v3_ga_flp_large_y_pred.npy")

