import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch_dct as dct
import matplotlib.pyplot as plt

class MLPCosine(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = dct.dct(x)
        print(X)
        return self.net(X)

class MLPFourier(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = torch.fft.fft(x,norm="ortho")          
        return self.net(X.real)
    
class MLPFourier(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = torch.fft.fft(x,norm="ortho")
        print(X.real)                
        return self.net(X.real)

class MLPFourierTime(nn.Module):
    def __init__(self, past, feature_dim, hidden=64, out_dim=2):
        super().__init__()
        self.past = past
        self.feat_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(past * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )

    def forward(self, x : torch.Tensor):
        # 1) UNFLATTEN
        x = x.view(x.shape[0], self.past, self.feat_dim)
        x = torch.fft.fft(x,dim=1).real
        x = x.reshape(x.shape[0], self.past * self.feat_dim)
        # 3) FLATTEN BACK and feedforward
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): return self.net(x)

class MLPTime(nn.Module):
    def __init__(self, past, features, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(past * features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): return self.net(x)


def load_dataset(path: str, cut=4000, seed: int = 0):
    # whitespace-separated; supports one or many rows
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    # reproducible shuffling
    rng = np.random.default_rng(seed)
    rng.shuffle(arr, axis=0)

    X = arr[:cut, :-2]   # inputs
    Y = arr[:cut, -2:]   # two independent targets in [0,1]

    print(f"Shuffled with seed={seed}, using first {cut} rows from {path}")
    return X, Y

def _build_windows_for_block(block_arr: np.ndarray, past: int):
    ins  = block_arr[:, :-2]   # (T, D_in)
    outs = block_arr[:, -2:]   # (T, 2)
    T, D_in = ins.shape

    pad = np.zeros((past, D_in), dtype=ins.dtype)
    padded = np.vstack([pad, ins])  # shape (T + past, D_in)

    # For each i, we want indices [i, i-1, ..., i-past]
    # shifted by +past to account for padding
    base_idx = np.arange(T) + past         # (T,)
    offsets  = np.arange(past)         # (past+1,)
    idx = base_idx[:, None] - offsets[None, :]  # (T, past+1)

    # Gather windows: X_block[i] = [ins[i], ins[i-1], ..., ins[i-past]]
    X_block = padded[idx]                  # (T, past+1, D_in)

    # Label at the current timestep i
    Y_block = outs                         # (T, 2)

    return X_block, Y_block

def load_time_series_dataset(path: str, past=5,blocks = 1000):
    X_blocks = []
    Y_blocks = []
    current = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # skip empty lines

            if line == "newlife":
                if current:
                    block_arr = np.stack(current).astype(np.float32)
                    X_b, Y_b = _build_windows_for_block(block_arr, past)
                    X_blocks.append(X_b)
                    Y_blocks.append(Y_b)
                    current = []
                    if len(X_blocks) >= blocks:
                        break
            else:
                # parse one data line: arbitrary number of floats, last 2 are outs
                vals = np.fromstring(line, sep=" ", dtype=np.float32)
                current.append(vals)

    # Handle last block if file doesn't end with "newlife"
    if current:
        block_arr = np.stack(current).astype(np.float32)
        X_b, Y_b = _build_windows_for_block(block_arr, past)
        X_blocks.append(X_b)
        Y_blocks.append(Y_b)

    # Concatenate all blocks
    X = np.concatenate(X_blocks, axis=0)   # (N_total, past+1, D_in)
    Y = np.concatenate(Y_blocks, axis=0)   # (N_total, 2)
    return X, Y

def make_loaders(X, Y, batch_size=64, val_frac=0.2, seed=0):
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    ds = TensorDataset(X_t, Y_t)

    gen = torch.Generator().manual_seed(seed)
    n = len(ds); n_val = int(n * val_frac); n_tr = max(n - n_val, 1)
    if n_val and n_tr < 1: n_val, n_tr = 0, n  # handle tiny datasets
    tr, va = random_split(ds, [n_tr, n_val], generator=gen) if n_val else (ds, None)

    # z-score inputs using train split only
    X_train = X_t[tr.indices] if isinstance(tr, torch.utils.data.Subset) else X_t
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_t.sub_(mean).div_(std)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False) if va else None
    return train_loader, val_loader, {"mean": mean, "std": std}

def compute_accuracy(model, loader, device=None, threshold=0.5):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = torch.sigmoid(model(xb))
            preds = (probs >= threshold).float()
            correct += (preds == yb).float().sum().item()
            total += yb.numel()
    return correct / total

def train_independent_bce(
    path: str,
    epochs=100,
    hidden=64,
    lr=1e-3,
    batch_size=64,
    seed=0,
    device=None,
    pos_weight=None,  # e.g., tensor([w0, w1]) for class imbalance
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    X, Y = load_dataset(path, seed=42)
    in_dim = X.shape[1]

    train_loader, val_loader, scaler = make_loaders(X, Y, batch_size=batch_size, seed=seed)

    models = [
        #MLPFourier, 
        MLPCosine, 
        #MLP
        ]
    for m in models:
        model = m(in_dim, hidden=hidden, out_dim=2).to(device)
        print(f"Trying model: {m.__name__}")
        #model = MLPFourier(in_dim, hidden=hidden, out_dim=2).to(device)
        #model = MLP(in_dim, hidden=hidden, out_dim=2).to(device)

        # BCE with logits supports soft targets in [0,1] and independent outputs
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def eval_loss(loader):
            if loader is None: return float("nan")
            model.eval(); total = n = 0
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    total += loss.item() * xb.size(0); n += xb.size(0)
            return total / max(n, 1)

        final_val = 0
        final_acc = 0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                acc = compute_accuracy(model, val_loader)
                print(f"Epoch {ep:3d}  val_loss: {eval_loss(val_loader):.4f} accuracy: {acc:.4f}")
                final_val = eval_loss(val_loader)
                final_acc = acc

        model.eval()
        print(f"Finished training model: {m.__name__} with val_loss: {final_val:.4f} accuracy: {final_acc:.4f}")

def train_timed_bce(
    path: str,
    epochs=100,
    past=5,
    hidden=64,
    lr=1e-3,
    batch_size=64,
    seed=0,
    blocks=100,
    device=None,
    pos_weight=None,  # e.g., tensor([w0, w1]) for class imbalance
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    X, Y = load_time_series_dataset(path, past=past,blocks=blocks)
    features = X.shape[2]
    X = X.reshape(X.shape[0], -1) # flatten time windows
    print(X.shape)
    train_loader, val_loader, scaler = make_loaders(X, Y, batch_size=batch_size, seed=seed)

    models = [
        MLPFourierTime,
        MLPTime
        ]
    
    all_histories = {}

    for m in models:
        model = m(past, features, hidden=hidden, out_dim=2).to(device)
        model_name = m.__name__
        val_losses = []
        val_accs = []
        print(f"Training model: {m.__name__}")
        #model = MLPFourier(in_dim, hidden=hidden, out_dim=2).to(device)
        #model = MLP(in_dim, hidden=hidden, out_dim=2).to(device)

        # BCE with logits supports soft targets in [0,1] and independent outputs
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def eval_loss(loader):
            if loader is None: return float("nan")
            model.eval(); total = n = 0
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    total += loss.item() * xb.size(0); n += xb.size(0)
            return total / max(n, 1)

        final_val = 0
        final_acc = 0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                acc = compute_accuracy(model, val_loader)
                print(f"Epoch {ep:3d}  val_loss: {eval_loss(val_loader):.4f} accuracy: {acc:.4f}")
                final_val = eval_loss(val_loader)
                final_acc = acc
                val_losses.append(final_val)
                val_accs.append(acc)

        model.eval()
        print(f"Finished training model: {m.__name__} with val_loss: {final_val:.4f} accuracy: {final_acc:.4f}")
        all_histories[model_name] = {
            "val_loss": val_losses,
            "acc": val_accs,
        }
    return all_histories

# ---------- Inference ----------
def predict_probs(model, scaler, x_row, device=None, threshold=0.5):
    """Returns independent probabilities and binary decisions per head."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(x_row, dtype=torch.float32)[None, :]
    mean, std = scaler["mean"], scaler["std"]
    x = (x - mean) / std
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # two numbers in [0,1]
    decisions = (probs >= threshold).astype(np.int32)
    return probs, decisions

def validate_dataset_generation():
    past = 5
    features = 8
    X, Y = load_time_series_dataset('out_data_id0.log', past=past)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    print("shape:",Xt.shape)
    N = 6
    print(Xt[N])
    print(Xt[N][:,4:5])
    print(Xt.shape[1]) # == past+1
    print(Xt[N].flatten())
    B = Xt[N].flatten().size(0)
    # 1) UNFLATTEN
    XtB = Xt[N].view(past, features)
    print(XtB)
    print(torch.fft.fft(Xt[N],dim=0).real)
    #print(torch.fft.fft(Xt[N][:,4:5],dim=0).real)
    print(torch.fft.fft(XtB,dim=0).real)

#validate_dataset_generation()
#train_independent_bce("out_lines.log", epochs=1, hidden=64, lr=1e-3)
#train_timed_bce("out_data_id0.log", epochs=10, hidden=10, lr=1e-2, blocks=100)

def sweep_timed_bce(path: str, plt_path="./"):
    
    # Search space
    hidden_sizes = [8, 16, 32, 64]
    lrs          = [1e-3, 3e-3, 1e-2]
    blocks_list  = [25, 50, 100, 200]
    batch_sizes  = [32, 64, 128]

    cfg_id = 0
    for hidden, lr, blocks, batch_size in itertools.product(
        hidden_sizes, lrs, blocks_list, batch_sizes
    ):
        cfg_id += 1
        print(f"\n=== Config: hidden={hidden}, lr={lr}, blocks={blocks}, batch_size={batch_size} ===")

        histories = train_timed_bce(
            path,
            epochs=10,          # keep small for sweep
            hidden=hidden,
            lr=lr,
            blocks=blocks,
            # LOW CONFIDENCE: only if your function accepts batch_size/seed/device
            batch_size=batch_size,
            seed=0,
        )
        # ------------------------------------------------
        # Plot ALL models for THIS config (accuracy)
        # ------------------------------------------------
        plt.figure(figsize=(7, 4))
        for model_name, h in histories.items():
            acc = h["acc"]
            plt.plot(acc, label=model_name)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Config #{cfg_id}  h={hidden}, lr={lr}, blocks={blocks}, bs={batch_size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # save one PNG per config (so we don’t pop up 10^n windows)
        fname = f"{plt_path}timed_bce_cfg{cfg_id}_h{hidden}_lr{lr}_b{blocks}_bs{batch_size}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved plot to {fname}")


sweep_timed_bce("out_data_id0.log", plt_path="plots/")