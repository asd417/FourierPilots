import os
import json
import numpy as np
import re
from collections import defaultdict

def parse_filename(fname):
    """
    Example filename:
    timed_bce_cfg1_h8_lr0.001_b25_bs32_rep5.json
    """
    pattern = r"cfg(\d+)_h(\d+)_lr([0-9.]+)_b(\d+)"
    m = re.search(pattern, fname)
    if not m:
        raise ValueError(f"Could not parse: {fname}")

    cfg, h, lr, b = m.groups()
    return {
        "cfg": int(cfg),
        "h": int(h),
        "lr": float(lr),
        "b": int(b)
    }
def collect_final_metrics(directory):
    # data[model]["h"][h_value] -> {acc: [...], val_loss: [...]}
    # data[model]["lr"][lr_value] -> {acc: [...], val_loss: [...]}
    # data[model]["b"][b_value] -> {acc: [...], val_loss: [...]}
    data = defaultdict(
        lambda: {
            "h":  defaultdict(lambda: {"acc": [], "val_loss": []}),
            "lr": defaultdict(lambda: {"acc": [], "val_loss": []}),
            "b":  defaultdict(lambda: {"acc": [], "val_loss": []}),
        }
    )

    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue

        meta = parse_filename(fname)
        if meta is None:
            continue

        h  = meta["h"]
        lr = meta["lr"]
        b  = meta["b"]

        path = os.path.join(directory, fname)
        with open(path, "r") as f:
            d = json.load(f)

        for model, metrics in d.items():
            final_acc  = metrics["acc"][-1]
            final_loss = metrics["val_loss"][-1]

            # group by h
            data[model]["h"][h]["acc"].append(final_acc)
            data[model]["h"][h]["val_loss"].append(final_loss)

            # group by lr
            data[model]["lr"][lr]["acc"].append(final_acc)
            data[model]["lr"][lr]["val_loss"].append(final_loss)

            # group by b
            data[model]["b"][b]["acc"].append(final_acc)
            data[model]["b"][b]["val_loss"].append(final_loss)

    return data

def hidden_layer_trend(data, path):
    import numpy as np
    import matplotlib.pyplot as plt
    # --------------------------
    # Final Accuracy (all models)
    # --------------------------
    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["h"].keys())
        final_acc = [np.mean(h_dict["h"][h]["acc"]) for h in hs]
        plt.plot(hs, final_acc, marker="o", label=model)

    plt.title("Final Accuracy vs Hidden Size (h)")
    plt.xlabel("h")
    plt.ylabel("Final Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/acc_vs_h.png")
    plt.close()

    # --------------------------
    # Final Validation Loss (all models)
    # --------------------------
    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["h"].keys())
        final_loss = [np.mean(h_dict["h"][h]["val_loss"]) for h in hs]
        plt.plot(hs, final_loss, marker="o", label=model)

    plt.title("Final Validation Loss vs Hidden Size (h)")
    plt.xlabel("h")
    plt.ylabel("Final Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/val_loss_vs_h.png")
    plt.close()

    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["lr"].keys())
        final_acc = [np.mean(h_dict["lr"][lr]["acc"]) for lr in hs]
        plt.plot(hs, final_acc, marker="o", label=model)

    plt.title("Final Accuracy vs Learning Rate (lr)")
    plt.xlabel("lr")
    plt.ylabel("Final Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/acc_vs_lr.png")
    plt.close()

    # --------------------------
    # Final Validation Loss (all models)
    # --------------------------
    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["lr"].keys())
        final_loss = [np.mean(h_dict["lr"][lr]["val_loss"]) for lr in hs]
        plt.plot(hs, final_loss, marker="o", label=model)

    plt.title("Final Validation Loss vs Learning Rate (lr)")
    plt.xlabel("lr")
    plt.ylabel("Final Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/val_loss_vs_lr.png")
    plt.close()

    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["b"].keys())
        final_acc = [np.mean(h_dict["b"][lr]["acc"]) for lr in hs]
        plt.plot(hs, final_acc, marker="o", label=model)

    plt.title("Final Accuracy vs Data Size")
    plt.xlabel("Data Size")
    plt.ylabel("Final Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/acc_vs_b.png")
    plt.close()

    # --------------------------
    # Final Validation Loss (all models)
    # --------------------------
    plt.figure(figsize=(9, 5))

    for model, h_dict in data.items():
        hs = sorted(h_dict["b"].keys())
        final_loss = [np.mean(h_dict["b"][lr]["val_loss"]) for lr in hs]
        plt.plot(hs, final_loss, marker="o", label=model)

    plt.title("Final Validation Loss vs Data Size")
    plt.xlabel("Data Size")
    plt.ylabel("Final Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/val_loss_vs_b.png")
    plt.close()

    print("Saved combined trend plots.")

def average_time_series_overall(directory):
    # model -> {metric -> [arrays]}
    storage = {}

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(directory, file), "r") as f:
            data = json.load(f)

        for model, metrics in data.items():
            if model not in storage:
                storage[model] = {"acc": [], "val_loss": []}

            storage[model]["acc"].append(np.array(metrics["acc"]))
            storage[model]["val_loss"].append(np.array(metrics["val_loss"]))

    # Compute horizontal (element-wise) average
    results = {}
    for model, metrics in storage.items():
        acc_stack = np.vstack(metrics["acc"])          # shape (N_files, T)
        loss_stack = np.vstack(metrics["val_loss"])    # shape (N_files, T)

        results[model] = {
            "acc": acc_stack.mean(axis=0).tolist(),        # avg at each time step
            "val_loss": loss_stack.mean(axis=0).tolist()
        }

    return results

import matplotlib.pyplot as plt

def plot_results_overall(results, path):
    # results structure:
    # { model_name: { "acc": [...], "val_loss": [...] } }

    # ----- Plot ACC -----
    plt.figure(figsize=(8, 5))
    for model, metrics in results.items():
        plt.plot(metrics["acc"], label=model)
    plt.title("Accuracy (Averaged)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/acc_overall.png")

    # ----- Plot VAL LOSS -----
    plt.figure(figsize=(8, 5))
    for model, metrics in results.items():
        plt.plot(metrics["val_loss"], label=model)
    plt.title("Validation Loss (Averaged)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/val_loss_overall.png")

results = average_time_series_overall("plots_timed/data")
plot_results_overall(results, "plots_timed")

results = collect_final_metrics("plots_timed/data")
hidden_layer_trend(results, "plots_timed")

results = average_time_series_overall("plots_independent/data")
plot_results_overall(results, "plots_independent")

results = collect_final_metrics("plots_independent/data")
hidden_layer_trend(results, "plots_independent")

