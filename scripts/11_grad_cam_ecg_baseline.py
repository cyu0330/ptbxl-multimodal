# scripts/11_grad_cam_ecg_baseline.py

import argparse
import os

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.ptbxl import PTBXLDataset
from src.interpretability.grad_cam_1d import GradCAM1D
from src.utils.seed import set_seed
from src.models.ecg_cnn import ECGCNN


def save_plot_ecg_with_cam_heatmap(ecg, cam, lead_idx, title, save_path):
    """Plot ECG with a red heatmap highlighting important regions."""
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam[cam < 0.2] = 0.0

    sig = ecg[lead_idx]
    T = sig.shape[-1]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(15, 4))
    cam_2d = np.expand_dims(cam, axis=0)

    ax.imshow(
        cam_2d,
        aspect="auto",
        cmap="Reds",
        alpha=0.7,
        extent=[0, T, sig.min(), sig.max()],
        origin="lower",
        interpolation="bilinear",
    )

    ax.plot(t, sig, color="black", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel(f"ECG (lead {lead_idx})")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[SAVE] Heatmap saved to: {save_path}")


def load_model_from_ckpt(cfg, ckpt_path, device):
    """Load baseline ECGCNN model from checkpoint."""
    data_cfg = cfg["data"]
    classes = data_cfg["labels"]
    in_leads = data_cfg.get("leads", 12)

    model = ECGCNN(in_leads=in_leads, feat_dim=256, num_labels=len(classes))

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] Model loaded. Missing:", missing)
    print("[INFO] Unexpected:", unexpected)

    model.to(device)
    model.eval()
    return model


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    out_dir = "outputs/gradcam"
    os.makedirs(out_dir, exist_ok=True)

    data_cfg = cfg["data"]
    classes = data_cfg["labels"]

    test_ds = PTBXLDataset(
        base_dir=data_cfg["base_dir"],
        split="test",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    print("[INFO] Test size:", len(test_ds))
    print("[INFO] Classes:", classes)

    model = load_model_from_ckpt(cfg, args.ckpt, device)

    target_layer = model.backbone[-1].net[0]
    grad_cam = GradCAM1D(model, target_layer)

    idx = args.index
    x, y = test_ds[idx]
    x_t = x.unsqueeze(0).to(device)
    signal_length = x.shape[-1]

    if args.class_name:
        class_name = args.class_name
        class_idx = classes.index(class_name)
    else:
        class_idx = args.class_idx
        class_name = classes[class_idx]

    print(f"[INFO] Running Grad-CAM on sample {idx}, class {class_name}")

    cam = grad_cam.generate_cam(
        input_tensor=x_t,
        class_idx=class_idx,
        signal_length=signal_length,
    )

    cam_save_path = os.path.join(out_dir, f"sample_{idx}_{class_name}_cam.npy")
    np.save(cam_save_path, cam.cpu().numpy())
    print(f"[SAVE] CAM saved to: {cam_save_path}")

    info_path = os.path.join(out_dir, f"sample_{idx}_{class_name}_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Sample index: {idx}\n")
        f.write(f"Class: {class_name}\n")
        f.write(f"Class idx: {class_idx}\n")
        f.write(f"ECG shape: {tuple(x.shape)}\n")
        f.write(f"CAM shape: {cam.shape}\n")
    print(f"[SAVE] Info saved to: {info_path}")

    plot_path = os.path.join(out_dir, f"sample_{idx}_{class_name}_plot.png")
    save_plot_ecg_with_cam_heatmap(
        ecg=x,
        cam=cam,
        lead_idx=args.lead,
        title=f"Grad-CAM | sample {idx} | class {class_name}",
        save_path=plot_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ecg_baseline.yaml")
    parser.add_argument("--ckpt", type=str,
                        default="outputs/ecg_baseline/ckpts/ecg_baseline_best.pth")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--class_name", type=str, default=None)

    main(parser.parse_args())
