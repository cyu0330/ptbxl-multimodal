# scripts/12_grad_cam_ecg_multimodal.py

import os
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset
from src.models.ecg_multimodal import ECGMultimodal


class GradCAM1D_ECGMultimodal:
    """Grad-CAM on the last conv layer of the ECG backbone."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(self, x_ecg, x_demo, class_idx, signal_length):
        """
        x_ecg:  [1, 12, T]
        x_demo: [1, 5]
        class_idx: target class index
        signal_length: original ECG length T
        """
        self.model.zero_grad()

        logits = self.model(x_ecg, x_demo)
        score = logits[:, class_idx].sum()
        score.backward()

        acts = self.activations      # [1, C, T']
        grads = self.gradients       # [1, C, T']

        weights = grads.mean(dim=-1, keepdim=True)  # [1, C, 1]

        cam = (weights * acts).sum(dim=1)           # [1, T']
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)                                 # [1, T]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0].cpu()


def compute_demo_importance(model, x_ecg, x_demo, class_idx):
    """
    Gradient * input on the demographic vector.
    Returns a length-5 numpy array normalised to [0, 1].
    """
    model.zero_grad()

    x_demo = x_demo.clone().detach().requires_grad_(True)

    logits = model(x_ecg, x_demo)
    score = logits[:, class_idx].sum()
    score.backward()

    grad = x_demo.grad[0].detach().cpu().numpy()
    val = x_demo.detach()[0].cpu().numpy()

    importance = np.abs(grad * val)
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def plot_ecg_and_demo_importance(
    ecg,
    cam,
    demo_importance,
    demo_feature_names,
    lead_idx,
    title,
    save_path,
):
    """
    ecg: [12, T]
    cam: [T]
    demo_importance: [D]
    demo_feature_names: list of length D
    """
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    sig = ecg[lead_idx]
    T = sig.shape[-1]
    t = np.arange(T)

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])

    cam_2d = np.expand_dims(cam, axis=0)
    ax1.imshow(
        cam_2d,
        aspect="auto",
        cmap="Reds",
        alpha=0.7,
        extent=[0, T, sig.min(), sig.max()],
        origin="lower",
        interpolation="bilinear",
    )
    ax1.plot(t, sig, color="black", linewidth=0.8)

    ax1.set_title(title)
    ax1.set_ylabel(f"ECG (lead {lead_idx})")

    ax2 = fig.add_subplot(gs[1, 0])

    y_pos = np.arange(len(demo_importance))
    ax2.barh(y_pos, demo_importance, color="salmon")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(demo_feature_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Relative importance")
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved Grad-CAM figure to: {save_path}")


def load_multimodal_model(cfg, ckpt_path, device):
    data_cfg = cfg["data"]
    model_cfg_all = cfg.get("model", {})
    model_cfg = model_cfg_all.get("ecg_multimodal", model_cfg_all.get("ecg_demo", {}))

    classes = data_cfg["labels"]
    in_leads = data_cfg.get("leads", 12)

    model = ECGMultimodal(
        in_leads=in_leads,
        num_labels=len(classes),
        ecg_feat_dim=model_cfg.get("ecg_feat_dim", 256),
        demo_hidden_dim=model_cfg.get(
            "demo_hidden_dim", model_cfg.get("demo_feat_dim", 64)
        ),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] Model loaded. missing:", missing)
    print("[INFO] unexpected:", unexpected)

    model.to(device)
    model.eval()
    return model


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    data_cfg = cfg["data"]
    base_dir = data_cfg["base_dir"]
    classes = data_cfg["labels"]

    test_ds = PTBXLECGMultimodalDataset(
        base_dir=base_dir,
        split="test",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    print("[INFO] ECG-MM test size:", len(test_ds))
    print("[INFO] Classes:", classes)

    model = load_multimodal_model(cfg, args.ckpt, device)

    target_layer = model.ecg_backbone.backbone[-1].net[0]
    grad_cam = GradCAM1D_ECGMultimodal(model, target_layer)

    idx = args.index
    x_ecg, x_demo, y = test_ds[idx]
    signal_length = x_ecg.shape[-1]

    x_ecg_t = x_ecg.unsqueeze(0).to(device)
    x_demo_t = x_demo.unsqueeze(0).to(device)

    if args.class_name:
        class_name = args.class_name
        class_idx = classes.index(class_name)
    else:
        class_idx = args.class_idx
        class_name = classes[class_idx]

    print(f"[INFO] Grad-CAM on sample {idx}, class {class_name}")

    cam = grad_cam.generate_cam(
        x_ecg=x_ecg_t,
        x_demo=x_demo_t,
        class_idx=class_idx,
        signal_length=signal_length,
    )

    demo_importance = compute_demo_importance(
        model,
        x_ecg_t,
        x_demo_t,
        class_idx=class_idx,
    )
    demo_feature_names = ["age", "sex", "height", "weight", "pacemaker"]

    out_dir = "outputs/gradcam_multimodal"
    os.makedirs(out_dir, exist_ok=True)

    cam_path = os.path.join(out_dir, f"sample_{idx}_{class_name}_cam.npy")
    np.save(cam_path, cam.numpy())
    print("[INFO] Saved CAM to:", cam_path)

    fig_path = os.path.join(out_dir, f"sample_{idx}_{class_name}_ecg_mm.png")
    plot_ecg_and_demo_importance(
        ecg=x_ecg,
        cam=cam,
        demo_importance=demo_importance,
        demo_feature_names=demo_feature_names,
        lead_idx=args.lead,
        title=f"ECG multimodal Grad-CAM | sample {idx} | class {class_name}",
        save_path=fig_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ecg_multimodal.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # change this to your own best checkpoint path
        default="outputs/ecg_multimodal/ckpts/ecg_multimodal_best.pth",
    )
    parser.add_argument("--index", type=int, default=10)
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--class_name", type=str, default="MI")
    main(parser.parse_args())
