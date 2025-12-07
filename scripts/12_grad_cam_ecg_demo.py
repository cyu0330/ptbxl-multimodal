# scripts/12_grad_cam_ecg_demo.py
#
# 用于多模态 ECG+Demo 模型的 Grad-CAM 可视化：
#  - 上半部分：ECG + 时间 Grad-CAM（红色背景）
#  - 下半部分：每个 demo 特征对该类别 logit 的相对重要度

import os
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_demo import PTBXLECGDemoDataset
from src.models.ecg_demo import ECGDemo


# ----------------------------------------------------
# 1. Grad-CAM for ECG+Demo（针对 ECG backbone 的最后一层卷积）
# ----------------------------------------------------


class GradCAM1D_ECGDemo:
    """
    Grad-CAM for ECGDemo model.
    target_layer: e.g. model.ecg_backbone.backbone[-1].net[0]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            # out: [B, C, T']
            self.activations = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            # grad_output[0]: [B, C, T']
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        # 用 full_backward_hook，避免 FutureWarning
        self.hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(self, x_ecg, x_demo, class_idx, signal_length):
        """
        x_ecg:  [1, 12, T]
        x_demo: [1, 5]
        class_idx: int, 目标类别索引
        signal_length: 原始 ECG 长度 T
        """
        self.model.zero_grad()

        logits = self.model(x_ecg, x_demo)  # [1, num_labels]
        score = logits[:, class_idx].sum()
        score.backward()

        # activations / gradients: [1, C, T']
        acts = self.activations       # [B, C, T']
        grads = self.gradients        # [B, C, T']

        # GAP over time → 通道权重
        weights = grads.mean(dim=-1, keepdim=True)    # [B, C, 1]

        cam = (weights * acts).sum(dim=1)             # [B, T']
        cam = F.relu(cam)

        # 插值回原始长度 T
        cam = F.interpolate(
            cam.unsqueeze(1),          # [B, 1, T']
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)                   # [B, T]

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 返回 [T]，只取第一个样本
        return cam[0].cpu()


# ----------------------------------------------------
# 2. 计算 demo 特征的重要度（梯度 × 输入）
# ----------------------------------------------------


def compute_demo_importance(model, x_ecg, x_demo, class_idx):
    """
    x_ecg:  [1, 12, T]  (on device)
    x_demo: [1, 5]      (on device)
    return: importance: [5] numpy array (abs(grad * input)，归一化到 0~1)
    """
    model.zero_grad()

    x_demo = x_demo.clone().detach().requires_grad_(True)

    logits = model(x_ecg, x_demo)          # [1, num_labels]
    score = logits[:, class_idx].sum()
    score.backward()

    grad = x_demo.grad[0].detach().cpu().numpy()   # [5]
    val = x_demo.detach()[0].cpu().numpy()         # [5]

    importance = np.abs(grad * val)                # 元素乘
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


# ----------------------------------------------------
# 3. 画图：上面 ECG+CAM，下面 demo 条形图
# ----------------------------------------------------


def plot_ecg_and_demo_importance(
    ecg, cam, demo_importance, demo_feature_names,
    lead_idx, title, save_path
):
    """
    ecg: [12, T] (numpy 或 Tensor)
    cam: [T]
    demo_importance: [D]
    demo_feature_names: list[str]，长度 D
    """
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    sig = ecg[lead_idx]
    T = sig.shape[-1]
    t = np.arange(T)

    # CAM 再归一化一次防万一
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # --- 上面：ECG + 红色背景 CAM ---
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

    # --- 下面：demo feature 重要度 ---
    ax2 = fig.add_subplot(gs[1, 0])

    y_pos = np.arange(len(demo_importance))
    ax2.barh(y_pos, demo_importance, color="salmon")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(demo_feature_names)
    ax2.invert_yaxis()  # 让第一个特征在最上面
    ax2.set_xlabel("Relative importance")
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[SAVE] ECG+Demo Grad-CAM figure saved to: {save_path}")


# ----------------------------------------------------
# 4. 加载 ECG+Demo 模型
# ----------------------------------------------------

def load_demo_model(cfg, ckpt_path, device):
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {}).get("ecg_demo", {})

    classes = data_cfg["labels"]
    in_leads = data_cfg.get("leads", 12)

    # ⚠️ 不要传 demo_dim，让它保持默认 5（age, sex, height, weight, pacemaker）
    # 只传 ecg_feat_dim，就和你训练时的配置兼容了
    model = ECGDemo(
        in_leads=in_leads,
        num_labels=len(classes),
        ecg_feat_dim=model_cfg.get("ecg_feat_dim", 256),
        # demo_hidden_dim 也可以显式传一下（可选）：
        # demo_hidden_dim=model_cfg.get("demo_feat_dim", 64),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] Model loaded. missing:", missing)
    print("[INFO] unexpected:", unexpected)

    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------
# 5. 主流程
# ----------------------------------------------------


def main(args):
    # 1) 读 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    data_cfg = cfg["data"]
    base_dir = data_cfg["base_dir"]
    classes = data_cfg["labels"]

    # 2) 数据集（test split）
    test_ds = PTBXLECGDemoDataset(
        base_dir=base_dir,
        split="test",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    print("[INFO] ECG+Demo test size:", len(test_ds))
    print("[INFO] Classes:", classes)

    # 3) 模型
    model = load_demo_model(cfg, args.ckpt, device)

    # target conv layer = ECG backbone 的最后一层 Conv1d
    target_layer = model.ecg_backbone.backbone[-1].net[0]
    grad_cam = GradCAM1D_ECGDemo(model, target_layer)

    # 4) 取一个样本
    idx = args.index
    x_ecg, x_demo, y = test_ds[idx]        # [12, T], [5], [L]
    signal_length = x_ecg.shape[-1]

    x_ecg_t = x_ecg.unsqueeze(0).to(device)    # [1, 12, T]
    x_demo_t = x_demo.unsqueeze(0).to(device)  # [1, 5]

    # 类别选择
    if args.class_name is not None and args.class_name != "":
        class_name = args.class_name
        class_idx = classes.index(class_name)
    else:
        class_idx = args.class_idx
        class_name = classes[class_idx]

    print(f"[INFO] Running ECG+Demo Grad-CAM on sample {idx}, class {class_name}")

    # 5) 生成时间 CAM
    cam = grad_cam.generate_cam(
        x_ecg=x_ecg_t,
        x_demo=x_demo_t,
        class_idx=class_idx,
        signal_length=signal_length,
    )

    # 6) 计算 demo 特征重要度
    demo_importance = compute_demo_importance(
        model,
        x_ecg_t,
        x_demo_t,
        class_idx=class_idx,
    )
    demo_feature_names = ["age", "sex", "height", "weight", "pacemaker"]

    # 7) 保存 CAM 数值 & 图像
    os.makedirs("outputs/gradcam_demo", exist_ok=True)

    cam_path = os.path.join(
        "outputs/gradcam_demo", f"sample_{idx}_{class_name}_cam.npy"
    )
    np.save(cam_path, cam.numpy())
    print("[SAVE] CAM saved to:", cam_path)

    fig_path = os.path.join(
        "outputs/gradcam_demo", f"sample_{idx}_{class_name}_ecg_demo.png"
    )
    plot_ecg_and_demo_importance(
        ecg=x_ecg,
        cam=cam,
        demo_importance=demo_importance,
        demo_feature_names=demo_feature_names,
        lead_idx=args.lead,
        title=f"ECG+Demo Grad-CAM | sample {idx} | class {class_name}",
        save_path=fig_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ecg_demo.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # ⚠️ 这里一定要改成你自己的 best 模型路径
        default="outputs\ecg_demo\ckpts\ecg_demo_20251206_222615_best.pth",
    )
    parser.add_argument("--index", type=int, default=10)
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--class_name", type=str, default="MI")
    main(parser.parse_args())
