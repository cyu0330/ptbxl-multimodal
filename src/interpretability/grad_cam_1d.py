# src/interpretability/grad_cam_1d.py

import torch
import torch.nn.functional as F


class GradCAM1D:
    def __init__(self, model, target_layer):
        """
        Args:
            model: your ECGCNN model (nn.Module)
            target_layer: the Conv1d layer to inspect, e.g.
                          model.backbone[-1].net[0]
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None   # A: (N, C, L')
        self.gradients = None     # dY/dA: (N, C, L')

        self._register_hooks()

    def _forward_hook(self, module, input, output):
        # output: (N, C, L')
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: (N, C, L')
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        # forward hook: save feature maps
        self.target_layer.register_forward_hook(self._forward_hook)
        # backward hook: save gradients wrt feature maps
        self.target_layer.register_backward_hook(self._backward_hook)

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        cam: (1, L') or (L',)
        """
        if cam.ndim == 1:
            cam = cam.unsqueeze(0)

        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val

        return cam.squeeze(0)

    def generate_cam(self, input_tensor, class_idx, signal_length=None):
        """
        Args:
            input_tensor: (1, C, L) e.g. (1, 12, 5000)
            class_idx: int, index of target class in logits
            signal_length: original signal length. If not None,
                           CAM will be upsampled to this length.

        Returns:
            1D CAM tensor of shape (signal_length,) or (L')
        """
        # 1) 清空梯度
        self.model.zero_grad()

        # 2) forward
        output = self.model(input_tensor)  # (1, num_labels) or (logits, z, ...)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # 3) 取对应类别的 logit
        score = logits[0, class_idx]

        # 4) backward
        score.backward(retain_graph=True)

        # 5) 取出 activations 和 gradients
        A = self.activations      # (1, C, L')
        dYdA = self.gradients     # (1, C, L')

        # 6) GAP over time: 得到每个通道的权重
        weights = dYdA.mean(dim=2, keepdim=True)  # (1, C, 1)

        # 7) 加权求和得到 CAM: (1, L')
        cam = (weights * A).sum(dim=1)  # (1, L')
        cam = torch.relu(cam)

        # 8) 归一化
        cam = self._normalize_cam(cam)  # (L',)

        # 9) 如果需要，插值回原始长度
        if signal_length is not None and cam.shape[-1] != signal_length:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),  # (1, 1, L')
                size=signal_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # (signal_length,)

        return cam
