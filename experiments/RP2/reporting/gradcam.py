"""
Grad-CAM++ Implementation for Multi-Architecture Pneumonia Detection
=====================================================================
Supports all 10 architectures used in this research:
  CNN   : ResNet-18, ResNet-50, DenseNet-121, VGG-19, EfficientNet-B0,
          MobileNetV3, ConvNeXt-Tiny
  ViT   : DeiT-Tiny, Swin-Tiny, ViT-B/16

Reference:
  Chattopadhyay et al. (2018) "Grad-CAM++: Improved Visual Explanations
  for Deep Convolutional Networks" — WACV 2018
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ─── Target Layer Registry ───────────────────────────────────────────────────
def _get_target_layer(model, model_name: str):
    """
    Returns the final feature-extraction layer for Grad-CAM++ hooks.
    For CNNs  → last convolutional/normalization layer.
    For ViTs  → last transformer block (attention output norm).
    """
    name = model_name.lower()

    if "resnet18" in name or "resnet50" in name:
        return model.layer4[-1]               # Last BasicBlock / Bottleneck

    elif "densenet121" in name:
        return model.features.denseblock4     # Last DenseBlock

    elif "vgg19" in name:
        return model.features[-1]             # Last MaxPool after last conv

    elif "efficientnetb0" in name or "efficientnet" in name:
        return model.features[-1]             # Last SiLU block

    elif "mobilenetv3" in name:
        return model.features[-1]             # Last InvertedResidual

    elif "convnext_tiny" in name or "convnext" in name:
        return model.features[-1]             # Last ConvNeXt stage

    elif "deit_tiny" in name or "deit" in name:
        return model.blocks[-1].norm1         # Last ViT block norm

    elif "swin_tiny" in name or "swin" in name:
        return model.layers[-1].blocks[-1]    # Last Swin block

    elif "vit_b_16" in name or "vit" in name:
        return model.blocks[-1].norm1         # Last ViT block norm

    else:
        # Fallback: try to find the last Conv2d layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError(f"Cannot find target layer for model: {model_name}")
        return last_conv


# ─── Grad-CAM++ Core ─────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation that works for both CNNs and Vision Transformers.

    For CNNs: uses standard Grad-CAM++ with alpha weighting.
    For Transformers: falls back to gradient-weighted attention map approach
                      since there are no spatial feature maps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Handle tuple outputs from Transformer blocks (e.g. Swin)
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _compute_cnn_gradcam_pp(self, target_class: int, input_size):
        """Standard Grad-CAM++ for CNN feature maps (B, C, H, W)."""
        grads = self.gradients           # (B, C, H, W)
        acts  = self.activations         # (B, C, H, W)

        # Grad-CAM++ alpha weights
        grads_squared = grads ** 2
        grads_cubed   = grads ** 3

        # Sum of activations per channel (denominator)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)

        eps = 1e-9
        alpha_denom = 2.0 * grads_squared + grads_cubed * sum_acts + eps
        alpha = grads_squared / alpha_denom               # (B, C, H, W)

        # Weights: relu(grad) weighted by alpha, then mean over spatial dims
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze()                               # (H, W)

        # Upsample to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, input_size[::-1])           # (H_in, W_in)

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

        return cam

    def _compute_vit_cam(self, input_size):
        """
        Gradient-weighted approach for Vision Transformer outputs.
        Activations shape: (B, N, C) where N = num_tokens.
        For ViT/DeiT: N = 1 + H/P * W/P.
        For Swin: N = H * W (no CLS token).
        """
        grads = self.gradients   # (B, N, C) or (B, N, C)
        acts  = self.activations

        # Grad-CAM++ weights over channel dim
        weights = grads.mean(dim=-1)          # (B, N)
        cam_tokens = (weights * acts.mean(dim=-1)).squeeze(0)  # (N,)

        # Try to figure out spatial layout
        num_tokens = cam_tokens.shape[0]

        # Check if ViT/DeiT (has CLS token at index 0)
        # Patch grid: (H_in / patch_size) ** 2 + 1 CLS
        # For 224x224 / 16 = 14x14 = 196 + 1 = 197
        has_cls = False
        grid = None
        for h in range(1, 20):
            for w in range(1, 20):
                if h * w + 1 == num_tokens:
                    has_cls = True
                    grid = (h, w)
                    break
                if h * w == num_tokens:
                    grid = (h, w)
                    break
            if grid:
                break

        if grid is None:
            # Fallback: 1D attention rollout-style
            cam_tokens = F.relu(cam_tokens).cpu().numpy()
            side = int(np.sqrt(num_tokens))
            cam_tokens = cam_tokens[:side * side].reshape(side, side)
        else:
            tokens = cam_tokens[1:] if has_cls else cam_tokens  # drop CLS
            cam_tokens = F.relu(tokens).cpu().numpy()
            cam_tokens = cam_tokens.reshape(grid[0], grid[1])

        cam = cv2.resize(cam_tokens.astype(np.float32), input_size[::-1])

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

        return cam

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Run forward + backward pass and compute the Grad-CAM++ map.

        Args:
            input_tensor : (1, 3, H, W) preprocessed image tensor.
            target_class : class index to explain (default: model's prediction).

        Returns:
            cam    : np.ndarray (H, W) in [0, 1]
            pred   : int predicted class
            probs  : list[float] class probabilities
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)                 # (1, num_classes)
        probs  = torch.softmax(output, dim=1)[0].tolist()
        pred   = int(torch.argmax(output, dim=1).item())

        if target_class is None:
            target_class = pred

        # Backward for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=False)

        # Decide CNN vs Transformer based on activation shape
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. "
                               "Check that target_layer is correct.")

        input_h = input_tensor.shape[2]
        input_w = input_tensor.shape[3]

        if self.activations.dim() == 4:
            # CNN: (B, C, H, W)
            cam = self._compute_cnn_gradcam_pp(target_class, (input_h, input_w))
        else:
            # Transformer: (B, N, C)
            cam = self._compute_vit_cam((input_h, input_w))

        return cam, pred, probs


# ─── Heatmap Overlay ─────────────────────────────────────────────────────────
def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Blend a Grad-CAM++ heatmap over the original X-ray image.

    Args:
        image_np : H x W x 3 uint8 RGB image.
        cam      : H x W float32 in [0, 1].
        alpha    : heatmap blend factor.
        colormap : OpenCV colormap.

    Returns:
        overlay  : H x W x 3 uint8 RGB blended image.
    """
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, colormap)        # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)    # → RGB

    # Ensure same size
    if image_np.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    overlay = (alpha * heatmap.astype(np.float32) +
               (1 - alpha) * image_np.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ─── Convenience function used by app.py ─────────────────────────────────────
def run_gradcam_pp(model: nn.Module, model_name: str,
                   input_tensor: torch.Tensor,
                   original_image_np: np.ndarray,
                   target_class: int = None):
    """
    High-level helper that wires together target-layer detection,
    Grad-CAM++ computation, and heatmap overlay.

    Args:
        model            : loaded PyTorch model on the correct device.
        model_name       : string key matching the architecture (e.g. "resnet18").
        input_tensor     : (1, 3, 224, 224) tensor on same device as model.
        original_image_np: (H, W, 3) uint8 RGB array for overlay.
        target_class     : class index to explain (None = predicted class).

    Returns:
        overlay  : np.ndarray (H, W, 3) uint8 blended image
        cam      : np.ndarray (H, W) raw heatmap [0,1]
        pred     : int predicted class index
        probs    : list[float] class probabilities [normal_prob, pneumonia_prob]
    """
    target_layer = _get_target_layer(model, model_name)
    explainer = GradCAMPlusPlus(model, target_layer)

    try:
        cam, pred, probs = explainer.generate(input_tensor, target_class)
    finally:
        explainer.remove_hooks()

    overlay = overlay_heatmap(original_image_np, cam)
    return overlay, cam, pred, probs
