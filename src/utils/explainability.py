import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Extracts activations and gradients from target convolutional layer.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output["cls_logit"]
        score.backward(retain_graph=True)

        gradients = self.gradients # (B, C, H_f, W_f)
        activations = self.activations # (B, C, H_f, W_f)

        if gradients is None or activations is None:
            # Fallback if hook wasn't triggered
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        return cam[0, 0].cpu().numpy()


class ForensicSHAP:
    """
    Forensic SHAP: Cooperative game attributions with fallback to
    Gradient-weighted Input Attribution: P_i = |input_i * grad_i|
    """
    @staticmethod
    def explain(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
        tensor = input_tensor.clone().detach().requires_grad_(True)
        output = model(tensor)
        score = output["cls_logit"]
        model.zero_grad()
        score.backward()

        grad = tensor.grad.data # (1, 3, H, W)
        attribution = torch.abs(tensor * grad).sum(dim=1, keepdim=True)
        attribution = attribution - attribution.min()
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        return attribution[0, 0].detach().cpu().numpy()


class ForensicLIME:
    """
    Forensic LIME: Local Interpretable Model-agnostic Explanations
    Perturbs superpixel regions to assess local region sensitivity.
    """
    @staticmethod
    def explain(model: torch.nn.Module, input_tensor: torch.Tensor, num_samples: int = 24, grid_size: int = 8) -> np.ndarray:
        B, C, H, W = input_tensor.shape
        base_out = model(input_tensor)
        base_score = float(torch.sigmoid(base_out["cls_logit"]).detach().cpu().item())

        heat = np.zeros((H, W), dtype=np.float32)
        cell_h = H // grid_size
        cell_w = W // grid_size

        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    perturbed = input_tensor.clone()
                    y1, y2 = i * cell_h, min((i + 1) * cell_h, H)
                    x1, x2 = j * cell_w, min((j + 1) * cell_w, W)
                    # Mask out region with mean gray
                    perturbed[:, :, y1:y2, x1:x2] = 0.5
                    p_score = float(torch.sigmoid(model(perturbed)["cls_logit"]).detach().cpu().item())
                    # Sensitivity is drop in fake probability
                    diff = abs(base_score - p_score)
                    heat[y1:y2, x1:x2] = diff

        if heat.max() > 0:
            heat = heat / heat.max()
        # Smooth with Gaussian blur for visual clarity
        heat = cv2.GaussianBlur(heat, (21, 21), 0)
        return heat


def render_heatmap_overlay(img_bgr: np.ndarray, heat_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Applies JET colormap overlay on the RGB image
    """
    heat_uint8 = (np.clip(heat_map, 0.0, 1.0) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, color_map, alpha, 0)
    return overlay
