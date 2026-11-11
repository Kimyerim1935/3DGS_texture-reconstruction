"""Real-data robust texture loss extracted from proposed_method/texture_losses.py."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RealDataRobustTextureLoss(nn.Module):
    """
    Mode I: Real-Data Robust Texture Guidance
    - Multi-Scale High-Frequency Decomposition (s=1, 2, 4)
    - Local Contrast Normalization: I_norm = (I - local_mean) / (local_std + eps)
    - Soft Texture Confidence M_tex
    - Photometric Reliability Gating R_photo = exp(-alpha_photo * E_low)
    - Combined Texture Weight W_tex = M_tex * R_photo
    - Robust Charbonnier Loss per scale: rho(x) = sqrt(x^2 + eps^2)
    - Weak Edge Preservation Loss L_edge = mean(|Grad(Render) - Grad(GT)|)
    """
    def __init__(self,
                 kernel_size=11,
                 sigma_max=2.0,
                 sigma_min=0.5,
                 scales=[1, 2, 4],
                 scale_weights=[0.5, 0.3, 0.2],
                 alpha_photo=5.0,
                 lambda_edge=0.03,
                 var_kernel=7,
                 var_thresh=0.02,
                 charbonnier_eps=1e-3,
                 norm_kernel=11,
                 adaptive_hf_weight=False,
                 adaptive_beta=5.0,
                 eps=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.scales = scales
        self.scale_weights = scale_weights
        self.alpha_photo = alpha_photo
        self.lambda_edge = lambda_edge
        self.var_kernel = var_kernel
        self.var_thresh = var_thresh
        self.charbonnier_eps = charbonnier_eps
        self.norm_kernel = norm_kernel
        self.adaptive_hf_weight = adaptive_hf_weight
        self.adaptive_beta = adaptive_beta
        self.eps = eps
        self.last_map_stats = None

    def compute_local_norm(self, img, ref_mean=None, ref_std=None):
        """
        If ref_mean and ref_std are provided (from GT), normalize img using ref_mean and ref_std.
        Otherwise, compute mean and std from img (used for GT).
        """
        c, h, w = img.shape
        if ref_mean is None or ref_std is None:
            img_4d = img.unsqueeze(0) # [1, C, H, W]
            pad = self.norm_kernel // 2
            kernel = torch.ones((c, 1, self.norm_kernel, self.norm_kernel), device=img.device) / (self.norm_kernel * self.norm_kernel)

            mean = F.conv2d(img_4d, kernel, padding=pad, groups=c)[0] # [C, H, W]
            mean_sq = F.conv2d(img_4d**2, kernel, padding=pad, groups=c)[0] # [C, H, W]
            var = torch.clamp(mean_sq - mean**2, min=0.0)
            std = torch.sqrt(var)
        else:
            mean = ref_mean
            std = ref_std

        std_clamped = torch.clamp(std, min=0.05)
        img_norm = (img - mean) / std_clamped
        img_norm = torch.clamp(img_norm, min=-3.0, max=3.0)
        return img_norm, mean, std

    def compute_texture_variance(self, img):
        """Return per-pixel local intensity variance for a CHW image."""
        c, _, _ = img.shape
        img_4d = img.unsqueeze(0)
        pad = self.var_kernel // 2
        kernel = torch.ones((c, 1, self.var_kernel, self.var_kernel), device=img.device) / (self.var_kernel * self.var_kernel)
        mean = F.conv2d(img_4d, kernel, padding=pad, groups=c)
        mean_sq = F.conv2d(img_4d.square(), kernel, padding=pad, groups=c)
        return (mean_sq - mean.square()).mean(dim=1)[0].clamp_min(0.0)

    def compute_texture_confidence(self, img):
        """
        img: [C, H, W] in [0, 1]
        Returns: [H, W] soft texture confidence map in [0, 1]
        """
        var = self.compute_texture_variance(img)
        m_tex = torch.sigmoid(50.0 * (var - self.var_thresh))
        return m_tex

    def compute_sobel_gradient(self, img):
        """
        img: [C, H, W] in [0, 1]
        Returns: [C, H, W] gradient magnitude along x and y
        """
        c, h, w = img.shape
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)

        img_4d = img.unsqueeze(0)
        grad_x = F.conv2d(img_4d, sobel_x, padding=1, groups=c)[0]
        grad_y = F.conv2d(img_4d, sobel_y, padding=1, groups=c)[0]
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_mag

    def forward(self, render_img, gt_img, step, max_steps, ref_start_step=500, lambda_H_max=0.1,
                hf_only=False, collect_stats=False, weight_mode="adaptive"):
        """
        render_img, gt_img: [C, H, W] in [0, 1]
        """
        import torchvision.transforms.functional as TF

        progress = step / max_steps
        base_sigma = self.sigma_max - progress * (self.sigma_max - self.sigma_min)

        # 1. Base LPF on raw images for L_low
        kernel_sz = [self.kernel_size, self.kernel_size]
        sigma_base = [base_sigma, base_sigma]
        render_lpf = TF.gaussian_blur(render_img, kernel_sz, sigma_base)
        gt_lpf = TF.gaussian_blur(gt_img, kernel_sz, sigma_base)

        l_low = render_img.new_zeros(()) if hf_only else F.l1_loss(render_lpf, gt_lpf)

        # Photometric reliability gating R_photo
        e_low = torch.abs(render_lpf - gt_lpf).mean(dim=0) # [H, W]
        r_photo = torch.exp(-self.alpha_photo * e_low).detach() # [H, W]

        # Texture confidence M_tex
        texture_variance = None
        if weight_mode == "rank_adaptive":
            texture_variance = self.compute_texture_variance(gt_img).detach()
            m_tex = torch.sigmoid(50.0 * (texture_variance - self.var_thresh))
        else:
            m_tex = self.compute_texture_confidence(gt_img)
        weight_maps = {
            "uniform": torch.ones_like(m_tex),
            "texture": m_tex,
            "reliability": r_photo,
            "adaptive": m_tex * r_photo,
        }
        rank_texture_gate = None
        rank_photo_gate = None
        if weight_mode == "rank_adaptive":
            # Calibrate each gate to this image's distribution instead of fixed
            # variance/error thresholds that shift with scene and resolution.
            pooled_size = (min(64, gt_img.shape[1]), min(64, gt_img.shape[2]))
            texture_values = F.adaptive_avg_pool2d(
                texture_variance[None, None], pooled_size
            ).reshape(-1)
            error_values = F.adaptive_avg_pool2d(
                e_low.detach()[None, None], pooled_size
            ).reshape(-1)
            texture_q = torch.quantile(texture_values, texture_values.new_tensor([0.70, 0.90]))
            error_q = torch.quantile(error_values, error_values.new_tensor([0.50, 0.90]))
            texture_span = (texture_q[1] - texture_q[0]).clamp_min(self.eps)
            error_span = (error_q[1] - error_q[0]).clamp_min(self.eps)
            rank_texture_gate = torch.sigmoid(4.0 * (texture_variance - texture_q[0]) / texture_span)
            rank_photo_gate = torch.sigmoid(4.0 * (error_q[1] - e_low.detach()) / error_span)
            w_tex = rank_texture_gate * rank_photo_gate
        elif weight_mode in weight_maps:
            w_tex = weight_maps[weight_mode]
        else:
            raise ValueError("weight_mode must be uniform, texture, reliability, adaptive, or rank_adaptive")

        # Optional structure-aware gate: reduce HF supervision where gradient
        # structure between render and GT is not yet aligned.
        if self.adaptive_hf_weight or not hf_only:
            grad_render = self.compute_sobel_gradient(render_img)
            grad_gt = self.compute_sobel_gradient(gt_img)
        else:
            grad_render = grad_gt = None
        if self.adaptive_hf_weight:
            grad_err = torch.abs(grad_render - grad_gt).mean(dim=0).detach()
            r_struct = torch.exp(-self.adaptive_beta * grad_err)
            w_tex = w_tex * r_struct
        else:
            r_struct = torch.ones_like(w_tex)

        # 2. Local Contrast Normalization (GT-Anchored)
        gt_norm, mean_gt, std_gt = self.compute_local_norm(gt_img)
        render_norm, _, _ = self.compute_local_norm(render_img, ref_mean=mean_gt.detach(), ref_std=std_gt.detach())

        # 3. Multi-Scale HF Decomposition & Charbonnier Loss
        l_tex_ms = 0.0
        scale_res_maps = []

        for scale, weight in zip(self.scales, self.scale_weights):
            cur_sig = base_sigma * scale
            cur_k = int(self.kernel_size * scale)
            if cur_k % 2 == 0:
                cur_k += 1

            sig_list = [cur_sig, cur_sig]
            k_list = [cur_k, cur_k]

            render_norm_lpf = TF.gaussian_blur(render_norm, k_list, sig_list)
            gt_norm_lpf = TF.gaussian_blur(gt_norm, k_list, sig_list)

            h_render = render_norm - render_norm_lpf
            h_gt = gt_norm - gt_norm_lpf

            diff_h = (h_render - h_gt).mean(dim=0) # [H, W]
            scale_res_maps.append(diff_h)

            # Charbonnier penalty
            rho_diff = torch.sqrt(diff_h**2 + self.charbonnier_eps**2) # [H, W]

            l_scale = (w_tex * rho_diff).sum() / (w_tex.sum() + self.eps)
            l_tex_ms = l_tex_ms + weight * l_scale

        # 4. Weak Edge Preservation Loss
        l_edge = (
            render_img.new_zeros(())
            if hf_only
            else F.l1_loss(grad_render, grad_gt)
        )

        # 5. Progressive 貫_H schedule (Mode E/F schedule)
        if step < ref_start_step:
            cur_lambda_H = 0.0
        else:
            k = 0.002
            cur_lambda_H = lambda_H_max * (1.0 / (1.0 + math.exp(-k * (step - ref_start_step))))

        diagnostics = {
            "render_norm": render_norm,
            "gt_norm": gt_norm,
            "m_tex": m_tex,
            "e_low": e_low,
            "r_photo": r_photo,
            "r_struct": r_struct,
            "w_tex": w_tex,
            "scale_res_maps": scale_res_maps,
            "grad_render": grad_render,
            "grad_gt": grad_gt
        }

        if collect_stats:
            maps_to_summarize = [
                ("M_tex", m_tex), ("R_photo", r_photo), ("E_low", e_low), ("W_tex", w_tex)
            ]
            if rank_texture_gate is not None:
                maps_to_summarize.extend([
                    ("M_rank", rank_texture_gate), ("R_rank", rank_photo_gate)
                ])
            self.last_map_stats = {
                name: self._summarize_map(value)
                for name, value in maps_to_summarize
            }

        return l_low, l_tex_ms, l_edge, cur_lambda_H, diagnostics

    @staticmethod
    def _summarize_map(value):
        values = value.detach().float().reshape(-1)
        quantiles = torch.quantile(values, values.new_tensor([0.05, 0.50, 0.95]))
        return {
            "mean": float(values.mean().item()),
            "p05": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p95": float(quantiles[2].item()),
            "max": float(values.max().item()),
        }
