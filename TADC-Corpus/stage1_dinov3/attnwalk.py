# -*- coding: utf-8 -*-
"""AttnWalk segmentation core used by Stage 1."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_multiotsu


class AttnWalkSegmenter:
    def __init__(self, k_neighbors=15, beta=0.30, max_iter=10, device="cuda"):
        self.k = k_neighbors
        self.beta = beta
        self.max_iter = max_iter
        self.device = device

    def _get_knn_affinity(self, feat):
        sim = torch.mm(feat, feat.t())
        k = min(self.k, sim.shape[1])
        topk_vals, topk_inds = torch.topk(sim, k, dim=1)
        W = torch.zeros_like(sim)
        W.scatter_(1, topk_inds, topk_vals)
        W = (W + W.t()) / 2.0
        W[W < 0] = 0
        return W

    def _get_pca_saliency(self, feat, attn_guide):
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        try:
            U, S, V = torch.svd(feat_centered)
            n_components = min(3, U.shape[1], S.shape[0])
            pcs = U[:, :n_components] * S[:n_components].unsqueeze(0)
        except Exception:
            return torch.norm(feat, dim=1)

        if attn_guide is not None:
            for i in range(pcs.shape[1]):
                pc = pcs[:, i]
                if (pc * attn_guide).sum() < 0:
                    pcs[:, i] = -pc

        for i in range(pcs.shape[1]):
            pc = pcs[:, i]
            pcs[:, i] = (pc - pc.min()) / (pc.max() - pc.min() + 1e-8)

        pca_map = pcs.mean(dim=1)
        pca_map = (pca_map - pca_map.min()) / (pca_map.max() - pca_map.min() + 1e-8)
        return pca_map

    def _propagate(self, P, seeds):
        Y = seeds.clone()
        for _ in range(self.max_iter):
            Y = (1 - self.beta) * torch.mv(P, Y) + self.beta * seeds
        return Y

    def segment(self, feat_np, attn_tensor, grid_size):
        h, w = grid_size
        N = h * w
        feat = torch.from_numpy(feat_np).to(self.device)
        feat = F.normalize(feat, dim=1)

        if attn_tensor is not None and attn_tensor.sum() > 1e-5:
            cls_attn = attn_tensor.flatten()
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        else:
            cls_attn = torch.zeros(N, device=self.device)

        pca_map = self._get_pca_saliency(feat, cls_attn)

        A_candidate = cls_attn
        S_pca_candidate = pca_map
        candidates = [A_candidate, S_pca_candidate]
        q_scores = []

        for C_map in candidates:
            C_np = C_map.cpu().numpy()
            C_uint8 = (C_np * 255).astype(np.uint8)
            t_otsu_val, _ = cv2.threshold(C_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t_star = t_otsu_val / 255.0
            fg_mask = C_map > t_star
            bg_mask = C_map <= t_star
            pi_1 = fg_mask.float().mean()
            pi_0 = bg_mask.float().mean()

            if pi_1 > 0 and pi_0 > 0:
                mu_1 = C_map[fg_mask].mean()
                mu_0 = C_map[bg_mask].mean()
                phi_f = pi_0 * pi_1 * torch.pow(mu_1 - mu_0, 2)
                phi_p = mu_1
            else:
                phi_f = torch.tensor(1e-5, device=self.device)
                phi_p = torch.tensor(1e-5, device=self.device)

            denom_spatial = C_map[fg_mask].sum() + 1e-10
            g_spatial = torch.where(
                fg_mask,
                C_map / denom_spatial,
                torch.tensor(0.0, device=self.device),
            )
            g_fg_only = g_spatial[fg_mask]
            H_sp = -torch.sum(g_fg_only * torch.log(g_fg_only + 1e-10))
            phi_c = torch.exp(-H_sp)
            q_C = phi_f * phi_p * phi_c
            q_scores.append(q_C)

        q_tensor = torch.stack(q_scores)
        tau = 1.0 / 5.0
        alpha_weights = F.softmax(q_tensor / tau, dim=0)
        alpha = alpha_weights[0]
        saliency_map = torch.pow(alpha * A_candidate + (1.0 - alpha) * S_pca_candidate, 2)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        W = self._get_knn_affinity(feat)
        deg = W.sum(dim=1)
        deg[deg == 0] = 1e-8
        P = W / deg.unsqueeze(1)

        sal_cpu = saliency_map.cpu().numpy()
        try:
            thresholds = threshold_multiotsu(sal_cpu, classes=3)
            bg_thresh = thresholds[0]
            fg_thresh = thresholds[1]
            fg_thresh = max(fg_thresh, sal_cpu.mean() + 1e-5)
        except Exception:
            sal_uint8 = (sal_cpu * 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fg_thresh = otsu_val / 255.0
            bg_thresh = fg_thresh * 0.3

        S_fg = (saliency_map > fg_thresh).float()
        if S_fg.sum() < 5:
            k_fallback = max(1, int(N * 0.01))
            val_fb, _ = torch.topk(saliency_map, k_fallback)
            fg_thresh = val_fb[-1]
            S_fg = (saliency_map >= fg_thresh).float()
            bg_thresh = fg_thresh * 0.3

        S_bg = (saliency_map < bg_thresh).float()
        border_mask = torch.zeros((h, w), device=self.device)
        border_mask[0, :] = 1
        border_mask[-1, :] = 1
        border_mask[:, 0] = 1
        border_mask[:, -1] = 1
        S_bg = torch.maximum(S_bg, border_mask.flatten())
        S_bg[S_fg > 0] = 0

        Y_fg = self._propagate(P, S_fg)
        Y_bg = self._propagate(P, S_bg)
        score_diff = Y_fg - Y_bg
        mask = (score_diff > 0).cpu().numpy().astype(np.uint8) * 255
        mask = mask.reshape(h, w)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        score_vis = score_diff.reshape(h, w).cpu().numpy()
        score_vis = (score_vis - score_vis.min()) / (score_vis.max() - score_vis.min() + 1e-8)
        return mask, score_vis, pca_map, cls_attn
