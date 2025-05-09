import numpy as np
import os
import cv2
from PIL import Image
import torch
from pathlib import Path

from .persp_conv import (
    equirectangular_to_perspective,
    perspective_to_equirectangular,
)
from .load_models import load_pipe
from .load_configs import load_config, PanoConfig

def build_depth_panorama(
    pano_bgr: np.ndarray,
    scene_type: str,
    pitch_map: dict[str, float],
    yaw_lists: dict[str, list[int]],
    fov_map: dict[str, float],
    crop_size: int,
    edge_sigma: float,
    center_bias: float,
    align_depth: bool,
):
    """
    Generates a dense depth panorama aligned to `pano_bgr`.
    Unseen pixels remain zero.
    """
    H, W = pano_bgr.shape[:2]

    # load the appropriate metric depth model
    pipe_metric = load_pipe(
        "pipe_metric_indoor" if scene_type.lower() == "indoor"
        else "pipe_metric_outdoor"
    )

    accum_depth = np.zeros((H, W), np.float32)
    accum_weight = np.zeros((H, W), np.float32)

    # precompute radial centre bias in crop coords
    yy, xx = np.indices((crop_size, crop_size))
    rr = np.sqrt((xx - crop_size/2)**2 + (yy - crop_size/2)**2)
    centre_weight = 1.0 - (rr/(crop_size/2))**center_bias
    centre_weight = np.clip(centre_weight, 0.0, 1.0)

    # iterate over rings and yaws
    for ring_name, pitch_deg in pitch_map.items():
        fov_deg = fov_map[ring_name]
        for yaw_deg in yaw_lists[ring_name]:
            # crop out a perspective image
            crop_bgr = equirectangular_to_perspective(
                pano_bgr,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=crop_size,
                height=crop_size,
            )
            # depth prediction (model expects RGB PIL image)
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                out = pipe_metric(pil_img)
            depth_crop = np.nan_to_num(
                np.array(out["depth"], dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            )

            # warp depth back to equirectangular
            depth_equi = perspective_to_equirectangular(
                depth_crop,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=W,
                height=H,
            )
            # convert to single-channel float
            depth_equi = cv2.cvtColor(depth_equi, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # build soft weight mask
            hard_mask = (depth_crop > 0).astype(np.float32)
            inner = cv2.erode(hard_mask, np.ones((5,5), np.uint8), iterations=1)
            soft_mask = cv2.GaussianBlur(inner, (0, 0), edge_sigma) * centre_weight

            mask_u8 = np.clip(soft_mask * 255, 0, 255).astype(np.uint8)
            fake_bgr = cv2.merge([mask_u8, mask_u8, mask_u8])
            weight_equi = perspective_to_equirectangular(
                fake_bgr,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=W,
                height=H,
            )[...,0].astype(np.float32) / 255.0

            # optional scale alignment
            if align_depth:
                overlap = (weight_equi > 0) & (accum_weight > 0)
                if overlap.any():
                    d_new = depth_equi[overlap]
                    d_old = accum_depth[overlap] / accum_weight[overlap]
                    q_old = np.quantile(d_old, [0.2, 0.8])
                    q_new = np.quantile(d_new, [0.2, 0.8])
                    denom = max(q_new[1] - q_new[0], 1e-6)
                    scale = (q_old[1] - q_old[0]) / denom
                    depth_equi *= scale

            # accumulate
            accum_depth  += depth_equi * weight_equi
            accum_weight += weight_equi

    # finalize: average weighted depth
    fused = np.zeros_like(accum_depth)
    mask_nz = accum_weight > 0
    fused[mask_nz] = accum_depth[mask_nz] / accum_weight[mask_nz]
    return fused, accum_weight

