import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
from .persp_conv import equirectangular_to_perspective
from .load_models import load_pipe


def process_view(
    crop_bgr: np.ndarray,
    view_tag: str,
    scene_type: str = "outdoor",
    debug_dir: Path | str = "debug_depth",
    q_lo: float = 0.20,
    q_hi: float = 0.80,
) -> np.ndarray:

    # --- prepare input for transformers pipeline -------------
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(crop_rgb)
    pil_img.save(f"{debug_dir}/{view_tag}_input_bgr.png")

    # --- pick the right depth pipelines ---------------------
    # pipe_rel    = load_pipe("pipe_rel")
    pipe_metric = load_pipe(
        "pipe_metric_indoor" if scene_type.lower() == "indoor"
        else "pipe_metric_outdoor"
    )

    with torch.inference_mode():
        # out_rel    = pipe_rel(pil_img)
        out_metric = pipe_metric(pil_img)

    # depth_rel_img    = out_rel["depth"]
    depth_metric_img = out_metric["depth"]

    # depth_rel    = np.array(depth_rel_img   , dtype=np.float32)
    # depth_rel = depth_rel.max() - depth_rel

    depth_metric = np.array(depth_metric_img, dtype=np.float32)

    # --- compute the 20–80% quantile alignment -------------
    # qrel_lo, qrel_hi = np.quantile(depth_rel,    [q_lo, q_hi])
    # qmet_lo, qmet_hi = np.quantile(depth_metric, [q_lo, q_hi])
    # scale = (qmet_hi - qmet_lo) / max(1e-6, (qrel_hi - qrel_lo))
    # depth_aligned = depth_rel * scale

    # --- optional: enforce ground ≥1.5m below camera -------
    # z_ground = depth_aligned[depth_aligned < np.median(depth_aligned)]
    # if z_ground.size:
    #     cam2ground = np.mean(z_ground)
    #     if cam2ground < 1.5:
    #         depth_aligned *= 1.5 / cam2ground

    # --- save debug visualizations -------------------------
    def norm_uint8(arr):
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # cv2.imwrite(f"{debug_dir}/{view_tag}_rel.png",     norm_uint8(depth_rel))
    cv2.imwrite(f"{debug_dir}/{view_tag}_metric.png",  norm_uint8(depth_metric))
    # cv2.imwrite(f"{debug_dir}/{view_tag}_aligned.png", norm_uint8(depth_aligned))

    return depth_metric


# ------------------------------------------------------------------
# 3. Whole panorama  →  dict{view_tag : depth}
# ------------------------------------------------------------------
def process_pano(
    pano_bgr: np.ndarray,
    pitch_map: dict[str, float],
    yaw_lists: dict[str, list[float]],
    fov_map: dict[str, float],
    scene_type: str = "outdoor",
    debug_dir: Path | str = "debug_depth",
) -> dict[str, np.ndarray]:

    os.makedirs(debug_dir, exist_ok=True)
    depths: dict[str, np.ndarray] = {}

    # how many pixels in pano = 1° of yaw
    px_per_deg = pano_bgr.shape[1] / 360.0

    for region, pitch in pitch_map.items():
        for yaw in yaw_lists[region]:
            fov = fov_map[region]

            # compute square view size from fov
            tile_size = max(32, int(px_per_deg * fov))

            # crop that square from the pano
            crop = equirectangular_to_perspective(
                pano_bgr,
                yaw=yaw,
                pitch=pitch,
                fov=fov,
                width=tile_size,
                height=tile_size
            )

            view_tag = f"{region}_{int(yaw)}"
            depth = process_view(
                crop_bgr=crop,
                view_tag=view_tag,
                scene_type=scene_type,
                debug_dir=debug_dir
            )
            depths[view_tag] = depth

    return depths