import numpy as np
import cv2
from PIL import Image
import torch
from utils.persp_conv import equirectangular_to_perspective, perspective_to_equirectangular
from utils.load_models import load_pipe

# ---  config -----------------------------------------------------------------
PITCH_MAP = {                        # degrees
    "atmosphere": 0.0,
    "sky_or_ceiling": 45.0,
    "ground_or_floor": -45.0,
}
YAW_LISTS = {                        # degrees
    "atmosphere":      [0, 45, 90, 135, 180, 225, 270, 315],
    "sky_or_ceiling":  [0, 90, 180, 270],
    "ground_or_floor": [0, 90, 180, 270],
}
FOV_MAP = {                          # degrees
    "atmosphere": 75.0,
    "sky_or_ceiling": 100.0,
    "ground_or_floor": 100.0,
}

CROP_SIZE = 1024                     # 1024×1024 crops (fits 12 GB GPU easily)

# -----------------------------------------------------------------------------


def build_depth_panorama(
    pano_bgr: np.ndarray,
    scene_type: str = "outdoor",
    pitch_map=PITCH_MAP,
    yaw_lists=YAW_LISTS,
    fov_map=FOV_MAP,
    crop_size: int = CROP_SIZE,
):
    """
    Returns a dense (H×W) depth panorama aligned to `pano_bgr`.
    No in-painting; unseen pixels remain 0.
    """
    H, W = pano_bgr.shape[:2]

    # (Step 3) load metric monocular model once
    pipe_metric = load_pipe(
        "pipe_metric_indoor" if scene_type.lower() == "indoor"
        else "pipe_metric_outdoor"
    )

    # accumulators for fusion
    accum_depth = np.zeros((H, W), np.float32)
    accum_weight = np.zeros((H, W), np.float32)

    # ------------------------------------------------------------------ loop
    for ring_name, pitch_deg in pitch_map.items():
        fov_deg = fov_map[ring_name]
        for yaw_deg in yaw_lists[ring_name]:

            # -------- Step 2: pano → perspective crop ----------------------
            crop_bgr = equirectangular_to_perspective(
                pano_bgr,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=crop_size,
                height=crop_size,
            )

            # torch / PIL expects RGB
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

            # -------- Step 3: depth prediction -----------------------------
            with torch.inference_mode():
                out = pipe_metric(pil_img)          # dict with key "depth"
            depth_crop = np.array(out["depth"], dtype=np.float32)  # H×W

            # guard against NaNs from the network
            depth_crop = np.nan_to_num(depth_crop, nan=0.0, posinf=0.0, neginf=0.0)

            # -------- Step 4a: warp depth back to equirect -----------------
            depth_equi = perspective_to_equirectangular(
                depth_crop,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=W,
                height=H,
            )

            if depth_equi.ndim == 3:
                depth_equi = cv2.cvtColor(depth_equi, cv2.COLOR_BGR2GRAY)

            # simple binary weight (1 ⇒ valid pixel, 0 ⇒ gap)
            valid_mask = depth_equi > 0

            # -------- Step 4b: fuse ----------------------------------------
            accum_depth[valid_mask] += depth_equi[valid_mask]
            accum_weight[valid_mask] += 1.0          # could be cosine weight

    # finish fusion: average where weight>0
    fused_depth = np.zeros_like(accum_depth)
    nonzero = accum_weight > 0
    fused_depth[nonzero] = accum_depth[nonzero] / accum_weight[nonzero]

    return fused_depth, accum_weight   # return weight if you want to inspect gaps

def main():
    import cv2
    import os
    import numpy as np

    # --- config: replace with your own file path ----------------------------
    pano_path = "pano.jpg"        # input panorama (must be equirectangular)
    out_npy_path = "pano_depth.npy"  # output depth map as NumPy
    out_vis_path = "depth_vis.jpg"   # still keep a JPEG viz if you like

    # --- load panorama ------------------------------------------------------
    pano_bgr = cv2.imread(pano_path, cv2.IMREAD_COLOR)
    if pano_bgr is None:
        raise FileNotFoundError(f"Could not read panorama from {pano_path}")
    print(f"[INFO] Loaded panorama: shape={pano_bgr.shape}")

    # --- build depth panorama -----------------------------------------------
    depth_pano, cov_mask = build_depth_panorama(
        pano_bgr,
        scene_type="outdoor",  # or "indoor" if appropriate
    )
    print(f"[INFO] Depth fusion complete. Saving to: {out_npy_path}")

    # --- save as .npy -------------------------------------------------------
    np.save(out_npy_path, depth_pano)
    print(f"[DONE] Depth panorama saved: {out_npy_path}")

    # --- optional: write a JPEG visualization ------------------------------
    non_holes = depth_pano[depth_pano > 0]
    if non_holes.size:
        dmin, dmax = non_holes.min(), non_holes.max()
        depth_vis = (depth_pano - dmin) / (dmax - dmin + 1e-6)
    else:
        depth_vis = np.zeros_like(depth_pano)
    cv2.imwrite(out_vis_path, (depth_vis * 255).astype(np.uint8))
    print(f"[DONE] Depth panorama visualization saved: {out_vis_path}")


if __name__ == "__main__":
    main()
