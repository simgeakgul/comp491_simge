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

# --- parameters ---------------------------------------------------
EDGE_SIGMA = 5      # px; controls feather width
CENTER_BIAS = 0.5   # <1 favours crop centre, >1 flattens weight
ALIGN_DEPTH = False   # turn on/off scale alignment
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

    # precompute centre bias (radial) in crop coords
    yy, xx = np.indices((crop_size, crop_size))
    rr = np.sqrt((xx - crop_size/2)**2 + (yy - crop_size/2)**2)
    centre_weight = 1.0 - (rr / (crop_size/2))**CENTER_BIAS
    centre_weight = np.clip(centre_weight, 0.0, 1.0)

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
           
            depth_crop = np.nan_to_num(
                np.array(out["depth"], dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            )


            # -------- Step 4a: warp depth back to equirect -----------------
            depth_equi = perspective_to_equirectangular(
                depth_crop,
                yaw=yaw_deg,
                pitch=pitch_deg,
                fov=fov_deg,
                width=W,
                height=H,
            )

            # persp to equir returns 3 channels
            depth_equi = cv2.cvtColor(depth_equi, cv2.COLOR_BGR2GRAY)
            depth_equi = depth_equi.astype(np.float32)

            hard_mask = (depth_crop > 0).astype(np.float32)

            kernel = np.ones((5,5), np.uint8)
            inner = cv2.erode(hard_mask, kernel, iterations=1)
            
            soft_mask = cv2.GaussianBlur(inner, (0, 0), EDGE_SIGMA)
            soft_mask *= centre_weight

            # stack into 3 channels so the warp sees a BGR image
            mask_u8 = np.clip(soft_mask*255, 0, 255).astype(np.uint8)
            fake_bgr = cv2.merge([mask_u8, mask_u8, mask_u8])            
           
            weight_equi = perspective_to_equirectangular(
                fake_bgr, yaw=yaw_deg, pitch=pitch_deg,
                fov=fov_deg, width=W, height=H
            )

            weight_equi = weight_equi[...,0].astype(np.float32) / 255.0

            # optional: align scale of new crop to existing fusion
            if ALIGN_DEPTH:
                overlap = (weight_equi > 0) & (accum_weight > 0)
                if overlap.any():
                    d_new = depth_equi[overlap]
                    d_old = accum_depth[overlap] / accum_weight[overlap]
                    q_old = np.quantile(d_old, [0.2, 0.8])
                    q_new = np.quantile(d_new, [0.2, 0.8])
                    # prevent division by zero
                    denom = max(q_new[1] - q_new[0], 1e-6)
                    scale = (q_old[1] - q_old[0]) / denom
                    depth_equi *= scale

            accum_depth  += depth_equi * weight_equi
            accum_weight += weight_equi


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