import numpy as np
import cv2
from utils.persp_conv import perspective_to_equirectangular


def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    *,
    blur_radius: int = 31,
    dilate_px: int = 3         # enlarge mask a few pixels before blurring
) -> np.ndarray:

    H, W = pano.shape[:2]

    # --- 1. Warp tile and mask back to pano coords --------------------------
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # make mask single-channel binary 0/1
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    bin_mask = (warped_mask > 127).astype(np.uint8)

    # --- 2. Feather the mask edge ------------------------------------------
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1,)*2)
        bin_mask = cv2.dilate(bin_mask, kernel)

    k = blur_radius | 1                 # make sure kernel is odd
    soft_mask = cv2.GaussianBlur(bin_mask*255, (k, k), 0).astype(np.float32)
    alpha = (soft_mask / 255.0)[..., None]   # H×W×1 in [0,1]

    # --- 3. Prevent black bleed: copy pano into tile outside the mask -------
    warped_tile_bg_fixed = warped_tile.copy()
    warped_tile_bg_fixed[bin_mask == 0] = pano[bin_mask == 0]

    # --- 4. Composite -------------------------------------------------------
    pano_f   = pano.astype(np.float32)
    tile_f   = warped_tile_bg_fixed.astype(np.float32)
    out_f    = (1.0 - alpha) * pano_f + alpha * tile_f

    return np.clip(out_f, 0, 255).astype(np.uint8)
