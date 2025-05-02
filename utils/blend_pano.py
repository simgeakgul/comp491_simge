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
    dilate_px: int = 3
) -> np.ndarray:

    H, W = pano.shape[:2]

    # 1) Warp tile & mask
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # make mask binary
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    bin_mask = (warped_mask > 127).astype(np.uint8)

    # 2) Morphological closing to eliminate tiny cracks
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px*2+1,)*2
        )
        bin_mask = cv2.dilate(bin_mask, kernel)
        bin_mask = cv2.erode(bin_mask, kernel)

    # 3) Feather edges
    k = blur_radius | 1
    soft_mask = cv2.GaussianBlur(bin_mask*255, (k, k), 0).astype(np.float32)
    alpha = (soft_mask / 255.0)[..., None]

    # 4) Fill any black holes in warped_tile with pano
    black_hole = np.all(warped_tile == 0, axis=2)
    if np.any(black_hole):
        warped_tile[black_hole] = pano[black_hole]

    # 5) Composite
    pano_f = pano.astype(np.float32)
    tile_f = warped_tile.astype(np.float32)
    out_f = (1.0 - alpha) * pano_f + alpha * tile_f

    return np.clip(out_f, 0, 255).astype(np.uint8)
