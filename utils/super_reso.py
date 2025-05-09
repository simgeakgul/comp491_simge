import cv2
import os
import numpy as np
from pathlib import Path
from .load_configs import load_config, PanoConfig
from .persp_conv import perspective_to_equirectangular

def build_border_mask(
    pano: np.ndarray,
    views: list[tuple[float, float, float]],
    border_px: int,
    center_fov: float | None = None,
    debug_path: str | Path = "border_mask.jpg",
) -> np.ndarray:
    H, W = pano.shape[:2]
    mask     = np.zeros((H, W), dtype=np.uint8)  # final seam‐lines
    coverage = np.zeros((H, W), dtype=np.uint8)  # “interiors” already seen

    # kernels for edge extraction and optional dilation
    k_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_px*2+1, border_px*2+1)
    )

    # collect all the yaw/pitch/fov triples
    all_views = []
    if center_fov is not None:
        all_views += [(0.0, 0.0, center_fov), (180.0, 0.0, center_fov)]
    all_views += views

    dummy = np.full((256, 256, 3), 255, dtype=np.uint8)

    for yaw, pitch, fov in all_views:
        # warp a full–white square into pano coords
        w = perspective_to_equirectangular(dummy, yaw=yaw, pitch=pitch,
                                           fov=fov, width=W, height=H)
        if w.ndim == 3:
            w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        tile = (w > 127).astype(np.uint8) * 255

        # figure out which pixels of this tile are *not* yet covered
        new_bits = cv2.bitwise_and(tile, cv2.bitwise_not(coverage))

        # extract just the boundary of those new bits
        border = cv2.morphologyEx(new_bits, cv2.MORPH_GRADIENT, k_border)
        if border_px > 1:
            border = cv2.dilate(border, k_dilate)

        # paint only the fresh border segments
        cv2.bitwise_or(mask, border, dst=mask)

        # now mark the entire tile interior as “covered” for future loops
        cv2.bitwise_or(coverage, tile, dst=coverage)

    cv2.imwrite(str(debug_path), mask)
    return mask


def save_pano_with_mask_overlay(
    pano: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (0, 0, 255),
    out_path: str = "debug_overlay.jpg",
) -> np.ndarray:
    bin_mask = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(pano, dtype=np.uint8)
    color_mask[bin_mask == 1] = color
    overlay = cv2.addWeighted(pano, 1.0, color_mask, alpha, 0)
    cv2.imwrite(out_path, overlay)
    return overlay


def split_pano_and_mask(
    pano: np.ndarray,
    mask: np.ndarray,
    vertical_num: int = 2,
    horizontal_num: int = 4
) -> dict[tuple[int,int], dict]:
    """
    Splits pano and mask into a grid of (vertical_num x horizontal_num) tiles.
    Returns a dict keyed by (row, col) with entries:
      {
        'pano': pano_tile,
        'mask': mask_tile,
        'coords': (y1, y2, x1, x2)
      }
    """
    H, W = pano.shape[:2]
    tile_h = H // vertical_num
    tile_w = W // horizontal_num
    tiles = {}

    for i in range(vertical_num):
        y1 = i * tile_h
        y2 = (i + 1) * tile_h if i < vertical_num - 1 else H
        for j in range(horizontal_num):
            x1 = j * tile_w
            x2 = (j + 1) * tile_w if j < horizontal_num - 1 else W

            pano_tile = pano[y1:y2, x1:x2].copy()
            mask_tile = mask[y1:y2, x1:x2].copy()
            tiles[(i, j)] = {
                'pano': pano_tile,
                'mask': mask_tile,
                'coords': (y1, y2, x1, x2)
            }
    return tiles

