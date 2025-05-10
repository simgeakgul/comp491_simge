import cv2
import json
import numpy as np
from pathlib import Path
from typing import Tuple

from .persp_conv import perspective_to_equirectangular
from .inpaint import load_mask_from_black, inpaint_image
import json
import cv2
import numpy as np


def _make_region_mask(h: int, w: int, region: str, pad: int) -> np.ndarray:
    """
    Create a binary (0/255) mask for *one* padded region of a square canvas.

    Parameters
    ----------
    h, w : full-canvas height and width   (h == w == n)
    region : {"top", "bottom", "left", "right"}
    pad    : pixel thickness of that region

    Returns
    -------
    uint8 mask of shape (h, w) with 255 in the region, else 0.
    """
    mask = np.zeros((h, w), np.uint8)
    if region == "top":        # first `pad` rows
        mask[:pad, :] = 255
    elif region == "bottom":   # last  `pad` rows
        mask[h - pad :, :] = 255
    elif region == "left":     # first `pad` cols
        mask[:, :pad] = 255
    elif region == "right":    # last  `pad` cols
        mask[:, w - pad :] = 255
    return mask


def complete_to_square(
    image_arr: np.ndarray,
    json_path: str,
    dilate_px: int,
    guidance_scale: float,
    steps: int,
) -> np.ndarray:
    """
    Expand `image_arr` to an n × n square using SD in-painting, driven by prompts
    stored in `json_path`.

    JSON structure
    --------------
    {
      "atmosphere":      "...",   # for left/right padding (portrait images)
      "sky_or_ceiling":  "...",   # for TOP padding   (landscape images)
      "ground_or_floor": "..."    # for BOTTOM padding
    }
    """
    h, w = image_arr.shape[:2]
    if h == w:
        return image_arr.copy()

    # --- load prompts --------------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as jf:
        prm = json.load(jf)

    n   = max(h, w)
    pad = (n - min(h, w)) // 2

    # --- place original in the middle of a black canvas ----------------------
    canvas = np.zeros((n, n, 3), dtype=image_arr.dtype)
    y0     = (n - h) // 2
    x0     = (n - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image_arr

    if w > h:
        # 1) TOP  — sky / ceiling
        top_mask = _make_region_mask(n, n, "top", pad)
        canvas   = inpaint_image(
            canvas, top_mask, prm["sky_or_ceiling"],
            dilate_px, guidance_scale, steps
        )

        # 2) BOTTOM — ground / floor
        bottom_mask = _make_region_mask(n, n, "bottom", pad)
        canvas      = inpaint_image(
            canvas, bottom_mask, prm["ground_or_floor"],
            dilate_px, guidance_scale, steps
        )

    else:  # h > w
        side_mask = load_mask_from_black(canvas)   
        canvas = inpaint_image(
            canvas, side_mask, prm["atmosphere"],
            dilate_px, guidance_scale, steps
        )

    return canvas



def center_image(img, fov_deg=90, out_w=4096, out_h=2048):
    pano = perspective_to_equirectangular(
        pers_img=img, yaw=0.0, pitch=0.0,
        fov=fov_deg, width=out_w, height=out_h
    )
    wrap = perspective_to_equirectangular(
        pers_img=img, yaw=180.0, pitch=0.0,
        fov=fov_deg, width=out_w, height=out_h
    )
    mask = wrap.any(axis=-1)
    pano[mask] = wrap[mask]
    return pano
