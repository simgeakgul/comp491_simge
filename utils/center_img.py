import cv2
import numpy as np
import json
import math

from .persp_conv import perspective_to_equirectangular
from .inpaint import pad_and_create_mask, inpaint_image

# ----------------------------------------------------------------------
# NEW helper
# ----------------------------------------------------------------------
def _inpaint_half(
    padded: np.ndarray, mask: np.ndarray, prompt: str,
    which: str,                 # "top" | "bottom"
    overlap_px: int,            # keep some context from the other half
    dilate_px: int, guidance: float, steps: int
) -> np.ndarray:

    H, W = padded.shape[:2]

    if which == "top":
        y0 = 0
        y1 = H // 2 + overlap_px
    else:                 # "bottom"
        y0 = H // 2 - overlap_px
        y1 = H

    # --- make height divisible by 8 so SD won’t crop it ---
    seg_h = y1 - y0
    seg_h -= seg_h % 8            # drop 0‑7 extra rows
    if which == "top":
        y1 = y0 + seg_h
    else:
        y0 = y1 - seg_h

    crop      = padded[y0:y1].copy()
    crop_mask = mask[y0:y1].copy()

    crop_out  = inpaint_image(
        crop, crop_mask, prompt,
        dilate_px=dilate_px,
        guidance_scale=guidance,
        steps=steps
    )

    padded[y0:y1] = crop_out      # same shape now → no broadcast error
    return padded




def complete_to_1024(
    image_arr: np.ndarray,
    prompts_path: str,
    dilate_px: int, 
    guidance_scale: float, 
    steps: int
) -> np.ndarray:

    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    side_prompt   = prompts['atmosphere']
    sky_prompt    = prompts['sky_or_ceiling']
    bottom_prompt = prompts['ground_or_floor']

    # --- Stage 0: scale so longest side == 1024, keep aspect ratio ---
    h, w = image_arr.shape[:2]
    scale = 1024.0 / max(h, w)
    if abs(scale - 1.0) > 1e-3:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image_arr = cv2.resize(image_arr, (new_w, new_h), interpolation=interp)

    # --- Stage 1: pad left/right if needed ---
    h, w = image_arr.shape[:2]
    missing_w = 1024 - w
    if missing_w > 2:
        left  = missing_w // 2
        right = missing_w - left
        padded, mask = pad_and_create_mask(
            image_arr, left, right, 0, 0
        )
        image_arr = inpaint_image(padded, mask, side_prompt, dilate_px, guidance_scale, steps)

    # --- Stage 2: pad top if needed ---
    h, w = image_arr.shape[:2]
    missing_h = 1024 - h
    if missing_h > 2:
        top = math.ceil(missing_h / 2)
        padded, mask = pad_and_create_mask(image_arr, 0, 0, top, 0)

        # in‑paint only the upper half we just created
        padded = _inpaint_half(
            padded, mask, sky_prompt,
            which="top",
            overlap_px=32,          # 16‑48 px of context is plenty
            dilate_px=dilate_px,
            guidance=guidance_scale,
            steps=steps
        )
        image_arr = padded        # continue with the updated frame

    # --- Stage 3: pad bottom if needed ---
    h, w = image_arr.shape[:2]
    missing_h = 1024 - h
    if missing_h > 2:
        bottom = missing_h
        padded, mask = pad_and_create_mask(image_arr, 0, 0, 0, bottom)

        padded = _inpaint_half(
            padded, mask, bottom_prompt,
            which="bottom",
            overlap_px=32,
            dilate_px=dilate_px,
            guidance=guidance_scale,
            steps=steps
        )
        image_arr = padded

    # --- Final: center‐crop any slight overshoot ---
    h, w = image_arr.shape[:2]
    if h != 1024 or w != 1024:
        y = max((h - 1024) // 2, 0)
        x = max((w - 1024) // 2, 0)
        image_arr = image_arr[y:y+1024, x:x+1024]

    return image_arr


def center_image(img, fov_deg=90, out_w=4096, out_h=2048):
    # front view
    pano = perspective_to_equirectangular(
        pers_img   = img,
        yaw        = 0.0,
        pitch      = 0.0,
        fov        = fov_deg,
        width  = out_w,
        height = out_h
    )

    # 180° yaw gives the “mirrored” halves at the left and right edges
    wrap = perspective_to_equirectangular(
        pers_img   = img,
        yaw        = 180.0,      # look backwards
        pitch      = 0.0,
        fov        = fov_deg,
        width  = out_w,
        height = out_h
    )

    # copy any non-black pixel from the second pass into the panorama
    mask = wrap.any(axis=-1)
    pano[mask] = wrap[mask]
    return pano


