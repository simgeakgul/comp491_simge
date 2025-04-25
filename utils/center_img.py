import cv2
import numpy as np
import json
import math

from .persp_conv import perspective_to_equirectangular
from .inpaint import pad_and_create_mask_reflect, inpaint_image

def complete_to_1024(
    image_arr: np.ndarray,
    prompts_path: str,
    feather: int = 30,
    guidance_scale: float = 10.0,
    steps: int = 50
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
        padded, mask = pad_and_create_mask_reflect(
            image_arr, left, right, 0, 0, feather
        )
        image_arr = inpaint_image(padded, mask, side_prompt, guidance_scale, steps)

    # --- Stage 2: pad top if needed ---
    h, w = image_arr.shape[:2]
    missing_h = 1024 - h
    if missing_h > 2:
        top = math.ceil(missing_h / 2)
        padded, mask = pad_and_create_mask_reflect(
            image_arr, 0, 0, top, 0, feather
        )
        image_arr = inpaint_image(padded, mask, sky_prompt, guidance_scale, steps)

    # --- Stage 3: pad bottom if needed ---
    h, w = image_arr.shape[:2]
    missing_h = 1024 - h
    if missing_h > 2:
        bottom = missing_h
        padded, mask = pad_and_create_mask_reflect(
            image_arr, 0, 0, 0, bottom, feather
        )
        image_arr = inpaint_image(padded, mask, bottom_prompt, guidance_scale, steps)

    # --- Final: center‐crop any slight overshoot ---
    h, w = image_arr.shape[:2]
    if h != 1024 or w != 1024:
        y = max((h - 1024) // 2, 0)
        x = max((w - 1024) // 2, 0)
        image_arr = image_arr[y:y+1024, x:x+1024]

    return image_arr


def center_image(
    img: np.ndarray,
    out_w: int = 4096,
    out_h: int = 2048,
    slice_fov: float = 45.0
) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Split image into left, center, right patches
    center_patch = img[cy-256:cy+256, cx-256:cx+256]
    left_raw = img[cy-256:cy+256, 0:256]  # Only 256 width
    left_patch = np.zeros((512, 512, 3), dtype=img.dtype)
    left_patch[:, 256:] = left_raw  # Fill the right half

    # RIGHT PATCH
    right_raw = img[cy-256:cy+256, w-256:w]  # Only 256 width
    right_patch = np.zeros((512, 512, 3), dtype=img.dtype)
    right_patch[:, :256] = right_raw  # Fill the left half

    cv2.imwrite("center.jpg", center_patch)
    cv2.imwrite("left.jpg", left_patch)
    cv2.imwrite("right.jpg", right_patch)

    # Project each patch with specified yaw
    center_proj = perspective_to_equirectangular(
        pers_img=center_patch,
        yaw=0.0,
        pitch=0.0,
        fov=slice_fov,
        width=out_w,
        height=out_h
    )
    left_proj = perspective_to_equirectangular(
        pers_img=left_patch,
        yaw=-45.0,  # Left side (yaw to the left)
        pitch=0.0,
        fov=slice_fov,
        width=out_w,
        height=out_h
    )
    right_proj = perspective_to_equirectangular(
        pers_img=right_patch,
        yaw=45.0,  # Right side (yaw to the right)
        pitch=0.0,
        fov=slice_fov,
        width=out_w,
        height=out_h
    )

    canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)

    for proj in [left_proj, center_proj, right_proj]:
        mask = (proj.sum(axis=-1) != 0)
        canvas[mask] = proj[mask]

    return canvas
