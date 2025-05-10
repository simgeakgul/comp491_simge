import cv2
import json
import numpy as np
from pathlib import Path
from typing import Tuple

from .persp_conv import perspective_to_equirectangular
from .inpaint import load_mask_from_black, inpaint_image


def _resize_and_center(
    img: np.ndarray, target: int = 1024
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    h, w = img.shape[:2]
    if h >= w:
        new_h, new_w = target, int(w * target / h)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top, left = 0, (target - new_w) // 2
    else:
        new_h, new_w = int(h * target / w), target
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top, left = (target - new_h) // 2, 0

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas[top: top + new_h, left: left + new_w] = resized
    return canvas, (top, left), (new_h, new_w)


def complete_to_1024(
    image_arr: np.ndarray,
    prompts_path: str,
    dilate_px: int,
    guidance_scale: float,
    steps: int,
) -> np.ndarray:
    # 1) centre on a 1024×1024 canvas
    canvas, (top, left), (new_h, new_w) = _resize_and_center(image_arr, 1024)
    if new_h == 1024 and new_w == 1024:
        return canvas                         # nothing to fill

    # 2) read prompts --------------------------------------------------------
    try:
        prompt_dict = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
        if not isinstance(prompt_dict, dict):
            raise ValueError
    except Exception:
        txt = Path(prompts_path).read_text(encoding="utf-8").strip()
        prompt_dict = {"atmosphere": txt,
                       "sky_or_ceiling": txt,
                       "ground_or_floor": txt}

    # 3) build one mask that covers *all* black pixels -----------------------
    mask = load_mask_from_black(canvas)
    if mask.max() == 0:
        return canvas                         # nothing to in-paint



    # 4) choose a single prompt ---------------------------------------------
    if new_w < 1024:                  # black bands are left & right
        prompt = prompt_dict.get("atmosphere", "")
    else:                             # black bands are top & bottom
        prompt = (
            (prompt_dict.get("sky_or_ceiling", "") + " ").strip() +
            prompt_dict.get("ground_or_floor", "")
        ).strip()

    # 5) one-shot in-paint ---------------------------------------------------
    canvas = inpaint_image(
        canvas, mask, prompt,
        dilate_px=dilate_px,
        guidance_scale=guidance_scale,
        steps=steps,
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
