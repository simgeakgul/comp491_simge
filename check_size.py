import json
import cv2
import math
import numpy as np
from utils.inpaint import pad_and_create_mask_reflect, inpaint_image

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

    # Stage 1: Expand sides (horizontal)
    h, w = image_arr.shape[:2]
    missing_w = 1024 - w
    if missing_w > 0:
        left = missing_w // 2
        right = missing_w - left
        padded, mask = pad_and_create_mask_reflect(
            image_arr, left, right, 0, 0, feather, reflect=True
        )

        cv2.imwrite("debug_sides_padded.png", padded)
        cv2.imwrite("debug_sides_mask.png", mask)
        image_arr = inpaint_image(
            padded, mask, side_prompt, guidance_scale, steps
        )

    # Stage 2: Expand sky (top) without reflect
    h, w = image_arr.shape[:2]
    missing_h = 1024 - h
    if missing_h > 0:
        top = math.ceil(missing_h / 2)
        padded, mask = pad_and_create_mask_reflect(
            image_arr, 0, 0, top, 0, feather, reflect=False
        )
        cv2.imwrite("debug_sky_padded.png", padded)
        cv2.imwrite("debug_sky_mask.png", mask)
        image_arr = inpaint_image(
            padded, mask, sky_prompt, guidance_scale, steps
        )

    # # Stage 3: Expand bottom (vertical) without reflect
    # h, w = image_arr.shape[:2]
    # missing_h = 1024 - h
    # if missing_h > 0:
    #     bottom = missing_h
    #     padded, mask = pad_and_create_mask_reflect(
    #         image_arr, 0, 0, 0, bottom, feather, reflect=False
    #     )
    #     cv2.imwrite("debug_bottom_padded.png", padded)
    #     cv2.imwrite("debug_bottom_mask.png", mask)
    #     image_arr = inpaint_image(
    #         padded, mask, bottom_prompt, guidance_scale, steps
    #     )

    # # Final: center-crop any overshoot
    # h, w = image_arr.shape[:2]
    # if h != 1024 or w != 1024:
    #     y = max((h - 1024) // 2, 0)
    #     x = max((w - 1024) // 2, 0)
    #     image_arr = image_arr[y:y+1024, x:x+1024]


    return image_arr


image = cv2.imread("input.jpg")

resized = complete_to_1024(
    image_arr = image,
    prompts_path = "prompts.json",
) 

cv2.imwrite("resized.jpg", resized)
