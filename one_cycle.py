import cv2
import os
import json
from PIL import Image
from pathlib import Path
from typing import Tuple, Union
import numpy as np                
from utils.persp_conv import (
    equirectangular_to_perspective,
    perspective_to_equirectangular
)
from utils.center_img import center_image, complete_to_1024
from utils.inpaint import load_mask_from_black, inpaint_image
from utils.blend_pano import blend_patch_into_pano

DEBUG_FOLDER = "debug_images"

def save_image(filename, image):
    full_path = os.path.join(DEBUG_FOLDER, filename)
    cv2.imwrite(full_path, image)

def one_cycle(
    pano: np.ndarray,
    yaw: float,
    pitch: float = 0.0,
    fov: float = 90.0,
    prompt: str = "",
    dilate_px: int = 16,
    guidance_scale: float = 8.0,
    steps: int = 50
) -> np.ndarray:

    persp = equirectangular_to_perspective(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=1024,
        height=512
    )

    save_image(f"1_persp_{int(yaw)}.jpg", persp)

    mask = load_mask_from_black(
        persp,
        dilate_px=dilate_px,
    )

    save_image(f"2_mask_{int(yaw)}.jpg", mask)

    # 3) inpaint that crop
    result = inpaint_image(
        image_arr=persp,
        mask_arr=mask,
        prompt=prompt,
        guidance_scale=guidance_scale,
        steps=steps
    )

    save_image(f"3_painted_{int(yaw)}.jpg", result)

    # 4) blend it back into pano
    pano_filled = blend_patch_into_pano(
        pano=pano,
        tile=result,
        mask=mask,
        dilate=dilate_px,
        yaw=yaw,
        pitch=pitch,
        fov=fov
    )

    return pano_filled


image = cv2.imread("input.jpg")
# resized = complete_to_1024(image_arr = image,  prompts_path = "prompts.json")

resized = cv2.imread("resized.jpg")
pano = center_image(resized)



with open('prompts.json', 'r') as file:
    prompts = json.load(file)


pitch_map = {
    "atmosphere":      0.0,    # horizontal band
    "sky_or_ceiling": 90.0,    # looking straight up
    "ground_or_floor": -90.0,  # looking straight down
}

# 3. Define all yaw angles per category  
horizontal_yaws = [0, 45, 90, 135, 180, 225, 270, 315]
sky_yaws        = [0, 90, 180, 270]
ground_yaws     = [0, 90, 180, 270]

# 4. Choose FOV per category (optional tweak)
fov_map = {
    "atmosphere":      90.0,
    "sky_or_ceiling": 120.0,
    "ground_or_floor":120.0,
}

# 5. Build a list of (prompt_key, yaw) pairs
view_list = []
for yaw in horizontal_yaws:
    view_list.append(("atmosphere", yaw))

# for yaw in sky_yaws:
#     view_list.append(("sky_or_ceiling", yaw))
# for yaw in ground_yaws:
#     view_list.append(("ground_or_floor", yaw))

# 6. Loop and call one_cycle
for prompt_key, yaw in view_list:
    pitch       = pitch_map[prompt_key]
    prompt_text = prompts[prompt_key]
    fov         = fov_map.get(prompt_key)
    pano = one_cycle(
        pano=pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        prompt=prompt_text,
        dilate_px=16,
        guidance_scale=8.0,
        steps=45
    )

    save_image(f"pano_{prompt_key}_{int(yaw)}.jpg", pano)




save_image("full_pano.jpg", pano)



