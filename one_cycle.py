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
os.makedirs(DEBUG_FOLDER, exist_ok=True)
id = 0

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

    global id
    view_tag = f"{int(id)}_pitch{int(pitch)}_yaw{int(yaw)}"

    persp = equirectangular_to_perspective(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=512,
        height=512
    )

    save_image(f"{view_tag}_0_persp.jpg", persp)

    mask = load_mask_from_black(persp)

    save_image(f"{view_tag}_1_mask.jpg", mask)

    # 3) inpaint that crop
    result = inpaint_image(
        image_arr=persp,
        mask_arr=mask,
        prompt=prompt,
        guidance_scale=guidance_scale,
        steps=steps
    )

    save_image(f"{view_tag}_2_painted.jpg", result)

    # 4) blend it back into pano
    pano_filled = blend_patch_into_pano(
        pano=pano,
        tile=result,
        mask=mask,
        yaw=yaw,
        pitch=pitch,
        fov=fov
    )

    save_image(f"{view_tag}_3_pano.jpg", pano_filled)
    id = id + 1
    return pano_filled


# image = cv2.imread("input.jpg")
# resized = complete_to_1024(image_arr = image,  prompts_path = "prompts.json")
# save_image("resized1.jpg", resized)

resized = cv2.imread("resized.jpg")
pano = center_image(resized, fov_deg=80)

with open('prompts.json', 'r') as file:
    prompts = json.load(file)


pitch_map = {
    "atmosphere":       0.0,    # horizontal band
    "sky_or_ceiling":   60.0,    # looking straight up
    "ground_or_floor": -60.0,  # looking straight down
}

# 3. Define all yaw angles per category  
horizontal_yaws = [45, -45, 90, -90, 135, -135, 180, -180]
sky_yaws        = [0, 90, 180, 270]
ground_yaws     = [0, 90, 180, 270]


# 4. Choose FOV per category (optional tweak)
fov_map = {
    "atmosphere":      80.0,
    "sky_or_ceiling":  120.0,
    "ground_or_floor": 120.0,
}

# 5. Build a list of (prompt_key, yaw) pairs
view_list = []
for yaw in horizontal_yaws:
    view_list.append(("atmosphere", yaw))

for yaw in sky_yaws:
    view_list.append(("sky_or_ceiling", yaw))

for yaw in ground_yaws:
    view_list.append(("ground_or_floor", yaw))

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
        guidance_scale=11.0,
        steps=50
    )

save_image("full_pano.jpg", pano)


