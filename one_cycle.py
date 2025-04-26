import cv2
import os
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
    prompt: str = (
        "Continue the alpine scene: pine trees and ground, "
        "matching the original artist’s soft brush strokes, lighting and color palette"
    ),
    dilate_px: int = 16,
    guidance_scale: float = 8.0,
    steps: int = 50
) -> np.ndarray:

    persp = equirectangular_to_perspective(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=512,
        height=256
    )

    p_name = f"1_persp_{int(yaw)}.jpg"
    save_image(p_name, persp)

    # 2) get hard & soft masks from the black areas
    mask = load_mask_from_black(
        persp,
        dilate_px=dilate_px,
    )

    save_image("2_hard_mask.jpg", hard)

    # 3) inpaint that crop
    result = inpaint_image(
        image_arr=persp,
        mask_arr=mask,
        prompt=prompt,
        guidance_scale=guidance_scale,
        steps=steps
    )

    r_name = f"4_painted_{int(yaw)}.jpg"
    save_image(r_name, result)

    # 4) blend it back into pano
    pano_filled = blend_patch_into_pano(
        pano=pano,
        tile=result,
        hard_mask=hard_mask,
        soft_mask=soft_mask,
        yaw=yaw,
        pitch=pitch,
        fov=fov
    )

    return pano_filled


image = cv2.imread("input.jpg")
# resized = complete_to_1024(image_arr = image,  prompts_path = "prompts.json")

resized = cv2.imread("resized.jpg")
pano = center_image(resized)
save_image("5_pano.jpg", pano)

pano0 = one_cycle(pano, yaw=45)
save_image("6_pano0.jpg", pano0)

# pano1 = one_cycle(pano0, yaw=315)
# save_image("pano1.jpg", pano1)

# pano2 = one_cycle(pano1, yaw=90)
# save_image("pano2.jpg", pano2)

# pano3 = one_cycle(pano2, yaw=270)
# save_image("pano3.jpg", pano3)

