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
    fov: float = 85.0,
    prompt: str = "",
    guidance_scale: float = 8.0,
    steps: int = 50
) -> np.ndarray:

    global id
    view_tag = f"{int(id)}_pitch{int(pitch)}_yaw{int(yaw)}"

    H, W = pano.shape[:2]
    px_per_degree = W / 360.0

    tile_w = max(32, int(px_per_degree * fov))  
    tile_h = tile_w  

    persp = equirectangular_to_perspective(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=tile_w,
        height=tile_h
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
