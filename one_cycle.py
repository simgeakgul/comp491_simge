import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Union
import numpy as np                
from utils.persp_conv import (
    equirectangular_to_perspective,
    equirectangular_to_cylindrical,
    perspective_to_equirectangular
)
from utils.center_img import center_image, complete_to_1024
from utils.inpaint import load_soft_hard_masks_from_black, inpaint_image
from utils.blend_pano import blend_patch_into_pano

def one_cycle(
    pano: np.ndarray,
    yaw: float,
    pitch: float = 0.0,
    fov: float = 90.0,
    prompt: str = (
        "Continue the alpine scene: pine trees and ground, "
        "matching the original artist’s soft brush strokes, lighting and color palette"
    ),
    threshold: int = 10,
    dilate_px: int = 32,
    feather: int = 64,
    guidance_scale: float = 11.0,
    steps: int = 50
) -> np.ndarray:

    H, W = pano.shape[:2]
    # pick a “window” half the pano’s size by default
    crop_w, crop_h = W // 2, H // 2

    # 1) cylindrical crop
    cyl = equirectangular_to_cylindrical(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=crop_w,
        height=crop_h
    )

    # 2) get hard & soft masks from the black areas
    image, hard_mask, soft_mask = load_soft_hard_masks_from_black(
        cyl,
        threshold=threshold,
        dilate_px=dilate_px,
        feather=feather
    )

    # 3) inpaint that crop
    result = inpaint_image(
        image_arr=cyl,
        mask_arr=soft_mask,
        prompt=prompt,
        guidance_scale=guidance_scale,
        steps=steps
    )

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

resized = complete_to_1024(image_arr = image,  prompts_path = "prompts.json")
pano = center_image(resized, fov_deg=90, out_w=2048, out_h=1024)

pano0 = one_cycle(pano, yaw=45)
cv2.imwrite("pano0.jpg", pano0)

# pano1 = one_cycle(pano0, yaw=315)
# cv2.imwrite("pano1.jpg", pano1)

# pano2 = one_cycle(pano1, yaw=90)
# cv2.imwrite("pano2.jpg", pano2)

# pano3 = one_cycle(pano2, yaw=270)
# cv2.imwrite("pano3.jpg", pano3)

