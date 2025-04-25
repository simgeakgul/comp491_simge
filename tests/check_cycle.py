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
from utils.center_img import center_image
from utils.inpaint import load_soft_hard_masks_from_black, inpaint_image
from utils.blend_pano import blend_patch_into_pano

# 1) save centered pano
image = cv2.imread("input.jpg")
pano = center_image(image, fov_deg=90, out_w=2048, out_h=1024)

# For debugging purposes, don't actually read from here.
cv2.imwrite("pano.jpg", pano)

# 3) cylindrical crop & save
cyl = equirectangular_to_cylindrical(
    pano,
    yaw=45,
    pitch=0,
    fov=90,
    width=1024,
    height=512
)

# For debugging purposes, don't actually read from here.
cv2.imwrite("flat.jpg", cyl)

# 4) build mask from black regions and save
image, hard_mask, soft_mask = load_soft_hard_masks_from_black(cyl, threshold=10)


# For debugging purposes, don't actually read from here.
cv2.imwrite("hard_mask.jpg", hard_mask)
cv2.imwrite("soft_mask.jpg", soft_mask)


result = inpaint_image(
    image_arr=cyl,
    mask_arr=soft_mask,
    prompt=(
        "Continue the alpine scene: pine trees and ground, "
        "matching the original artist’s soft brush strokes, lighting and color palette"
    ),
    guidance_scale=11.0,
    steps=50
)


# For debugging purposes, don't actually read from here.
cv2.imwrite("wide_out.jpg", result)

# 3) Blend back
pano0 = blend_patch_into_pano(
    pano    = pano,
    tile    = result,
    hard_mask = hard_mask,
    soft_mask = soft_mask,
    yaw=45, pitch=0, fov=90
)

# For debugging purposes, don't actually read from here.
cv2.imwrite("pano0.jpg", pano0)

