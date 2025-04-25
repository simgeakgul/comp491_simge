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
from utils.inpaint import load_image_and_mask_from_black, load_soft_hard_masks_from_black, inpaint_image


def save_pano_crop(
    pano_path: Union[str, Path],
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int,
    bgr: bool = True      
) -> np.ndarray:
    """
    Reads a panorama from `pano_path`, crops a perspective view
    of size (width×height), and returns it.
    """
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(f"Could not open panorama at {pano_path}")

    if not bgr:
        pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)

    crop = equirectangular_to_perspective(
        pano,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        width=width,
        height=height
    )
    return crop


# 1) generate & save centered pano
image = cv2.imread("input.jpg")
pano = center_image(image, fov_deg=90, out_w=2048, out_h=1024)
cv2.imwrite("pano.jpg", pano)

# 2) crop a perspective view and save it
persp = save_pano_crop(
    pano_path="pano.jpg",
    yaw=45,
    pitch=0,
    fov=90,
    width=1024,
    height=512,
    bgr=True
)
cv2.imwrite("view.jpg", persp, [cv2.IMWRITE_JPEG_QUALITY, 95])

# 3) cylindrical crop & save
cyl = equirectangular_to_cylindrical(
    pano,
    yaw=45,
    pitch=0,
    fov=90,
    width=1024,
    height=512
)
cv2.imwrite("flat.jpg", cyl)

# 4) build mask from black regions and save
image, hard_mask, soft_mask = load_soft_hard_masks_from_black("flat.jpg", threshold=10)
hard_mask.save("hard_mask.jpg")
soft_mask.save("soft_mask.jpg")

# result = inpaint_image(
#     image=image,
#     mask=soft_mask,
#     prompt=(
#         "Continue the alpine scene: pine trees and ground, "
#         "matching the original artist’s soft brush strokes, lighting and color palette"
#     ),
#     guidance_scale=11.0,
#     steps=50
# )

# result.save("wide_out.jpg")



def blend_patch_into_pano(
    pano: np.ndarray,
    tile: Union[str, Path, np.ndarray, Image.Image],
    hard_mask: Union[str, Path, Image.Image],  # likewise accept path or image
    soft_mask: Union[str, Path, Image.Image],
    yaw: float,
    pitch: float,
    fov: float
) -> np.ndarray:
    """
    pano      : (H, W, 3) uint8 panorama
    tile      : path or PIL/np of the in-painted view
    hard_mask : path or PIL image of the 0/255 mask
    soft_mask : path or PIL image of the blurred mask
    yaw,pitch,fov : same params used to render the tile
    """

    # 0) Load tile if it’s a path
    if isinstance(tile, (str, Path)):
        tile = Image.open(str(tile)).convert("RGB")

    # 1) Convert tile to BGR numpy if PIL
    if isinstance(tile, Image.Image):
        tile = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)

    # 2) Load masks if they’re paths
    if isinstance(hard_mask, (str, Path)):
        hard_mask = Image.open(str(hard_mask)).convert("L")
    if isinstance(soft_mask, (str, Path)):
        soft_mask = Image.open(str(soft_mask)).convert("L")

    # 3) Get pano dimensions
    H, W = pano.shape[:2]

    # 4) Reproject the in-painted tile back onto the panorama canvas
    patch = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # 5) Warp both masks back to pano
    hm_arr = np.array(hard_mask, dtype=np.uint8)
    sm_arr = np.array(soft_mask, dtype=np.uint8)

    # stack to 3-channel so we can reuse the same API
    hm3 = np.stack([hm_arr]*3, axis=-1)
    sm3 = np.stack([sm_arr]*3, axis=-1)

    warped_hm3 = perspective_to_equirectangular(
        hm3, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_sm3 = perspective_to_equirectangular(
        sm3, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # 6) Extract single-channel boolean and float masks
    warped_hard = warped_hm3[...,0] > 128             # boolean mask
    warped_soft = warped_sm3[...,0].astype(np.float32) / 255.0  # 0..1

    # 7) Build the final alpha: only inside the hard region, using the soft values
    alpha = np.zeros((H, W), np.float32)
    alpha[warped_hard] = warped_soft[warped_hard]
    alpha = alpha[..., None]  # shape (H, W, 1)

    # 8) Alpha-blend
    pano_f  = pano.astype(np.float32)
    patch_f = patch.astype(np.float32)
    pano_new = pano_f * (1 - alpha) + patch_f * alpha

    return pano_new.astype(np.uint8)


# 3) Blend back
pano0 = blend_patch_into_pano(
    pano    = pano,
    tile    = "wide_out.jpg",
    hard_mask = hard_mask,
    soft_mask = soft_mask,
    yaw=45, pitch=0, fov=90
)
cv2.imwrite("pano0.jpg", pano0)