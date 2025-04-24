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
from utils.inpaint import load_image_and_mask_from_black, inpaint_image

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
    width=1280,
    height=720,
    bgr=True
)
cv2.imwrite("view.jpg", persp, [cv2.IMWRITE_JPEG_QUALITY, 95])

# 3) cylindrical crop & save
cyl = equirectangular_to_cylindrical(
    pano,
    yaw=45,
    pitch=0,
    fov=90,
    width=1600,
    height=800
)
cv2.imwrite("flat.jpg", cyl)

# 4) build mask from black regions and save
image, mask = load_image_and_mask_from_black("flat.jpg", threshold=10)
mask.save("mask.jpg")


result = inpaint_image(
    image=image,
    mask=mask,
    prompt=(
        "Continue the alpine scene: pine trees and ground, "
        "matching the original artist’s soft brush strokes, lighting and color palette"
    ),
    guidance_scale=11.0,
    steps=50
)

result.save("wide_out.jpg")
print(">> Wide_out.jpg written.")





def blend_patch_into_pano(
    pano: np.ndarray,          # (H, W, 3) current panorama (uint8)
    tile: np.ndarray,      # (Hp, Wp, 3) freshly in-painted perspective view
    yaw: float, pitch: float, fov: float,
) -> np.ndarray:
    """
    1. Re-project the perspective tile back onto equirectangular coords
       using your `perspective_to_equirectangular`.
    2. Build a soft alpha mask where the tile covers valid pixels.
    3. Alpha-blend the tile into `pano` and return the new panorama.
    """

    if isinstance(tile, Image.Image):  # PIL ➜ np ➜ BGR
        tile = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)

    H, W = pano.shape[:2]

    H, W = pano.shape[:2]

    # 1. warp tile back onto a blank equirect canvas

    patch = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # 2. create a simple “valid-pixel” mask (anything non-zero)
    mask = np.any(patch != 0, axis=-1)

    # 3. hard overwrite
    pano[mask] = patch[mask]
    return pano

pano0 = blend_patch_into_pano(
    pano = pano,
    tile = result,
    yaw=45, pitch=0, fov=90,
      
)

cv2.imwrite("pano0.jpg", pano0)