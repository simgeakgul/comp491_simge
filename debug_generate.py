import cv2
from pathlib import Path
from typing import Tuple, Union
import numpy as np                
from utils.persp_conv import equirectangular_to_perspective, equirectangular_to_cylindrical
from utils.center_img import center_image

import cv2
import numpy as np
from typing import Tuple, Union

def save_pano_crop(
    pano_path: str | Path,
    out_path: str | Path,
    yaw: float,
    pitch: float,
    fov: float,
    resolution: Union[int, Tuple[int, int]] = 1024,
    bgr: bool = True                # True if panorama is BGR (OpenCV default)
) -> None:
    """
    Extract a perspective crop from an equirectangular panorama and save it.

    Parameters
    ----------
    pano_path   : Input panorama file (equirectangular, 2:1 aspect).
    out_path    : Where to write the JPEG crop.
    yaw         : Heading in degrees (0° = look forward, + = turn right).
    pitch       : Pitch in degrees (+ = look up, − = look down).
    fov         : Horizontal FOV of the crop in degrees.
    resolution  : Output size.  Either single int (square) or (W, H) tuple.
    bgr         : Set False if the panorama is RGB instead of BGR.
    """
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(pano_path)

    if not bgr:
        pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)

    crop = equirectangular_to_perspective(
        pano, yaw=yaw, pitch=pitch, fov=fov, resolution=resolution
    )

    # JPEG quality 95 is a good balance; tweak if needed
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

image = cv2.imread("input.jpg")
pano = center_image(image, fov_deg=90, out_w=2048, out_h=1024)
cv2.imwrite("pano.jpg", pano)

save_pano_crop(
    pano_path="pano.jpg",          # 360×180 source image
    out_path="view.jpg",                # file to write
    yaw=45,                             
    pitch=0,                            # keep level
    fov=90,                             # 90° horizontal field-of-view
    resolution=(1280, 720)              # 16:9 output
)

cyl_crop = equirectangular_to_cylindrical(
                 pano, yaw=45, pitch=0, fov=90, resolution=(1600, 800))
cv2.imwrite("flat.jpg", cyl_crop)

