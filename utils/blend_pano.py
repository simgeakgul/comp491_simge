import numpy as np   
import cv2
from utils.persp_conv import perspective_to_equirectangular 
def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    dilate: int,
    yaw: float,
    pitch: float,
    fov: float
) -> np.ndarray:
    """
    Paste only the original (non-dilated) inpainted pixels from `tile` back into `pano`.
    
    - Erodes `mask` by `dilate` to undo the dilation step.
    - Warps both the inpainted `tile` and the eroded mask back to the panorama.
    - Copies only the eroded-mask pixels from the warped tile into the pano.
    """
    # 1) remove dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    mask_nodilate = cv2.erode(mask, kernel, iterations=1)

    # 2) warp tile and eroded mask back into pano coords
    H, W = pano.shape[:2]
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    # ensure single-channel mask for warping
    warped_mask = perspective_to_equirectangular(
        mask_nodilate, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    # binarize
    mask_bool = warped_mask > 127

    # 3) composite
    pano_filled = pano.copy()
    pano_filled[mask_bool] = warped_tile[mask_bool]

    return pano_filled