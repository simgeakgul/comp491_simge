import numpy as np   
import cv2
from utils.persp_conv import perspective_to_equirectangular 

def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float
) -> np.ndarray:
    """
    Warps `tile` and the *original* mask back into pano coordinates,
    and composites only where mask==255.
    """
    H, W = pano.shape[:2]

    # warp both the inpainted tile and the original mask
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # composite
    pano_out = pano.copy()
    mask_bool = warped_mask > 127
    pano_out[mask_bool] = warped_tile[mask_bool]

    return pano_out

def put_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    blur_kernel_size: int = 15  # <- Adjustable
) -> np.ndarray:
    """
    Warps tile and mask back into pano coordinates,
    and blends smoothly based on blurred mask edges.
    """
    H, W = pano.shape[:2]

    # Warp both the inpainted tile and the original mask
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # Ensure mask is single-channel if it's not already
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)

    # Normalize mask to [0,1]
    warped_mask = warped_mask.astype(np.float32) / 255.0

    # Blur the mask to smooth the transition
    warped_mask = cv2.GaussianBlur(warped_mask, (blur_kernel_size, blur_kernel_size), 0)

    # Expand mask to 3 channels if needed
    if pano.ndim == 3 and pano.shape[2] == 3:
        warped_mask = np.expand_dims(warped_mask, axis=-1)

    # Blend pano and tile according to the mask
    pano_out = (1.0 - warped_mask) * pano + warped_mask * warped_tile
    pano_out = np.clip(pano_out, 0, 255).astype(np.uint8)

    return pano_out
