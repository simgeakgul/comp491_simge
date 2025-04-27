import numpy as np   
import cv2
from utils.persp_conv import perspective_to_equirectangular 

def put_patch_into_pano(
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



def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    border_width: int = 16
) -> np.ndarray:
    """
    Warps `tile` and `mask` back into pano coords, then does
    a smooth, border-only blend over `border_width` px.
    """
    H, W = pano.shape[:2]

    # 1) Warp both the inpainted tile and the original mask
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )

    # 2) Ensure mask is single-channel uint8
    #    If warped_mask is 3-channel, convert to gray:
    if warped_mask.ndim == 3 and warped_mask.shape[2] == 3:
        warped_mask_gray = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    else:
        warped_mask_gray = warped_mask.copy()
    # 3) Binary mask: 0 or 255, dtype=uint8
    mask_bin = ((warped_mask_gray > 127) * 255).astype(np.uint8)

    # 4) Distance transform *inside* the mask
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)

    # 5) Build alpha matte: alpha = dist/border_width, clipped [0,1]
    alpha = np.clip(dist.astype(np.float32) / float(border_width), 0.0, 1.0)

    # 6) Composite only inside mask
    pano_out = pano.astype(np.float32)
    inside = mask_bin.astype(bool)
    a3 = alpha[..., None]  # expand to 3 channels

    pano_out[inside] = (
        a3[inside] * warped_tile[inside].astype(np.float32)
        + (1.0 - a3[inside]) * pano_out[inside]
    )

    return np.clip(pano_out, 0, 255).astype(np.uint8)