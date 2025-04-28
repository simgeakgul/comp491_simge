import numpy as np   
import cv2
from cv2.detail import GraphCutSeamFinder, MultiBandBlender
from utils.persp_conv import perspective_to_equirectangular 



def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    blur_radius: int = 31
) -> np.ndarray:
    """
    1) Warp the inpainted `tile` and the original binary `mask` back into pano coords.
    2) Blur the warped mask to get a soft alpha matte.
    3) Alpha-blend warped_tile into pano using that soft alpha, removing hard seams.
    """
    H, W = pano.shape[:2]

    # 1) Warp both tile and mask
    warped_tile = perspective_to_equirectangular(
        tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    warped_mask = perspective_to_equirectangular(
        mask, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H
    )
    # single-channel mask
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    # binary 0/255
    bin_mask = (warped_mask > 127).astype(np.uint8) * 255

    # 2) Blur to create soft alpha (0…255)
    #    Kernel size must be odd
    k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    soft_mask = cv2.GaussianBlur(bin_mask, (k, k), 0).astype(np.float32)

    # normalize alpha to [0..1]
    alpha = soft_mask / 255.0
    alpha = alpha[..., None]  # shape H×W×1

    # 3) Composite
    pano_f = pano.astype(np.float32)
    tile_f = warped_tile.astype(np.float32)

    pano_out = (1.0 - alpha) * pano_f + alpha * tile_f
    return np.clip(pano_out, 0, 255).astype(np.uint8)

