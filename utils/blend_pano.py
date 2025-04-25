import numpy as np   
from utils.persp_conv import perspective_to_equirectangular 

def blend_patch_into_pano(
    pano: np.ndarray,
    tile: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float
) -> np.ndarray:

    H, W = pano.shape[:2]

    # 1) Reproject the tile into the panorama frame
    patch = perspective_to_equirectangular(tile, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H)

    # 2) Prepare masks as 0/255 uint8
    hm_bin = (hard_mask.astype(bool)).astype(np.uint8) * 255
    sm_f = soft_mask.astype(np.float32)
    sm_arr = (sm_f * 255).astype(np.uint8) if sm_f.max() <= 1.0 else sm_f.astype(np.uint8)

    # 3) Stack to 3‐channel so we can reuse the projection API
    hm3 = np.stack([hm_bin]*3, axis=-1)
    sm3 = np.stack([sm_arr]*3, axis=-1)

    # 4) Warp both masks back onto the pano
    warped_hm3 = perspective_to_equirectangular(hm3, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H)
    warped_sm3 = perspective_to_equirectangular(sm3, yaw=yaw, pitch=pitch, fov=fov, width=W, height=H)

    # 5) Extract boolean hard mask and float soft mask
    warped_hard = warped_hm3[..., 0] > 128
    warped_soft = warped_sm3[..., 0].astype(np.float32) / 255.0

    # 6) Build alpha map (H×W×1)
    alpha = np.zeros((H, W), np.float32)
    alpha[warped_hard] = warped_soft[warped_hard]
    alpha = alpha[..., None]

    # 7) Alpha-blend
    pano_f  = pano.astype(np.float32)
    patch_f = patch.astype(np.float32)
    blended = pano_f * (1 - alpha) + patch_f * alpha

    return blended.astype(np.uint8)