import cv2
import numpy as np
from copy import deepcopy
from typing import Tuple

# ───────────────────────────────────────────────────────────────
# My implementations
from utils.persp_conv import (
    equirectangular_to_perspective,
    perspective_to_equirectangular,
    
)

from utils.inpaint import inpaint_image

# ───────────────────────────────────────────────────────────────
#  PARAMETERS

PANO_PATH          = "anchored.jpg"        # the anchored equirectangular you saved
ATMOS_PROMPT       = "ATMOSPHERE PROMPT"
SKY_PROMPT         = "SKY PROMPT"
GROUND_PROMPT      = "GROUND PROMPT"
OUT_PANO_PATH      = "full_pano.jpg"

OUT_W, OUT_H       = 2048, 1024            # final pano size (2:1)
SIDE_FOV           = 85                    # °  for horizontal band
TOPBOT_FOV         = 120                   # °  for sky & ground
PERS_SIZE          = 1024                  # square perspective resolution
ALPHA_FEATHER      = 30                    # px for feather blend

# ───────────────────────────────────────────────────────────────
def alpha_blend(base: np.ndarray, patch: np.ndarray, mask: np.ndarray, feather: int) -> np.ndarray:
    """Feather‑blend 'patch' into 'base' using 'mask'."""
    if feather > 0:
        k = max(1, feather // 2 * 2 + 1)  # make kernel size odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    base32, patch32, mask32 = [x.astype(np.float32) for x in (base, patch, mask)]
    mask32 = mask32 / 255.0               # ensure it's in [0,1]
    if mask32.ndim == 2:
        mask32 = mask32[:, :, None]      # expand to (H, W, 1)

    blend = base32 * (1 - mask32) + patch32 * mask32
    return blend.astype(base.dtype)


# ───────────────────────────────────────────────────────────────
def create_mask(h: int, w: int, border_px: int = 0) -> np.ndarray:
    """Return a white mask the size of the perspective image, optionally erasing borders."""
    m = np.ones((h, w), np.uint8)*255
    if border_px > 0:
        m[:border_px] = m[-border_px:] = m[:, :border_px] = m[:, -border_px:] = 0
    return m

# ───────────────────────────────────────────────────────────────
def main() -> None:

    pano = cv2.imread(PANO_PATH, cv2.IMREAD_COLOR)          # BGR
    pano = cv2.resize(pano, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # 1. DEFINE CAMERA POSES
    poses: list[Tuple[float,float,float,str]] = []

    # 8 side views (yaw 0…315, pitch 0)
    for yaw in range(0, 360, 45):
        poses.append((yaw, 0, SIDE_FOV, ATMOS_PROMPT))

    # 4 sky views
    for yaw in (0, 90, 180, 270):
        poses.append((yaw,  90, TOPBOT_FOV, SKY_PROMPT))
        poses.append((yaw, -90, TOPBOT_FOV, GROUND_PROMPT))

    # ------------------------------------------------------------------
    # 2. PROCESS EACH VIEW
    for idx, (yaw, pitch, fov, prompt) in enumerate(poses):
        # 2‑a  crop perspective
        persp = equirectangular_to_perspective(
            pano[..., ::-1],   # convert BGR→RGB if your fn expects RGB
            yaw, pitch, fov, PERS_SIZE
        )[..., ::-1]          # back to BGR for OpenCV

        # 2‑b  create + save mask
        mask = create_mask(persp.shape[0], persp.shape[1])
        mask_path   = f"tmp_mask_{idx}.png"
        persp_path  = f"tmp_persp_{idx}.jpg"
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(persp_path, persp)

        # 2‑c  inpaint
        inpainted_path = f"tmp_inpaint_{idx}.jpg"
        inpaint_image(persp_path, mask_path, prompt, inpainted_path)
        inpainted = cv2.imread(inpainted_path)

        # 2‑d  warp back to equirect
        patch = perspective_to_equirectangular(
            inpainted[..., ::-1], yaw, pitch, fov, OUT_W, OUT_H
        )[..., ::-1]   # back to BGR

        # binary mask where patch ≠ 0
        patch_mask = (patch.sum(-1, keepdims=True) > 0).astype(np.uint8)

        # 2‑e  alpha‑blend
        pano = alpha_blend(pano, patch, patch_mask, ALPHA_FEATHER)

    # ------------------------------------------------------------------
    # 3. SAVE / OPTIONAL PARTIAL DENOISE
    cv2.imwrite(OUT_PANO_PATH, pano)
    print(f"Saved full panorama to {OUT_PANO_PATH}")

    # If you want partial denoise with SD, feed OUT_PANO_PATH as full image
    # and run last 30% timesteps using the ATMOS_PROMPT.

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
