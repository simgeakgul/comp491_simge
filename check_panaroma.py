import cv2
import numpy as np
from typing import Tuple

# ───────────────────────────────────────────────────────────────
# Your projection helpers
from utils.persp_conv import (
    equirectangular_to_perspective,
    perspective_to_equirectangular,
)
# ───────────────────────────────────────────────────────────────
PANO_PATH      = "anchored.jpg"      # anchored equirectangular
GRID_PATH      = "grid.jpg"          # checkerboard debug texture
OUT_PANO_PATH  = "full_pano.jpg"

#  RESOLUTION / FOV
OUT_W, OUT_H   = 2048, 1024          # panorama 2:1
PERS_SIZE      = 1024                # square perspective patches
SIDE_FOV       = 85                  # equatorial views
TOPBOT_FOV     = 120                 # sky / ground

ALPHA_FEATHER  = 30                  # feather‑blend in pixels
# ───────────────────────────────────────────────────────────────
def alpha_blend(base: np.ndarray,
                patch: np.ndarray,
                mask:  np.ndarray,
                feather: int) -> np.ndarray:
    """Feather‑blend patch into base using binary mask."""
    if feather > 0:
        k = max(1, feather // 2 * 2 + 1)       # odd kernel size
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    base32, patch32, mask32 = [x.astype(np.float32) for x in (base, patch, mask)]
    mask32 = mask32 / 255.0
    if mask32.ndim == 2:                       # (H,W) → (H,W,1)
        mask32 = mask32[:, :, None]
    return (base32 * (1 - mask32) + patch32 * mask32).astype(base.dtype)

# ───────────────────────────────────────────────────────────────
def main() -> None:
    # 0.  load images
    pano = cv2.imread(PANO_PATH, cv2.IMREAD_COLOR)
    pano = cv2.resize(pano, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    grid = cv2.imread(GRID_PATH, cv2.IMREAD_COLOR)
    grid = cv2.resize(grid, (PERS_SIZE, PERS_SIZE))

    # 1. define 16 camera poses
    poses: list[Tuple[float, float, float, str]] = []
    for yaw in range(0, 360, 45):                         # 8 side views
        poses.append((yaw, 0,   SIDE_FOV, "side"))
    for yaw in (0, 90, 180, 270):                         # 4 sky + 4 ground
        poses.append((yaw,  90, TOPBOT_FOV, "sky"))
        poses.append((yaw, -90, TOPBOT_FOV, "ground"))

    # 2. process each view
    for idx, (yaw, pitch, fov, tag) in enumerate(poses):
        # 2‑a project checkerboard to equirectangular
        patch = perspective_to_equirectangular(
            grid[..., ::-1],  # BGR→RGB if your func expects RGB
            yaw, pitch, fov,
            OUT_W, OUT_H
        )[..., ::-1]          # back to BGR

        # 2‑b save patch for visual debug
        # cv2.imwrite(f"patch_{idx:02d}_{tag}.png", patch)

        # 2‑c binary mask where patch present
        patch_mask = (patch.sum(-1, keepdims=True) > 0).astype(np.uint8)*255

        # 2‑d blend into panorama
        # pano = alpha_blend(pano, patch, patch_mask, ALPHA_FEATHER)
        pano = alpha_blend(pano, patch, patch_mask, 0)

    # 3. write full panorama
    cv2.imwrite(OUT_PANO_PATH, pano)
    print(f"Saved blended panorama to {OUT_PANO_PATH}")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
