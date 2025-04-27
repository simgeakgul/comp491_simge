import os
import cv2
import numpy as np
from utils.persp_conv import (
    equirectangular_to_perspective,
    perspective_to_equirectangular
)

def test_full_reconstruction(
    pano: np.ndarray,
    horizontal_yaws: list,
    sky_yaws: list,
    ground_yaws: list,
    fov_map: dict,
    width: int = 512,
    height: int = 512,
    debug_folder: str = "debug_reconstruct"
) -> np.ndarray:
    """
    1) Creates a black canvas same size as pano
    2) For each view in horizontal / sky / ground:
         a) Crop a perspective
         b) Warp it back to equirectangular (patch)
         c) Paste patch onto canvas
         d) Save crop, patch, and canvas step image
    3) Save final reconstructed pano
    """
    os.makedirs(debug_folder, exist_ok=True)
    canvas = np.zeros_like(pano)

    # Build list of (yaw, pitch, key) triples
    views = []
    for yaw in horizontal_yaws:
        views.append((yaw, 0.0, "atmosphere"))
    for yaw in sky_yaws:
        views.append((yaw, 90.0, "sky_or_ceiling"))
    for yaw in ground_yaws:
        views.append((yaw, -90.0, "ground_or_floor"))

    for idx, (yaw, pitch, key) in enumerate(views):
        fov = fov_map[key]

        # 1) Crop perspective
        persp = equirectangular_to_perspective(
            pano, yaw=yaw, pitch=pitch, fov=fov,
            width=width, height=height
        )
        cv2.imwrite(f"{debug_folder}/step{idx:02d}_crop_{key}_{int(yaw)}_{int(pitch)}.jpg", persp)

        # 2) Warp back to pano patch
        patch = perspective_to_equirectangular(
            persp,
            yaw=yaw, pitch=pitch, fov=fov,
            width=pano.shape[1],
            height=pano.shape[0]
        )
        cv2.imwrite(f"{debug_folder}/step{idx:02d}_patch_{key}_{int(yaw)}_{int(pitch)}.jpg", patch)

        # 3) Paste onto canvas
        mask = (patch.sum(axis=2) > 0)
        canvas[mask] = patch[mask]
        cv2.imwrite(f"{debug_folder}/step{idx:02d}_canvas_{key}_{int(yaw)}_{int(pitch)}.jpg", canvas)

    # 4) Save final result
    cv2.imwrite(f"{debug_folder}/final_reconstructed_pano.jpg", canvas)
    return canvas

horizontal_yaws = [45, 90, 135, 225, 270, 315]
sky_yaws        = [0, 90, 180, 270]
ground_yaws     = [0, 90, 180, 270]

fov_map = {
    "atmosphere":      90.0,
    "sky_or_ceiling": 120.0,
    "ground_or_floor":120.0,
}

pano = cv2.imread("pano.jpg")

reconstructed = test_full_reconstruction(
    pano=pano,
    horizontal_yaws=horizontal_yaws,
    sky_yaws=sky_yaws,
    ground_yaws=ground_yaws,
    fov_map=fov_map,
    width=512,
    height=512,
    debug_folder="debug_reconstruct"
)

