import os
import cv2
import numpy as np
from utils.load_configs import load_config, PanoConfig
from utils.super_reso import build_border_mask, save_pano_with_mask_overlay, split_pano_and_mask
from utils.inpaint import inpaint_image


def main():
    base = "test_folders/achilles"
    cfg  = load_config(os.path.join(base, "config.yaml"))

    pano_path = os.path.join(base, "pano.jpg")
    pano      = cv2.imread(pano_path)
    all_views = []
    for category, yaw_list in [
        ("atmosphere",      cfg.horizontal_yaws),
        ("sky_or_ceiling",  cfg.sky_yaws),
        ("ground_or_floor", cfg.ground_yaws),
    ]:
        pitch = cfg.pitch_map[category]
        fov   = cfg.fov_map.get(category, cfg.fovdeg)
        for yaw in yaw_list:
            all_views.append((yaw, pitch, fov))

    seam_mask = build_border_mask(
        pano       = pano,
        views      = all_views,
        border_px  = cfg.border_px,
        center_fov = cfg.fovdeg,
        debug_path = os.path.join(base, "border_mask.jpg")
    )

    vertical_num   = 2
    horizontal_num = 4
    tiles = split_pano_and_mask(pano, seam_mask, vertical_num, horizontal_num)

    fixed_pano = np.zeros_like(pano)

    prompt         = "Semales transaction"
    dilate_px      = 1
    guidance_scale = cfg.guidance_scale
    steps          = cfg.steps

    for (row, col), info in tiles.items():
        pano_tile = info['pano']
        mask_tile = info['mask']
        y1, y2, x1, x2 = info['coords']

        fixed_tile = inpaint_image(
            image_arr      = pano_tile,
            mask_arr       = mask_tile,
            prompt         = prompt,
            dilate_px      = dilate_px,
            guidance_scale = guidance_scale,
            steps          = steps
        )

        fixed_pano[y1:y2, x1:x2] = fixed_tile

    fixed_path = os.path.join(base, "fixed_pano.jpg")
    cv2.imwrite(fixed_path, fixed_pano)

    debug_fixed_folder = os.path.join(base, "debug_fixed_tiles")
    os.makedirs(debug_fixed_folder, exist_ok=True)
    for (row, col), info in tiles.items():
        y1, y2, x1, x2 = info['coords']
        cv2.imwrite(
            os.path.join(debug_fixed_folder, f"fixed_r{row}_c{col}.jpg"),
            fixed_pano[y1:y2, x1:x2]
        )

if __name__ == "__main__":
    main()
