import os
import cv2
import numpy as np
from utils.load_configs import load_config
from utils.depth_prediction import build_depth_panorama

def main():
    # load yaml config into dataclass
    base = "test_folders/landscape"
    cfg  = load_config(os.path.join(base, "config.yaml"))

    # build yaw_lists dict from cfg
    yaw_lists = {
        "atmosphere":      cfg.horizontal_yaws,
        "sky_or_ceiling":  cfg.sky_yaws,
        "ground_or_floor": cfg.ground_yaws,
    }

    # read panorama
    pano_path = os.path.join(base, "fixed_pano.jpg")
    pano_bgr = cv2.imread(pano_path, cv2.IMREAD_COLOR)

    if pano_bgr is None:
        raise FileNotFoundError(f"Could not read panorama: {pano_path}")

    # compute depth panorama
    depth_pano, cov_mask = build_depth_panorama(
        pano_bgr    = pano_bgr,
        scene_type  = cfg.in_out,         
        pitch_map   = cfg.pitch_map,
        yaw_lists   = yaw_lists,
        fov_map     = cfg.fov_map,
        crop_size   = cfg.crop_size,
        edge_sigma  = cfg.edge_sigma,
        center_bias = cfg.center_bias,
        align_depth = cfg.align_depth,
    )

    # save NumPy and visualization
    depth_map = os.path.join(base, "depth_pano.jpg")
    # dept_points = os.path.join(base, "pano_depth.npy")
    # np.save(dept_points, depth_pano)

    vis = np.zeros_like(depth_pano)
    nz = depth_pano > 0
    if nz.any():
        dmin, dmax = depth_pano[nz].min(), depth_pano[nz].max()
        vis[nz] = (depth_pano[nz] - dmin) / (dmax - dmin + 1e-6)
    cv2.imwrite(depth_map, (vis * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
