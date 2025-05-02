import numpy as np

def backproject_depth_to_camera(depth: np.ndarray, fov: float) -> np.ndarray:
    """
    Given a depth map (H×W) in meters and the horizontal field‑of‑view in degrees,
    return a (H*W, 3) array of 3D points in the camera’s local frame.
    """
    H, W = depth.shape
    # pixel coordinates
    i, j = np.indices((H, W), dtype=np.float32)
    # principal point at center
    cx, cy = W / 2.0, H / 2.0
    # focal length in pixels
    f = W / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    # camera‑space coordinates
    x = (j - cx) * depth / f
    y = (i - cy) * depth / f
    z = depth
    pts_cam = np.stack((x, y, z), axis=-1)  # H×W×3
    return pts_cam.reshape(-1, 3)           # (H*W)×3

def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix that first pitches (about X), then yaws (about Y).
    Yaw, pitch in degrees.
    """
    y, p = np.deg2rad(yaw), np.deg2rad(pitch)
    # pitch around X
    Rx = np.array([[1,         0,          0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]])
    # yaw around Y
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [         0, 1,        0],
                   [-np.sin(y), 0, np.cos(y)]])
    return Ry @ Rx

def pano_to_pointcloud(
    pano_bgr: np.ndarray,
    depths: dict[str, np.ndarray],
    pitch_map: dict[str, float],
    yaw_lists: dict[str, list[float]],
    fov_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per‑view depth maps into one global point cloud with colors.
    Returns:
      points: (N,3) float32 array of XYZ in panorama frame
      colors: (N,3) uint8 array of RGB
    """
    all_pts = []
    all_cols = []
    for region, pitch in pitch_map.items():
        for yaw in yaw_lists[region]:
            tag = f"{region}_{int(yaw)}"
            depth = depths[tag]               # H×W depth in meters
            H, W = depth.shape
            fov = fov_map[region]

            # 1) back‑project to camera frame
            pts_cam = backproject_depth_to_camera(depth, fov)  # (H*W)×3

            # 2) rotate into world/panorama frame
            R = rotation_matrix(yaw=yaw, pitch=pitch)
            pts_world = (R @ pts_cam.T).T                     # (H*W)×3

            # 3) sample color from the same view
            #    re‑crop colors from panorama
            crop_rgb = equirectangular_to_perspective(
                pano_bgr, yaw=yaw, pitch=pitch,
                fov=fov, width=W, height=H
            )
            cols = crop_rgb.reshape(-1, 3)                   # (H*W)×3

            all_pts.append(pts_world)
            all_cols.append(cols)

    # concatenate all views
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    # (optional) filter out points at zero depth or outliers here…

    return points, colors
