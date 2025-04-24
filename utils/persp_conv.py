import cv2
import numpy as np

def equirectangular_to_perspective(
    equi_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    resolution: tuple[int, int] | int
) -> np.ndarray:
    """
    Convert an equirectangular panorama to a perspective view.

    Parameters
    ----------
    equi_img  : (H, W, 3/4) uint8/float32 BGR or RGB array.
    yaw       : Yaw (heading) in degrees. +yaw rotates camera to the right.
    pitch     : Pitch in degrees. +pitch looks up, −pitch looks down.
    fov       : Horizontal field‑of‑view in degrees.
    resolution: Either (width, height) or a single int for square output.

    Returns
    -------
    pers      : Perspective image (height, width, channels) same dtype as input.
    """
    # ---------------------------- parameters ----------------------------
    if isinstance(resolution, int):
        W = H = resolution
    else:
        W, H = resolution
    equi_h, equi_w = equi_img.shape[:2]

    # convert to radians
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    fov   = np.deg2rad(fov)

    # vertical fov from aspect ratio
    fov_v = 2 * np.arctan(np.tan(fov/2) * (H / W))

    # ---------------------------- pixel grid ----------------------------
    x = np.linspace(-np.tan(fov/2),  np.tan(fov/2),  W, dtype=np.float32)
    y = np.linspace( np.tan(fov_v/2),-np.tan(fov_v/2), H, dtype=np.float32)
    xs, ys = np.meshgrid(x, y)                     # shape (H, W)

    # camera‑space directions (before rotation)
    zs = np.ones_like(xs)
    dirs = np.stack([xs, ys, zs], axis=-1)         # (H, W, 3)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # ---------------------------- rotate by yaw & pitch -----------------
    # Rotation around y (yaw), then x (pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    R_yaw   = np.array([[ cy, 0., sy],
                        [0., 1., 0.],
                        [-sy, 0., cy]], dtype=np.float32)

    R_pitch = np.array([[1., 0.,  0.],
                        [0., cp, -sp],
                        [0., sp,  cp]], dtype=np.float32)

    R = R_pitch @ R_yaw
    dirs = dirs @ R.T                                     # (H, W, 3)

    # ---------------------------- dirs → spherical ----------------------
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])          # [-π, π]
    lat = np.arcsin(np.clip(dirs[..., 1], -1., 1.))       # [-π/2, π/2]

    # map to pixel coords in equirectangular
    u = (lon + np.pi) / (2 * np.pi) * equi_w             # [0, W)
    v = (np.pi/2 - lat) / np.pi * equi_h                  # [0, H)

    # ---------------------------- remap ----------------------------
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    pers  = cv2.remap(equi_img, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_WRAP)
    return pers


def perspective_to_equirectangular(
    pers_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    out_width: int,
    out_height: int
) -> np.ndarray:
    """
    Warp a square or rectangular perspective image *back* onto an
    equirectangular canvas.

    Parameters
    ----------
    pers_img   : (Hp, Wp, C) uint8 / float32 perspective (BGR or RGB).
    yaw        : Heading of the perspective camera in **degrees** (+ = turn right).
    pitch      : Pitch in **degrees** (+ = look up, − = look down).
    fov        : Horizontal field‑of‑view of the perspective camera in **degrees**.
    out_width  : Width  of the full equirectangular panorama (e.g. 4096).
    out_height : Height of the full equirectangular panorama (e.g. 2048).

    Returns
    -------
    pano_patch : (out_height, out_width, C) same dtype as input,  
                 containing the perspective view re‑projected to equirectangular.
                 Pixels outside the perspective’s FOV are left **black** (zeros).
    """

    Hp, Wp = pers_img.shape[:2]

    # ---------------------- pre‑compute constants ----------------------
    # Convert to radians
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    fov_h = np.deg2rad(fov)
    # vertical FOV given aspect ratio
    fov_v = 2 * np.arctan(np.tan(fov_h / 2) * (Hp / Wp))
    tan_h = np.tan(fov_h / 2)
    tan_v = np.tan(fov_v / 2)

    # Rotation matrices (world -> cam)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    R_yaw   = np.array([[ cy, 0., sy],
                        [0.,  1., 0.],
                        [-sy, 0., cy]], dtype=np.float32)
    R_pitch = np.array([[1., 0.,  0.],
                        [0., cp, -sp],
                        [0., sp,  cp]], dtype=np.float32)

    R = (R_pitch @ R_yaw).T          # inverse: world→cam

    # ---------------------- build pano pixel grid ----------------------
    # Spherical angles for each pano pixel
    j  = np.linspace(0, out_width-1,  out_width, dtype=np.float32)
    i  = np.linspace(0, out_height-1, out_height, dtype=np.float32)
    jj, ii = np.meshgrid(j, i)                               # (H, W)

    lon = (jj / out_width)  * 2*np.pi - np.pi                # [-π, π]
    lat =  np.pi/2 - (ii / out_height) * np.pi               # [-π/2, π/2]

    # World‑space unit vectors
    xw = np.cos(lat) * np.sin(lon)
    yw = np.sin(lat)
    zw = np.cos(lat) * np.cos(lon)
    dirs_world = np.stack([xw, yw, zw], axis=-1)            # (H, W, 3)

    # Rotate into camera space
    dirs_cam = dirs_world @ R.T                             # (H, W, 3)

    # Perspective projection
    xc = dirs_cam[..., 0]
    yc = dirs_cam[..., 1]
    zc = dirs_cam[..., 2]

    # points in front of camera
    z_positive = zc > 0

    # Normalised image‑plane coords
    x_img =  xc / zc
    y_img =  yc / zc

    in_fov = (np.abs(x_img) <= tan_h) & (np.abs(y_img) <= tan_v) & z_positive

    # Map to perspective pixel coordinates
    u =  ( x_img / tan_h + 1) * 0.5 * (Wp-1)
    v = (-y_img / tan_v + 1) * 0.5 * (Hp-1)

    # ---------------------- build remap grids --------------------------
    map_x = np.full_like(u, -1, dtype=np.float32)  # -1 ⇒ cv2 will leave pixel black
    map_y = np.full_like(v, -1, dtype=np.float32)
    map_x[in_fov] = u[in_fov].astype(np.float32)
    map_y[in_fov] = v[in_fov].astype(np.float32)

    # ---------------------- warp perspective → pano --------------------
    pano_patch = cv2.remap(
        pers_img,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return pano_patch

def equirectangular_to_cylindrical(
    equi_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,                 # horizontal FOV of the desired view (deg)
    resolution: Union[int, Tuple[int, int]] = 1024
) -> np.ndarray:
    """
    Extract a cylindrical (not rectilinear) view from an equirectangular panorama.
    Horizontal lines stay horizontal, verticals stay vertical, and side–stretch
    is greatly reduced, at the cost of curved diagonals.
    """
    # ---------- geometry ----------
    if isinstance(resolution, int):
        W = H = resolution
    else:
        W, H = resolution

    fov = np.deg2rad(fov)
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    f = (W / 2) / np.tan(fov / 2)          # focal length in pixels
    equi_h, equi_w = equi_img.shape[:2]

    # ---------- output pixel grid ----------
    x = np.arange(W, dtype=np.float32) - W/2     # centred coords
    y = np.arange(H, dtype=np.float32) - H/2
    xs, ys = np.meshgrid(x, y)                   # shape (H,W)

    # cylindrical → spherical
    lon = xs / f                                 # rad
    lat = np.arctan(ys / f)

    # apply camera yaw / pitch
    lon += yaw
    lat += pitch

    # map spherical → equirectangular indices
    u = (lon + np.pi) / (2*np.pi) * equi_w
    v = (np.pi/2 - lat) / np.pi * equi_h

    map_x = u.astype(np.float32)
    map_y = -v.astype(np.float32)

    cyl = cv2.remap(equi_img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
    return cyl