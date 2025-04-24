import cv2
import numpy as np

def equirectangular_to_perspective(
    equi_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    """
    Convert an equirectangular panorama to a perspective view.

    Parameters
    ----------
    equi_img : (H, W, C) uint8/float32 panorama (BGR or RGB)
    yaw      : heading in degrees (+ = turn right)
    pitch    : pitch in degrees (+ = look up)
    fov      : horizontal FOV in degrees
    width    : output image width in pixels
    height   : output image height in pixels

    Returns
    -------
    pers     : (height, width, C) same dtype as input
    """
    equi_h, equi_w = equi_img.shape[:2]

    # to radians
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    fov   = np.deg2rad(fov)

    # compute vertical FOV from aspect ratio
    fov_v = 2 * np.arctan(np.tan(fov/2) * (height / width))

    # pixel grid in camera space
    x = np.linspace(-np.tan(fov/2),  np.tan(fov/2),  width,  dtype=np.float32)
    y = np.linspace( np.tan(fov_v/2), -np.tan(fov_v/2), height, dtype=np.float32)
    xs, ys = np.meshgrid(x, y)            # (H, W)
    dirs = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # rotate by yaw (Y) then pitch (X)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_yaw   = np.array([[ cy, 0., sy],[0.,1.,0.],[-sy,0.,cy]], dtype=np.float32)
    R_pitch = np.array([[1.,0.,0.],[0.,cp,-sp],[0.,sp,cp]], dtype=np.float32)
    dirs = dirs @ (R_pitch @ R_yaw).T     # (H, W, 3)

    # to spherical
    lon = np.arctan2(dirs[...,0], dirs[...,2])      # [-π,π]
    lat = np.arcsin(np.clip(dirs[...,1], -1., 1.))  # [-π/2,π/2]

    # map to equirectangular coords
    u = (lon + np.pi) / (2*np.pi) * equi_w
    v = (np.pi/2 - lat)   / np.pi      * equi_h

    # remap
    return cv2.remap(
        equi_img,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )


def perspective_to_equirectangular(
    pers_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    """
    Re-project a perspective image back onto an equirectangular canvas.

    Parameters
    ----------
    pers_img : (Hp, Wp, C) uint8/float32 perspective (BGR or RGB)
    yaw      : heading in degrees (+ = turn right)
    pitch    : pitch in degrees (+ = look up)
    fov      : horizontal FOV in degrees
    width    : output panorama width in pixels
    height   : output panorama height in pixels

    Returns
    -------
    pano_patch : (height, width, C) same dtype as input.
                 Pixels outside the FOV remain black.
    """
    Hp, Wp = pers_img.shape[:2]

    # to radians
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    fov_h = np.deg2rad(fov)
    fov_v = 2 * np.arctan(np.tan(fov_h/2) * (Hp / Wp))
    tan_h = np.tan(fov_h/2)
    tan_v = np.tan(fov_v/2)

    # rotation (world→cam)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R = (np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]],dtype=np.float32) @
         np.array([[ cy,0, sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)).T

    # pano pixel grid → world dirs
    j = np.linspace(0, width-1,  width, dtype=np.float32)
    i = np.linspace(0, height-1, height, dtype=np.float32)
    jj, ii = np.meshgrid(j,i)
    lon =  (jj/width)  * 2*np.pi - np.pi
    lat =  np.pi/2 - (ii/height) * np.pi
    xw = np.cos(lat)*np.sin(lon)
    yw = np.sin(lat)
    zw = np.cos(lat)*np.cos(lon)
    dirs = np.stack([xw,yw,zw], axis=-1)  # (H, W, 3)

    # to camera space
    dc = dirs @ R.T
    xc, yc, zc = dc[...,0], dc[...,1], dc[...,2]
    in_view = (zc>0) & (np.abs(xc/zc)<=tan_h) & (np.abs(yc/zc)<=tan_v)

    # map to pers coords
    u = (( xc/zc / tan_h + 1)*0.5)*(Wp-1)
    v = ((-yc/zc / tan_v + 1)*0.5)*(Hp-1)

    map_x = np.full_like(u, -1, dtype=np.float32)
    map_y = np.full_like(v, -1, dtype=np.float32)
    map_x[in_view] = u[in_view]
    map_y[in_view] = v[in_view]

    return cv2.remap(
        pers_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def equirectangular_to_cylindrical(
    equi_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    """
    Extract a cylindrical view from an equirectangular panorama.

    Parameters
    ----------
    equi_img : (H, W, C) panorama
    yaw       : heading in degrees (+ = turn right)
    pitch     : pitch in degrees (+ = look up)
    fov       : horizontal FOV in degrees
    width     : output width in pixels
    height    : output height in pixels

    Returns
    -------
    cyl       : (height, width, C) same dtype as input
    """
    # to radians
    fov   = np.deg2rad(fov)
    yaw   = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    f = (width / 2) / np.tan(fov / 2)
    equi_h, equi_w = equi_img.shape[:2]

    x = np.arange(width,  dtype=np.float32) - width/2
    y = np.arange(height, dtype=np.float32) - height/2
    xs, ys = np.meshgrid(x, y)

    lon = xs / f
    lat = np.arctan(ys / f)
    lon += yaw
    lat += pitch

    u = (lon + np.pi) / (2*np.pi) * equi_w
    v = -(np.pi/2 - lat)  / np.pi       * equi_h

    return cv2.remap(
        equi_img,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
