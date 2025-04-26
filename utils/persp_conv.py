import cv2
import numpy as np
from utils.Equirec2Perspec import Equirectangular
from utils.Perspec2Equirec import Perspective

def equirectangular_to_perspective(
    equi_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    # now takes the array directly
    equ = Equirectangular(equi_img)
    # note: GetPerspective(fov, theta, phi, H, W)
    persp = equ.GetPerspective(fov, yaw, pitch, height, width)
    return persp

def perspective_to_equirectangular(
    pers_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    # feed the image array straight in
    per = Perspective(pers_img, fov, yaw, pitch)
    # GetEquirec(H, W) returns (equirect, mask)
    equirec, _ = per.GetEquirec(height, width)
    return equirec
