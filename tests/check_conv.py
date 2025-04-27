import cv2
import utils.Equirec2Perspec as E2P
from utils.persp_conv import equirectangular_to_perspective

pano = cv2.imread("pano.jpg")
equ = E2P.Equirectangular(pano)
img = equ.GetPerspective(120, 0, 0, 1280, 1280)
cv2.imwrite("perspective_0_0.png", img)

test = equirectangular_to_perspective(
    pano,
    0,    # yaw
    0,    # pitch
    120,  # fov
    1280, # width
    1280  # height
)
cv2.imwrite("persp_test.png", test)
