import cv2
from utils.depth_prediction import process_pano, process_view
from utils.load_models import load_pipe


PITCH_MAP = {
    "atmosphere": 0.0,
    "sky_or_ceiling": 60.0,
    "ground_or_floor": -60.0,
}
YAW_LISTS = {
    "atmosphere": [45, -45, 135, -135, 90, -90, 180, -180],
    "sky_or_ceiling": [0, 90, 180, 270],
    "ground_or_floor": [0, 90, 180, 270],
}
FOV_MAP = {
    "atmosphere": 80.0,
    "sky_or_ceiling": 110.0,
    "ground_or_floor": 110.0,
}


pano = cv2.imread("pano1.jpg")

depths = process_pano(pano, PITCH_MAP, YAW_LISTS, FOV_MAP, scene_type="outdoor")







