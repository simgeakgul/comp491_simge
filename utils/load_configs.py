from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class PanoConfig:
    pitch_map:          dict[str, float]
    horizontal_yaws:    list[int]
    sky_yaws:           list[int]
    ground_yaws:        list[int]
    fov_map:            dict[str, float]
    guidance_scale:     float
    steps:              int
    dilate_pixel:       int
    fovdeg:             float
    border_px:          int

def load_config(path: str | Path = "config.yaml") -> PanoConfig:
    data = yaml.safe_load(Path(path).read_text())
    return PanoConfig(
        pitch_map      = data["pitch_map"],
        horizontal_yaws= data["horizontal_yaws"],
        sky_yaws       = data["sky_yaws"],
        ground_yaws    = data["ground_yaws"],
        fov_map        = data["fov_map"],
        guidance_scale = data["guidance_scale"],
        steps          = data["steps"],
        dilate_pixel   = data["dilate_pixel"],
        fovdeg         = data["fovdeg"],
        border_px      = data["border_px"],
    )
