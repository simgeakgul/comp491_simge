import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class PanoConfig:
    
    out_w: int
    out_h: int

    pitch_map:          Dict[str, float]
    horizontal_yaws:    List[int]
    sky_yaws:           List[int]
    ground_yaws:        List[int]
    fov_map:            Dict[str, float]
    guidance_scale:     float
    steps:              int
    dilate_pixel:       int
    fovdeg:             float
    border_px:          int

    crop_size:          int
    edge_sigma:         float
    center_bias:        float
    align_depth:        bool
    in_out:             str

def load_config(path: str | Path = "config.json") -> PanoConfig:
    data = json.loads(Path(path).read_text())
    return PanoConfig(

        out_w          = data["out_w"],
        out_h          = data["out_h"],

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

        crop_size      = data["crop_size"],
        edge_sigma     = data["edge_sigma"],
        center_bias    = data["center_bias"],
        align_depth    = data["align_depth"],
        in_out         = data["in_out"],
    )
