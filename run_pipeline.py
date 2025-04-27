import cv2
import json
import argparse
import os

from utils.center_img import center_image, complete_to_1024
from one_cycle import one_cycle

# --- Stable constants ---
PROMPTS_PATH    = "prompts.json"
INPUT_PANO      = "resized.jpg"
OUTPUT_PANO     = "full_pano.jpg"
PITCH_MAP       = {
    "atmosphere":       0.0,
    "sky_or_ceiling":  60.0,
    "ground_or_floor": -60.0,
}
HORIZONTAL_YAWS = [45, -45, 135, -135, 90, -90, 180, -180]
SKY_YAWS        = [0, 90, 180, 270]
GROUND_YAWS     = [0, 90, 180, 270]
FOV_MAP         = {
    "atmosphere":      80.0,
    "sky_or_ceiling": 110.0,
    "ground_or_floor": 110.0,
}
DILATE_PX       = 16
GUIDANCE_SCALE  = 11.0
STEPS           = 50


def generate_full_pano(img_path: str, fovdeg: float) -> None:

    # 1) Load image
    image = cv2.imread(img_path)
    resized = complete_to_1024(image_arr = image,  prompts_path = PROMPTS_PATH)
    pano0 = center_image(resized, fovdeg)

    # 2) Load prompts
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)

    # 3) Build list of (category, yaw)
    views = []
    for y in HORIZONTAL_YAWS:
        views.append(("atmosphere", y))
    for y in SKY_YAWS:
        views.append(("sky_or_ceiling", y))
    for y in GROUND_YAWS:
        views.append(("ground_or_floor", y))

    # 4) Apply one_cycle for each view
    for category, yaw in views:
        pano = one_cycle(
            pano=pano0,
            yaw=yaw,
            pitch=PITCH_MAP[category],
            fov=FOV_MAP.get(category, fovdeg),
            prompt=prompts[category],
            dilate_px=DILATE_PX,
            guidance_scale=GUIDANCE_SCALE,
            steps=STEPS
        )

    # 5) Save result
    save_image(OUTPUT_PANO, pano)
    print(f"Saved full pano to {OUTPUT_PANO}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a full 360° pano via repeated inpainting cycles"
    )
    parser.add_argument(
        "input_image",
        help="Path to your input (centered/resized) pano image"
    )
    parser.add_argument(
        "fovdeg",
        type=float,
        help="Base field of view (in degrees) for the atmosphere passes"
    )
    args = parser.parse_args()
    generate_full_pano(args.input_image, args.fovdeg)


if __name__ == "__main__":
    main()