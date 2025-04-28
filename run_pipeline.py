import cv2
import json
import os

from utils.center_img import center_image, complete_to_1024
from utils.one_cycle import one_cycle, save_image

PITCH_MAP       = {
    "atmosphere":       0.0,
    "sky_or_ceiling":   60.0,
    "ground_or_floor": -60.0,
}


HORIZONTAL_YAWS = [45, -45, 135, -135, 90, -90, 180, -180]
SKY_YAWS        = [0, 90, 180, 270]
GROUND_YAWS     = [0, 90, 180, 270]


FOV_MAP         = {
    "atmosphere":      80.0,
    "sky_or_ceiling":  110.0,
    "ground_or_floor": 110.0,
}


GUIDANCE_SCALE  = 11.0
STEPS           = 50

def generate_full_pano( img_path: str,
                        out_path:str,
                        prompts_path:str,
                        fovdeg: float) -> None:

    # 1) Load image
    image = cv2.imread(img_path)
    resized = complete_to_1024(image_arr = image,  prompts_path = prompts_path)
    pano = center_image(resized, fovdeg)
    save_image("in_pano.jpg", pano)

    # 2) Load prompts
    with open(prompts_path, "r") as f:
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
            pano=pano,
            yaw=yaw,
            pitch=PITCH_MAP[category],
            fov=FOV_MAP.get(category, fovdeg),
            prompt=prompts[category],
            guidance_scale=GUIDANCE_SCALE,
            steps=STEPS
        )

    # 5) Save result
    save_image(out_path, pano)
    print(f"Saved full pano to {out_path}")



def get_files(folder_path: str) -> (str, str, str):

    img_path = os.path.join(folder_path, "input.jpg")
    prompts_path = os.path.join(folder_path, "prompts.json")
    pano_path = os.path.join(folder_path, "pano.jpg")

    return img_path, prompts_path, pano_path



def main():

    img_path, prompts_path, pano_path = get_files("landscape")

    generate_full_pano( img_path    = img_path,
                        out_path    = pano_path,
                        prompts_path= prompts_path,
                        fovdeg      = 90.0)

    
if __name__ == "__main__":
    main()


