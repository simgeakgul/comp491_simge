import cv2
import json
import argparse
import os
from utils.center_img import center_image, complete_to_square
from utils.one_cycle import one_cycle, save_image
from utils.load_configs import load_config, PanoConfig

def generate_full_pano(
    img_path: str,
    out_path: str,
    prompts_path: str,
    cfg: PanoConfig
) -> None:
    # 1) read and super-res
    image = cv2.imread(img_path)


    resized = complete_to_square(
        image_arr     = image,
        json_path     = prompts_path,
        dilate_px     = cfg.dilate_pixel,
        guidance_scale= cfg.guidance_scale,
        steps         = cfg.steps,
    )

    # 2) center the front view
    pano = center_image(resized, cfg.fovdeg, cfg.out_w, cfg.out_h)
    save_image("00_init.jpg", pano)

    # 3) load prompts
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    # 4) collect all one_cycle views
    views: list[tuple[str,int]] = []
    for y in cfg.horizontal_yaws:
        views.append(("atmosphere", y))
    for y in cfg.sky_yaws:
        views.append(("sky_or_ceiling", y))
    for y in cfg.ground_yaws:
        views.append(("ground_or_floor", y))

    # 5) run each patch
    for category, yaw in views:
        pano = one_cycle(
            pano           = pano,
            yaw            = yaw,
            pitch          = cfg.pitch_map[category],
            fov            = cfg.fov_map.get(category, cfg.fovdeg),
            prompt         = prompts[category],
            guidance_scale = cfg.guidance_scale,
            steps          = cfg.steps,
            dilate_px      = cfg.dilate_pixel,
        )

    # 6) write final pano
    cv2.imwrite(out_path, pano)


def get_files(folder_path: str) -> tuple[str,str,str]:
    return (
        os.path.join(folder_path, "input.jpg"),
        os.path.join(folder_path, "prompts.json"),
        os.path.join(folder_path, "pano.jpg"),
        os.path.join(folder_path, "config.json")
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a full panorama for a given folder"
    )
    p.add_argument(
        "--base",
        required=True,
        help="path to the folder containing input.jpg, config.yaml, etc."
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base

    img_path, prompts_path, pano_path, conf_path = get_files(base)
    cfg = load_config(conf_path)

    generate_full_pano(
        img_path     = img_path,
        out_path     = pano_path,
        prompts_path = prompts_path,
        cfg          = cfg
    )
    print(f"Panorama saved to {pano_path}")


if __name__ == "__main__":
    main()