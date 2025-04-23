#!/usr/bin/env python
"""
Anchored panorama synthesis from a single perspective photo
===========================================================

Implements Part-1 (“Panorama Generation”) of:
  Schwarz et al., *A Recipe for Generating 3D Worlds From a Single Image* (2025)

Steps
-----
1.  Project the input image to the centre of an equirectangular canvas
    *and* duplicate it to the backside (yaws 0° and 180°).
2.  Generate three directional prompts with a VLM (atmosphere / sky / ground).
3.  Out-paint 16 perspective crops (8 side, 4 sky, 4 ground) with ControlNet
    in-painting, guided by the three prompts.
4.  Write intermediate debug images and the final 360×180° panorama.

Author:  GPT-4-o (“o3”) – April 2025
"""

# ---------- standard libs ----------------------------------------------------
from pathlib import Path
import argparse, os, cv2, numpy as np
from PIL import Image
from tqdm import tqdm
import json

# ---------- your helper functions (already provided) -------------------------
from utils.center_img   import center_image
# from utils.generate_prompt import generate_three_prompts
from utils.inpaint      import pad_and_create_mask_reflect, inpaint_image
from utils.persp_conv   import (
    equirectangular_to_perspective, perspective_to_equirectangular
)

# ---------- CAMERA GRID (values exactly as in the paper) ---------------------
SIDE_VIEWS = [0, 45, 90, 135, 180, 225, 270, 315]               # yaw°
SKY_VIEWS  = [(0, 60), (90, 90), (180, 120), (270, 90)]         # (yaw,pitch)
GRND_VIEWS = [(0,-60), (90,-90), (180,-120), (270,-90)]

SIDE_FOV   = 85      # °
TOPBOT_FOV = 120     # °
VIEW_RES   = 1024    # perspective resolution (square)

# -----------------------------------------------------------------------------


def blend_patch(pano: np.ndarray, patch: np.ndarray) -> None:
    """
    In-place overwrite of pano pixels where `patch` is non-black.
    """
    mask = (patch.sum(axis=-1) > 0)
    pano[mask] = patch[mask]


def save_jpg(arr: np.ndarray, path: Path) -> None:
    """Utility: BGR/float/uint8 tolerant → uint8 & save as JPEG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def main(args):
    # --------------------------------------------------------------------- #
    # 0. House-keeping & I/O                                                #
    # --------------------------------------------------------------------- #
    out_dir = Path(args.out)
    dbg_dir = out_dir / "debug_pano"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(args.input).convert("RGB")
    img_np  = np.array(img_pil)

    # --------------------------------------------------------------------- #
    # 1.  Initialise panorama & anchor duplication                          #
    # --------------------------------------------------------------------- #
    print("[Step-1] Creating anchored panorama …")
    pano = center_image(img_np, fov_deg=args.fov,
                        out_w=args.width, out_h=args.height)

    # duplicate to backside (yaw = 180°)
    backside = perspective_to_equirectangular(
        img_np, yaw=180, pitch=0, fov=args.fov,
        out_width=args.width, out_height=args.height
    )
    blend_patch(pano, backside)

    save_jpg(pano, dbg_dir / "00_pano_init.jpg")

    # --------------------------------------------------------------------- #
    # 2.  Prompt generation                                                 #
    # --------------------------------------------------------------------- #
    print("[Step-2] Generating three directional prompts …")

    with open('prompts.json') as json_file:
        prompts = json.load(json_file)

    for k, v in prompts.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------------------- #
    # 3.  Progressive out-painting                                          #
    # --------------------------------------------------------------------- #
    view_id = 0  # used for debug filenames

    def process_view(yaw, pitch, fov, prompt):
        nonlocal pano, view_id

        # (a) extract perspective crop
        pers = equirectangular_to_perspective(
            pano, yaw=yaw, pitch=pitch, fov=fov, resolution=VIEW_RES
        )
        cv2.imwrite(dbg_dir / f"{view_id:02d}_crop_raw.jpg", cv2.cvtColor(pers, cv2.COLOR_RGB2BGR))

        pers_pil = Image.fromarray(pers)

        # (b) build mask → white where pixels are *still* black (= need fill)
        missing = (pers.sum(-1) == 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(missing).convert("L")

        # if nothing is missing, skip
        if missing.max() == 0:
            return

        # quick feathering for seamless borders
        pers_pad, mask_pad = pad_and_create_mask_reflect(
            pers_pil, *([64] * 4), feather=25
        )
        mask_pad.paste(mask_pil, (64, 64))

        # (c) in-paint
        inp_dbg  = pers_pad.copy()
        out_pil  = inpaint_image(
            pers_pad, mask_pad, prompt,
            guidance_scale=args.guidance, steps=args.steps
        )

        # save debug: what model saw & produced
        save_jpg(np.array(inp_dbg),  dbg_dir / f"{view_id:02d}_in_model.jpg")
        save_jpg(np.array(out_pil),  dbg_dir / f"{view_id:02d}_out_model.jpg")

        # (d) remove padding, convert back to numpy
        out_np  = np.array(out_pil)[64:-64, 64:-64]  # undo mirror pad

        # (e) re-project & blend into panorama
        patch = perspective_to_equirectangular(
            out_np, yaw=yaw, pitch=pitch, fov=fov,
            out_width=args.width, out_height=args.height
        )
        blend_patch(pano, patch)

        # save panorama snapshot
        save_jpg(pano, dbg_dir / f"{view_id:02d}_pano.jpg")

        view_id += 1

    # ---------- horizontal band ------------------------------------------
    print("[Step-3a] In-painting side views …")
    for yaw in SIDE_VIEWS:
        process_view(yaw, 0, SIDE_FOV, prompts["atmosphere"])

    # ---------- sky -------------------------------------------------
    print("[Step-3b] In-painting sky …")
    for yaw, pitch in SKY_VIEWS:
        process_view(yaw, pitch, TOPBOT_FOV, prompts["sky_or_ceiling"])

    # ---------- ground ----------------------------------------------------
    print("[Step-3c] In-painting ground …")
    for yaw, pitch in GRND_VIEWS:
        process_view(yaw, pitch, TOPBOT_FOV, prompts["ground_or_floor"])



    # --------------------------------------------------------------------- #
    # 4.  Save final panorama                                               #
    # --------------------------------------------------------------------- #
    save_jpg(pano, out_dir / "final_pano.jpg")
    print(f"Finished!  Full panorama written to {out_dir/'final_pano.jpg'}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Panorama synthesis (Anchored)")
    p.add_argument("input", type=str,               help="Path to perspective input image")
    p.add_argument("-o", "--out", type=str, default=".",
                   help="Output directory (default = current)")
    p.add_argument("--width",  type=int, default=2048, help="Panorama width (2:1 aspect)")
    p.add_argument("--height", type=int, default=1024, help="Panorama height")
    p.add_argument("--fov",    type=float, default=90, help="Initial FOV of input image")
    p.add_argument("--guidance", type=float, default=10.0, help="Classifier-free guidance")
    p.add_argument("--steps",    type=int, default=50,   help="DDIM steps for in-painting")

    main(p.parse_args())
