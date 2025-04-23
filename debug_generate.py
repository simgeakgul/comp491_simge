from pathlib import Path
import argparse, cv2, numpy as np
from PIL import Image
from tqdm import tqdm
import json


from utils.center_img import center_image
from utils.generate_prompt import generate_three_prompts
from utils.inpaint import pad_and_create_mask_reflect, inpaint_image
from utils.persp_conv import equirectangular_to_perspective, perspective_to_equirectangular


# ----------------------------------------------------------------------------
# 1.  CAMERA GRID FOR PROGRESSIVE OUTPAINTING   (values from the paper)
# ----------------------------------------------------------------------------
SIDE_VIEWS = [ 0, 45, 90, 135, 180, 225, 270, 315 ]      # yaw in °
SKY_VIEWS  = [ (0,   60), (90,  90), (180, 120), (270,  90) ]   # (yaw,pitch)
GRND_VIEWS = [ (0,  -60), (90, -90), (180,-120), (270, -90) ]

SIDE_FOV = 85   # °
TOPBOT_FOV = 120
VIEW_RES = 1024  # square perspective resolution

# α-blending helper -----------------------------------------------------------
def alpha_blend(base, patch, mask):
    """
    Blend patch → base where mask∈[0,255].
    All images uint8, shape (H,W,3). Works in-place on `base`.
    """
    if mask.ndim == 2:            # make 3-channel
        mask = np.repeat(mask[:, :, None], 3, axis=2)
    # convert to float32 for blending
    base[:] = (patch.astype(np.float32) * (mask/255.0) +
               base.astype(np.float32)  * (1 - mask/255.0)).astype(np.uint8)

# ----------------------------------------------------------------------------

def build_panorama(image_path: Path,
                   out_width: int = 4096,
                   out_height: int = 2048,
                   guidance: float = 10.0,
                   steps: int = 50,
                   debug_dir: Path | None = Path("debug_pano")):
    """
    Builds the panorama *and* writes intermediate JPEGs so you can
    inspect progress visually.

    ── Saved files ───────────────────────────────────────────
    debug_pano/
        00_anchor.jpg         — after anchoring (input + backside duplicate)
        01_after_view_00.jpg  — after first inpainted view is merged
        02_after_view_01.jpg  — after second inpainted view is merged
        ...
        XX_final.jpg          — final panorama, same as function return
    """
    if debug_dir:                         # make directory once
        debug_dir.mkdir(exist_ok=True, parents=True)

    # 1. LOAD IMAGE ------------------------------------------------------------
    src = Image.open(image_path).convert("RGB")

    # 2. CREATE BASE PANORAMA (anchored) --------------------------------------
    pano = center_image(np.array(src), fov_deg=90,
                        out_width=out_width, out_height=out_height)

    back_patch = perspective_to_equirectangular(
        pers_img=np.array(src),
        yaw=180, pitch=0, fov=90,
        out_width=out_width, out_height=out_height
    )
    pano = np.maximum(pano, back_patch)

    if debug_dir:
        Image.fromarray(pano).save(debug_dir / "00_anchor.jpg", quality=90)

    # 3. PROMPTS (unchanged) ---------------------------------------------------
    with open("prompts.json") as f:
        prompts = json.load(f)

    final_pano   = pano.copy()
    accum_weight = np.zeros((out_height, out_width), np.float32)

    # 4. BUILD QUEUE -----------------------------------------------------------
    queue = (
        [(y, p, TOPBOT_FOV, "sky_or_ceiling")  for y, p in SKY_VIEWS ] +
        [(y, p, TOPBOT_FOV, "ground_or_floor") for y, p in GRND_VIEWS] +
        [(y, 0, SIDE_FOV, "atmosphere")        for y      in SIDE_VIEWS]
    )

    # 5. PROCESS VIEWS ---------------------------------------------------------
    for idx, (yaw, pitch, fov, key) in enumerate(
            tqdm(queue, desc="Outpainting views")):

        crop = equirectangular_to_perspective(
            pano, yaw=yaw, pitch=pitch, fov=fov,
            resolution=(VIEW_RES, VIEW_RES)
        )
        crop_img = Image.fromarray(crop)

        pad_px = int(VIEW_RES * 0.15)
        padded, mask = pad_and_create_mask_reflect(
            crop_img, pad_px, pad_px, pad_px, pad_px, feather=25
        )

        result = inpaint_image(
            image=padded,
            mask=mask,
            prompt=prompts[key],
            guidance_scale=guidance,
            steps=steps
        )

        result = result.crop((pad_px, pad_px,
                              pad_px + VIEW_RES, pad_px + VIEW_RES))
        result_np = np.array(result)

        patch = perspective_to_equirectangular(
            result_np, yaw=yaw, pitch=pitch, fov=fov,
            out_width=out_width, out_height=out_height)
        weight = perspective_to_equirectangular(
            np.full_like(result_np, 255),
            yaw=yaw, pitch=pitch, fov=fov,
            out_width=out_width, out_height=out_height)[:, :, 0]

        alpha_blend(final_pano, patch, weight)
        accum_weight += weight.astype(np.float32) / 255.0

        # save after this view
        if debug_dir:
            Image.fromarray(final_pano).save(
                debug_dir / f"{idx+1:02d}_after_view_{idx:02d}.jpg",
                quality=90
            )

    # 6. NORMALISE & CLEANUP ---------------------------------------------------
    nz = accum_weight > 0
    final_pano[nz] = (final_pano[nz].astype(np.float32) /
                      accum_weight[nz, None]).astype(np.uint8)

    strip_w = int(out_width * (10/360))
    cx      = out_width // 2
    final_pano[:, cx-strip_w//2:cx+strip_w//2] = cv2.GaussianBlur(
        final_pano[:, cx-strip_w//2:cx+strip_w//2],
        (0, 0), sigmaX=5, sigmaY=5)

    if debug_dir:
        Image.fromarray(final_pano).save(debug_dir / "ZZ_final.jpg", quality=95)

    return Image.fromarray(final_pano)


# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out",   type=Path, default="panorama.jpg")
    ap.add_argument("--width", type=int,  default=4096)
    ap.add_argument("--height",type=int,  default=2048)
    ap.add_argument("--steps", type=int,  default=50)
    ap.add_argument("--gscale",type=float,default=10.0, help="guidance_scale")
    args = ap.parse_args()

    pano = build_panorama(
        args.image, args.width, args.height,
        guidance=args.gscale, steps=args.steps
    )
    pano.save(args.out, quality=95)
    print(f"\nSaved panorama → {args.out}")

if __name__ == "__main__":
    main()