#!/usr/bin/env python
"""
Test harness for:
  • equirectangular_to_perspective
  • perspective_to_equirectangular

Usage
-----
python test_projection.py pano.jpg --out debug_out

What you get
------------
debug_out/
├── pano_input.jpg         # the panorama we start with (RGB)
├── view_00_pers.jpg       # raw perspective crop (BGR)         ┐
├── view_00_back.jpg       # crop re-inserted into panorama     ├── per view
├── view_00_diff.jpg       # abs-error heatmap                  ┘
└── ... (one triplet per view)
In the console you also see MSE per view and the overall round-trip error.

If you omit --img, the script builds a synthetic checkerboard pano so you can
check geometry without needing a real 360.
"""

import cv2, argparse, math
import numpy as np
from pathlib import Path
from typing import Sequence, Tuple

# ------------------------------------------------------------------ #
#  bring in the two projection functions (same file or import)       #
# ------------------------------------------------------------------ #
from utils.persp_conv import (          # rename this to where you saved them
    equirectangular_to_perspective,
    perspective_to_equirectangular
)
# ------------------------------------------------------------------ #


# Convenience -----------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    return diff[mask].mean() if mask.any() else 0.0


def make_checkerboard(w: int, h: int, squares: int = 16) -> np.ndarray:
    """Synthetic 2:1 checkerboard for quick geometry sanity-check."""
    # alternating 0 / 255 tiles in UV space
    tile_w, tile_h = w // squares, h // (squares // 2)
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(squares // 2):
        for x in range(squares):
            if (x + y) & 1:
                img[
                    y*tile_h:(y+1)*tile_h,
                    x*tile_w:(x+1)*tile_w
                ] = 255
    return img


# Camera grid used for round-trip test ----------------------------------------
YAW_LIST   = [ 0,  45,  90, 135, 180, 225, 270, 315 ]  # °
PITCH_LIST = [ 0,  45, -45 ]                           # °
FOV        = 90                                        # °


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # 0.  Load or synthesize panorama
    # ----------------------------------------------------------------------
    if args.img is None:
        pano_rgb = make_checkerboard(args.width, args.height)
        print("[info] No --img given → using synthetic checkerboard pano.")
    else:
        pano_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
        assert pano_bgr is not None, f"Could not read {args.img}"
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    H, W = pano_rgb.shape[:2]
    cv2.imwrite(str(out_dir/"pano_input.jpg"), cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR))

    # ----------------------------------------------------------------------
    # 1.  Round-trip test for each view
    # ----------------------------------------------------------------------
    global_mask = np.zeros((H, W), bool)   # where we ever write back
    errs = []

    view_idx = 0
    for pitch in PITCH_LIST:
        for yaw in YAW_LIST:
            # ---- forward projection ---------------------------------
            pers = equirectangular_to_perspective(
                pano_rgb, yaw=yaw, pitch=pitch, fov=FOV, resolution=args.res
            )

            cv2.imwrite(str(out_dir/f"view_{view_idx:02d}_pers.jpg"),
                        cv2.cvtColor(pers, cv2.COLOR_RGB2BGR))

            # ---- backward -------------------------------------------
            patch = perspective_to_equirectangular(
                pers, yaw=yaw, pitch=pitch, fov=FOV,
                out_width=W, out_height=H
            )
            # we mark where patch placed anything
            mask = (patch.sum(-1) > 0)
            global_mask |= mask

            cv2.imwrite(str(out_dir/f"view_{view_idx:02d}_back.jpg"),
                        cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            # ---- diff & metrics -------------------------------------
            err_map = np.abs(pano_rgb.astype(np.int16) - patch.astype(np.int16)).sum(-1)
            err_norm = (err_map / err_map.max() * 255).astype(np.uint8) if err_map.max() else err_map
            cv2.imwrite(str(out_dir/f"view_{view_idx:02d}_diff.jpg"),
                        cv2.applyColorMap(err_norm, cv2.COLORMAP_JET))

            view_mse = mse(pano_rgb, patch, mask)
            errs.append(view_mse)
            print(f"view {view_idx:02d}  yaw={yaw:>3}°  pitch={pitch:>3}°  MSE={view_mse:.2f}")

            view_idx += 1

    overall = sum(errs) / len(errs)
    print(f"\n[summary] Avg round-trip MSE over {len(errs)} views: {overall:.2f}")
    full_cover = global_mask.mean() * 100
    print(f"[summary] Panorama area covered by all back-projections: {full_cover:.1f}%")

    print(f"Debug images saved in: {out_dir.resolve()}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Round-trip test for pano ⇄ perspective conversions")
    ap.add_argument("--img",   type=str, help="Equirectangular panorama (2:1). If omitted, uses synthetic checkerboard.")
    ap.add_argument("--out",   type=str, default="debug_out", help="Directory for debug images")
    ap.add_argument("--width",  type=int, default=2048, help="Width of synthetic pano (if --img not given)")
    ap.add_argument("--height", type=int, default=1024, help="Height of synthetic pano")
    ap.add_argument("--res",    type=int, default=512,  help="Resolution of square perspective crops")
    args = ap.parse_args()
    main(args)
