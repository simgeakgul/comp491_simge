import cv2
import numpy as np
from .persp_conv import perspective_to_equirectangular

def center_image(img, fov_deg=90, out_w=2048, out_h=1024):
    # front view
    pano = perspective_to_equirectangular(
        pers_img   = img,
        yaw        = 0.0,
        pitch      = 0.0,
        fov        = fov_deg,
        out_width  = out_w,
        out_height = out_h
    )

    # 180° yaw gives the “mirrored” halves at the left and right edges
    wrap = perspective_to_equirectangular(
        pers_img   = img,
        yaw        = 180.0,      # look backwards
        pitch      = 0.0,
        fov        = fov_deg,
        out_width  = out_w,
        out_height = out_h
    )

    # copy any non-black pixel from the second pass into the panorama
    mask = wrap.any(axis=-1)
    pano[mask] = wrap[mask]
    return pano


if __name__ == "__main__":
    # === Step 1: Load your perspective image ===
    input_path = "../input.jpg" 
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Failed to load image. Check your path.")

    # === Step 2: Project to partial equirectangular ===
    equirect = center_image(image, fov_deg=90)

    # === Step 3: Save the result ===
    cv2.imwrite("partial_panorama.jpg", equirect)
    print("Saved to partial_panorama.jpg")
