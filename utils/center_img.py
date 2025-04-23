import cv2
import numpy as np

def center_image(image, fov_deg=90, out_width=2048, out_height=1024):
    """
    Projects a perspective image onto an equirectangular panorama.

    Args:
        image (np.ndarray): Input image (assumed perspective).
        fov_deg (float): Field of view in degrees.
        out_width (int): Width of the output panorama.
        out_height (int): Height of the output panorama.

    Returns:
        np.ndarray: Output equirectangular image.
    """
    h, w = out_height, out_width
    fov_rad = np.deg2rad(fov_deg)

    # Assume square pixels and use horizontal FOV
    focal = 0.5 * w / np.tan(0.5 * fov_rad)

    # Create grid for the equirectangular image
    u = np.linspace(-np.pi, np.pi, w)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, h)
    theta, phi = np.meshgrid(u, v)

    # Convert spherical coords to Cartesian (unit sphere)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    # Project to perspective plane
    Xp = focal * (x / z) + (image.shape[1] / 2)
    Yp = focal * (y / z) + (image.shape[0] / 2)
    flip_mask = np.abs(theta) > (np.pi / 2)
    Yp[flip_mask] = image.shape[0] - Yp[flip_mask]

    # Map from source perspective image to target pano
    map_x = Xp.astype(np.float32)
    map_y = Yp.astype(np.float32)

    # Do remapping
    equirect = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return equirect

if __name__ == "__main__":
    # === Step 1: Load your perspective image ===
    input_path = "input.jpg"  # replace with your image path
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Failed to load image. Check your path.")

    # === Step 2: Project to partial equirectangular ===
    equirect = center_image(image, fov_deg=90)

    # === Step 3: Save the result ===
    cv2.imwrite("partial_panorama.jpg", equirect)
    print("Saved to partial_panorama.jpg")
