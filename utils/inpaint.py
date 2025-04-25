from PIL import Image, ImageOps, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2


def pad_and_create_mask_reflect(
    image: np.ndarray,
    left: int, right: int,
    top: int, bottom: int,
    feather: int = 30,
    reflect: bool = True
) -> (np.ndarray, np.ndarray):
    
    H, W = image.shape[:2]
    new_H, new_W = H + top + bottom, W + left + right

    # 1) create blank canvas and paste original
    padded = np.zeros((new_H, new_W, 3), dtype=image.dtype)
    padded[top:top+H, left:left+W] = image

    if reflect:
        if left > 0:
            padded[top:top+H, :left] = np.fliplr(image[:, :left])
        if right > 0:
            padded[top:top+H, left+W:] = np.fliplr(image[:, W-right:W])
        if top > 0:
            padded[:top, :] = np.flipud(padded[top:top+top, :])
        if bottom > 0:
            padded[top+H:, :] = np.flipud(padded[top+H-bottom:top+H, :])

    mask = np.full((new_H, new_W), 255, dtype=np.uint8)
    mask[top:top+H, left:left+W] = 0

    k = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), sigmaX=feather)


    return padded, mask


def load_image_and_mask_from_black(
    arr: np.ndarray,
    threshold: int = 10,
    dilate_px: int = 32,
    feather: int = 64
) -> (np.ndarray, np.ndarray):

    image = arr.copy()

    black = np.all(image <= threshold, axis=2).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    grown = cv2.dilate(black, kernel, iterations=1)

    # convert to 0–255 mask
    mask = (grown * 255).astype(np.uint8)

    # apply Gaussian blur (kernel size must be odd)
    ksize = (feather * 2 + 1, feather * 2 + 1)
    mask = cv2.GaussianBlur(mask, ksize, sigmaX=feather)

    return image, mask


def load_soft_hard_masks_from_black(
    arr: np.ndarray,
    threshold: int = 10,
    dilate_px: int = 32,
    feather: int = 64
) -> (np.ndarray, np.ndarray, np.ndarray):


    image = arr.copy()
    black = np.all(image <= threshold, axis=2).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
    )
    grown = cv2.dilate(black, kernel, iterations=1)

    hard_mask = (grown * 255).astype(np.uint8)

    ksize = (feather * 2 + 1, feather * 2 + 1)
    soft_mask = cv2.GaussianBlur(hard_mask, ksize, sigmaX=feather)

    return image, hard_mask, soft_mask


def inpaint_image(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    prompt: str,
    guidance_scale: float = 10.0,
    steps: int = 50,
    device: str = None
) -> np.ndarray:

    # choose device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.feature_extractor.do_resize = False
    pipe.feature_extractor.size = None
    pipe.safety_checker = None

    # convert numpy → PIL
    image = Image.fromarray(image_arr).convert("RGB")
    mask = Image.fromarray(mask_arr).convert("L")

    # ensure dimensions are multiples of 8
    w, h = image.size
    w8, h8 = w - (w % 8), h - (h % 8)
    image = image.crop((0, 0, w8, h8))
    mask  = mask.crop((0, 0, w8, h8))

    # run inpainting
    out_pil = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h8,
        width=w8,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images[0]

    return np.array(out_pil)