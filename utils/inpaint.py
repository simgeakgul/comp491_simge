# inpaint_utils.py
from PIL import Image, ImageOps, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2


def pad_and_create_mask_reflect(
    image: Image.Image,
    left: int, right: int,
    top: int, bottom: int,
    feather: int = 30
) -> (Image.Image, Image.Image):
    """
    Mirror-pad `image` on each side, then build a mask that’s white
    in the padded regions and black over the original, feathered by `feather` px.
    """
    W, H = image.size
    new_W, new_H = W + left + right, H + top + bottom

    padded = Image.new("RGB", (new_W, new_H))
    padded.paste(image, (left, top))
    padded.paste(image.crop((0,0,left,H)).transpose(Image.FLIP_LEFT_RIGHT), (0, top))
    padded.paste(image.crop((W-right,0,W,H)).transpose(Image.FLIP_LEFT_RIGHT), (left+W, top))
    padded.paste(padded.crop((0, top, new_W, top+top)).transpose(Image.FLIP_TOP_BOTTOM), (0,0))
    padded.paste(padded.crop((0, top+H-bottom, new_W, top+H)).transpose(Image.FLIP_TOP_BOTTOM), (0, top+H))

    mask = Image.new("L", (new_W, new_H), 255)
    mask.paste(0, (left, top, left+W, top+H))

    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return padded, mask

def load_image_and_mask_from_black(
    path: str,
    threshold: int = 10,
    dilate_px: int = 32,     
    feather: int = 64         # bigger feather
):
    image = Image.open(path).convert("RGB")
    arr   = np.array(image)

    black = np.all(arr <= threshold, axis=2).astype(np.uint8)

    # 1. Grow the mask *into* the picture ← overlap
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    grown   = cv2.dilate(black, kernel, iterations=1)

    mask = Image.fromarray(grown * 255, mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return image, mask

def load_soft_hard_masks_from_black(
    path: str,
    threshold: int = 10,
    dilate_px: int = 32,
    feather: int = 64
):
    """
    Returns:
      image      : PIL RGB
      hard_mask  : binary PIL L mask (0 or 255), after dilation
      soft_mask  : blurred PIL L mask, used for alpha-blend
    """
    image = Image.open(path).convert("RGB")
    arr   = np.array(image)

    black = np.all(arr <= threshold, axis=2).astype(np.uint8)

    # 1) Grow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    grown  = cv2.dilate(black, kernel, iterations=1)  # values 0 or 1

    hard_mask = Image.fromarray((grown * 255).astype(np.uint8), mode="L")

    # 2) Feather for alpha
    soft_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=feather))

    return image, hard_mask, soft_mask



def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    guidance_scale: float = 10.0,
    steps: int = 50,
    device: str = None
) -> Image.Image:
    """
    Runs SD inpainting on `image` using `mask` and returns the result.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    pipe.feature_extractor.do_resize = False
    pipe.feature_extractor.size = None
    pipe.safety_checker = None

    w, h = image.size
    w, h = w - w % 8, h - h % 8
    img_crop = image.crop((0,0,w,h))
    mask_crop = mask.crop((0,0,w,h))

    out = pipe(
        prompt=prompt,
        image=img_crop,
        mask_image=mask_crop,
        height=h,
        width=w,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images[0]

    return out
