from .load_models import load_pipe
from PIL import Image, ImageOps, ImageFilter
import torch
import numpy as np
import cv2

pipe = load_pipe(model_id = "diffusion")

def pad_and_create_mask(
    image: np.ndarray,
    left: int, right: int,
    top: int, bottom: int,
) -> (np.ndarray, np.ndarray):

    H, W = image.shape[:2]
    new_H, new_W = H + top + bottom, W + left + right
    padded = np.zeros((new_H, new_W, 3), dtype=image.dtype)
    padded[top:top+H, left:left+W] = image
    mask = np.full((new_H, new_W), 255, dtype=np.uint8)
    mask[top:top+H, left:left+W] = 0

    return padded, mask


def load_mask_from_black(
    arr: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    mask = np.all(arr == 0, axis=2).astype(np.uint8) * 255
    return mask

def feather_mask(mask: np.ndarray, feather_px: int = 16) -> np.ndarray:
    """
    Convert a binary 0/255 mask into a blurred 0‑255 alpha mask.
    SD‑inpaint treats 0   = keep image,
                     255 = replace with noise,
              any 0<val<255 = blend, so style bleeds across the seam.
    """
    if feather_px <= 0:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), feather_px)

def inpaint_image(
    image_arr: np.ndarray,
    mask_arr:  np.ndarray,
    prompt: str,
    dilate_px: int,
    guidance_scale: float,
    steps: int
) -> np.ndarray:

    # 1) dilate mask ---------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    dilated_mask = cv2.dilate(mask_arr, kernel, iterations=1)
    dilated_mask = feather_mask(dilated_mask, feather_px=16)

    # 2) numpy → PIL ---------------------------------------------------------
    image_pil = Image.fromarray(cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB))
    mask_pil  = Image.fromarray(dilated_mask).convert("L")

    # 3) ***PAD*** so w,h are multiples of 8 (no cropping)
    w, h = image_pil.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    if pad_w or pad_h:
        image_pil = ImageOps.expand(image_pil, (0, 0, pad_w, pad_h), fill=0)
        mask_pil  = ImageOps.expand(mask_pil,  (0, 0, pad_w, pad_h),  fill=255)

    # 4) run diffusion -------------------------------------------------------
    NEG_PROMPT = (
        "glitch, jpeg artefacts, unrealistic shadows, border, frame, text, "
        "watermark, cropped, deformed perspective"
    )
    out_pil = pipe(
        prompt              = prompt,
        negative_prompt     = NEG_PROMPT,
        image               = image_pil,
        mask_image          = mask_pil,
        height              = image_pil.height,
        width               = image_pil.width,
        guidance_scale      = guidance_scale,
        num_inference_steps = steps,
        noise_alpha         = 0.25,
    ).images[0]

    # 5) remove the padding so size == original ------------------------------
    out_pil = out_pil.crop((0, 0, w, h))

    # 6) PIL → numpy ---------------------------------------------------------
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)