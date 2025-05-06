from PIL import Image, ImageOps, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2

device ="cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
pipe.feature_extractor.do_resize = False
pipe.feature_extractor.size = None
pipe.safety_checker = None


def pad_and_create_mask(
    image: np.ndarray,
    left: int, right: int,
    top: int, bottom: int,
) -> (np.ndarray, np.ndarray):

    H, W = image.shape[:2]
    new_H, new_W = H + top + bottom, W + left + right

    # 1) Create black canvas and blit the image
    padded = np.zeros((new_H, new_W, 3), dtype=image.dtype)
    padded[top:top+H, left:left+W] = image

    # 2) Mask: white where you WANT to inpaint (the new padding),
    #    black where you want to keep (the original image)
    mask = np.full((new_H, new_W), 255, dtype=np.uint8)
    mask[top:top+H, left:left+W] = 0

    return padded, mask


def load_mask_from_black(
    arr: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    mask = np.all(arr == 0, axis=2).astype(np.uint8) * 255
    return mask

def inpaint_image(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    prompt: str,
    dilate_px: int,
    guidance_scale: float,
    steps: int
) -> np.ndarray:
    """
    - Dilates mask_arr for more context during diffusion.
    - Runs StableDiffusion inpaint.
    - Returns the full inpainted crop.
    """
    # 1) dilate the original mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    dilated_mask = cv2.dilate(mask_arr, kernel, iterations=1)

    # 2) prepare PIL inputs
    image_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil  = Image.fromarray(dilated_mask).convert("L")

    # 3) crop to multiples of 8
    w, h = image_pil.size
    w8, h8 = w - w % 8, h - h % 8
    image_pil = image_pil.crop((0, 0, w8, h8))
    mask_pil  = mask_pil.crop((0, 0, w8, h8))

    NEG_PROMPT = (
    "glitch, jpeg artefacts, unrealistic shadows, border, frame, text, watermark, "
    "cropped, deformed perspective"
    )

    # 4) run inpainting pipeline
    out_pil = pipe(
        prompt              = prompt,
        negative_prompt     = NEG_PROMPT,
        image               = image_pil,
        mask_image          = mask_pil,
        height              = h8,
        width               = w8,
        guidance_scale      = guidance_scale,
        num_inference_steps = steps,
    ).images[0]

    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)