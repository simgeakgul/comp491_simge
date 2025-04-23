from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

def inpaint_image(image_path: str,
                  mask_path: str,
                  prompt: str,
                  output_path: str = "inpainted_output.jpg"):
    # load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None

    # load _without_ any resize
    rgb_img  = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")

    # ensure dimensions are multiples of 8
    w, h = rgb_img.size
    w -= w % 8; h -= h % 8

    # pass your own size into the pipeline
    result = pipe(
        prompt=prompt,
        image=rgb_img,
        mask_image=mask_img,
        height=h,
        width=w,
        guidance_scale=7.5,
        num_inference_steps=50,
    ).images[0]

    result.save(output_path)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
