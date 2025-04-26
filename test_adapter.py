from diffusers import AutoPipelineForInpainting
import torch
from PIL import Image
device="cuda"

def snap_to_64(img):
    w, h = img.size
    w64 = (w // 64) * 64
    h64 = (h // 64) * 64
    return img.crop((0, 0, w64, h64))



pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    torch_dtype=torch.float16,
    ).to(device)


pipeline.load_ip_adapter("h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)


image = mask_image = Image.open("1_persp_45.jpg")
image = snap_to_64(image)

ip_image = Image.open("input.jpg")
ip_image = snap_to_64(ip_image)

mask_image = Image.open("2_mask_45.jpg")
mask_image = snap_to_64(mask_image)

prompt = "Continue the alpine scene: pine trees and reflective lake, matching the original artist’s soft brush strokes, lighting and color palette"

image_out = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
    num_inference_steps=30,
    strength=0.7,
).images[0]

image_out.save("test_sdxl.jpg")