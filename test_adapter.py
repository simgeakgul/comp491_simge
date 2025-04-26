from diffusers import AutoPipelineForInpainting
import torch
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16)


# pipeline.to(device)
# pipeline.load_ip_adapter("h94/IP-Adapter", 
#     subfolder="sdxl_models", 
#     weight_name="ip-adapter_sdxl.bin")
# pipeline.set_ip_adapter_scale(0.6)

# image = Image.open("debug_images/1_persp_45.jpg")
# mask_image = Image.open("debug_images/2_mask_45.jpg")

# prompt = "Continue the alpine scene: pine trees and ground"
# generator = torch.Generator(device="cuda").manual_seed(0)

# image = pipeline(
#   prompt=prompt,
#   image=image,
#   mask_image=mask_image,
#   guidance_scale=8.0,
#   num_inference_steps=20,  # steps between 15 and 30 work well for us
#   strength=0.99,  # make sure to use `strength` below 1.0
#   generator=generator,
# ).images[0]

# image.save("test_sdxl.jpg")