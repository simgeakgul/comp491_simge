from diffusers import AutoPipelineForInpainting
import torch

pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    torch_dtype=torch.float16,
    force_download=True,
    resume_download=False,
    ).to("cuda")

# pipeline.load_ip_adapter("h94/IP-Adapter", 
#     subfolder="sdxl_models", 
#     weight_name="ip-adapter_sdxl.bin")
# pipeline.set_ip_adapter_scale(0.6)