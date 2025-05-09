from transformers import pipeline, AutoImageProcessor
from diffusers import StableDiffusionInpaintPipeline
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipe(model_id: str):

    if model_id == "llava_model":
            
        MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        model     = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)

        return model

    elif model_id == "llava_processor":
        MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(MODEL_ID, use_fast=True)


    elif model_id == "diffusion":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        pipe.feature_extractor.do_resize = False
        pipe.feature_extractor.size = None
        pipe.safety_checker = None

        return pipe

    elif model_id == "pipe_rel":
        ## Depth model
        processor_depth = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf",
            use_fast=False
        )
        pipe_depth = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Large-hf",
            image_processor=processor_depth,
            device=device
        )
        return pipe_depth

    elif model_id == "pipe_metric_indoor":
        ## Indoor metric
        processor_indoor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            use_fast=False
        )
        pipe_metric_indoor = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            image_processor=processor_indoor,
            device=device
        )
        return pipe_metric_indoor

    elif model_id == "pipe_metric_outdoor":
        ## Outdoor metric
        processor_outdoor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            use_fast=False
        )
        pipe_metric_outdoor = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            image_processor=processor_outdoor,
            device=device
        )
        return pipe_metric_outdoor

    else:
        print("Model not found.")

