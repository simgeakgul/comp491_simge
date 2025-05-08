from transformers import pipeline, AutoImageProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipe(model_id: str):
    if model_id == "pipe_rel":
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

