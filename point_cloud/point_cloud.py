from transformers import pipeline, AutoImageProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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

