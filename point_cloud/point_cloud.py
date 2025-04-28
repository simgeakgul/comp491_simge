from transformers import pipeline
from PIL import Image

# load pipe
pipe = pipeline(task="depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Small-hf",
    use_fast=False)

# load image
image = Image.open("input.jpg")

# inference
depth = pipe(image)["depth"]
print(type(depth))
depth.save('depth.jpg')
