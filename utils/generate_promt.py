#!/usr/bin/env python
"""
Generate three scene‑level descriptions from a single image, following
Schwarz et al. (2025):
  1. Coarse atmosphere (ignore main objects / people)
  2. Sky / ceiling appearance
  3. Ground / floor appearance
Usage:
    python pano_prompts.py path/to/image.jpg
"""

import sys, requests, torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"

# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model     = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

# ──────────────────────────────────────────────────────────────────────────────
def query_llava(img: Image.Image, question: str, max_new_tokens: int = 60) -> str:
    """Ask LLaVA one multimodal question and return its answer string."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text",  "text": question},
            {"type": "image"}                                    # keep order
        ]
    }]
    chat   = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=chat, return_tensors="pt").to(device, dtype)
    out    = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).strip()

def generate_three_prompts(img: Image.Image) -> dict:
    """Return the three prompts needed for the Anchored panorama pipeline."""
    return {
        "atmosphere": query_llava(
            img,
            "Describe the overall atmosphere, lighting and colour palette of "
            "this scene, **ignoring specific objects or people**."
        ),
        "sky_or_ceiling": query_llava(
            img,
            "Describe only the SKY if it is outdoors, or only the CEILING if it "
            "is indoors. Mention colours, cloud patterns, lighting, etc."
        ),
        "ground_or_floor": query_llava(
            img,
            "Describe only the GROUND if it is outdoors, or only the FLOOR if it "
            "is indoors. Mention material, texture, colours, and notable details."
        )
    }

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python pano_prompts.py <image_path_or_URL>")

    path = sys.argv[1]
    img  = Image.open(path) if not path.startswith("http") else \
           Image.open(requests.get(path, stream=True).raw)

    prompts = generate_three_prompts(img)
    print("\n--- PROMPTS FOR PANORAMA GENERATION ---")
    for k, v in prompts.items():
        print(f"[{k.upper()}]\n{v}\n")
