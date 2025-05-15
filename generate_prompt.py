import torch
import os
import argparse
import json
from PIL import Image
from utils.load_configs import load_config, PanoConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
      
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
dtype  = torch.float16 if device == "cuda" else torch.float32

model     = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

processor = LlavaNextProcessor.from_pretrained(MODEL_ID, use_fast=True)

def query_llava(img: Image.Image, question: str, max_new_tokens: int = 64) -> str: 

    conversation = [{
        "role": "user",
        "content": [
            {"type": "text",  "text": question},
            {"type": "image"}
        ]
    }]

    chat   = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=chat, return_tensors="pt").to(device, dtype)
    out    = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(out[0], skip_special_tokens=True).strip()


def generate_three_prompts(img: Image.Image, prompts_path: str, in_out: str) -> dict:

    def clean_response(text: str) -> str:
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[-1].strip()
        return text.strip()

    sys_prompt = "Build simple sentences with simple words."

    prompts = {
        "atmosphere": query_llava(
            img,
            "Describe only the background and  overall atmosphere "
            "Do not mention center objects."
            f"{sys_prompt}"
            
        ),

        "sky_or_ceiling": query_llava(
            img,
            f"Describe only the {'sky' if in_out == 'outdoor' else 'ceiling'} "
            f"for this {in_out} scene. "
            f"{sys_prompt}"
        ),

        "ground_or_floor": query_llava(
            img,
            f"Describe only the {'ground' if in_out == 'outdoor' else 'floor'} "
            f"for this {in_out} scene."
            "**ignoring specific objects or people**."
            f"{sys_prompt}"
        )
    }

    prompts_clean = {k: clean_response(v) for k, v in prompts.items()}

    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_clean, f, indent=2, ensure_ascii=False)

    return prompts_clean


def parse_args():
    p = argparse.ArgumentParser(description="Generate prompts for a given folder")
    p.add_argument(
        "--base",
        required=True,
        help="path to the folder containing input.jpg, config.yaml, etc."
    )
    return p.parse_args()


def main():

    args = parse_args()
    base = args.base

    image_path   = os.path.join(base, "input.jpg")
    prompts_path = os.path.join(base, "prompts.json")
    conf_path    = os.path.join(base, "config.yaml")

    cfg     = load_config(conf_path)
    in_out  = cfg.in_out

    img = Image.open(image_path).convert("RGB")
    generate_three_prompts(img, prompts_path, in_out)
    print(f"Prompts saved to {prompts_path}")

if __name__ == "__main__":
    main()