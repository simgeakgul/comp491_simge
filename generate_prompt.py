import torch
import os
import argparse
import json
from PIL import Image
from utils.load_configs import load_config, PanoConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

def build_message(image, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a vision-language assistant that builds concise, comma-separated scene descriptions. Important rule: ignore center objects and people." 
        },

        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    return messages


def query_qwen(image, prompt):
    messages = build_message(image, prompt)

    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip() 

def generate_three_prompts(img, prompts_path, in_out):
    prompts = {
        "atmosphere": query_qwen(
            img,
            f"Describe the overall {'background scene' if in_out == 'indoor' else 'landscape'}  and style" 
        ),

        "sky_or_ceiling": query_qwen(
            img,
            f"Describe only the {'sky' if in_out == 'outdoor' else 'ceiling'} " 
        ),

        "ground_or_floor": query_qwen(
            img,
            f"Describe only the {'ground' if in_out == 'outdoor' else 'floor'} "
        )
    }

    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

def generate_caption(image_path):

    img = Image.open(image_path).convert("RGB")
    caption = query_qwen(
            img,
            "What would be the backgorund music for this? Start with 'The sound of...'." )
    return caption


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
    conf_path    = os.path.join(base, "config.json")

    cfg     = load_config(conf_path)
    in_out  = cfg.in_out

    img = Image.open(image_path).convert("RGB")
    generate_three_prompts(img, prompts_path, in_out)
    print(f"Prompts saved to {prompts_path}")

if __name__ == "__main__":
    main()