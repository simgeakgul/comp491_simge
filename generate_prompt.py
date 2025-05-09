import torch
import os
import json
from PIL import Image
from utils.load_models import load_pipe


model = load_pipe("llava_mmodel")
processor = load_pipe("llava_processor")

def query_llava(img: Image.Image, question: str, max_new_tokens: int = 100) -> str:

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


def generate_three_prompts(img: Image.Image, prompts_path: str) -> dict:

    def clean_response(text: str) -> str:
        """Removes 'USER:' and 'ASSISTANT:' prefixes if they exist."""
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[-1].strip()
        return text.strip()

    prompts = {
        "atmosphere": query_llava(
            img,
            "Describe the overall atmosphere, lighting and colour palette of "
            "this scene, **ignoring specific objects or people** build two sentences."
        ),
        "sky_or_ceiling": query_llava(
            img,
            "Describe only the SKY if it is outdoors, or only the CEILING if it "
            "is indoors. Build two sentences."
        ),
        "ground_or_floor": query_llava(
            img,
            "Describe only the GROUND if it is outdoors, or only the FLOOR if it "
            "is indoors. Build two sentences."
        )
    }
    prompts_clean = {k: clean_response(v) for k, v in prompts.items()}

    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_clean, f, indent=2, ensure_ascii=False)

    return prompts_clean

def main():

    base = "test_folders/achilles"
    image_path = os.path.join(base, "input.jpg")
    prompts_path = os.path.join(base, "prompts.json")

    img = Image.open(image_path).convert("RGB")

    generate_three_prompts(img, prompts_path)
    print("Prompts saved to prompts.json")

if __name__ == "__main__":
    main()