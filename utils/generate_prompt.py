import sys, requests, torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json

def query_llava(img: Image.Image, question: str, max_new_tokens: int = 100) -> str:
    """
        Args:
        img (Image.Image):
            A PIL Image object that the model will ‘see’ alongside your question.
        question (str):
            The natural-language prompt or question you want to ask about the image.
        max_new_tokens (int, optional):
            The maximum number of tokens the model is allowed to generate in its reply.
            Defaults to 60.

    Returns:
        str:
            The generated text response from LLaVA, decoded into a plain Python string
            with special tokens removed and whitespace stripped.
    """
    
    MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    processor = LlavaNextProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model     = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

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


def generate_three_prompts(img: Image.Image) -> dict:

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

    with open("prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts_clean, f, indent=2, ensure_ascii=False)

    return prompts_clean

def main():
    if len(sys.argv) < 2:
        print("Usage: python prompt_gen.py <image_path_or_URL>")
        sys.exit(1)

    image_path = sys.argv[1]

    if image_path.startswith("http"):
        img = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    generate_three_prompts(img)
    print("✅ Prompts saved to prompts.json")

if __name__ == "__main__":
    main()