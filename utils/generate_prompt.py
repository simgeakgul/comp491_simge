import sys, requests, torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def query_llava(img: Image.Image, question: str, max_new_tokens: int = 60) -> str:
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

    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
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

    """
    Args:
        img (Image.Image):
            The PIL Image you wish to analyze.

    Returns:
        dict:
            A dictionary with exactly three keys:
            - "atmosphere"       (str): Overall atmosphere, lighting, and colour palette.
            - "sky_or_ceiling"   (str): If outdoors, a sky description; if indoors, a ceiling description.
            - "ground_or_floor"  (str): If outdoors, ground details; if indoors, floor details.
    """
    
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
