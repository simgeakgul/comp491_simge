import os
import torch
import scipy
import argparse
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile as wavfile
from generate_prompt import generate_caption

def create_sound(caption):
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16,
    ).to("cuda")

    audio = pipe(
        prompt=caption,
        negative_prompt="Low Quality.",
        num_inference_steps=500,
        guidance_scale=8,
        audio_length_in_s=10.0
    ).audios[0]

    return audio
    

def parse_args():
    p = argparse.ArgumentParser(description="Generate sound for a given folder")
    p.add_argument(
        "--base",
        required=True,
        help="path to the folder containing input.jpg, config.yaml, etc."
    )
    return p.parse_args()

def main():
    args = parse_args()
    base = args.base
    audio_path = os.path.join(base, "soundscape.wav")
    image_path   = os.path.join(base, "input.jpg")

    caption = generate_caption(image_path)
    print(caption)
    audio = create_sound(caption)

    wavfile.write(audio_path, rate=16000, data=audio)

    print(f"Sound saved to {audio_path}")

if __name__ == "__main__":
    main()
        

