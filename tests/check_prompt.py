#!/usr/bin/env python3
"""
Script to verify that the `generate_three_prompts` function works as intended.
Loads a test image, generates three prompts, checks their structure, and prints them.
"""
from PIL import Image
from utils.generate_prompt import generate_three_prompts


def main():
    # Path to your test image (update as needed)
    TEST_IMAGE_PATH = "input.jpg"

    try:
        img = Image.open(TEST_IMAGE_PATH)
    except FileNotFoundError:
        print(f"Error: '{TEST_IMAGE_PATH}' not found.")
        return

    # Generate prompts
    prompts = generate_three_prompts(img)

    # Expected prompt keys
    expected_keys = {"atmosphere", "sky_or_ceiling", "ground_or_floor"}

    # Check that all expected keys are present
    if set(prompts.keys()) != expected_keys:
        print(f"Unexpected prompt keys: {prompts.keys()}")

    # Validate and print each prompt
    for key in sorted(expected_keys):
        prompt = prompts.get(key)
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"[ERROR] Prompt for '{key}' is empty or invalid.")
        else:
            print(f"\n=== {key.upper()} PROMPT ===")
            print(prompt)

    print("\nAll prompts generated and checked.")


if __name__ == "__main__":
    main()
