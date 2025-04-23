from PIL import Image
from utils.inpaint import pad_and_create_mask_reflect, inpaint_image

def main():
    orig = Image.open("input.jpg").convert("RGB")

    padded, mask = pad_and_create_mask_reflect(
        orig,
        left=200, right=200,
        top=0, bottom=0,
        feather=30
    )
    padded.save("padded.jpg")
    mask.save("padded_mask.png")

    result = inpaint_image(
        image=padded,
        mask=mask,
        prompt=(
            "Continue the alpine scene: pine trees and reflective lake, "
            "matching the original artist’s soft brush strokes, lighting and color palette"
        ),
        guidance_scale=11.0,
        steps=60
    )

    result.save("wide_out.jpg")
    print("✅ wide_out.jpg written.")

if __name__ == "__main__":
    main()
