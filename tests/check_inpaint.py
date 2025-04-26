import cv2
from utils.inpaint import pad_and_create_mask, inpaint_image

def main():

    orig = cv2.imread("input.jpg")
   
    padded, mask = pad_and_create_mask(
        orig,
        left=200, right=200,
        top=0, bottom=0,
    )

    cv2.imwrite("padded.jpg", padded)
    cv2.imwrite("padded_mask.png", mask)

    result = inpaint_image(
        image_arr=padded,
        mask_arr=mask,
        prompt=(
            "Continue the alpine scene: pine trees and reflective lake, "
            "matching the original artist’s soft brush strokes, lighting and color palette"
        ),
        guidance_scale=8.0,
        steps=50
    )

    cv2.imwrite("wide_out.jpg", result)
    print("✅ wide_out.jpg written.")

if __name__ == "__main__":
    main()
