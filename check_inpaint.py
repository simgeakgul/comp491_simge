from PIL import Image
from utils.inpaint import inpaint_image
def pad_and_create_mask(image: Image.Image,
                        left: int, right: int,
                        top: int, bottom: int
                       ) -> (Image.Image, Image.Image):
    """
    Pads `image` on each side by the given pixel counts,
    and returns (padded_image, mask_image) ready for Stable Diffusion inpainting.

    - padded_image is RGB with white background in the new areas.
    - mask_image is L-mode: white where you want to inpaint (the padded strips),
      black where you want to keep the original.
    """
    orig_w, orig_h = image.size
    new_w = orig_w + left + right
    new_h = orig_h + top + bottom

    # 1) Create a white canvas and paste original
    padded_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded_img.paste(image, (left, top))

    # 2) Create mask: start all white (inpaint everywhere), then black out original
    mask_img = Image.new("L", (new_w, new_h), 255)
    mask_img.paste(0, (left, top, left + orig_w, top + orig_h))

    return padded_img, mask_img

from PIL import Image

# load your original
img = Image.open("input.jpg").convert("RGB")

# pad 200px on each side horizontally, none vertically
padded, mask = pad_and_create_mask(img, left=200, right=200, top=0, bottom=0)

# save out and feed into your inpaint function
padded.save("padded.jpg")
mask.save("padded_mask.png")

# now call your existing inpaint_image:
inpaint_image("padded.jpg", "padded_mask.png", prompt="continuation of the original image with it's style", output_path="wide_out.jpg")
