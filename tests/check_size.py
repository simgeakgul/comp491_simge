from utils.center_image import complete_to_1024



image = cv2.imread("input.jpg")

result = complete_to_1024(
    image_arr = image,
    prompts_path = "prompts.json",
) 


cv2.imwrite("resized.jpg", result)



