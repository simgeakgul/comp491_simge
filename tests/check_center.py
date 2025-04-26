from utils.center_img import center_image #,  complete_to_1024
import cv2

# image = cv2.imread("input.jpg")
# resized = complete_to_1024(image_arr = image,  prompts_path = "prompts.json")
# cv2.imwrite("resized.jpg", resized)

resized = cv2.imread("resized.jpg")
pano = center_image(resized)

cv2.imwrite("pano.jpg", pano)