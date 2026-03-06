from cpd_isp import ImagePair
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_a = cv2.imread("tests/input_images_4/IMG_1.jpg")
img_b = cv2.imread("tests/input_images_4/IMG_2.jpg")

pic = ImagePair(img_a, img_b)

cv2.imwrite("tests/output_images/test3.png", pic.image)

# plt.imshow(np.clip(pic.image / 255.0, 0.0, 1.0))
# plt.show()

