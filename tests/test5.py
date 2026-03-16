import cv2
from cpd_isp import ImagePair
import matplotlib.pyplot as plt


picB = cv2.imread("tests/input_images_6/B.jpg", cv2.IMREAD_COLOR)
picA = cv2.imread("tests/input_images_6/A.jpg", cv2.IMREAD_COLOR)


pic = ImagePair(picA, picB)

plt.imshow(pic.image)
plt.show()
