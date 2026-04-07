import cv2
from cpd_isp import ImagePair, RawImage
import matplotlib.pyplot as plt


picA = cv2.imread("tests/input_images_7/A.png", cv2.IMREAD_COLOR)
picB = cv2.imread("tests/input_images_7/B.png", cv2.IMREAD_COLOR)

pic = ImagePair(picB, picA)

cv2.imwrite("tests/output_images/test9.png", pic.image)
# cv2.imshow("Test", pic.image)
# cv2.waitKey(0)        # Wait for any keypress
# cv2.destroyAllWindows()
