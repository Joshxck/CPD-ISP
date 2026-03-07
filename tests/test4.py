import cv2
import numpy as np
from cpd_isp import CorrectedImage, ImageStitcher, DxfGenerator, ImagePair, RawImage
import matplotlib.pyplot as plt

def getPhoto(id, pic):
    photos = (("IMG_1_A.jpg","IMG_1_B.jpg"),
              ("IMG_2_A.jpg","IMG_2_B.jpg"),
              ("IMG_3_A.jpg","IMG_3_B.jpg"),
              ("IMG_4_A.jpg","IMG_4_B.jpg"),
              ("IMG_5_A.jpg","IMG_5_B.jpg"),
              ("IMG_6_A.jpg","IMG_6_B.jpg"),
              ("IMG_7_A.jpg","IMG_7_B.jpg"),
              ("IMG_8_A.jpg","IMG_8_B.jpg"),
              ("IMG_9_A.jpg","IMG_9_B.jpg"),
              ("IMG_10_A.jpg","IMG_10_B.jpg"),
              ("IMG_11_A.jpg","IMG_11_B.jpg"))
    # photos = ("IMG_1366.png",
    #           "IMG_1367.png",
    #           "IMG_1368.png",
    #           "IMG_1369.png",
    #           "IMG_1370.png",
    #           "IMG_1371.png",
    #           "IMG_1372.png",
    #           "IMG_1373.png")
    return cv2.imread(f"tests/input_images_5/{photos[id][pic]}", cv2.IMREAD_COLOR)

def getImageObj(id):
    # pic1 = RawImage(getPhoto(id, 0))
    # pic2 = RawImage(getPhoto(id, 1))

    pic = ImagePair(getPhoto(id, 0), getPhoto(id, 1))

    return pic

st = ImageStitcher(getImageObj(0), margin=220, blend_width=20)

n = 10
for i in range(1, n + 1):
    st.add_image(getImageObj(i))

cv2.imwrite("tests/output_images/Output.png", st.canvas)
# plt.tight_layout()
# plt.show()

# plt.imshow(img1.processed_resized_image, cmap='gray')  # add cmap='gray' for grayscale
# plt.imshow(img2.processed_resized_image, cmap='gray')  # add cmap='gray' for grayscale
# plt.colorbar()  # useful to see the actual pixel value range
# plt.show()

#cv2.imwrite("tests/output_images/test1.png", img1.processed_resized_image)
# cv2.imwrite("tests/output_images/test2.png", img2.corrected_image)

# st = ImageStitcher(getImageObj(0), margin=220, blend_width=20)

# st.add_image(getImageObj(1))

# cv2.imwrite("tests/output_images/test.png", .img_contours)