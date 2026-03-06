import cv2
import numpy as np
from cpd_isp import CorrectedImage, ImageStitcher, DxfGenerator
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


def getPhoto(id):
    photos = ("IMG_1346.jpg",
              "IMG_1347.jpg",
              "IMG_1348.jpg",
              "IMG_1349.jpg",
              "IMG_1350.jpg",
              "IMG_1351.jpg",
              "IMG_1352.jpg",
              "IMG_1353.jpg",
              "IMG_1354.jpg",
              "IMG_1355.jpg",
              "IMG_1356.jpg")
    # photos = ("IMG_1366.png",
    #           "IMG_1367.png",
    #           "IMG_1368.png",
    #           "IMG_1369.png",
    #           "IMG_1370.png",
    #           "IMG_1371.png",
    #           "IMG_1372.png",
    #           "IMG_1373.png")
    return cv2.imread(f"tests/input_images/{photos[id]}", cv2.IMREAD_COLOR)

def getImageObj(id):
    pic = CorrectedImage(getPhoto(id))
    pic.perspectiveWarpResizeRaw()
    return pic

img1 = getImageObj(8)
img2 = getImageObj(9)

st = ImageStitcher(img1, margin=220, blend_width=20)
st.add_image(img2)

img1.processed_resized_image = match_histograms(img1.processed_resized_image, img2.processed_resized_image)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im1 = axes[0].imshow(img1.processed_resized_image, cmap='gray')
axes[0].set_title('Image 1')
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(img2.processed_resized_image, cmap='gray')
axes[1].set_title('Image 2')
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# plt.imshow(img1.processed_resized_image, cmap='gray')  # add cmap='gray' for grayscale
# plt.imshow(img2.processed_resized_image, cmap='gray')  # add cmap='gray' for grayscale
# plt.colorbar()  # useful to see the actual pixel value range
# plt.show()

#cv2.imwrite("tests/output_images/test1.png", img1.processed_resized_image)
# cv2.imwrite("tests/output_images/test2.png", img2.corrected_image)

# st = ImageStitcher(getImageObj(0), margin=220, blend_width=20)

# st.add_image(getImageObj(1))

# cv2.imwrite("tests/output_images/test.png", .img_contours)