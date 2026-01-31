import cv2
import numpy as np
from cpd_isp import CorrectedImage, ImageStitcher, DxfGenerator

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
    return cv2.imread(f"tests/input_images/{photos[id]}", cv2.IMREAD_COLOR)

def getImageObj(id):
    pic = CorrectedImage(getPhoto(id))
    pic.perspectiveWarpResizeRaw()
    return pic


st = ImageStitcher(getImageObj(0), margin=220, blend_width=20)

n = 10
for i in range(1, n + 1):
    st.add_image(getImageObj(i))

cv2.imwrite("tests/output_images/Output.png", st.canvas)

# cv2.imshow("Output Image", st.canvas)

dxf = DxfGenerator(st.canvas, 450, 350)

dxf.get_contours()

dxf.plot_contours()

cv2.imwrite("tests/output_images/Contours.png", dxf.img_contours)