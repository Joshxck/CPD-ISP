import cv2
from cpd_isp import ImagePair, RawImage, ImageStitcher, DxfGenerator
import matplotlib.pyplot as plt


A1 = cv2.imread("tests/output_images_5/A1.png", cv2.IMREAD_COLOR)
A2 = cv2.imread("tests/output_images_5/A2.png", cv2.IMREAD_COLOR)
A3 = cv2.imread("tests/output_images_5/A3.png", cv2.IMREAD_COLOR)
A4 = cv2.imread("tests/output_images_5/A4.png", cv2.IMREAD_COLOR)
A5 = cv2.imread("tests/output_images_5/A5.png", cv2.IMREAD_COLOR)
A6 = cv2.imread("tests/output_images_5/A6.png", cv2.IMREAD_COLOR)
A7 = cv2.imread("tests/output_images_5/A7.png", cv2.IMREAD_COLOR)
A8 = cv2.imread("tests/output_images_5/A8.png", cv2.IMREAD_COLOR)

B1 = cv2.imread("tests/output_images_5/B1.png", cv2.IMREAD_COLOR)
B2 = cv2.imread("tests/output_images_5/B2.png", cv2.IMREAD_COLOR)
B3 = cv2.imread("tests/output_images_5/B3.png", cv2.IMREAD_COLOR)
B4 = cv2.imread("tests/output_images_5/B4.png", cv2.IMREAD_COLOR)
B5 = cv2.imread("tests/output_images_5/B5.png", cv2.IMREAD_COLOR)
B6 = cv2.imread("tests/output_images_5/B6.png", cv2.IMREAD_COLOR)
B7 = cv2.imread("tests/output_images_5/B7.png", cv2.IMREAD_COLOR)
B8 = cv2.imread("tests/output_images_5/B8.png", cv2.IMREAD_COLOR)

pic1 = ImagePair(B1, A1)
pic2 = ImagePair(A2, B2)
pic3 = ImagePair(B3, A3)
pic4 = ImagePair(A4, B4)
pic5 = ImagePair(B5, A5)
pic6 = ImagePair(A6, B6)
pic7 = ImagePair(B7, A7)
pic8 = ImagePair(A8, B8)

#pic1.phaseCorrelationPreProcess(True,50)

#cv2.imwrite("tests/output_images/test10.png", pic1.processed_image)

st = ImageStitcher(pic1, 50, 20)
st.add_image(pic2)
st.add_image(pic3)
st.add_image(pic4)
st.add_image(pic5)
st.add_image(pic6)
st.add_image(pic7)
st.add_image(pic8)

cv2.imwrite("tests/output_images/test20.png", st.canvas)

dxf = DxfGenerator(st.canvas, 150, 150)

dxf.get_contours(2000)

dxf.plot_contours()

cv2.imwrite("tests/output_images/Contours_2.png", dxf.img_contours)

dxf.contours_to_merged_dxf("tests/output_images/ouput_2.dxf")