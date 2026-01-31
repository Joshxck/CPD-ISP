import cv2
import numpy as np

class DxfGenerator:
    def __init__(self, img, x_margin, y_margin):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(binary)

        x, y, w, h = cv2.boundingRect(coords)

        self.img = img[y+y_margin:y+h-y_margin, x+x_margin:x+w-x_margin]

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    
    def get_contours(self, min_area=5000):
        blurred = cv2.medianBlur(self.gray, 5)
        blurred = cv2.GaussianBlur(blurred, (81,81), 1)

        _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        edges = cv2.Canny(thresh, 100, 170)

        kernel = np.ones((7,7), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edges_closed,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )

        epsilon = 1.0  # adjust for precision

        outline = max(contours, key=cv2.contourArea)
        self.outline = cv2.approxPolyDP(outline, epsilon, True)

        big_contours = [
            c for c in contours
            if cv2.contourArea(c) > min_area
        ]

        

        big_contours_sorted = sorted(big_contours, key=cv2.contourArea, reverse=True)
        self.big_contours = [cv2.approxPolyDP(c, epsilon, True) for c in big_contours_sorted]
    
    def plot_contours(self):
        self.img_contours = self.img

        cv2.drawContours(
            self.img_contours,
            self.big_contours,  # list of contours
            -1,
            (0, 0, 255),           # red (BGR)
            2                      # thickness
        )


