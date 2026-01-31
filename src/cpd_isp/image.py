import cv2
import numpy as np
from cv2 import aruco

class CorrectedImage:
    def __init__(self, image):
        self.image = image
        
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, aruco_params)

        self.corners, self.ids, _ = detector.detectMarkers(image)

        # Sort them corners by ID assuming IDs 0–3 are top-right, top-left, bottom-left, bottom-right
        id_to_corner = {int(id_): c for id_, c in zip(self.ids.flatten(), self.corners)}
        
        required_ids = [0, 1, 2, 3]
        if not all(i in id_to_corner for i in required_ids):
            raise RequiredMarkersMissingError(id_to_corner.keys())

        p_tr = id_to_corner[0][0][0]  # top right
        p_tl = id_to_corner[1][0][0]  # top left
        p_bl = id_to_corner[2][0][0]  # bottom left
        p_br = id_to_corner[3][0][0]  # bottom rightd

        self.raw_pts = np.float32([p_tr, p_tl, p_bl, p_br])
    
        width_top = np.linalg.norm(p_tr - p_tl)
        width_bottom = np.linalg.norm(p_br - p_bl)
        self.raw_width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(p_tl - p_bl)
        height_right = np.linalg.norm(p_tr - p_br)
        self.raw_height = int(max(height_left, height_right))

        self.corrected_image = None
        self.corrected_width = None
        self.corrected_height = None
        self.processed_image = None

    def perspectiveWarpResize(self, width, height):
        new_pts = np.float32([
            [width, 0],        # top right
            [0, 0],            # top left
            [0, height],       # bottom left
            [width, height],   # bottom right
        ])
    
        M = cv2.getPerspectiveTransform(self.raw_pts, new_pts)
        self.corrected_image = cv2.warpPerspective(self.image, M, (width, height))

        self.corrected_width = width
        self.corrected_height = height

    def perspectiveWarpResizeRaw(self):
        self.perspectiveWarpResize(self.raw_width, self.raw_height)

    def phaseCorrelationPreProcess(self, is_old, threshold=140):
        w = self.corrected_width
        overlap_w = int(self.corrected_width / 2)

        mask = np.zeros(self.corrected_image.shape[:2], dtype="uint8")

        gray = cv2.cvtColor(self.corrected_image, cv2.COLOR_RGB2GRAY)
        
        if is_old:
            cv2.rectangle(mask, (w-overlap_w, 0), (w, self.corrected_height), 255, -1)
        else:
            cv2.rectangle(mask, (0, 0), (overlap_w, self.corrected_height), 255, -1)

        gray = cv2.bitwise_and(gray, gray, mask=mask)

        edge_mask = gray > threshold

        thresh = gray * edge_mask

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(thresh)
        
        self.processed_image = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # hann_window = self._create_hann_window(enhanced.shape)
        # self.processed_image = enhanced.astype(np.float32) * hann_window

    def resizeImage(self, target_width, target_height):
        if target_height > self.corrected_height or target_width > self.corrected_width:
            raise ValueError("Target size must be <= original size")

        self.processed_resized_image = self.processed_image[0:target_height, 0:target_width]

    def process(self, target_width, target_height, is_old, threshold=140):
        self.phaseCorrelationPreProcess(is_old, threshold)
        self.resizeImage(target_width, target_height)

    def _create_hann_window(self, shape):
        h, w = shape
        hann_h = np.hanning(h)
        hann_w = np.hanning(w)
        window = np.outer(hann_h, hann_w)
        return window


class ArucoError(Exception):
    pass


class RequiredMarkersMissingError(ArucoError):
    def __init__(self, scanned_ids):
        self.scanned_ids = scanned_ids
        super().__init__(f"Ids required are [0, 1, 2, 3]; Ids scanned: {scanned_ids}")

