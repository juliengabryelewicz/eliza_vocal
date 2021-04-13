import cv2
import dlib
from face_recognition import utils as ut
import numpy as np
from typing import Tuple, List, Optional

LANDMARK_POINTS_SIZE = 68
CLAHE_CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)

class LandMarker:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

    def __init__(self, landmark_predictor_path: str):
        self.shape_predictor = dlib.shape_predictor(landmark_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()

    def from_image_to_landmarks(self, image, exclude_vector_base: bool = False):
        face_detections = self.face_detector(image, 1)
        if len(face_detections) == 0:
            return []
        face_landmarks_list = []
        for k, d in enumerate(face_detections):
            shape = self.shape_predictor(image, d)
            x_list = tuple(float(shape.part(i).x) for i in range(LANDMARK_POINTS_SIZE))
            y_list = tuple(float(shape.part(i).y) for i in range(LANDMARK_POINTS_SIZE))
            face_landmarks_list.append(tuple(ut.from_points_to_vectors(x_list, y_list, exclude_vector_base)))
        return face_landmarks_list

    def from_image_to_landmark_points(self, image: np.ndarray):
        face_detections = self.face_detector(image, 1)
        if len(face_detections) < 1:
            return [None]
        landmark_points_list = []
        for (i, rect) in enumerate(face_detections):
            shape = self.shape_predictor(image, rect)
            landmark_points_list.append(ut.from_shape_to_np(shape, size=LANDMARK_POINTS_SIZE))
        return landmark_points_list

    def from_image_to_rectangles(self, image: np.ndarray):
        detections = self.face_detector(image, 1)
        if len(detections) == 0:
            return [(0, 0, 0, 0)]
        rectangles = []
        for (i, rect) in enumerate(detections):
            (x, y, w, h) = ut.rectangular_to_bounding_box(rect)
            rectangles.append((x, y, w, h))
        return rectangles
