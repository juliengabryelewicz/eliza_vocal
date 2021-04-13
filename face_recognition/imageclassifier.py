from face_recognition.landmarker import LandMarker
from face_recognition.dataset import Dataset
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from typing import Tuple, List

class ImageClassifier:
    def __init__(self, csv_path: str, landmarker: LandMarker):
        ds = Dataset.from_csv(file_path=csv_path)
        self.landmarker = landmarker
        self.dataset = ds
        self.classifier = SVC(kernel='linear')
        self.classifier.fit(ds.without_labels, ds.factorized_labels)

    def predict_emotion(self, image: np.ndarray):
        face_landmarks_list = self.landmarker.from_image_to_landmarks(image, exclude_vector_base=True)
        predicted_labels = []
        if not face_landmarks_list:
            return predicted_labels
        for face_landmarks in face_landmarks_list:
            predicted_class_idx = self.classifier.predict(X=[face_landmarks])
            predicted_classes = self.dataset.unique_labels[predicted_class_idx]
            predicted_labels.append(predicted_classes[0])
        return predicted_labels

    def face_rectangle_extraction(self, image: np.ndarray):
        return self.landmarker.from_image_to_rectangles(image=image)

    def landmark_points_extraction(self, image: np.ndarray) -> List[np.ndarray]:
        return self.landmarker.from_image_to_landmark_points(image)