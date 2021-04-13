import cv2
from eliza.analyzer import Analyzer
from face_recognition.imageclassifier import ImageClassifier
from face_recognition.landmarker import LandMarker
import json
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from nlu.nlu import Nlu
import numpy as np
import pickle
import pyaudio
from speech_emotion_recognition.utils import extract_feature
import threading
from vosk import Model, KaldiRecognizer

class ElizaApp(App):

    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    FACE_LANDMARK_MODEL = 'data/face_emotion/shape_predictor_68_face_landmarks.dat'
    FACE_EMOTION_MODEL = 'data/face_emotion/dataset.csv'
    SPEECH_EMOTION_MODEL = 'data/speech_emotion/speech_emotion.model'
    NLU_MODEL = 'data/nlu/dataset.json'
    VOICE_RECOGNITION_MODEL = 'data/voice_recognition/fr'

    vidCapture = None

    def build(self):
        # main informations for alice (emotion user, intent...)
        self.eliza = Analyzer()
        self.face_emotion = None
        self.speech_emotion = None
        self.nlu_intent = None
        self.eliza_answer = Label(text='Bonjour, je suis ELIZA \nComment puis-je vous aider?', font_size='20sp')

        #preparing kivy interface
        self.img=Image()
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img)
        layout.add_widget(self.eliza_answer)

        #preparing openCV
        self.capture = cv2.VideoCapture(0)

        #Loading all models
        landmarker = LandMarker(landmark_predictor_path=self.FACE_LANDMARK_MODEL)
        classifier = ImageClassifier(csv_path=self.FACE_EMOTION_MODEL, landmarker=landmarker)
        self.classifier = classifier
        self.speech_emotion_model = pickle.load(open(self.SPEECH_EMOTION_MODEL, "rb"))
        self.nlu = Nlu(self.NLU_MODEL)
        
        # add timeouts
        Clock.schedule_interval(self.update, 1.0/33.0)
        Clock.schedule_interval(self.read_face_recognition, 5)
        threading.Thread(target = self.start_microphone).start()

        return layout

    def transform_image(self, image: np.ndarray):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = self.CLAHE.apply(gray_image)
        return resized_image

    # speech recognition
    def start_microphone(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        model = Model(self.VOICE_RECOGNITION_MODEL)
        rec = KaldiRecognizer(model, 16000)
        while True:
            data = stream.read(8000, exception_on_overflow = False)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if len(result) > 0:
                    if result["text"] != "":
                        self.read_emotion_recognition_speech(data)
                        self.read_intent_nlu(result["text"])
                    Logger.info('Vocal message: '+result["text"])


    # NLU (Natural Language Understanding)
    def read_intent_nlu(self, data):
        result = self.nlu.parse(data)
        if result["intent"]["intentName"] == None:
            self.nlu_intent = "None"
        else:
            self.nlu_intent = result["intent"]["intentName"]
        Logger.info("Nlu result:", result)
        self.add_eliza_information()
        self.change_eliza_answer()

    #SER (Speech Emotion Recognition)
    def read_emotion_recognition_speech(self, data):
        features = extract_feature(data, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        result = self.speech_emotion_model.predict(features)[0]
        if self.speech_emotion != result:
            self.speech_emotion = result
            Logger.info("Predicted emotion speech: "+result)

    # FER (Face Emotion Recognition)
    def read_face_recognition(self, dt):
        ret, frame = self.capture.read()
        predicted_labels = self.classifier.predict_emotion(image=self.transform_image(image=frame))
        rectangles = self.classifier.face_rectangle_extraction(image=frame)
        landmark_points_list = self.classifier.landmark_points_extraction(image=frame)
        for lbl, rectangle, lm_points in zip(predicted_labels, rectangles, landmark_points_list):
            if self.face_emotion != predicted_labels[0]:
                Logger.info('Predicted emotion face: '+predicted_labels[0])
                self.face_emotion = predicted_labels[0]
                self.add_eliza_information()
                self.change_eliza_answer()

    #adapt ELIZA answer depending on all informations (FER, SER, and NLU)
    def change_eliza_answer(self):
        self.eliza_answer.text = self.eliza.get_eliza_answer()

    def add_eliza_information(self):
        self.eliza.add_information(self.face_emotion, self.speech_emotion, self.nlu_intent)
        
    # Show every frame of the camera
    def update(self, dt):
        ret, frame = self.capture.read()
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture1

if __name__ == '__main__':
    ElizaApp().run()
    cv2.destroyAllWindows()
