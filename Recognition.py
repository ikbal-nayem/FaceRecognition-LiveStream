import os
import joblib
import time
from PIL import Image
from face_recognition import preprocessing


file_loc = os.path.dirname(os.path.abspath(__file__))
face_recogniser = joblib.load(os.path.join(file_loc, 'model', 'face_recogniser.pkl'))
preprocess = preprocessing.ExifOrientationNormalize()

pT = 0
def recognize_cv2(img):
    cT = time.time()
    img = Image.fromarray(img)
    print(cT-pT)
    pT = cT
    img = preprocess(img)
    faces = face_recogniser(img)
    return [{
                'top_prediction': face['top_prediction'],
                'bounding_box': face['bb']
            } for face in faces]
