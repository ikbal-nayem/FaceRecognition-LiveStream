import os
import joblib
from PIL import Image
from face_recognition import preprocessing


file_loc = os.path.dirname(os.path.abspath(__file__))
face_recogniser = joblib.load(os.path.join(file_loc, 'model', 'face_recogniser.pkl'))
preprocess = preprocessing.ExifOrientationNormalize()


def recognize_cv2(img):
    img = Image.fromarray(img)
    img = preprocess(img)
    faces = face_recogniser(img)
    faces_data = [{
                    'top_prediction': face['top_prediction'],
                    'bounding_box': face['bb']
                } for face in faces]
    return {"faces": faces_data}
