import os
import cv2
import joblib
from datetime import datetime
from PIL import Image
from face_recognition import preprocessing


file_loc = os.path.dirname(os.path.abspath(__file__))
face_recogniser = joblib.load(os.path.join(file_loc, 'model', 'face_recogniser.pkl'))
preprocess = preprocessing.ExifOrientationNormalize()


def cv2ToImage(frame):
        date = datetime.now()
        image_name = "{}.jpg".format(date.isoformat())
        image = cv2.imencode(frame, cv2.IMREAD_UNCHANGED)[1]
        print(image)
        return image.tobytes()
        # return (image_name, image[1].tobytes(), 'image/jpeg', {'Expires': '0'})


def recognize_cv2(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # pil_img = Image.open(cv2ToImage(frame))
    img = preprocess(pil_img)
    faces = face_recogniser(img)
    return [{
                'top_prediction': face['top_prediction'],
                'bounding_box': face['bb']
            } for face in faces]
