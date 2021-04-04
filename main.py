import cv2
import requests
import time
from Recognition import recognize_cv2

print('Starting webcam...')
capture = cv2.VideoCapture(0)
capture.set(3, 720)
capture.set(4, 480)
pTime = 0
while True:
    check, frame = capture.read()
    faces = recognize_cv2(frame)
    if len(faces) > 0:
      for face in faces:
        top = int(face['bounding_box']['top'])
        bottom = int(face['bounding_box']['bottom'])
        left = int(face['bounding_box']['left'])
        right = int(face['bounding_box']['right'])
        cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 0), 1)
        confidence = "{:.2f}".format(face['top_prediction']['confidence']*100)
        face_label = str(f"{face['top_prediction']['label']} ({confidence})") if confidence>90 else "?"
        cv2.putText(frame, face_label, (left, top), cv2.LINE_AA, .5, (100,0,200), 2)
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(frame, f"fps: {fps}", (40, 40), cv2.LINE_AA, .5, (100,100,20), 2)
    cv2.imshow('Camera Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

capture.release()
cv2.destroyAllWindows()