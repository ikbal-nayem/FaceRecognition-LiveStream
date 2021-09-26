import cv2
import requests
import time
from Recognition import recognize_cv2


MIN_CONFIDANCE = 95

print('Starting webcam...')
# capture = cv2.VideoCapture('rtsp://192.168.0.135:8554/live0.264')
capture = cv2.VideoCapture(0)
pTime = 0
while True:
    check, frame = capture.read()
    if check:
      frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
      faces = recognize_cv2(frame)
      if len(faces) > 0:
        for face in faces:
          top = int(face['bounding_box']['top'])
          bottom = int(face['bounding_box']['bottom'])
          left = int(face['bounding_box']['left'])
          right = int(face['bounding_box']['right'])
          cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 0), 1)
          confidence = "{:.2f}".format(face['top_prediction']['confidence']*100)
          face_label = str(f"{face['top_prediction']['label']} ({confidence})") if float(confidence)>MIN_CONFIDANCE else "?"
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