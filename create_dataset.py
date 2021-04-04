import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import os


name = str(input("Person Name: "))
DATASET_PATH = os.path.join("datasets", name)
if not os.path.isdir(DATASET_PATH):
  os.mkdir(DATASET_PATH)

mtcnn = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9])

image_no = 0
capture = cv2.VideoCapture(0)
count = 0
while True:
  count += 1
  check, frame = capture.read()
  frame = cv2.resize(frame, (400, 300))
  faces, _ = mtcnn.detect(Image.fromarray(frame))
  if faces is not None and count%7 == 0:
    image_no += 1
    cv2.imwrite(os.path.join(DATASET_PATH, f"{name}_{image_no}.jpg"), frame)
    if image_no == 100:
      break

  image_text = f"Number of image taken {image_no} for {name}"
  cv2.putText(frame, image_text, (20, 20), cv2.LINE_AA, .5, (100,0,200), 1)
  if faces is not None:
    for (x, y, w, h) in faces:
      x, y, w, h = int(x), int(y), int(w), int(h)
      cv2.rectangle(frame, (x,y), (w,h), (200,100,0), 2)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()