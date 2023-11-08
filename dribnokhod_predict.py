from ultralytics import YOLO
from datetime import datetime
import time

model = YOLO("runs/detect/train2/weights/best.pt")  # загрузите предварительно обученную модель YOLOv8n

start_time = datetime.now()

model.predict(source="test\\images\\youtube-33_jpg.rf.05783b47767c3a08450ffbb9fa131ec2.jpg")  # предсказать по изображению

print(datetime.now() - start_time)

start_time = datetime.now()

model.predict(source="test\\images\\youtube-33_jpg.rf.05783b47767c3a08450ffbb9fa131ec2.jpg")  # предсказать по изображению

print(datetime.now() - start_time)


