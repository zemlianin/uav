from ultralytics import YOLO
 
model = YOLO("runs/detect/train2/weights/best.pt")  # загрузите предварительно обученную модель YOLOv8n

model.predict(source="test\\images\\youtube-33_jpg.rf.05783b47767c3a08450ffbb9fa131ec2.jpg")  # предсказать по изображению
