from ultralytics import YOLO
import torch
import glob
 
model = YOLO("runs/detect/train3/weights/best.pt")  # загрузите предварительно обученную модель YOLOv8n


path_test= "test/images"

imgsz = 640  # Размер изображения
batch_size = 16  # Размер пакета

def detect():
    for iter in  glob.glob(path_test + '/*'):
        result = model.predict(source=f"{iter}")
        print(result)   # box with xyxy format, (N, 4)
      # print(result.boxes.xywh)   # box with xywh format, (N, 4)
      #  print(result.boxes.xyxyn)  # box with xyxy format but normalized, (N, 4)
      #  print(result.boxes.xywhn)  # box with xywh format but normalized, (N, 4)
      #  print(result.boxes.conf)   # confidence score, (N, 1)
      #  print(result.boxes.cls)    # cls, (N, 1)
      
        
# Запуск тестирования
detect()