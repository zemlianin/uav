from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")  # загрузите предварительно обученную модель YOLOv8n
 
model.train(data="./data.yaml", imgsz=640, epochs=400)  # обучите модель
model.val()  # оцените производительность модели на наборе проверки
model.export(format="onnx")  # экспортируйте модель в формат ONNX

#model.predict(image="test\\images\\youtube-33_jpg.rf.05783b47767c3a08450ffbb9fa131ec2.jpg")  # предсказать по изображению
