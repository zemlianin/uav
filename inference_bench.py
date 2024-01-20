from ultralytics import YOLO
import glob
import pandas as pd
import sys
from deepsparse import compile_model

model = compile_model(
    "runs/detect/train/weights/best.pt"
)  # загрузите предварительно обученную модель YOLOv8n


path_test = "test_gen/images"

imgsz = 640  # Размер изображения
batch_size = 16  # Размер пакета

RESULTS = []


def detect():
    for iter in glob.glob(path_test + "/*"):
        result = model.predict(source=f"{iter}")
        RESULTS.append(dict(result[0].speed))
        # print(result)   # box with xyxy format, (N, 4)
    # print(result.boxes.xywh)   # box with xywh format, (N, 4)
    #  print(result.boxes.xyxyn)  # box with xyxy format but normalized, (N, 4)
    #  print(result.boxes.xywhn)  # box with xywh format but normalized, (N, 4)
    #  print(result.boxes.conf)   # confidence score, (N, 1)
    #  print(result.boxes.cls)    # cls, (N, 1)


# Запуск тестирования
for _ in range(100):
    detect()

print("\n" * 50)
sys.stdout.flush()

df = pd.DataFrame.from_records(RESULTS)
df["total_ms"] = df["preprocess"] + df["inference"] + df["postprocess"]
print(df.describe())
