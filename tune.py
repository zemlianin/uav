from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.tune(
    data="datasets/data.yaml",
    imgsz=640,
    epochs=20,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
    iterations=100,
    batch=-1,
    # degrees=21,
    # perspective=0.0005,
    # flipud=0.5,
    # fliplr=0.5,
    # mosaic=0.2,
    # erasing=0.1,
)
# model.val()
# model.export(format="onnx")
