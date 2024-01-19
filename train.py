from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="./datasets/data.yaml",
    imgsz=640,
    epochs=400,
    lr0=0.00257,
    lrf=0.00785,
    optimizer="AdamW",
    momentum=0.87412,
    weight_decay=0.00033,
    warmup_epochs=3.21521,
    warmup_momentum=0.90524,
    box=7.72674,
    cls=0.74341,
    dfl=1.30398,
    hsv_h=0.00795,
    hsv_s=0.89761,
    hsv_v=0.62797,
    degrees=0.0,
    translate=0.06111,
    scale=0.44044,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.3121,
    mosaic=0.91954,
    mixup=0.0,
    copy_paste=0.0,
    cache=True,
    batch=-1,
    workers=12,
)
model.val()
model.export(format="onnx")
