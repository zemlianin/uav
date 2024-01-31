import supervision as sv
import numpy as np
from ultralytics import YOLO
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


TEST_FILES = {
    "testing/bradley-1/48.1964217, 37.6707864.mp4": "test_results/bradley1.mp4",
    "testing/btr-1/video_2023-11-15_20-10-30.mp4": "test_results/btr1.mp4",
    "testing/car-1/готово.mp4": "test_results/car1.mp4",
    "testing/car-2/Sequence 81_3.mp4": "test_results/car2.mp4",
    "testing/car-3/video_2023-11-15_20-13-03.mp4": "test_results/car3.mp4",
    "testing/convoy-1/1,12.mp4": "test_results/convoy1.mp4",
    "testing/leopard-1/Sequence 26_1.mp4": "test_results/leopard1.mp4",
    "testing/ukr-1/video_2023-11-15_19-56-23.mp4": "test_results/ukr1.mp4",
    "testing/ukr-2/video_2023-11-15_20-19-17.mp4": "test_results/urk2.mp4",
}


def process_frame(frame: np.ndarray, _, model) -> np.ndarray:
    results = model(frame, imgsz=640, conf=0.6)[0]
    detections = sv.Detections.from_ultralytics(results)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    if detections:
        print("==" * 20)
        print(detections)
        print(type(detections[0]))
        print("==" * 20)

    labels = [
        f"{model.names[item[0]]} {item[1]:0.2f}"
        for item in zip(detections.class_id, detections.confidence)
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame


def predict_video(args) -> None:
    source, target = args
    model = YOLO("runs/detect/train/weights/best.pt")
    callback = partial(process_frame, model=model)
    sv.process_video(
        source_path=source,
        target_path=target,
        callback=callback,
    )


with Pool() as pool:
    items = list(TEST_FILES.items())
    for _ in tqdm(pool.imap_unordered(predict_video, items), total=len(items)):
        ...
