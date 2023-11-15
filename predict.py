from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import time
import cv2
import supervision as sv

# Load images for training
test_img_0 = cv2.imread("./test_gen/images/test_0.jpg")
test_img_1 = cv2.imread("./test_gen/images/test_1.jpg")

# Load pre-trained YOLOv8 model
model = YOLO("runs/detect/train19/weights/best.pt")

# Predict and display results for the first image
start_time = datetime.now()
result_0 = model.predict(test_img_0)
print("Time taken for prediction on test_0.jpg:", datetime.now() - start_time)

# Display the results as an image
detections = sv.Detections.from_ultralytics(result_0[0])

box_annotator = sv.BoxAnnotator()

# Make sure 'image' is the correct variable
annotated_frame = box_annotator.annotate(
    scene=test_img_0.copy(),  # Assuming 'image' is the correct variable
    detections=detections
)

cv2.imshow("UAV Image Recognition", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Predict and display results for the second image
start_time = datetime.now()
result_1 = model.predict(test_img_1)
print("Time taken for prediction on test_1.jpg:", datetime.now() - start_time)

# Display the results as an image
detections = sv.Detections.from_ultralytics(result_1[0])

# Make sure 'image' is the correct variable
annotated_frame = box_annotator.annotate(
    scene=test_img_1.copy(),  # Assuming 'image' is the correct variable
    detections=detections
)

cv2.imshow("UAV Image Recognition", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
