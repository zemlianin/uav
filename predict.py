import os
from ultralytics import YOLO
from datetime import datetime
import cv2
import supervision as sv

# Directory containing images
# image_dir = "test_gen/images/"
image_dir = "./testing/car-2"

# Load pre-trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Define class names
classes = ['BRT', 'DOM', 'DST', 'GHM', 'HMN', 'LBT']

# Create a window for display
cv2.namedWindow("UAV Image Recognition", cv2.WINDOW_NORMAL)

# Iterate through all files in the directory in ascending order
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(".jpg"):
        # Load the image
        image_path = os.path.join(image_dir, filename)
        test_img = cv2.imread(image_path)

        # Predict on the image
        start_time = datetime.now()
        result = model.predict(test_img)
        print(f"Time taken for prediction on {filename}: {datetime.now() - start_time}")

        # Display the results as an image
        detections = sv.Detections.from_ultralytics(result[0])

        box_annotator = sv.BoxAnnotator()

        # Generate labels based on detection information
        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        # Make sure 'image' is the correct variable
        annotated_frame = box_annotator.annotate(
            scene=test_img.copy(),  # Assuming 'image' is the correct variable
            detections=detections,
            labels=labels
        )

        cv2.imshow("UAV Image Recognition", annotated_frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()
