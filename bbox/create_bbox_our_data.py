import cv2
import os
from datetime import datetime

# Path to your directory containing RGB images
image_dir = "../capturing/captured_frames"
output_dir = "./annotations"
os.makedirs(output_dir, exist_ok=True)

# Globals for drawing bounding boxes
drawing = False
start_point = None
bounding_boxes = []

# Mouse callback for drawing bounding boxes
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, bounding_boxes, temp_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a rectangle
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Show the rectangle as you draw
            temp_image = image.copy()
            cv2.rectangle(temp_image, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotator", temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the rectangle
        drawing = False
        end_point = (x, y)
        x_min, y_min = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x_max, y_max = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

        # Calculate normalized center, width, and height
        x_center = (x_min + x_max) / 2 / image.shape[1]
        y_center = (y_min + y_max) / 2 / image.shape[0]
        width = (x_max - x_min) / image.shape[1]
        height = (y_max - y_min) / image.shape[0]

        # Save bounding box
        bounding_boxes.append((x_center, y_center, width, height))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Annotator", image)


images = [
    f for f in os.listdir(image_dir)
    if f.startswith("rgb")
]

# Sort by timestamp
images.sort(key=lambda x: datetime.strptime(
    f"{x.split('_')[1]}_{x.split('_')[2].replace('.png', '')}",  # Combine date and time parts
    "%Y%m%d_%H%M%S%f"
))

# Annotation loop
for img_file in images:
    img_path = os.path.join(image_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image {img_file}. Skipping...")
        continue

    temp_image = image.copy()
    bounding_boxes = []  # Reset bounding boxes for this image
    print(f"Annotating: {img_file}")

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", draw_rectangle)
    cv2.imshow("Annotator", image)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter: Save annotations and move to next image
            # Save bounding boxes to text file with same name as rgb image
            txt_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + ".txt")
            with open(txt_file, "w") as f:
                for bbox in bounding_boxes:
                    f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            break

        elif key == ord('z'):  # Ctrl+Z: Undo last bounding box
            if bounding_boxes:
                bounding_boxes.pop()
                image = temp_image.copy()  # Reset image
                for bbox in bounding_boxes:
                    # Re-draw all bounding boxes
                    x_center = bbox[0] * image.shape[1]
                    y_center = bbox[1] * image.shape[0]
                    width = bbox[2] * image.shape[1]
                    height = bbox[3] * image.shape[0]
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imshow("Annotator", image)
            else:
                print("No bounding boxes to undo.")

        elif key == 27:  # ESC: Exit program
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
