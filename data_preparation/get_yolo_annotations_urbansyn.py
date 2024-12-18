import os
import json
import cv2

# Kaggle directories
image_dir = "/kaggle/input/urbansyn/rgb/rgb"
bbox_dir = "/kaggle/input/urbansyn/bbox2d/bbox2d"
output_dir = "/kaggle/working/yolo-labels"
os.makedirs(output_dir, exist_ok=True)

# Classes to include (only pedestrians for YOLO class 0)
included_classes = {"person": 0}  # Map labels to YOLO class IDs

print(f"Total JSON files in bbox2d: {len(os.listdir(bbox_dir))}")
print(f"Total image files in rgb: {len(os.listdir(image_dir))}")

# Process each bbox JSON file
for bbox_file in os.listdir(bbox_dir):
    if bbox_file.endswith(".json"):
        # get file name of rgb image corresponding to the bbox file
        image_name = bbox_file.replace("bbox2d_", "rgb_").replace(".json", ".png")
        image_path = os.path.join(image_dir, image_name)
        bbox_path = os.path.join(bbox_dir, bbox_file)
        
        # Load the image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_name} not found in {image_dir}. Skipping...")
            continue
        img_height, img_width = image.shape[:2]

        # Load the bounding box data
        try:
            with open(bbox_path, "r") as f:
                bbox_data = json.load(f)
        except Exception as e:
            print(f"Error reading {bbox_path}: {e}")
            continue
        
        # Prepare YOLO annotations
        yolo_lines = []
        for obj in bbox_data:
            label = obj.get("label")
            bbox = obj.get("bbox")
            # If bbox class=person then continue
            if label in included_classes and bbox:
                x_min, x_max = bbox["xMin"], bbox["xMax"]
                y_min, y_max = bbox["yMin"], bbox["yMax"]

                # Convert to YOLO format
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Append the annotation
                class_id = included_classes[label]
                yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Write YOLO annotations only if there are bounding boxes
        if yolo_lines:
            output_file = os.path.join(output_dir, image_name.replace(".png", ".txt"))
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"Processed {image_name}")
        else:
            pass