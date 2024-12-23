import cv2
import os
import numpy as np

def create_rgbd_tiff_from_annotations(annotation_folder, rgb_folder, depth_folder):
    rgb_dir = "./rgbd/images"
    rgbd_dir = "./rgbd/depth"
    labels_dir = "./rgbd/labels"


    for sub_dir in ["train", "val"]:
        os.makedirs(os.path.join(rgb_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(rgbd_dir, sub_dir), exist_ok=True)    
        os.makedirs(os.path.join(labels_dir, sub_dir), exist_ok=True)    

    # Get list of annotation files
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    # Process each annotation file
    for i, annotation_file in enumerate(annotation_files):
        base_name = annotation_file.split(".")[0].split("rgb_")[1]
        
        is_train = (i % 10) < 8  # First 8 go to train, next 2 to val
        target_dir = "train" if is_train else "val"

        rgb_file = f"rgb_{base_name}.png"
        depth_file = f"depth_{base_name}.png"

        rgb_path = os.path.join(rgb_folder, rgb_file)
        depth_path = os.path.join(depth_folder, depth_file)
        annotation_path = os.path.join(annotation_folder, annotation_file)

        print(rgb_path)
        print(depth_path)

        if not os.path.exists(rgb_path):
            print(f"Missing rgb file for {base_name}. Skipping...")
        if not os.path.exists(depth_path):
            print(f"Missing depth file for {base_name}. Skipping...")
            continue

        # Load the RGB and depth images
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if rgb_image is None or depth_image is None:
            print(f"Failed to load {rgb_file} or {depth_file}. Skipping...")
            continue

        # Ensure the depth image has the same dimensions as the RGB image
        if rgb_image.shape[:2] != depth_image.shape:
            print(f"Resizing depth image {depth_file} to match RGB dimensions.")
            depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Stack the RGB and depth images into a 4-channel image
        #rgbd_image = np.dstack(rgb_image, depth_image)

        # Save the RGB-D TIFF image
        #rgbd_output_path = os.path.join(rgbd_dir, target_dir, f"{base_name}.tiff")
        #cv2.imwrite(rgbd_output_path, rgbd_image)
        rgbd_output_path = os.path.join(rgbd_dir, target_dir, rgb_file)
        cv2.imwrite(rgbd_output_path, depth_image)

        # Save the RGB image
        rgb_output_path = os.path.join(rgb_dir, target_dir, rgb_file)
        cv2.imwrite(rgb_output_path, rgb_image)

        # Save the annotation file
        annotation_output_path = os.path.join(labels_dir, target_dir, annotation_file)
        with open(annotation_path, "r") as src, open(annotation_output_path, "w") as dst:
            dst.write(src.read())

        print(f"Processed: {rgb_file}, {depth_file}, {annotation_file}")

# Input folders
annotation_folder = "../bbox/annotations"             # Path to folder with annotations
rgb_folder = "../capturing/captured_frames/"      # Path to folder with RGB images
depth_folder = "../data_preparation/normalized_depth" # Path to folder with normalized grayscale depth images


# Run the function
create_rgbd_tiff_from_annotations(annotation_folder, rgb_folder, depth_folder)
