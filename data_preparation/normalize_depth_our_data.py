import cv2
import numpy as np
import os
from scipy.spatial import KDTree

# Precompute COLORMAP_JET RGB values and build a KDTree
def create_reverse_colormap_jet():
    grayscale = np.arange(256, dtype=np.uint8).reshape(1, -1)

    # Apply COLORMAP_JET to get the corresponding RGB values
    colormap_jet = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET).reshape(-1, 3)

    # Used for nearest-neighbor lookup
    kd_tree = KDTree(colormap_jet)

    return kd_tree, colormap_jet, np.arange(256, dtype=np.uint8)

# Convert a COLORMAP_JET image back to a linear grayscale image using KDTree for approximate matching.
def reverse_colormap_jet_image(colormap_image, kd_tree, colormap_jet, grayscale_values):
    # Reshape the image into a list of RGB values
    reshaped_image = colormap_image.reshape(-1, 3)

    # Find the closest RGB values in the KDTree
    _, indices = kd_tree.query(reshaped_image, k=1)

    # Map indices to grayscale values
    grayscale_image = grayscale_values[indices].reshape(colormap_image.shape[:2])

    return grayscale_image

# Process all COLORMAP_JET depth images in a folder and save them as grayscale
def process_all_depth_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Precompute the COLORMAP_JET KDTree and grayscale indices
    kd_tree, colormap_jet, grayscale_values = create_reverse_colormap_jet()

    # Process each image in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            colormap_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if colormap_image is None:
                continue

            # Convert COLORMAP_JET image to grayscale
            grayscale_image = reverse_colormap_jet_image(colormap_image, kd_tree, colormap_jet, grayscale_values)

            # Normalize the grayscale depth map
            grayscale_normalized = cv2.normalize(grayscale_image, None, 0, 255, cv2.NORM_MINMAX)
            grayscale_normalized = grayscale_normalized.astype(np.uint8)

            cv2.imwrite(output_path, grayscale_normalized)

input_folder = "./aligned_depth" 
output_folder = "./normalized_depth"

process_all_depth_images(input_folder, output_folder)
