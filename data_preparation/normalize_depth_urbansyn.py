import OpenEXR
import Imath
import numpy as np
import cv2
import os

# Load an EXR file and extract the depth channel 'Y'
def load_exr_to_depth_image(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)

    # Get header information
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Extract the 'Y' channel (depth channel)
    depth_data = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))

    # Convert the channel data to a numpy array
    depth_image = np.frombuffer(depth_data, dtype=np.float32).reshape(height, width)
    return depth_image

def normalize_depth_to_grayscale_log(depth_image, scale=10):
    # Replace invalid or infinite values
    depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

    # Find the depth range (ignore zeros)
    min_depth = np.min(depth_image[depth_image > 0])
    max_depth = np.max(depth_image)

    if max_depth - min_depth == 0:
        # Depth range is zero. Creating a black image
        return np.zeros_like(depth_image, dtype=np.uint8)

    # Apply logarithmic transformation with scaling
    depth_image = np.clip(depth_image, min_depth, max_depth)  # Ensure values are in range
    scaled_depth = scale * (depth_image - min_depth) / (max_depth - min_depth)  # Scale depths
    log_depth = np.log1p(scaled_depth)  # log(1 + scaled depth)
    log_max = np.log1p(scale)  # log(1 + scale) ensures full range

    # Normalize to 0–255
    normalized_image = (255 * log_depth / log_max).astype(np.uint8)
    return normalized_image

# Process all EXR depth files and save normalized grayscale images
def process_exr_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".exr"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".exr", ".png"))

            try:
                depth_image = load_exr_to_depth_image(input_path)
                normalized_image = normalize_depth_to_grayscale_log(depth_image)

                # Save as grayscale PNG
                cv2.imwrite(output_path, normalized_image)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

input_folder = "/kaggle/input/urbansyn/depth/depth" # Folder with EXR files
output_folder = "/kaggle/working/normalized_depth"  # Folder to save normalized grayscale images

process_exr_files(input_folder, output_folder)