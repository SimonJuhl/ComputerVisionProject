import pyrealsense2 as rs
import cv2
import os
from datetime import datetime
import numpy as np

# Define output folder for saving images
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# Configure the pipeline for the Intel RealSense camera
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB and Depth streams
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

print("Press SPACE to capture. ESC to exit.")

try:
    while True:
        # Wait for frames (depth + color)
        frames = pipeline.wait_for_frames()

        # Extract the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Check if both frames are valid
        if not color_frame or not depth_frame:
            continue

        # Convert RealSense frames to numpy arrays (OpenCV compatible)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Make depth image a colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show live RGB and depth images
        cv2.imshow("RGB Stream", color_image)
        cv2.imshow("Depth Stream", depth_colormap)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("Exiting...")
            break
        elif key == 32:  # SPACE key
            # Generate file names w. timestamps
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]  # Accurate to milliseconds
            color_filename = os.path.join(output_folder, f"rgb_{timestamp}.png")
            depth_filename = os.path.join(output_folder, f"depth_{timestamp}.png")

            # Save
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_colormap)

            print(f"Captured and saved {color_filename} and {depth_filename}")

except KeyboardInterrupt:
    print("Stopped by user. Exiting...")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")
