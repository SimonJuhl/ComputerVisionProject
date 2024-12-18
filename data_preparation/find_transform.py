import os
import cv2
import numpy as np

# Corresponding points from find_corresponding_points.py
rgb_points = np.array([
    (36, 493), (110, 670), (496, 666), (469, 238),
    (764, 528), (1009, 664), (1209, 282), (1101, 152),
    (37, 397), (430, 492), (1033, 381), (614, 308),
    (712, 304), (514, 197), (224, 213), (233, 54), (1218, 148)
], dtype=np.float32)

depth_points = np.array([
    (218, 454), (249, 577), (525, 574), (521, 278),
    (728, 485), (882, 574), (1037, 312), (958, 206),
    (213, 385), (493, 450), (909, 382), (617, 326),
    (689, 325), (550, 246), (349, 257), (358, 140), (1045, 224)
], dtype=np.float32)

# Compute the homography matrix using RANSAC to handle inaccuracies
homography_matrix, mask = cv2.findHomography(depth_points, rgb_points, cv2.RANSAC)

image_dir = "../capturing/captured_frames/"
output_dir = "./aligned_depth/"
os.makedirs(output_dir, exist_ok=True)

images = [
    f for f in os.listdir(image_dir)
    if f.startswith("depth")
]


for img_file in images:
    img_path = os.path.join(image_dir, img_file)
    depth_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if depth_image is None:
        raise ValueError("Error loading images. Check file paths.")

    # Warp the depth image to align it with the RGB image
    aligned_depth = cv2.warpPerspective(depth_image, homography_matrix, (1280, 720))

    # Save the aligned depth image
    new_path = os.path.join(output_dir, img_file)
    cv2.imwrite(new_path, aligned_depth)


print("Homography Matrix:")
print(homography_matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()
