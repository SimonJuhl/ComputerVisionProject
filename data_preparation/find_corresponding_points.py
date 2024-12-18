import cv2

# Store clicked points
rgb_points = []
depth_points = []
click_flag = 0  # To determine which image to click on (0: RGB, 1: Depth)

def click_event(event, x, y, flags, param):
    global click_flag, rgb_points, depth_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_flag == 0:  # RGB Image
            print(f"RGB Image Point: ({x}, {y})")
            rgb_points.append((x, y))
            click_flag = 1
        elif click_flag == 1:  # Depth Image
            depth_x = x - 1280  # Subtract 1280 to map back to depth image coordinates
            print(f"Depth Image Point: ({depth_x}, {y})")
            depth_points.append((depth_x, y))
            click_flag = 0 

def show_images(rgb_image_path, depth_image_path):
    global click_flag

    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path)

    if rgb_image is None or depth_image is None:
        print("Error loading images")
        return

    # Combine images side-by-side for simultaneous display
    combined_image = cv2.hconcat([rgb_image, depth_image])

    # Display combined image
    window_name = "Select Points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined_image)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    # Print the collected points
    print("\nFinal RGB Points:", rgb_points)
    print("Final Depth Points:", depth_points)

    cv2.destroyAllWindows()

rgb_image_path = "../capturing/captured_frames/rgb_20241214_210036135.png"
depth_image_path = "../capturing/captured_frames/depth_20241214_210036135.png"

# Run the function
show_images(rgb_image_path, depth_image_path)





# rgb_image_path = "../capturing/captured_frames/rgb_20241214_203756227.png"
# depth_image_path = "../capturing/captured_frames/depth_20241214_203756227.png"
# Final RGB Points: [(36, 493), (110, 670), (496, 666), (469, 238), (764, 528), (1009, 664), (1209, 282), (1101, 152)]
# Final Depth Points: [(218, 454), (249, 577), (525, 574), (521, 278), (728, 485), (882, 574), (1037, 312), (958, 206)]


# rgb_image_path = "../capturing/captured_frames/rgb_20241214_210036135.png"
# depth_image_path = "../capturing/captured_frames/depth_20241214_210036135.png"
# Final RGB Points: [(37, 397), (430, 492), (1033, 381), (614, 308), (712, 304), (514, 197), (224, 213), (233, 54), (1218, 148)]
# Final Depth Points: [(213, 385), (493, 450), (909, 382), (617, 326), (689, 325), (550, 246), (349, 257), (358, 140), (1045, 224)]