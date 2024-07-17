import cv2
import numpy as np

# Define HSV color range for white light (adjust as needed)
lower_white = np.array([0, 0, 170])
upper_white = np.array([180, 25, 255])

# Define purple color for replacement
purple = (255, 0, 127)
dot_radius = 5

def track_white_light(cap):
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Convert to HSV colorspace for better color detection
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Create mask for white color
  mask = cv2.inRange(hsv, lower_white, upper_white)

  # Find contours in the mask
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest contour (assuming white light is the biggest)
  largest_contour = None
  max_area = 0
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
      max_area = area
      largest_contour = cnt

  # Check if a large enough contour is found
  if largest_contour is not None:
    # Get center of the contour (assuming white light is circular)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = np.int32([x, y])

    # Draw a purple dot at the center
    cv2.circle(frame, center, dot_radius, purple, -1)

      # Black out everything else (optional)
  frame = cv2.bitwise_and(frame, frame, mask=mask)

  return frame

if __name__ == "__main__":
  # Open multiple video captures (replace indexes if needed)
  cap1 = cv2.VideoCapture(1)
  cap2 = cv2.VideoCapture(2)
  cap3 = cv2.VideoCapture(3)

  # Define a window with a grid layout for 3 cameras (adjust as needed)
  rows, cols = 2, 2  # 2 rows, 2 columns for a 3x3 grid (excluding empty space)
  window_name = "Multi-Camera White Light Tracking"
  # Set a larger window size (adjust width and height as desired)
  window_width = 800
  window_height = 600
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window_name, window_width, window_height) 

  while True:
    # Capture frames from each camera
    frame1 = track_white_light(cap1)
    frame2 = track_white_light(cap2)
    frame3 = track_white_light(cap3)

    # Resize frames if needed to fit the grid layout
    frame_height, frame_width, _ = frame1.shape  # Assuming all frames have same size
    new_width = int(window_width / cols)
    new_height = int(window_height / rows)
    frame1 = cv2.resize(frame1, (new_width, new_height))
    frame2 = cv2.resize(frame2, (new_width, new_height))
    frame3 = cv2.resize(frame3, (new_width, new_height))

    # Create a canvas for displaying all frames
    canvas = np.zeros((rows * new_height, cols * new_width, 3), dtype=np.uint8)

    # Place each frame in its respective position on the canvas
    canvas[0:new_height, 0:new_width] = frame1
    canvas[0:new_height, new_width:2*new_width] = frame2
    canvas[new_height:2*new_height, 0:new_width] = frame3  # Assuming 3 cameras

    # Display the combined canvas
    cv2.imshow(window_name, canvas)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
      break

  # Release captures and close windows
  cap1.release()
