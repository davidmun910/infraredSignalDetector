import cv2
import numpy as np

def open_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Camera {camera_id} could not be opened.")
    return cap

def detect_and_draw_circle(frame, color):
    # Convert frame to grayscale for circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the outer circle in purple
            cv2.circle(frame, (x, y), r, (128, 0, 128), 2)
            # Draw the center of the circle (purple filled)
            cv2.circle(frame, (x, y), 2, (128, 0, 128), -1)
    
    return frame

def main():
    # Open three camera streams
    cap1 = open_camera(1)
    cap2 = open_camera(2)
    cap3 = open_camera(3)

    while True:
        # Read frames from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # Check if frames are successfully captured
        if not ret1:
            print("Failed to grab frame from camera 0")
            break
        if not ret2:
            print("Failed to grab frame from camera 1")
            break
        if not ret3:
            print("Failed to grab frame from camera 2")
            break

        # Detect and draw circles on each frame
        frame1_with_circles = detect_and_draw_circle(frame1, (255, 0, 0))  # Blue circles for camera 0
        frame2_with_circles = detect_and_draw_circle(frame2, (0, 255, 0))  # Green circles for camera 1
        frame3_with_circles = detect_and_draw_circle(frame3, (0, 0, 255))  # Red circles for camera 2

        # Display the frames with circles
        cv2.imshow('Camera 0', frame1_with_circles)
        cv2.imshow('Camera 1', frame2_with_circles)
        cv2.imshow('Camera 2', frame3_with_circles)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
