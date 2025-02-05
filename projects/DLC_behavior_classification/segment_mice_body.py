import cv2
import numpy as np

# Function to segment the mouse using edge detection
def segment_mouse(video_path, output_path):
    # Read the video from the AVI file
    video = cv2.VideoCapture(video_path)

    # Get video properties (frame width, frame height, and FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define codec for the output video file (XVID)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create a VideoWriter object to write the segmented frames to a new video file
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define lower and upper thresholds for mouse color (in BGR format)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([60, 60, 60])

    # Iterate through the frames in the video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to extract dark regions
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Apply a mask to keep only the dark mouse
        mouse = cv2.bitwise_and(frame, frame, mask=binary)

        # Detect edges using Canny edge detector
        edges = cv2.Canny(mouse, 50, 150)

        # Write the segmented frame to the output video file
        output_video.write(mouse)

        # Display segmented mouse and edges
        cv2.imshow('Segmented Mouse', mouse)
        cv2.imshow('Edges', edges)

        # Break the loop by pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture, close windows, and release the output video writer
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

# Call the function with the path to the AVI video and the desired output file path
segment_mouse(r'D:\Tail_E186\Tail_221029_E186-Adina-2023-01-19\230314_E200.avi', r'D:\Tail_E186\Tail_221029_E186-Adina-2023-01-19\230314_E200_seg.avi')
