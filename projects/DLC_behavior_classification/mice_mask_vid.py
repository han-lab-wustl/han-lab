import cv2
import rembg

# Read the video file
video_path = r"D:\Tail_E186\Tail_221029_E186-Adina-2023-01-19\230314_E200.avi"
cap = cv2.VideoCapture(video_path)

# Create a VideoWriter to save the processed video with transparency
output_path = r"D:\Tail_E186\Tail_221029_E186-Adina-2023-01-19\230314_E200_body.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height), True)

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Remove the background using rembg
    msk = rembg.remove(frame)
    msk = msk[:,:,-1]
    # msks.append(msk)
    # Convert the frame back to BGR
    # msk = cv2.cvtColor(msk, cv2.COLOR_RGB2BGR)
    
    # Write the frame with transparency to the output video
    out.write(msk)
    
    # Display the resulting frame
    # plt.imshow(msk)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Destroy any open windows
cv2.destroyAllWindows()