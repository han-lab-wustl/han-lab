import cv2
import yaml

def get_crop_and_edit_config_file(vidpth, yaml_file):
# Load the AVI video

    cap = cv2.VideoCapture(vidpth)

    # Get the frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the frame to draw the rectangle on
    frame_number = 100  # Change this to the desired frame number

    # Initialize variables for rectangle coordinates
    rect_coords = None

    # Function to handle mouse events
    def draw_rectangle(event, x, y, flags, param):
        global rect_coords
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_coords = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            rect_coords.append((x, y))
            cv2.rectangle(frame, rect_coords[0], rect_coords[1], (0, 255, 0), 2)

    # Loop through the frames until the desired frame
    for i in range(frame_number):
        ret, frame = cap.read()

    # Create a window and bind the mouse callback function
    cv2.namedWindow('Draw Rectangle')
    cv2.setMouseCallback('Draw Rectangle', draw_rectangle)

    # Display the frame and wait for user input
    while True:
        cv2.imshow('Draw Rectangle', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Get the rectangle coordinates
    if rect_coords:
        x1, y1 = rect_coords[0]
        x2, y2 = rect_coords[1]
        print(f"Rectangle coordinates: ({x1}, {y1}), ({x2}, {y2})")
    else:
        print("No rectangle drawn.")

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Edit the .yaml file
    lines_to_edit = [205, 206, 207, 208]  # Change these to the desired line numbers

    # Read the existing YAML file
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Update the YAML data with the rectangle coordinates
    if rect_coords:
        yaml_data['x1'] = x1
        yaml_data['y1'] = y1
        yaml_data['x2'] = x2
        yaml_data['y2'] = y2

    # Write the updated YAML data back to the file
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

    print(f"Edited {yaml_file} at lines {lines_to_edit}")
    
    return rect_coords