import cv2
import pandas as pd

# Load the predictions
predictions_df = pd.read_csv('predictions.csv')

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reduce the size of the video display
scale_percent = 50  # percent of original size
new_width = int(frame_width * scale_percent / 100)
new_height = int(frame_height * scale_percent / 100)

# Define the codec and create VideoWriter object to save the output video
out = cv2.VideoWriter('output_with_predictions.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Check if the current frame has a prediction
    if frame_index in predictions_df['frame'].values:
        # Get the prediction for the current frame
        prediction = predictions_df.loc[predictions_df['frame'] == frame_index, 'value'].values[0]
        print(f"Frame {frame_index}: Prediction = {prediction}")  # Debugging line

        # Overlay the prediction on the frame at the top
        text = f"Prediction: {prediction}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the frame with the prediction overlay
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame with Predictions', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
