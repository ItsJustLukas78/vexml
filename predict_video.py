# import os
#
# from ultralytics import YOLO
# import cv2
#
#
# VIDEOS_DIR = os.path.join('.', 'videos')
#
# video_path = os.path.join(VIDEOS_DIR, 'match4.mp4')
# video_path_out = '{}_Output.mp4'.format(video_path)
#
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
#
# model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')
#
# # Load a model
# model = YOLO(model_path)  # load a custom model
#
# threshold = 0.5
#
# colordict = {
#     0: (19, 69, 139), # brown robot
#     1: (0, 255, 0), # green triball
#     2: (0, 0, 200), # red triball
#     3: (200, 0, 0), # blue triball
#     4: (0, 0, 255), # light red goal
#     5: (255, 0, 0) # light blue goal
# }
#
# while ret:
#
#     results = model(frame)[0]
#
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#
#         if score > threshold:
#
#             color = colordict[int(class_id)]
#
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
#
#     out.write(frame)
#     ret, frame = cap.read()
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()



import os
import cv2
from ultralytics import YOLO
from sort.sort import Sort

# Path settings
VIDEO_PATH = os.path.join('.', 'videos', 'match4.mp4')
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')
OUTPUT_VIDEO_PATH = '{}_Output.mp4'.format(VIDEO_PATH)

colordict = {
    0: (19, 69, 139),  # brown robot
    1: (0, 255, 0),     # green triball
    2: (0, 0, 200),     # red triball
    3: (200, 0, 0),     # blue triball
    4: (0, 0, 255),     # light red goal
    5: (255, 0, 0)      # light blue goal
}

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # Adjust threshold as needed
            detections.append([x1, y1, x2, y2, score, int(class_id)])

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        color = colordict[int(obj_id)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
