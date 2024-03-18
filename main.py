from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=5)  # train the model




#file parsing stuff

#
# import cv2
#
# video_path = '/Users/lukassomwong/PycharmProjects/vexml/videos/match3.mp4'
# output_folder = '/Users/lukassomwong/PycharmProjects/vexml/data/images/test/'
#
# cap = cv2.VideoCapture(video_path)
# frame_count = 0
#
# # while (frame_count <= 2000):
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #     if frame_count >= 1000:
# #         cv2.imwrite(output_folder + f'frame_{str(frame_count).zfill(6)}.jpg', frame)
# #     frame_count += 1
# #
# #
# #
# # cap.release()
# # cv2.destroyAllWindows()
