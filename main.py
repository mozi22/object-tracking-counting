import sys
import cv2
from core.ObjectDetection import ObjectDetection
from sort import Sort
import utils

root = '/app'
samples_directory = root + '/samples' # Directory for test data

sys.path.append(root)
sys.path.append(samples_directory)

video_file = samples_directory + '/vid1.MOV'
video_capture = cv2.VideoCapture(video_file)

img_size_reduction_proportion = 0.7

objDet = ObjectDetection()


areas = utils.get_areas()

tracker = Sort(areas)  # create instance of the SORT tracker


while True:
    ret, frame = video_capture.read()

    # reduce frame size.
    frame_width = int(frame.shape[1] * img_size_reduction_proportion)
    frame_height = int(frame.shape[0] * img_size_reduction_proportion)
    frame = cv2.resize(frame,(frame_width, frame_height))

    detection_results = objDet.run(frame)
    detection_results = utils.filter_results(detection_results)

    frame = utils.draw_detected_objects(frame, detection_results)
    frame = utils.draw_areas_of_interest(frame, areas)
    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
