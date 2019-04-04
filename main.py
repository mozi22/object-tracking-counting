import sys
import cv2
from core.ObjectDetection import ObjectDetection
from sort import Sort
import utils
import time
from core.FeatureTracking import FeatureTracking

root = '/app'
samples_directory = root + '/samples' # Directory for test data

sys.path.append(root)
sys.path.append(samples_directory)

# video_file = samples_directory + '/short_vdos/dining_entrance/test_case1.mp4'
video_file = samples_directory + '/vid1.MOV'
video_capture = cv2.VideoCapture(video_file)
video_capture.set(cv2.CAP_PROP_FPS, 5)
fps = video_capture.get(cv2.CAP_PROP_FPS)

img_size_reduction_proportion = 0.7

objDet = ObjectDetection()


areas = utils.get_areas()

# tracker = Sort(areas)  # create instance of the SORT tracker
tracker = FeatureTracking(areas)

paused = False

while True:




    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        paused = not paused

    if paused == True:
        continue


    ret, frame = video_capture.read()

    # reduce frame size.
    frame_width = int(frame.shape[1] * img_size_reduction_proportion)
    frame_height = int(frame.shape[0] * img_size_reduction_proportion)
    frame = cv2.resize(frame,(frame_width, frame_height))

    detection_results = objDet.run(frame)
    detection_results = utils.filter_results(detection_results)

    # include tracking here
    areas, tracker_results = tracker.run(detection_results,
                                        frame,
                                        frame_width,
                                        frame_height,
                                        time.time())

    detection_results = []
    for result in tracker_results:

        detection_results.append((result.id.encode('utf-8'), 0.0, (result.location_history[-1].x - (result.bounding_box[0]/2), result.location_history[-1].y - (result.bounding_box[1]/2), result.bounding_box[0], result.bounding_box[1])))

    frame = utils.draw_detected_objects(frame, detection_results)
    frame = utils.draw_areas_of_interest(frame, areas)
    utils.draw_numbers(frame, areas)
    cv2.imshow('', frame)

    if key == ord('q'):
        break
