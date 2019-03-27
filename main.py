import sys
import cv2
import numpy as np
from core.ObjectDetection import ObjectDetection
from shapely.geometry.polygon import Polygon

root = '/app'
samples_directory = root + '/samples' # Directory for test data

sys.path.append(root)
sys.path.append(samples_directory)

video_file = samples_directory + '/vid1.MOV'
video_capture = cv2.VideoCapture(video_file)

img_size_reduction_proportion = 0.7

objDet = ObjectDetection()

# responsible for drawing the bounding boxes and class names
def draw_detected_objects(frame, predicted_boxes):
    font_box_color = [255, 0, 0]
    for box in predicted_boxes:
        class_name = box[0].decode('utf-8')
        if class_name in ['person', 'bicycle']:

            x, y, w, h = box[2]

            cv2.rectangle(frame,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        font_box_color)
            cv2.putText(frame, box[0].decode('utf-8'), (int(x), int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_box_color, 2, cv2.LINE_AA)
    return frame

def draw_areas_of_interest(frame, areas, colorss):
    weight = 0.4
    if weight > 0:
        overlay = frame.copy()
        weight1 = 1 - weight

        for idx, area in enumerate(areas):
            x, y = area.exterior.coords.xy

            temp = []

            for i in range(len(x)):
                temp.append([x[i], y[i]])

            np_area = np.array(temp)

            # np_area[:, 0] *= image_width
            # np_area[:, 1] *= image_height

            pts = np_area.astype(int)
            vrx = pts.reshape((-1, 1, 2))

            cv2.polylines(overlay, [pts], True, colorss[idx], 3)
            cv2.fillPoly(overlay, [vrx], color=colorss[idx])
            cv2.addWeighted(overlay, weight, frame, weight1, 0, frame)

    return frame



polygons = list()
colors = list()

# cross line area
polygon1 = Polygon([(350, 594), (645, 475), (720, 471), (508, 605)])
colors.append([255,0,255])

# AOI 1
polygon2 = Polygon([(516, 605), (728, 471), (1113, 436), (1148, 651)])
colors.append([0,0,255])

# AOI 2
polygon3 = Polygon([(350, 594), (1148, 651), (1154, 754), (11, 751)])
colors.append([0,255,0])

# AOI 3
polygon4 = Polygon([(720, 471), (862, 350), (957, 343), (1113, 436)])
colors.append([0,165,255])


polygons.append(polygon1)
polygons.append(polygon2)
polygons.append(polygon3)
polygons.append(polygon4)

while True:
    ret, frame = video_capture.read()

    # reduce frame size.
    frame_width = int(frame.shape[1] * img_size_reduction_proportion)
    frame_height = int(frame.shape[0] * img_size_reduction_proportion)
    frame = cv2.resize(frame,(frame_width, frame_height))

    detection_results = objDet.run(frame)

    frame = draw_detected_objects(frame, detection_results)
    frame = draw_areas_of_interest(frame, polygons, colors)
    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
