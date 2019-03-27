import numpy as np
from shapely.geometry.polygon import Polygon
from dotmap import DotMap
import cv2

def get_areas():
    areas = list()

    # cross line area
    dm1 = DotMap(polygon=Polygon([(350, 594), (645, 475), (720, 471), (508, 605)]),
           color=[255, 0, 255], inside=0, outside=0)

    # AOI 1
    dm2 = DotMap(polygon=Polygon([(516, 605), (728, 471), (1113, 436), (1148, 651)]),
           color=[0, 0, 255], inside=0, outside=0)

    # AOI 2
    dm3 = DotMap(polygon=Polygon([(350, 594), (1148, 651), (1154, 754), (11, 751)]),
           color=[0, 255, 0], inside=0, outside=0)

    # AOI 3
    dm4 = DotMap(polygon=Polygon([(720, 471), (862, 350), (957, 343), (1113, 436)]),
           color=[0, 165, 255], inside=0, outside=0)

    areas.append(dm1)
    areas.append(dm2)
    areas.append(dm3)
    areas.append(dm4)

    return areas

def filter_results(predicted_boxes):
    filtered_boxes = []

    for box in predicted_boxes:
        class_name = box[0].decode('utf-8')
        if class_name in ['person', 'bicycle']:
            filtered_boxes.append(box)
    return filtered_boxes


# responsible for drawing the bounding boxes and class names
def draw_detected_objects(frame, predicted_boxes):
    font_box_color = [255, 0, 0]
    for box in predicted_boxes:
        x, y, w, h = box[2]

        cv2.rectangle(frame,
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2)),
                    font_box_color)
        cv2.putText(frame, box[0].decode('utf-8'), (int(x), int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_box_color, 2, cv2.LINE_AA)
    return frame

def draw_areas_of_interest(frame, areas):
    weight = 0.4
    if weight > 0:
        overlay = frame.copy()
        weight1 = 1 - weight

        for idx, area in enumerate(areas):
            x, y = area.polygon.exterior.coords.xy

            temp = []

            for i in range(len(x)):
                temp.append([x[i], y[i]])

            np_area = np.array(temp)

            # np_area[:, 0] *= image_width
            # np_area[:, 1] *= image_height

            pts = np_area.astype(int)
            vrx = pts.reshape((-1, 1, 2))

            cv2.polylines(overlay, [pts], True, area.color, 3)
            cv2.fillPoly(overlay, [vrx], color=area.color)
            cv2.addWeighted(overlay, weight, frame, weight1, 0, frame)

    return frame
