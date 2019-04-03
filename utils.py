import numpy as np
from shapely.geometry.polygon import Polygon
from dotmap import DotMap
import cv2

def get_areas():
    areas = list()

    # dm1 = DotMap(name='pink_cross_line', closed=False, enabled=True, polygon=Polygon([(989, 350), (489, 336), (0, 755), (1104, 755)]),
    #        color=[255, 0, 255], inside=0, outside=0)

    # # cross line area
    dm1 = DotMap(name='pink_cross_line', closed=False, enabled=True, polygon=Polygon([(0, 580), (0, 252), (720, 441), (508, 555)]),
           color=[255, 0, 255], inside=0, outside=0)

    # AOI 1
    dm2 = DotMap(name='red_closed', closed=True, enabled=False, polygon=Polygon([(516, 605), (728, 471), (1113, 436), (1148, 651)]),
           color=[0, 0, 255], inside=0, outside=0)

    # AOI 2
    dm3 = DotMap(name='green_closed', closed=True, enabled=False, polygon=Polygon([(350, 594), (1148, 651), (1154, 754), (11, 751)]),
           color=[0, 255, 0], inside=0, outside=0)

    # AOI 3
    dm4 = DotMap(name='orange_closed', closed=True, enabled=False, polygon=Polygon([(720, 471), (862, 350), (957, 343), (1113, 436)]),
           color=[0, 165, 255], inside=0, outside=0)


    if dm1.enabled == True:
        areas.append(dm1)
    if dm2.enabled == True:
        areas.append(dm2)
    if dm3.enabled == True:
        areas.append(dm3)
    if dm4.enabled == True:
        areas.append(dm4)

    return areas


def draw_tracked_objects(frame, tracked_objs):
    predicted_boxes = []
    for obj in tracked_objs:

        # get the mid point of the tracked obj from the bottom mid point


        predicted_boxes.append((obj.id.encode('utf-8'),0.0,(obj.location_history[-1].x, (obj.location_history[-1].y - obj.bounding_box[1] / 2), obj.bounding_box[0], obj.bounding_box[1])))

    predicted_boxes = filter_results(predicted_boxes)
    return draw_detected_objects(frame, predicted_boxes)

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
    bike_color = [0, 0, 255]
    for box in predicted_boxes:
        name, probability, coords = box
        x, y, w, h = map(int, coords)

        if 'bicycle' in name.decode('utf-8'):
            cv2.rectangle(frame,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        bike_color)
            cv2.putText(frame, box[0].decode('utf-8'), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bike_color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        font_box_color)

            cv2.putText(frame, box[0].decode('utf-8'), (int(x), int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_box_color, 2, cv2.LINE_AA)
    return frame

def draw_areas_of_interest(frame, areas):
    weight = 0.2
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

