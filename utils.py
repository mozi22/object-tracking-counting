import numpy as np
from shapely.geometry.polygon import Polygon
from dotmap import DotMap
import cv2

def get_areas():
    areas = list()

    # perfect copy for bicycle but not for people
    # dm1 = DotMap(name='pink_cross_line', closed=False, enabled=True, polygon=Polygon([(0, 500), (0, 452), (800, 441), (638, 605)]),
    #        color=[255, 0, 255], inside=0, outside=0)

    # full area
    # dm1 = DotMap(name='Pink', closed=False, enabled=True, polygon=Polygon([(0, 530), (0, 442), (800, 441), (638, 585)]),
    #        color=[255, 0, 255], inside=0, outside=0)

    dm1 = DotMap(name='Enter', closed=False, enabled=True, polygon=Polygon([(301, 609), (623, 478), (800, 441), (638, 585)]),
           color=[255, 0, 255], inside=0, outside=0)

    # AOI 1
    dm2 = DotMap(name='Red', closed=True, enabled=True, polygon=Polygon([(642, 585), (805, 441), (1213, 436), (1248, 651)]),
           color=[0, 0, 255], inside=0, outside=0)

    # AOI 2
    dm3 = DotMap(name='Green', closed=True, enabled=True, polygon=Polygon([(640, 588), (1246, 654), (1154, 754), (11, 751)]),
           color=[0, 255, 0], inside=0, outside=0)

    # AOI 3
    dm4 = DotMap(name='Orange', closed=True, enabled=True, polygon=Polygon([(747, 433), (862, 350), (957, 343), (1113, 436)]),
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

def draw_numbers(frame, areas):

    cv2.rectangle(frame,(20,20),(430,305),[162,97,69], cv2.FILLED)

    text_x, text_y = 40, 60
    vertical_text_gap = 35
    for area in areas:

        if area.name == 'Orange' or area.name == 'Green':
            continue

        if area.closed == False:
            cv2.putText(frame, area.name + ' Person + : ' + str(area.counters['p_in']), (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
            cv2.putText(frame, area.name + ' Person - : ' + str(area.counters['p_out']), (text_x, text_y + vertical_text_gap), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
            cv2.putText(frame, area.name + ' Bicycle + : ' + str(area.counters['b_in']), (text_x, text_y + vertical_text_gap * 2), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
            cv2.putText(frame, area.name + ' Bicycle - : ' + str(area.counters['b_out']), (text_x, text_y + vertical_text_gap * 3), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
        else:
            cv2.putText(frame, area.name + ' Inside + : ' + str(area.inside), (text_x, text_y + (vertical_text_gap * 4)), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
            cv2.putText(frame, area.name + ' Time Person : ' + str(round(area.total_time_spent_person,2)) + 's', (text_x, text_y + (vertical_text_gap * 5)), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)
            cv2.putText(frame, area.name + ' Time Bicycle : ' + str(round(area.total_time_spent_bike,2)) + 's', (text_x, text_y + (vertical_text_gap * 6)), cv2.FONT_HERSHEY_DUPLEX, 0.8, [255, 255, 255], 2)

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

