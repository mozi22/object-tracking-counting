from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from core.KalmanTracker import KalmanTracker
import time


class FeatureTrackingObject:

    def __init__(self, id, x, y, box_width, box_height, features):
        # how many previous steps you want to keep track of
        self.history_limit = 5

        self.location_history = list()
        self.timestamp_history = list()
        self.threshold = 1

        self.region = None

        self.last_seen = time.time()
        self.kalman_tracker = KalmanTracker(self.get_kalman_location(x, y, box_width, box_height))
        self.penalty = 0

        # used to compare how far behind we need to make comparison
        self.features_history = list()

        bottom_location = self.get_bottom_mid_location(x, y, box_height)
        self.location_history.append(bottom_location)
        self.timestamp_history.append(time.time())
        self.features_history.append(features)

        self.id = id
        self.bounding_box = (box_width, box_height)
        self.extrapolated_location = Point(x, y)

        self.direction = 0

    '''
        Since we want to keep track of user history i.e last N locations or last N timestamps he was detected on.
        We don't want to overwhelm our system with keeping  track of all the tracked values. Hence we limit the 
        values in the list using this function. self.history_limit is used for setting the value of N.
    '''

    def pop_from_location_history(self):
        if len(self.location_history) > self.history_limit:
            self.location_history.pop(0)
            self.timestamp_history.pop(0)
            self.features_history.pop(0)
            return True
        return False

    def get_bottom_mid_location(self, x, y, box_height):
        return Point(x, (y + (box_height / 2)))

    '''
        Since kalman accepts bounding_box in x1,y1 (top-left),x2,y2 (bottom-right) format. YOLO returns the center
        x,y and width and height. This function converts the YOLO results to kalman format.

        @x: center x of bounding_box
        @y: center y of bounding_box
        @w: width of bounding_box
        @h: height of bounding_box
    '''

    def get_kalman_location(self, x, y, w, h):
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    '''
        Revert what get_kalman_location function does. 

        @x1: top left x of bounding_box
        @y1: top left y of bounding_box
        @x2: bottom right x of bounding_box
        @y2: bottom right y of bounding_box
    '''

    def revert_kalman_location(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        return int(x1 + w / 2), int(y1 + h / 2), int(x2 - w / 2), int(y2 - h / 2)

    '''
        called when the objects are paired in the FeatureTracking.py. We update the paired objects new location
        and other attributes in this function with the latest detected results for the frame.

        @x: center x of bounding_box as returned by YOLO
        @y: center y of bounding_box as returned by YOLO
        @box_width: bounding_box width as returned by YOLO
        @box_height: bounding_box height as returned by YOLO
        @features: calculated new histgoram features of the detected object
    '''

    def update_attributes(self, x, y, box_width, box_height, features):
        new_time = time.time()

        new_location = self.get_bottom_mid_location(x, y, box_height)
        new_feature_history = features
        self.kalman_tracker.update(self.get_kalman_location(x, y, box_width, box_height))
        self.location_history.append(new_location)
        self.features_history.append(new_feature_history)

        # self.location_history[0] = new_location
        # self.features_history[0] = new_feature_history

        self.pop_from_location_history()
        self.bounding_box = (box_width, box_height)

        self.last_seen = new_time

        self.extrapolated_location = self.calculate_extrapolated_points()

        self.timestamp_history.append(new_time)

    '''
        update the attributes with respect to kalman.predict() if we know that the person does exist but he is not 
        detected.

        @features: features detected w.r.t the extrapolated location
    '''

    def update_ghost_attributes(self, features):
        new_time = time.time()
        self.location_history.append(self.extrapolated_location)
        self.features_history.append(features)
        self.extrapolated_location = self.calculate_extrapolated_points()
        self.pop_from_location_history()

        self.timestamp_history.append(new_time)

    '''
        extrapolate the location of the object using kalman_tracker
    '''

    def calculate_extrapolated_points(self):
        pos = self.kalman_tracker.predict()
        x, y, w, h = self.revert_kalman_location(pos[0], pos[1], pos[2], pos[3])
        return Point(x, y)
