from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
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
        self.penalty = 0

        # used to compare how far behind we need to make comparison
        self.features_history = list()

        # used for changing box direction in tracking
        self.closeness_history = list()

        bottom_location = self.get_bottom_mid_location(x, y, box_width, box_height)
        self.location_history.append(bottom_location)
        self.timestamp_history.append(time.time())
        self.features_history.append(features)

        self.id = id
        self.bounding_box = (box_width, box_height)
        self.extrapolated_location = Point(x,y)

        self.direction = 0


    def pop_from_location_history(self):
        if len(self.location_history) > self.history_limit:
            self.location_history.pop(0)
            self.timestamp_history.pop(0)
            self.features_history.pop(0)
            self.closeness_history.pop(0)
            return True
        return False

    def get_bottom_mid_location(self, x, y, box_width, box_height):
        return Point((x + (box_width / 2)), (y + (box_height / 2)))

    def update_attributes(self, x, y, box_width, box_height, features, closeness):

        new_time = time.time()

        new_location = self.get_bottom_mid_location(x, y, box_width, box_height)
        new_closeness = closeness
        new_feature_history = features

        self.location_history.append(new_location)
        self.closeness_history.append(new_closeness)
        self.features_history.append(new_feature_history)

        # self.location_history[0] = new_location
        # self.closeness_history[0] = new_closeness
        # self.features_history[0] = new_feature_history

        self.pop_from_location_history()
        self.bounding_box = (box_width, box_height)

        self.last_seen = new_time

        # we need atleast 2 points to extrapolate the 3rd point.
        if len(self.timestamp_history) > 3:
            self.extrapolated_location = self.calculate_extrapolated_points(
                                                                            self.location_history[-1],
                                                                            box_width, box_height)
        else:
            self.extrapolated_location = Point(x, y)

        self.timestamp_history.append(new_time)

    def update_ghost_attributes(self, features, closeness):

        new_time = time.time()
        self.location_history.append(self.extrapolated_location)
        self.closeness_history.append(closeness)
        self.features_history.append(features)
        self.pop_from_location_history()



        # we need atleast 2 points to extrapolate the 3rd point.
        if len(self.timestamp_history) > 3:
            self.extrapolated_location = self.calculate_extrapolated_points(
                                                                            self.location_history[-1],
                                                                            self.bounding_box[0], self.bounding_box[1])


        self.timestamp_history.append(new_time)


    def calculate_extrapolated_points(self, curr_loc, width, height):

        # directions = [
        #                 ['+', '+'],
        #                 ['+', '-'],
        #                 ['-', '+'],
        #                 ['-', '-'],
        # ]

        if self.closeness_history[-1] > self.closeness_history[-3]:
            self.direction += 1

            if self.direction > 3:
                self.direction = 0

        # minimizing the bhattacharya distance by moving the box in all 4 directions
        # how fast you want to move the box can be controlled using self.threshold
        if self.direction == 0:
            x = curr_loc.x + self.threshold
            y = curr_loc.y + self.threshold
        elif self.direction == 1:
            x = curr_loc.x + self.threshold
            y = curr_loc.y - self.threshold
        elif self.direction == 2:
            x = curr_loc.x - self.threshold
            y = curr_loc.y + self.threshold
        else:
            x = curr_loc.x - self.threshold
            y = curr_loc.y - self.threshold

        return self.get_bottom_mid_location(x,y,width,height)
