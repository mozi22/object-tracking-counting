import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import numpy as np
from core.FeatureTrackingObject import FeatureTrackingObject
from scipy.misc import imsave
class FeatureTracking:


    def __init__(self, areas):


        self.id = 0
        self.areas = areas
        self.old_frame = None

        self.known_objects = []
        self.time_to_live = 1

        self.bhattacharya_distance_limit = 0.35

        for area in self.areas:
            area.polygon = Polygon(area.polygon)
            area.inside = 0
            area.counters = {}


    def translate(self, name):
        return name.decode("utf-8")


    # we have to denormalize the polygon saved in config for use.
    def denormalize_polygon(self, polygon, width, height):
        np_area = np.array(polygon)

        np_area[:, 0] *= width
        np_area[:, 1] *= height

        return np.rint(np_area)

    # for closed areas
    def is_inside(self, obj, area):
        try:
            if len(obj.location_history) > 3:
                if  area.polygon.contains(obj.location_history[0]) and \
                    not area.polygon.contains(obj.location_history[1]) and \
                    not area.polygon.contains(obj.location_history[2]) and \
                    not area.polygon.contains(obj.location_history[3]) and \
                    not area.polygon.contains(obj.location_history[4]):
                    # the first time, this person was detected inside and now he is outside.
                    area.outside += 1
                    self.update_counter(area.counters, area.interest, '-')
                    return 'Out'
                elif not area.polygon.contains(obj.location_history[0]) and \
                    area.polygon.contains(obj.location_history[1]) and \
                    area.polygon.contains(obj.location_history[2]) and \
                    area.polygon.contains(obj.location_history[3]) and \
                    area.polygon.contains(obj.location_history[4]):

                    self.update_counter(area.counters, area.interest, '+')
                    # the first time, this person was detected outside and now he is inside.
                    area.inside += 1
                    imsave('mozi.jpg',self.get_sub_frame_using_bounding_box_results(obj.location_history[4].x,obj.location_history[4].y,obj.bounding_box[0],obj.bounding_box[1]))
                    return 'In'
        except:
            pass

    def update_counter(self, counters, name, suff):
        counter = name + ' ' + suff
        if not counter in counters:
            counters[counter] = 0
        counters[counter] += 1
        return counters[counter]

    def filter_person_objects(self,detections):

        person_obj = []
        for detection in detections:
            name, probability, coords = detection
            name = self.translate(name)
            if name == 'vehicle' or name == 'person':
                person_obj.append(detection)

        return person_obj


    def create_new_object(self,det_obj):

        name, probability, x, y, width, height = self.get_params_of_detection(det_obj)

        name = self.translate(name) + str(self.id)
        features = self.calculatePersonHistograms(x, y, width, height)

        new_obj = FeatureTrackingObject(name, x, y, width, height, features)

        self.id += 1
        self.known_objects.append(new_obj)


    def get_params_of_detection(self, det):
        name, probability, coords = det
        x, y, width, height = map(int, coords)

        return name, probability, x, y, width, height


    def perform_pairing(self, detected, knowns):

        n_detected = len(detected)
        n_knowns = len(knowns)

        # create a bhattacharya matrix and a distance matrix
        bhattacharya_matrix = np.zeros((n_detected, n_knowns))

        for i, det in enumerate(detected):

            name, probability, x, y, width, height = self.get_params_of_detection(det)

            features = self.calculatePersonHistograms(x,y,width,height)

            for j, known in enumerate(knowns):
                # save the features distance between the two boxes.
                bhattacharya_matrix[i,j] = self.calculateCloseness(known.features_history[-1], features)

        if bhattacharya_matrix.shape[1] == 0 or bhattacharya_matrix.size == 0:
            # empty known_objects
            return [], list(range(len(detected))), [], []

        # print(bhattacharya_matrix)
        return self.find_pairs(bhattacharya_matrix)


    def find_pairs(self, bhattacharya_matrix):

        bhattacharya_matrix[np.where(bhattacharya_matrix  == 0)] = 0.001
        bhattacharya_matrix[np.where(bhattacharya_matrix > 0.35)] = 10
        # get min values in each row
        result = np.multiply(bhattacharya_matrix, bhattacharya_matrix == np.min(bhattacharya_matrix, 1)[:, None])

        df = pd.DataFrame(result.T)

        row_sum = np.sum(df, axis=1)
        df1 = df[row_sum != 0].drop_duplicates()
        df0 = df[row_sum == 0]

        index0 = df0.index.values
        index1 = df1.index.values

        results = np.arange(0, (np.max(np.hstack([index0, index1]))))
        results2 = np.hstack([index0, index1])

        remove_objs = np.setdiff1d(results, results2)

        result = pd.concat([df1, df0]).sort_index().values.T

        # set all the 0 values as 10
        result[np.where(result == 0)] = 10

        # get min values in each col ( will be less than zero )
        result = np.multiply(result, result == np.min(result, 0)[None, :])

        result[np.where(result == 10)] = None
        result[np.where(result == 0)] = None

        pairs = np.argwhere(~np.isnan(result))

        pairs_values = result[~np.isnan(result)]
        pairs_values = pairs_values[:, np.newaxis]

        pairs = np.concatenate((pairs, pairs_values), axis=1)
        known_but_not_detected = np.argwhere(np.all(np.isnan(result), axis=0)).flatten()
        new_obj = np.argwhere(np.all(np.isnan(result), axis=1)).flatten()

        return pairs, new_obj, known_but_not_detected, remove_objs

    # get the coordinates of the part only where the bounding box appears.
    def get_x1_y1_x2_y2(self, x,y, width, height):

        division = 3

        y1 = int(y - (height / division))
        y2 = int(y + (height / division))
        x1 = int(x - (width / division))
        x2 = int(x + (width / division))

        if y1 < 0:
            y1 = 0
        if y2 < 0:
            y2 = 0
        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0

        return x1, y1, x2, y2

    def get_sub_frame_using_bounding_box_results(self, x, y, width, height):

        x1, y1, x2, y2 = self.get_x1_y1_x2_y2(x, y, width, height)

        return self.frame[y1:  y2, x1: x2]



    def calculateCloseness(self, prevFeatures, currentFeatures):
        return cv2.compareHist(prevFeatures, currentFeatures, cv2.HISTCMP_BHATTACHARYYA)

    def calculatePersonHistograms(self, x, y, width, height):

        sub_frame = self.get_sub_frame_using_bounding_box_results(x, y, width, height)
        return cv2.calcHist([sub_frame], [0], None, [256], [0, 256])


    # this function checks if there are less detections and more knowns. The present knowns are still in the image or not.
    # there can be 3 possibilities.
    # 1) the object  exists but not detected
    # 2) the object died

    def check_ghost_known_objects(self, ghost_knowns):

        # 1.a) take the Histogram of extrapolated point.
        # 1.b) compare with the histogram of last point.t)
        # 1.c) if the Bh dist is less than its still there.
        # 1.d) else it died

        to_be_deleted = []
        for ghost_known in ghost_knowns:

            ghost_obj = self.known_objects[ghost_known]


            old_features = ghost_obj.features_history[-1]
            features = self.calculatePersonHistograms(ghost_obj.extrapolated_location.x,
                                                      ghost_obj.extrapolated_location.y,
                                                      ghost_obj.bounding_box[0],
                                                      ghost_obj.bounding_box[1])


            ghost_closeness = self.calculateCloseness(old_features, features)

        # if an object was last seen more than time_to_live it means our box is just roaming around there without
        # the person/car being present there. Hence remove it.
        if ghost_closeness > self.bhattacharya_distance_limit:
            to_be_deleted.append(ghost_known)
        else:
            self.known_objects[ghost_known].update_ghost_attributes(features, ghost_closeness)
        self.delete_objs(to_be_deleted)

    def delete_objs(self,to_be_deleted):
        for index in sorted(to_be_deleted, reverse=True):
            del self.known_objects[index]

    def update_current_paired_objects(self, paired, detected):

        for pair in paired:
            detected_obj_index = int(pair[0])
            known_obj_index = int(pair[1])
            distance_among_them = pair[2]

            name, probability, x, y, width, height = self.get_params_of_detection(detected[detected_obj_index])
            features = self.calculatePersonHistograms(x, y, width, height)

            self.known_objects[known_obj_index].update_attributes(x, y, width, height, features, distance_among_them)

    def change_area_count(self, trk, area):
        if area.polygon.contains(trk.location_history[-1]):
            area.inside += 1

    def run(self, detections, frame, frame_w, frame_h, timestamp):

        # detections = self.filter_person_objects(detections)

        self.timestamp = timestamp
        self.frame = frame
        self.frame_w = frame_w
        self.frame_h = frame_h


        if len(self.known_objects) == 0 and len(detections) == 0:
            for result in detections:

                self.create_new_object(result)
            # n objects found
            return self.known_objects

        paired, not_paired, ghost_known, to_be_deleted = self.perform_pairing(detections, self.known_objects)

        self.delete_objs(to_be_deleted)

        # create new objects for the ones not paired.
        for non_paired_idx in not_paired:
            self.create_new_object(detections[non_paired_idx])

        self.update_current_paired_objects(paired,detections)

        # if detected objects are less than known objects this means 3 possibilities. Look at the function call for
        # the possibilities.
        if len(ghost_known) > 0:
            self.check_ghost_known_objects(ghost_known)



        for area in self.areas:
            if area.closed == True:
                area.inside = 0
            for obj in self.known_objects:
                if area.closed == True:
                    if len(obj.location_history) > 2:
                        self.change_area_count(obj, area)
                else:
                    self.is_inside(obj, area)
            # if area.name == 'pink_cross_line':
            #     print('Name =', area.name, 'Inside =', area.inside, 'Outside =', area.outside, 'Counters =', area.counters)

        # print(self.areas)

        return self.areas


