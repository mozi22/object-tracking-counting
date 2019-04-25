import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time
import pandas as pd
import numpy as np
import common.aggregation_pb2 as AggregationProto
import common.NanoREQ as nanoreq
from core.FeatureTrackingObject import FeatureTrackingObject
import math


class FeatureTracking:

    def __init__(self, tracker_settings, camera, aggregation_node):

        self.tracker_settings = tracker_settings
        self.camera = camera
        self.objects_map = tracker_settings.objects_map
        self.id = 0
        self.areas = tracker_settings.areas
        self.old_frame = None

        # list of objects currently tracked by the tracker
        self.known_objects = []

        # how much time to wait until killing an object if its not detected again.
        self.time_to_live = 1
        self.aggregator = nanoreq.NanoREQ(aggregation_node)

        # if a ghost object is further away from this distance, kill it.
        self.kill_distance_limit = 200

        # set the initial values for each area for this video.
        for area in self.areas:
            area.polygon = Polygon(area.polygon)
            area.inside = 0
            area.counters = {}

    '''
        used for mapping `truck`, `car` and other names to a single name `vehicle`

        @name: the name of the detected object by YOLO.
    '''

    def translate(self, name):
        name = name.decode("utf-8")
        if name in self.objects_map:
            return self.objects_map[name]
        else:
            return name

    '''
        The whole tracking works on denormalized locations. But since the points of the areas in the config file
        are saved as normalized we use this function to check if a person is inside an area or not(or crossed an area
        or not). Hence we normalize the location whenever needed.

        @location_x: the x location of the object
        @location_y: the y location of the object
    '''

    def normalize_location(self, location_x, location_y):
        return Point(location_x / self.frame_w, location_y / self.frame_h)

    '''

        the function is called for open areas only. The function is responsible for incrementing the inside count
        for an area if the last 3 positions of the person were outside of the area while the next two positions
        are inside the area. The other way around for counting outside count. 

        @obj: the tracked obj, one of the objects from self.known_objects.
    '''

    def is_inside(self, obj):
        try:
            if len(obj.location_history) > 3:

                for id, area in enumerate(self.areas):

                    if area.polygon.contains(
                            self.normalize_location(obj.location_history[0].x, obj.location_history[0].y)) and \
                            area.polygon.contains(
                                self.normalize_location(obj.location_history[1].x, obj.location_history[1].y)) and \
                            area.polygon.contains(
                                self.normalize_location(obj.location_history[2].x, obj.location_history[2].y)) and \
                            not area.polygon.contains(
                                self.normalize_location(obj.location_history[3].x, obj.location_history[3].y)) and \
                            not area.polygon.contains(
                                self.normalize_location(obj.location_history[4].x, obj.location_history[4].y)) and \
                            area.interest in obj.id:
                        # the first time, this person was detected inside and now he is outside.
                        area.inside -= 1
                        self.update_counter(area.counters, area.interest, '-')
                        return 'Out'
                    elif not area.polygon.contains(
                            self.normalize_location(obj.location_history[0].x, obj.location_history[0].y)) and \
                            not area.polygon.contains(
                                self.normalize_location(obj.location_history[1].x, obj.location_history[1].y)) and \
                            not area.polygon.contains(
                                self.normalize_location(obj.location_history[2].x, obj.location_history[2].y)) and \
                            area.polygon.contains(
                                self.normalize_location(obj.location_history[3].x, obj.location_history[3].y)) and \
                            area.polygon.contains(
                                self.normalize_location(obj.location_history[4].x, obj.location_history[4].y)) and \
                            area.interest in obj.id:

                        self.update_counter(area.counters, area.interest, '+')
                        # the first time, this person was detected outside and now he is inside.
                        area.inside += 1
                        return 'In'
        except:
            pass

    '''
        update the counter of people crossing inside or outside of an area. Only used for open areas.

        @counters: the counter for an area
        @name: name of the type of object, `person` or `vehicle`.
        @suff: inside(+) or outside(-) ?, 
    '''

    def update_counter(self, counters, name, suff):
        counter = name + ' ' + suff
        if not counter in counters:
            counters[counter] = 0
        counters[counter] += 1
        return counters[counter]

    '''
        if a video is responsible for detecting only person or vehicle, this function takes care of it.

        @detections: detection results(array) from YOLO
        @only_one: default(both) means consider both vehicle and person
                   vehicle means track only vehicles
                   person means tracking only person
    '''

    def filter_person_objects(self, detections, only_one='both'):

        person_obj = []
        for detection in detections:
            name, probability, coords = detection
            name = self.translate(name)
            if only_one != 'both':
                if name == only_one:
                    person_obj.append(detection)
            else:
                if name == 'vehicle' or name == 'person':
                    person_obj.append(detection)

        return person_obj

    '''
        function is responsible for creating new objects detected for the first time.
        it simply appends the objects in self.known_objects array which keeps a list
        of the currently tracked objects.

        @det_obj: Detection result(single object) from YOLO
    '''

    def create_new_object(self, det_obj):

        name, probability, x, y, width, height = self.get_params_of_detection(det_obj)

        name = self.translate(name) + str(self.id)
        features = self.calculatePersonHistograms(x, y, width, height)

        new_obj = FeatureTrackingObject(name, x, y, width, height, features)

        self.id += 1
        self.known_objects.append(new_obj)

    '''
        Its a helper function that separates out the fields in a detection result of YOLO
        @det: A detection result(single object) from YOLO
    '''

    def get_params_of_detection(self, det):
        name, probability, coords = det
        x, y, width, height = map(int, coords)

        return name, probability, x, y, width, height

    '''
        calculate the euclidean distance between two points
        @x1: x location of point 1
        @y1: y location of point 1
        @x2: x location of point 2
        @y2: y location of point 2
    '''

    def distance_calc(self, x1, y1, x2, y2):
        return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

    '''
        This function is responsible for pairing close objects with respect to features(histograms), distances and
        predicted distances.

        The function calculates a matrix of the following shape (len(detected_objects),len(known_objects))
        where detected objects are sent by YOLO and known_objects are known by this tracker in the self.known_objects
        array.

        It calculates a matrix of 3 distances.

        - features_matrix: responsible for comparing the histogram of the object seen last time and in this frame
        - distance_matrix: responsible for comparing distance between the last time object was seen and in this frame
        - kalman_pred_matrix: responsible for comparing distances with respect to the last known and the predicted 
                              location according to kalman tracker.
    '''

    def perform_pairing(self, detected, knowns):

        n_detected = len(detected)
        n_knowns = len(knowns)

        # create a bhattacharya matrix and a distance matrix
        features_matrix = np.zeros((n_detected, n_knowns))
        distance_matrix = np.zeros((n_detected, n_knowns))
        kalman_pred_matrix = np.zeros((n_detected, n_knowns))

        for i, det in enumerate(detected):

            name, probability, x, y, width, height = self.get_params_of_detection(det)

            features = self.calculatePersonHistograms(x, y, width, height)

            for j, known in enumerate(knowns):
                # save the features distance between the two boxes.
                features_matrix[i, j] = self.calculateCloseness(known.features_history[-1], features)
                distance_matrix[i, j] = self.distance_calc(known.location_history[-1].x, known.location_history[-1].y,
                                                           x, y)
                kalman_pred_matrix[i, j] = self.distance_calc(known.extrapolated_location.x,
                                                              known.extrapolated_location.y, x, y)

        if features_matrix.shape[1] == 0 or features_matrix.size == 0 and \
                distance_matrix.shape[1] == 0 or distance_matrix.size == 0 and \
                kalman_pred_matrix.shape[1] == 0 or kalman_pred_matrix.size == 0:
            # empty known_objects
            return [], list(range(len(detected))), [], []

        f_pairs, f_new_obj, f_known_but_not_detected, f_remove_objs = self.find_pairs(features_matrix)
        k_pairs, k_new_obj, k_known_but_not_detected, k_remove_objs = self.find_pairs(kalman_pred_matrix,
                                                                                      distance_limit=200, constant=9999)
        d_pairs, d_new_obj, d_known_but_not_detected, d_remove_objs = self.find_pairs(distance_matrix,
                                                                                      distance_limit=200, constant=9999)

        pairs, matrix_idx = self.return_most_likely_pairs(f_pairs, k_pairs, d_pairs)

        if matrix_idx == 0:
            new_obj = f_new_obj
            known_but_not_detected = f_known_but_not_detected
            remove_objs = f_remove_objs
        elif matrix_idx == 1:
            new_obj = k_new_obj
            known_but_not_detected = k_known_but_not_detected
            remove_objs = k_remove_objs
        else:
            new_obj = d_new_obj
            known_but_not_detected = d_known_but_not_detected
            remove_objs = d_remove_objs

        return pairs, new_obj, known_but_not_detected, remove_objs

    '''
        our goal is to identify maximum pairs and less new_objects. This function compares the 3 matrices
        to see if they have detected the same results(same pairs). If yes, this means we are very sure
        that the two pairs belong to each other. If that doesnt happen, we simply give priority to the
        pairs created using the kalman prediction matrix. There can be 3 possibilities.

        Take the example of the following pair matrix.

        [[ 0, 1, ... ]
        [ 1, 0, ... ]        
        [ 2, 2, ... ]]

        The matrix represents in the first row that the detected object 0 is paired with known_object 1 and so on for 
        the other rows. In the matrix ... represents the distance, could be the feature distance or euclidean distance
        which we are not interested in this function.

        The best case for us will be that all 3 matrices return us similar matrix. But there can be other scenarios
        also. Namely 3.

        1) All 3 pair matrix return the same result.
        2) 2 pair matrix return the same result.
        3) None of the matrix return the same result ( maybe one can find more pairs and other can find less pairs ).

        In case of step 1 & 2, the majority wins. Hence we agree that those are pairs that really exist.
        In case of step 3, we simply give priority to the pairs created using the euclidean distance between the 
        kalman predicted location of the object and the last location the object was seen on.

        @fp: the feature distance matrix
        @kp: the kalman prediction distance matrix
        @nd: the normal distance matrix ( from last location of the object )
    '''

    def return_most_likely_pairs(self, fp, kp, nd):

        combined_matrices = np.array((fp, kp, nd))

        combined_matrices_pairs = np.array([fp.shape[0], kp.shape[0], nd.shape[0]])

        '''
            @matrice_idx_with_max_pairs could be [0,1,2] or [0,2] or [0]
            0 = fp
            1 = kp
            2 = nd

            [0,2] means fp and nd make the max pairs
        '''
        matrice_idx_with_max_pairs = np.where(combined_matrices_pairs == combined_matrices_pairs.max())[0]

        if matrice_idx_with_max_pairs.shape[0] > 1:
            # more than 1 array making pair, if all 3 are making equal pairs thats good, but we are only interested,
            # if 2 makes it thats enough for us to decide that these pairs are the same
            if np.all(combined_matrices[matrice_idx_with_max_pairs[0]][:, 0] == combined_matrices[
                                                                                    matrice_idx_with_max_pairs[1]][:0]) \
                    and np.all(combined_matrices[matrice_idx_with_max_pairs[0]][:, 1] == combined_matrices[
                                                                                             matrice_idx_with_max_pairs[
                                                                                                 1]][:, 1]):
                return combined_matrices[matrice_idx_with_max_pairs[0]], matrice_idx_with_max_pairs[0]
            else:
                # this means they make the same pairs but the indices are different for e.g
                # fp makes a pair between 0,1 but kp makes a pair between 1,0. while 2,2 is same in both
                # in this case we'll give priority to kp
                return kp, 1
        else:
            # only 1 array making max pairs
            # in this case we'll give priority to kp
            return kp, 1

    '''
        consider an example of the following feature distance matrix.

        [[0.3, 0.7, 0.31 ],
         [0.27, 0.1, 0.02 ],
         [0.17, 0.41, 0.42 ]]

        The rows represent the detected objects while the column represents the known objects. The job of find_pairs is
        to find the pairs in the following format.

        A=[[1, 2, 0.3],
          [2, 0, 0.1]]

        The minimum distance is found row and column wise to ensure that only 1 detected object is paired with 1
        known_objects. For e.g a situation like in B cannot happen.

        B=[[1, 1, 0.3],
          [2, 1, 0.1]]

         Which means known_object 1 is paired with detected object 1 and 2. Hence in the above example(A) detected 
         object 1 is paired with 2 while detected object 2 is paired with 0.

        @calculated_matrix: could be any of the 3 matrices calculated in perform_pairing function.
        @distance_limit: If obj1 and obj2 has a distance < distance_limit than they can be considered the same object.
        @constant: its a helper field used to find minimum distances column wise and row wise. Should be set to 9999
                   for euclidean distances but 10 for bhattacharya_distance.
    '''

    def find_pairs(self, calculated_matrix, distance_limit=0.35, constant=10):

        calculated_matrix[np.where(calculated_matrix == 0)] = 0.001
        calculated_matrix[np.where(calculated_matrix > distance_limit)] = constant
        # get min values in each row
        result = np.multiply(calculated_matrix, calculated_matrix == np.min(calculated_matrix, 1)[:, None])

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
        result[np.where(result == 0)] = constant

        # get min values in each col ( will be less than zero )
        result = np.multiply(result, result == np.min(result, 0)[None, :])

        result[np.where(result == constant)] = None
        result[np.where(result == 0)] = None

        pairs = np.argwhere(~np.isnan(result))

        pairs_values = result[~np.isnan(result)]
        pairs_values = pairs_values[:, np.newaxis]

        pairs = np.concatenate((pairs, pairs_values), axis=1)
        known_but_not_detected = np.argwhere(np.all(np.isnan(result), axis=0)).flatten()
        new_obj = np.argwhere(np.all(np.isnan(result), axis=1)).flatten()

        return pairs, new_obj, known_but_not_detected, remove_objs

    '''
        the function is responsible for returning the top left and bottom right corner of the bounding box.

        @x: the center x location of the bounding_box
        @y: the center y location of the bounding_box
        @width: width of the bounding_box
        @height: height of the bounding_box
    '''

    def get_x1_y1_x2_y2(self, x, y, width, height):

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

    '''
        returns the area from the frame where the bounding_box appears.

        @x: the center x location of the bounding_box
        @y: the center y location of the bounding_box
        @width: width of the bounding_box
        @height: height of the bounding_box
    '''

    def get_sub_frame_using_bounding_box_results(self, x, y, width, height):

        x1, y1, x2, y2 = self.get_x1_y1_x2_y2(x, y, width, height)

        return self.frame[y1:  y2, x1: x2]

    '''
        returns the BHATTACHARYA_DISTANCE between two histograms.

        @prevFeatures: represent the previous features of an object calculated using calculatePersonHistograms function.
        @currentFeatures: represent the previous features of an object calculated using calculatePersonHistograms 
                          function.
    '''

    def calculateCloseness(self, prevFeatures, currentFeatures):
        return cv2.compareHist(prevFeatures, currentFeatures, cv2.HISTCMP_BHATTACHARYYA)

    '''
        returns the histogram of features for a bounding_box.

        @x: the center x location of the bounding_box
        @y: the center y location of the bounding_box
        @width: width of the bounding_box
        @height: height of the bounding_box
    '''

    def calculatePersonHistograms(self, x, y, width, height):

        sub_frame = self.get_sub_frame_using_bounding_box_results(x, y, width, height)
        return cv2.calcHist([sub_frame], [0], None, [256], [0, 256])

    '''
        there are some objects which might not be detected but we still need to track them. This function is responsible
        for it. It uses the extrapolated location calculated using kalman.predict(). This function deletes an object
        if he has not been detected for a long time, more than time_to_live or if he is too far from the last time
        he was detected ( kill_distance_limit ).

        @ghost_knowns: list of self.known_objects which were not detected in this frame but were detected in 
                        previous frames.
    '''

    def check_ghost_known_objects(self, ghost_knowns):

        to_be_deleted = []
        for ghost_known in ghost_knowns:
            ghost_obj = self.known_objects[ghost_known]

            # old_features = ghost_obj.features_history[-1]
            features = self.calculatePersonHistograms(ghost_obj.extrapolated_location.x,
                                                      ghost_obj.extrapolated_location.y,
                                                      ghost_obj.bounding_box[0],
                                                      ghost_obj.bounding_box[1])

            # ghost_closeness = self.calculateCloseness(old_features, features)
            ghost_closeness = self.distance_calc(ghost_obj.extrapolated_location.x,
                                                 ghost_obj.extrapolated_location.y,
                                                 ghost_obj.location_history[-1].x,
                                                 ghost_obj.location_history[-1].y)

        # if an object was last seen more than time_to_live it means our box is just roaming around there without
        # the person/car being present there. Hence remove it.
        if ghost_closeness > self.kill_distance_limit or self.time_to_live < time.time() - ghost_obj.last_seen:
            to_be_deleted.append(ghost_known)
        else:
            self.known_objects[ghost_known].update_ghost_attributes(features)
        self.delete_objs(to_be_deleted)

    '''
        deletes the known_objects which are no longer tracked.

        @to_be_deleted: a list() with the indexes of objects to be deleted.
    '''

    def delete_objs(self, to_be_deleted):
        for index in sorted(to_be_deleted, reverse=True):
            del self.known_objects[index]

    '''
        the new locations and other attributes are updated for the paired objects in this function.

        @paired: a matrix representing the pairs in the form mentioned in the documentation of return_most_likely_pairs
                 function.

        @detected: the detection results from YOLO.
    '''

    def update_current_paired_objects(self, paired, detected):

        for pair in paired:
            # represent the detected object index
            detected_obj_index = int(pair[0])

            # represent the known_object index paired with the detected object using the perform_pairing function.
            known_obj_index = int(pair[1])

            # this could be euclidean distance or feature distance.
            distance_among_them = pair[2]

            name, probability, x, y, width, height = self.get_params_of_detection(detected[detected_obj_index])
            features = self.calculatePersonHistograms(x, y, width, height)

            self.known_objects[known_obj_index].update_attributes(x, y, width, height, features)

    '''
        this is called for closed regions only. The function is responsible for counting someone as inside in a closed
        area if he is present inside the area. In closed area we reset the number of people inside an area to 0 and
        individually count each person to be inside for each frame. If he/she is inside we increment the count.

        @trk: the tracked object
        @area: the DotMap area object. 
    '''

    def change_area_count(self, trk, area):
        if area.polygon.contains(self.normalize_location(trk.location_history[-1].x,
                                                         trk.location_history[-1].y)) and area.interest in trk.id:
            area.inside += 1

    '''
        This is the heart of the algorithm which coordinates between all other functions.

        @detections: bounding boxes(array) returned by the YOLO
        @frame: current frame to be processed
        @frame_w: width of the frame
        @frame_h: height of the frame
        @timestamp: current timestamp (time.time())
    '''

    def run(self, detections, frame, frame_w, frame_h, timestamp):

        detections = self.filter_person_objects(detections, 'vehicle')

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

        self.update_current_paired_objects(paired, detections)

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
                        self.notify_aggregator(area, area.inside)
                else:
                    in_or_out = self.is_inside(obj)
                    self.notify_aggregator(area, 1 if in_or_out == 'In' else -1)

        return self.areas

    '''
        sends the compiled tracking results to the aggregator

        @area: the DotMap area object containing all of its useful attributes.
        @count: represents if someone entered or left. +1 represents entered while -1 represents left.
                In closed area it represents the number of people inside an area.
    '''

    def notify_aggregator(self, area, count):
        data = AggregationProto.Aggregation()
        data.timestamp = self.timestamp
        self.timestamp += 0.000001
        data.camera = self.camera
        data.region_name = area.name
        data.closed = area.closed
        data.klass = area.interest
        data.count = count
        self.aggregator.send(data.SerializeToString())
