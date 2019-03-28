"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
from data_association import associate_detections_to_trackers
import feature_matching as fm
from shapely.geometry import Point
import csv

class Sort:

  def __init__(self, polygons, max_age=5, min_hits=3 ):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.trackerids = 0
    self.areas = polygons

    # for area_id, area in enumerate(self.areas):
    #   polygon = self.denormalize_polygon(area.polygon, width, height)
    #   area.polygon = Polygon(polygon)
    #   area.id = area_id
    #   area.inside = 0

    # how many previous positions you want to keep
    self.keep_history = 5


  def write_csv(self, file, values):
    with open(file , mode='a') as frames_file:
      frame_writer = csv.writer(frames_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      frame_writer.writerow(values)

  # for closed areas
  def is_inside(self, obj):
    try:
      if len(obj.location_history) > 3:
        for id, area in enumerate(self.areas):
          # print('locH', len(obj.location_history))
          if  area.polygon.contains(obj.location_history[1]) and \
            not area.polygon.contains(obj.location_history[2]) and \
            not area.polygon.contains(obj.location_history[3]) and \
            not area.polygon.contains(obj.location_history[4]):
            # the first time, this person was detected inside and now he is outside.


            # print(area.polygon.contains(obj.location_history[1]))
            # print(area.polygon.contains(obj.location_history[2]))
            # print(area.polygon.contains(obj.location_history[3]))
            # print(area.polygon.contains(obj.location_history[4]))
            # print('boy')
            area.inside -= 1
            self.write_csv('exp/tests/csv/main_dining_closed.csv', [obj.id, self.frame_count, 0, 1])
          elif not area.polygon.contains(obj.location_history[1]) and \
            area.polygon.contains(obj.location_history[2]) and \
            area.polygon.contains(obj.location_history[3]) and \
            area.polygon.contains(obj.location_history[4]):

            # the first time, this person was detected outside and now he is inside.
            # print(area.polygon.contains(obj.location_history[1]))
            # print(area.polygon.contains(obj.location_history[2]))
            # print(area.polygon.contains(obj.location_history[3]))
            # print(area.polygon.contains(obj.location_history[4]))
            area.inside += 1
            self.write_csv('exp/tests/csv/main_dining_closed.csv', [obj.id, self.frame_count, 1, 0])
    except:
        pass


  def get_bottom_mid_point(self, x1, x2, y2):

      w = x2 - x1
      bottom_mid_x = int(x1 + (w / 2))
      bottom_mid_y = int(y2)

      return bottom_mid_x, bottom_mid_y

  # we have to denormalize the polygon saved in config for use.
  def denormalize_polygon(self, polygon, width, height):
    np_area = np.array(polygon)

    np_area[:, 0] *= width
    np_area[:, 1] *= height

    return np.rint(np_area)

  def change_area_count(self, trk, area):
      if not area.polygon.contains(trk.location_history[-1]):
          area.inside -= 1



  def update(self,dets,img=None):


    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) #for kal!
      #print(pos)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)

    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if dets != []:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]

          features = fm.calculatePersonHistograms(dets[d,:][0][0],
                                                  dets[d,:][0][1],
                                                  dets[d,:][0][2],
                                                  dets[d,:][0][3],
                                                  img)

          trk.update(dets[d,:][0], features,img) ## for dlib re-intialize the trackers ?!

      #create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        features = fm.calculatePersonHistograms(dets[i][0],
                                     dets[i][1],
                                     dets[i][2],
                                     dets[i][3],
                                     img)
        self.trackers.append(KalmanBoxTracker(dets[i,:], features, self.trackerids))
        self.trackerids += 1

        # bottom_mid_x, bottom_mid_y = self.get_bottom_mid_point(dets[i,:][0], dets[i,:][2], dets[i,:][3])
        # if self.areas[0].polygon.contains(Point(bottom_mid_x,bottom_mid_y)):
        #     self.write_csv('/app/tests/csv/main_dining_closed.csv', ['obj', self.frame_count, 1, 0])

    i = len(self.trackers)
    in_count = i

    for area in self.areas:
        area.inside = in_count
        for trk in reversed(self.trackers):
            if dets == []:
              trk.update([],img)
            d = trk.get_state()

            # if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            ret.append(np.concatenate((d,[trk.id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            bottom_mid_x, bottom_mid_y = self.get_bottom_mid_point(ret[-1][-1][0], ret[-1][-1][2], ret[-1][-1][3])
            trk.location_history.append(Point(bottom_mid_x, bottom_mid_y))


            if trk.history_limit >= self.keep_history:
              trk.location_history.pop(0)
            else:
              trk.history_limit += 1

            i -= 1

            if area.closed == True:
                if len(trk.location_history) > 2:
                    self.change_area_count(trk, area)
            else:
                self.is_inside(trk)


                        # self.write_csv('/app/tests/csv/main_dining_closed.csv', ['obj', self.frame_count, 0, 1])

            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                pos = trk.predict(img)
                features = fm.calculatePersonHistograms(pos[0],
                                                        pos[1],
                                                        pos[2],
                                                        pos[3],
                                                        img)

                closeness = fm.calculateCloseness(trk.last_features, features)

                # scipy.misc.imsave('tests/test_images/frame' + str(trk.id) + '_' + str(self.frame_count) + '.jpg',
                #                   fm.get_sub_frame_using_bounding_box_results(pos[0],
                #                                                               pos[1],
                #                                                               pos[2],
                #                                                               pos[3],
                #                                                               img))
                if closeness > 0.25:
                    # pos = trk.predict(img)
                    # bottom_mid_x, bottom_mid_y = self.get_bottom_mid_point(pos[0], pos[2], pos[3])
                    #
                    #
                    # if self.areas[0].polygon.contains(Point(bottom_mid_x, bottom_mid_y)):
                    #     self.write_csv('/app/tests/csv/main_dining_closed.csv', ['obj', self.frame_count, 0, 1])

                    self.trackers.pop(i)
            # cv2.circle(img, (bottom_mid_x, bottom_mid_y), 5, (0, 0, 255), -1)

    print(self.areas)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))