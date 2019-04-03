"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
import core.feature_matching as fm
from core.kalman_tracker import KalmanBoxTracker
from data_association import associate_detections_to_trackers
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
class Sort:

  def __init__(self, areas, max_age=1, min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.areas = areas
    for area in self.areas:
        area.polygon = Polygon(area.polygon)
        area.inside = 0
        area.counters = {}
    # how many previous positions you want to keep
    self.keep_history = 5

  # for closed areas
  def is_inside(self, obj, area):

      if len(obj.location_history) > self.keep_history - 1:
        if  area.polygon.contains(obj.location_history[0]) and \
            area.polygon.contains(obj.location_history[1]) and \
            area.polygon.contains(obj.location_history[2]) and \
            not area.polygon.contains(obj.location_history[3]) and \
            not area.polygon.contains(obj.location_history[4]):
            # the first time, this person was detected inside and now he is outside.
            area.outside += 1
            self.update_counter(area.counters, area.interest, obj.obj_type + '-')
            return 'Out'

        elif not area.polygon.contains(obj.location_history[0]) and \
            not area.polygon.contains(obj.location_history[1]) and \
            not area.polygon.contains(obj.location_history[2]) and \
            area.polygon.contains(obj.location_history[3]) and \
            area.polygon.contains(obj.location_history[4]):
            # the first time, this person was detected outside and now he is inside.
            area.inside += 1
            self.update_counter(area.counters, area.interest, obj.obj_type + ' +')
            return 'In'
        try:
            if len(obj.location_history) > self.keep_history - 1:
                pass
        except:
            pass

  def update_counter(self, counters, name, suff):
      counter = name + ' ' + suff
      if not counter in counters:
          counters[counter] = 0
      counters[counter] += 1
      return counters[counter]

  def translate(self, name):
    name = name.decode("utf-8")
    if name in self.objects_map:
      return self.objects_map[name]
    else:
      return name

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
      if area.polygon.contains(trk.location_history[-1]):
          area.inside += 1

  def unique(self,list1):

      # intilize a null list
      unique_list = []

      # traverse for all elements
      for x in list1:
          # check if exists in unique_list or not
          if x not in unique_list:
              unique_list.append(x)
              # print list

      return unique_list

  def delete_objs(self,to_be_deleted):
      for index in sorted(to_be_deleted, reverse=True):
          del self.trackers[index]

  def run(self,dets,img, frame_width, frame_height, timestamp=None):


    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """

    detections = np.empty((0, 6))

    for obj, score, bounds in dets:
        x, y, w, h = bounds

        if obj.decode('utf-8') == 'person':
            name = 0
        else:
            name = 1

        detections = np.vstack((detections, (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2), score, name)))

    dets = detections

    self.timestamp = timestamp
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) #for kal!
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)

    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
        if len(self.trackers) > 0:
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
        self.trackers.append(KalmanBoxTracker(dets[i,:], features, dets[i,5]))

    to_be_deleted = []
    for area in self.areas:
        if area.closed:
            area.inside = 0
        for idx, trk in enumerate(reversed(self.trackers)):
            if dets == []:
              trk.update([],img)
            d = trk.get_state()

            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            bottom_mid_x, bottom_mid_y = self.get_bottom_mid_point(ret[-1][-1][0], ret[-1][-1][2], ret[-1][-1][3])
            trk.location_history.append(Point(bottom_mid_x, bottom_mid_y))


            if trk.history_limit >= self.keep_history:
                if len(self.trackers) > 0:
                    trk.location_history.pop(0)
            else:
              trk.history_limit += 1

            if area.closed == True:
                if len(trk.location_history) > 1:
                    self.change_area_count(trk, area)
            else:
                self.is_inside(trk, area)

            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                pos = trk.predict(img)
                features = fm.calculatePersonHistograms(pos[0],
                                                        pos[1],
                                                        pos[2],
                                                        pos[3],
                                                        img)


                closeness = fm.calculateCloseness(trk.last_features, features)

                # if distance between last time we saw this object and current estimation is more than 25. Kill it.
                if closeness > 0.25:
                    if len(self.trackers) > 0:
                        to_be_deleted.append(idx)

        print('Name =', area.name, 'Inside =', area.inside, 'Outside =', area.outside, 'Counters =', area.counters)

    to_be_deleted = self.unique(to_be_deleted)

    self.delete_objs(to_be_deleted)

    if(len(ret)>0):
      return self.areas, self.trackers
    return self.areas, self.trackers

