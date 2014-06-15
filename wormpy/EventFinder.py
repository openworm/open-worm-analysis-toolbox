# -*- coding: utf-8 -*-
"""
  EventFinder.py
  
  A module for finding the motion state of each frame of a worm video.

  Contents
  ---------------------------------------
  This module contains definitions for the following:

  Classes:
    EventFinder
      __init__
      get_events

    EventList
      __init__
      num_events @property
      get_event_mask
      merge

    EventOutputStructure
      __init__
      get_feature_struct
      

  Usage
  ---------------------------------------
  LocomotionFeatures.get_motion_codes() calculates the motion codes for the
  worm, and to do so, for each of the possible motion states (forward, 
  backward, paused) it creates an instance of the EventFinder class,
  sets up appropriate parameters, and then calls EventFinder.get_events()
  to obtain an instance of EventList, the "result" class.
  
  Then to format the result appropriately, the EventOutputStructure class is 
  instantiated with our "result" and then get_feature_struct is called.
  
  So the flow from within LocomotionFeatures.get_motion_codes() is:
    # (approximately):
    ef = EventFinder()
    event_list = ef.get_events()
    me = EventOutputStructure(event_list)
    return me.get_feature_struct

  EventOutputStructure is used by not just get_motion_codes but also ...
  DEBUG (add the other uses (e.g. upsilons))
  
  Notes
  ---------------------------------------
  See https://github.com/openworm/movement_validation/blob/master/
  documentation/Yemini%20Supplemental%20Data/Locomotion.md#2-motion-states
  for a plain-English description of a motion state.

  The code for this module came from several files in the 
  @event_finder and @event_ss folders from:
  https://github.com/JimHokanson/SegwormMatlabClasses/blob/
    master/%2Bseg_worm/%2Bfeature/

"""

import numpy as np
import operator
from itertools import groupby
from wormpy import config



class EventFinder:
  """
  class EventFinder

  To use this, create an instance, then specify the options.  Default
  options are initialized in __init__.
  
  Then call get_events() to obtain an EventList instance 
  containing the desired events from a given block of data.
  
  """

  def __init__(self):
    # Temporal thresholds
    self.min_frames_threshold = None #(scalar or [1 x n_frames])
    self.include_at_frames_threshold = config.INCLUDE_AT_FRAMES_THRESHOLD    

    self.max_inter_frames_threshold = None #(scalar or [1 x n_frames])
    self.include_at_inter_frames_threshold = \
            config.INCLUDE_AT_INTER_FRAMES_THRESHOLD

    
    # Space (distance) and space&time (speed) thresholds
    self.min_distance_threshold = None #(scalar or [1 x n_frames])
    self.max_distance_threshold = None #(scalar or [1 x n_frames])
    self.include_at_distance_threshold = config.INCLUDE_AT_DISTANCE_THRESHOLD

    self.min_speed_threshold = None #(scalar or [1 x n_frames])
    self.max_speed_threshold = None #(scalar or [1 x n_frames])
    self.include_at_speed_threshold = config.INCLUDE_AT_SPEED_THRESHOLD

      
  
  def get_events(self, speed_data, distance_data=None):
    """
    Obtain the events implied by event_data, given how this instance
    of EventFinder has been configured.
    
    Parameters
    ---------------------------------------
    speed_data   : 1-d numpy array of length n
      Gives the per-frame instantaneous speed as % of skeleton length
    distance_data   : 1-d numpy array of length n (optional)
      Gives the per-frame distance travelled as % of skeleton length
      If not specified, speed_data will be used to derive distance_data,
      since speed = distance x time.
    
    Returns
    ---------------------------------------
    An instance of class EventList

    Notes:
    ---------------------------------------
    If the first/last event are solely preceded/followed by NaN
    frames, these frames are swallowed into the respective event.

    Formerly getEvents.m.  Originally it was findEvent.m.  

    """
    # Override distance_data with speed_data if it was not provided
    if not distance_data.size:
      distance_data = speed_data
        
    # For each frame, determine if it matches our speed threshold criteria
    speed_mask = self.get_speed_threshold_mask(speed_data)
    
    # Convert our mask into the indices of the "runs" of True, that is 
    # of the data matching our above speed criteria
    event_candidates = self.get_start_stop_indices(speed_data, speed_mask)

    # Possible short circuit: if we have absolutely no qualifying events 
    # in event_data, just exit early.
    if not event_candidates.size:
      return EventList()
      
    # In this function we remove time gaps between events if the gaps 
    # are too small (max_inter_frames_threshold)
    if self.include_at_inter_frames_threshold:
      event_candidates = self.remove_gaps(event_candidates, 
                                          self.max_inter_frames_threshold, 
                                          operator.le)
    else: # the threshold is exclusive
      event_candidates = self.remove_gaps(event_candidates, 
                                          self.max_inter_frames_threshold, 
                                          operator.lt)

    # Remove events that aren't at least
    # self.min_frames_threshold in length
    event_candidates = \
      self.remove_too_small_events(event_candidates)
      

  
    # For each candidate event, sum the instantaneous speed at all 
    # frames in the event, and decide if the worm moved enough distance
    # for the event to qualify as genuine.
    # i.e. Filter events based on data sums during event
    event_candidates = \
      self.remove_events_by_data_sum(event_candidates, distance_data)

    return EventList(event_candidates)

  
  def get_speed_threshold_mask(self, event_data):
    """
    Get possible events between the speed thresholds.  Return a mask
    
    Parameters
    ---------------------------------------
    event_data: 1-d numpy array of instantaneous worm speeds
    
    Returns
    ---------------------------------------
    A 1-d boolean numpy array masking any instantaneous speeds falling
    frame-by-frame within the boundaries specified by 
    self.min_speed_threshold and self.max_speed_threshold, 
    which are themselves 1-d arrays.
  
    Notes
    ---------------------------------------
    Formerly h__getPossibleEventsByThreshold, in 
    seg_worm/feature/event_finder/getEvents.m
    
    """

    # Start with a mask that's all True since if neither min or max thresholds 
    # were set there was nothing to mask.
    event_mask = np.ones((len(event_data)), dtype=bool)

    # If min_speed_threshold has been initialized to something...
    if self.min_speed_threshold != None:  
        if self.include_at_speed_threshold:
            event_mask = event_data >= self.min_speed_threshold
        else:
            event_mask = event_data > self.min_speed_threshold
    
    # If max_speed_threshold has been initialized to something...
    if self.max_speed_threshold != None:
        if self.include_at_speed_threshold:
            event_mask = event_mask & (event_data <= self.max_speed_threshold)
        else:
            event_mask = event_mask & (event_data < self.max_speed_threshold)

    return event_mask  


  def get_start_stop_indices(self, event_data, event_mask):
    """
    From a numpy event mask, get the start and stop indices.  For
    example:

      0 1 2 3 4 5   <- indices
      F F T T F F   <- event_mask
    F F F T T F F F <- bracketed_event_mask
          s s       <- start and stop
    So in this case we'd have as an output [(2,3)], for the one run 
    that starts at 2 and ends at 3.
    
    Parameters
    ---------------------------------------
    event_data: 1-d float numpy array 
      Instantaneous worm speeds
    event_mask: 1-d boolean numpy array 
      True if the frame is a possible event candidate for this event.
    
    Returns
    ---------------------------------------
    event_candidates: 2-d int numpy array
      An array of tuples giving the start and stop, respectively,
      of each run of Trues in the event_mask.
  
    Notes
    ---------------------------------------
    Formerly h__getStartStopIndices, in 
    seg_worm/feature/event_finder/getEvents.m
    
    """
    # Make sure our parameters are of the correct type and dimension
    assert(type(event_data) == np.ndarray)
    assert(type(event_mask) == np.ndarray)
    assert(len(np.shape(event_data)) == 1)
    assert(np.shape(event_data) == np.shape(event_mask))
    assert(event_mask.dtype == bool)
    assert(event_data.dtype == float)
    
    # We concatenate falses to ensure event starts and stops at the edges
    # are caught
    bracketed_event_mask = np.concatenate([[False], event_mask, [False]])


    # Let's obtain the "x-coordinates" of the True entries.
    # e.g. If our bracketed_event_mask is
    # [False, False, False, True, False True, True, True, False], then
    # we obtain the array [3, 5, 6, 7]
    x = np.flatnonzero(bracketed_event_mask)

    # Group these together using a fancy trick from 
    # http://stackoverflow.com/questions/2154249/, since
    # the lambda function x:x[0]-x[1] on an enumerated list will
    # group consecutive integers together
    # e.g. [[(0, 3)], [(1, 5), (2, 6), (3, 7)]]
    x_grouped = [list(group) for key, group in groupby(enumerate(x), 
                                                       lambda i:i[0]-i[1])]

    # We want to know the first element from each "run", and the last element
    # e.g. [[3, 4], [5, 7]]
    event_candidates = [(i[0][1], i[-1][1]) for i in x_grouped]

    # Early exit if we have no starts and stops at all
    if not event_candidates:
      return np.array(event_candidates)
    
    # If a run of NaNs precedes the first start index, all the way back to 
    # the first element, then revise our first (start, stop) entry to include
    # all those NaNs.
    if np.all(np.isnan(event_data[:event_candidates[0][0]])):
      event_candidates[0] = (0, event_candidates[0][1])
    
    # Same but with NaNs succeeding the final end index.    
    if np.all(np.isnan(event_data[event_candidates[-1][1]:])):
      event_candidates[-1] = (event_candidates[-1][0], event_data.size-1)
      
    return np.array(event_candidates)

  
  def remove_gaps(self, event_candidates, threshold, 
                  comparison_operator):
    """
    Remove time gaps in the events that are smaller/larger than a given 
    threshold value.
    
    That is, apply a greedy right-concatenation to any (start, stop) duples 
    within threshold of each other.
    
    Parameters
    ---------------------------------------
    event_candidates: a list of (start, stop) duples
      The start and stop indexes of the events
    threshold: int
      Number of frames to do the comparison on
    comparison_operator: a comparison function
      One of operator.lt, le, ge, gt
    
    Returns
    ---------------------------------------
    A new numpy array of (start, stop) duples giving the indexes 
    of the events with the gaps removed.
    
    """
    assert(comparison_operator == operator.lt or
           comparison_operator == operator.le or
           comparison_operator == operator.gt or
           comparison_operator == operator.ge)

    new_event_candidates = []
    num_groups = np.shape(event_candidates)[0]
    
    i = 0
    while(i < num_groups):
      # Now advance through groups to the right, 
      # continuing as long as they satisfy our comparison operator
      j = i 
      while(j+1 < num_groups and comparison_operator(
             event_candidates[j+1][0] - event_candidates[j][1],
             threshold)):
        j += 1

      # Add this largest possible start/stop duple to our NEW revised list        
      new_event_candidates.append((event_candidates[i][0], 
                                   event_candidates[j][1]))
      i = j + 1
    
    return np.array(new_event_candidates)
  
  
  
  def remove_too_small_events(self, event_candidates):
    """
    This function filters events based on time (really sample count)
    
    Parameters
    ---------------------------------------
    event_candidates: numpy array of (start, stop) duples    
    
    Returns
    ---------------------------------------
    numpy array of (start, stop) duples

    """
    if not self.min_frames_threshold:
      return event_candidates
    
    event_num_frames = event_candidates[:,1] - event_candidates[:,0] + 1

    events_to_remove = np.zeros(len(event_num_frames), dtype=bool)

    if self.include_at_frames_threshold:
      events_to_remove = event_num_frames <= self.min_frames_threshold
    else:
      events_to_remove = event_num_frames < self.min_frames_threshold

    return event_candidates[np.flatnonzero(~events_to_remove)]
    
    
  def remove_events_by_data_sum(self, event_candidates, distance_data):
    """
    This function removes events by data sum.  An event is only valid
    if the worm has moved a certain minimum proportion of its mean length 
    over the course of the event.

    For example, a forward motion state is only a forward event if the 
    worm has moved at least 5% of its mean length over the entire period.
    
    For a given event candidate, to calculate the worm's movement we 
    SUM the worm's distance travelled per frame (distance_data) over the 
    frames in the event.
    
    Parameters
    ---------------------------------------
    event_candidates: numpy array of (start, stop) duples    
    
    Returns
    ---------------------------------------
    numpy array of (start, stop) duples
      A subset of event_candidates with only the qualifying events.
    
    Notes
    ---------------------------------------
    Formerly h____RemoveEventsByDataSum
   
    """
    # If we've established no distance thresholds, we have nothing to 
    # remove from event_candidates
    if self.min_distance_threshold == None and \
       self.max_distance_threshold == None:
       return event_candidates

    # Sum over the event and threshold data so we know exactly how far
    # the worm did travel in each candidate event, and also the min/max
    # distance it MUST travel for the event to be considered valid
    # --------------------------------------------------------
    
    num_runs = np.shape(event_candidates)[0]

    # Sum the actual distance travelled by the worm during each candidate event
    event_sums = np.empty(num_runs, dtype=float)
    for i in range(num_runs):
      event_sums[i] = np.nansum(distance_data
                  [event_candidates[i][0]:(event_candidates[i][1]+1)])

    # self.min_distance_threshold contains a 1-d n-element array of
    # skeleton lengths * 5% or whatever proportion we've decided the 
    # worm must move for our event to be valid.  So to figure out the 
    # threshold for a given we event, we must take the MEAN of this 
    # threshold array.
    #
    # Note that we test if min_distance_threshold event contains any 
    # elements, since we may have opted to simply not include this 
    # threshold at all.
    min_threshold_sums = np.empty(num_runs, dtype=float)
    if self.min_distance_threshold != None:
      for i in range(num_runs):
        min_threshold_sums[i] = np.nanmean(self.min_distance_threshold
                  [event_candidates[i][0]:(event_candidates[i][1]+1)])

    # Same procedure as above, but for the maximum distance threshold.  
    max_threshold_sums = np.empty(num_runs, dtype=float)
    if self.max_distance_threshold != None:
      for i in range(num_runs):
        max_threshold_sums[i] = np.nanmean(self.max_distance_threshold
                  [event_candidates[i][0]:(event_candidates[i][1]+1)])


    # Actual filtering of the candidate events
    # --------------------------------------------------------
    
    events_to_remove = np.zeros(num_runs, dtype=bool)

    # Remove events where the worm travelled too little
    if self.min_distance_threshold != None:
      if self.include_at_distance_threshold:
        events_to_remove = (event_sums <= min_threshold_sums)
      else:
        events_to_remove = (event_sums < min_threshold_sums)
      
    # Remove events where the worm travelled too much
    if self.max_distance_threshold != None:
      if self.include_at_distance_threshold:
        events_to_remove = events_to_remove | \
                           (event_sums >= max_threshold_sums)
      else:
        events_to_remove = events_to_remove | \
                           (event_sums > max_threshold_sums)

    return event_candidates[np.flatnonzero(~events_to_remove)]
  







class EventList:
  """
  A list of events.

  (An event is simply a contiguous subset of frame indices.)

  You can ask for a representation of the event list as
  1) a sequence of (start, stop) duples
  2) a boolean array of length num_frames with True for all event frames
  
  Properties
  ---------------------------------------
  start_Is: 1-d numpy array
  end_Is: 1-d numpy array
  starts_and_stops: 2-d numpy array
  num_events: int
  num_events_for_stats: int
  last_frame: int
  

  Methods
  ---------------------------------------
  get_event_mask: returns 1-d boolean numpy array
  merge: returns an EventList instance

  Notes
  ---------------------------------------
  
  Previous name:
  seg_worm.feature.event_ss ("ss" stands for "simple structure")

  @MichaelCurrie: in @JimHokanson's original code there were lines
  to change start_Is and end_Is from column to row vectors, if
  necessary.  Because here we use numpy arrays, they are not 
  treated as matrices so we don't need to care.

  
  @JimHokanson: I was going to leave this class as just a Matlab 
  structure but there is at least one function that would be better 
  as a method of this class.
  
  @JimHokanson: This class is the encapsulation of the raw 
  substructure, or output from finding the event.
  
  """
  def __init__(self, event_starts_and_stops=None):
    # self.start_Is and self.end_Is will be the internal representation
    # of the events within this class.
    self.start_Is = None
    self.end_Is = None
    
    # Check if our events array exists and there is at least one event
    if event_starts_and_stops != None and event_starts_and_stops.size != 0:
      self.start_Is = event_starts_and_stops[:,0]
      self.end_Is   = event_starts_and_stops[:,1]
    
    if(self.start_Is == None):
      self.start_Is = np.array([], dtype=int)

    if(self.end_Is == None):
      self.end_Is = np.array([], dtype=int)
  
  @property
  def starts_and_stops(self):
    """
    Returns the start_Is and end_Is as a single numpy array
    """
    s_and_s = np.array([self.start_Is, self.end_Is])

    # check that we didn't have s_and_s = [None, None] or something
    if len(np.shape(s_and_s)) == 2:
      # We need the first dimension to be the events, and the second
      # to be the start / end, not the other way around.
      # i.e. we want it to be n x 2, not 2 x n
      s_and_s = np.rollaxis(s_and_s, 1)
      return s_and_s
    else:
      return np.array([])

  @property
  def __len__(self):
    """ 
    Return the number of events stored by a given instance of this class.

    Notes
    ---------------------------------------
    Formerly n_events
    
    """
    return len(self.start_Is)

  @property
  def last_event_frame(self):
    """
    Return the frame # of end of the final event
    
    Notes
    ---------------------------------------
    Note that the events represented by a given instance 
    must have come from a video of at least this many 
    frames.

    """
    # Check if the end_Is have any entries at all
    if self.end_Is != None and self.end_Is.size != 0:
      return self.end_Is[-1]
    else:
      return 0

  @property
  def num_events_for_stats(self):
    """
    Compute the number of events, excluding the partially recorded ones.
    """
    value = self.__len__
    if value > 1:
      if self.start_Is[0] == 0:
        value = value - 1
      if self.end_Is[-1] == self.num_video_frames:
        value = value - 1

    return value

  def get_event_mask(self, num_frames=None):
    """
    Obtain a boolean array of length num_frames, where all events 
    within num_frames are set to True and other frames marked False
    
    Returns an array with True entries only between 
    start_Is[i] and end_Is[i], for all i such that 
    0 <= end_Is[i] < num_frames    
    
    Parameters
    ---------------------------------------
    num_frames: int (optional)
      The number of frames to use in the mask
      If num_frames is not given, a mask just large enough to accomodate
      all the events is returned (i.e. of length self.last_event_frame+1)
    
    Returns
    ---------------------------------------
    1-d boolean numpy array of length num_frames
    
    """
    # @JimHokanson TODO
    # seg_worm.events.events2stats - move here
    # fromStruct - from the old struct version ...
    
    # Create empty array of all False, as large as 
    # it might possibly need to be
    if not num_frames:
      num_frames = self.last_event_frame + 1
      
    mask = np.zeros(max(self.last_event_frame + 1, num_frames), dtype='bool')

    for i_event in range(self.__len__):
      mask[self.start_Is[i_event]:self.end_Is[i_event]+1] = True
    
    return mask[0:num_frames]

  @classmethod
  def merge(cls, obj1, obj2):
    """
    Merge two EventList instances, effectively merging lists of events into
    a larger list.

    Merges two instances of the EventList class together via concatenation
    
    Acts as a factory, producing a third instance of the EventList class
    that is the concatenation of the first two, with the start indices 
    blended and properly in order.

    Parameters
    ---------------------------------------
    cls: The static class parameter, associated with @classmethod
    obj1: EventList instance
    obj2: EventList instance
    
    Returns
    ---------------------------------------
    EventList: A new EventList instance
    
    """
    all_starts = np.concatenate(obj1.start_Is, obj2.start_Is)
    all_ends   = np.concatenate(obj1.end_Is,   obj2.end_Is)
    
    # @JimHokanson TODO: Would be good to check that events don't overlap ...
    
    new_starts = np.sort(all_starts)
    order_I = np.argsort(all_starts)
        
    new_ends   = all_ends[order_I]
    
    # Since we have sorted and intermingled the two sets of events, we
    # have lost information about which instance the events are a part of
    # at some point we could choose to alter this method and return this
    # variable since it stores information about which instance the 
    # element is a part of:
    #is_first   = np.concatenate(np.ones(obj1.num_events, dtype='bool'),
    #                            np.ones(obj2.num_events, dtype='bool'))
    #is_first_object = is_first[order_I]
    
    return EventList(new_starts, new_ends)




class EventListForOutput(EventList):
  """
  EventListForOutput
  
  Inherits from EventList
  
  Returns the structure that matches the form seen in 
  the feature files

  Notes
  ---------------------------------------
  Formerly seg_worm.feature.event
  See also seg_worm.events.events2stats

  Known Uses
  ---------------------------------------
  posture.coils
  locomotion.turns.omegas
  locomotion.turns.upsilons
  locomotion.motion.forward  
  locomotion.motion.backward
  locomotion.motion.paused
    
    %.frames    - event_stats (from event2stats)
    %.frequency -
    
    
  """
  """
    properties
        fps
        n_video_frames
        
        %INPUTS
        %------------------------------------------------------------------
        start_Is %[1 n_events]
        end_Is   %[1 n_events]
        data_sum_name %[1 n_events]
        inter_data_sum_name %[1 n_events], last value is NaN
        
        
    end
    
    
  """
 
  def __init__(self, event_list, distance_per_frame, 
               data_sum_name, inter_data_sum_name):
    """
    Initialize an instance of EventListForOutput
    
    Parameters
    ---------------------------------------
    event_list: EventList instance
      A list of all events

    distance_per_frame: A 1-d numpy array with dtype=float
      This data is used for computations, it is either:
      1) distance
        From: worm.locomotion.velocity.midbody.speed
        distance = abs(speed / fps);

        Known users:
        seg_worm.feature_helpers.posture.getCoils
        seg_worm.feature_helpers.locomotion.getWormMotionCodes

      or
      
      2) Other uses, not currently documented here?
      
    data_sum_name: string
      When retrieving the final structure, two names must be used 
      for the field that contains the sum of the input data
      (e.g. distance if it is speed data that is being summed):
       - data_sum_name for the distance the worm travelled during 
                       the event
       - inter_data_sum_name for the distance the worm travelled 
                             BETWEEN events
  
    inter_data_sum_name: string
      See above definition for data_sum_name
      
      
    Notes
    ---------------------------------------
    Former parameters: event_ss, fps, data, data_sum_name, inter_data_sum_name
    
    @JimHokanson: Some of this code is based on event2stats.
    
    """    
    EventList.__init__(self, event_list.starts_and_stops)

    self.distance_per_frame = distance_per_frame    
    self.data_sum_name = data_sum_name
    self.inter_data_sum_name = inter_data_sum_name

    # Calculate the features
    self.calculate_features()
    

  def calculate_features(self):
    """
    Calculate the outputs

    Notes
    ---------------------------------------
    This code was formerly part of __init__
    
    """
    # Exit early if there are no events to report
    if not self.__len__:    # i.e. if len(self.start_Is) == 0:
      return

    self.event_durations       = (self.end_Is - self.start_Is + 1) \
                                 / config.FPS
    self.inter_event_durations = (self.start_Is[1:] - self.end_Is[:-1] - 1) \
                                 / config.FPS

    # Calculate data_sum_values by adding up the 
    # distance_per_frame for each event
    if self.data_sum_name:
      self.data_sum_values = np.zeros(self.__len__)
      for i in range(self.__len__):
        self.data_sum_values[i] = \
          np.nansum(self.distance_per_frame
            [self.start_Is[i]:self.end_Is[i]])

    # Calculate inter_data_sum_values by adding up the 
    # distance_per_frame BETWEEN each event
    if self.inter_data_sum_name:
      self.inter_data_sum_values = np.zeros(self.__len__-1)
      for i in range(self.__len__-1):
        self.inter_data_sum_values[i] = \
          np.nansum(self.distance_per_frame
            [self.end_Is[i]+1:self.start_Is[i+1]-1])

    self.total_time = self.num_video_frames / config.FPS
    self.frequency = self.num_events_for_stats / self.total_time

    self.time_ratio = np.nansum(self.event_durations) / self.total_time
    if self.data_sum_name:
      self.data_ratio = np.nansum(self.data_sum_values) \
                        / np.nansum(self.distance_per_frame)

  @property
  def num_video_frames(self):
    return len(self.distance_per_frame)

  def get_event_mask(self):
    """
    Return a numpy boolean array corresponding to the events

    Returns
    ---------------------------------------
    A 1-d numpy boolean array
    
    Notes
    ---------------------------------------

    EventListForOutput has overridden its superclass, EventList's,
    get_event_mask with its own here, since self.distance_per_frame 
    gives it the precise number of frames so it no longer needs to 
    accept it as a parameter.
    
    """
    EventList.get_event_mask(self.num_video_frames)

  
  def get_feature_dict(self):
    """
    Returns the structure that matches the form seen in 
    the feature files

    Returns
    ---------------------------------------
    A dictionary with entries:
    
      .frames
          .start
          .end
          .time
          .interTime
          .(data_sum_name) - if specified
          .(inter_data_sum_name) - if specified
      .frequency

    event_durations %[1 n_events]
    inter_event_durations %[1 n_events], last value is NaN
    
    These two properties are missing if the input names
    are empty:
      data_sum_values
      inter_data_sum_values
    
    total_time
    frequency
    time_ratio
    data_ratio (might not exist if .data_sum_name is not specified)


    Notes
    ---------------------------------------
    This bit of code is meant to replace all of the little
    extra stuff that was previously being done after getting
    the event frames and converting them to stats
    
    For more information on the output, see events2stats in the 
    SegWorm codebase.
    
    Formerly getFeatureStruct

    """
    feature_dict = {}
 
    # Early exit if the number of events is 0
    if not self.__len__:
      feature_dict = {'frames':[], 'frequency':[], 'timeRatio':[]}
      return feature_dict

    #---------------------------------
    frames = {'start': self.start_Is,
              'end': self.end_Is,
              'time': self.event_durations,
              'interTime': self.inter_event_durations}
    
    # If data_sum_name actually is set to something other than None:
    if self.data_sum_name:
      frames[self.data_sum_name] = self.data_sum_values
    
    # If inter_data_sum_name actually is set to something other than None:
    if self.inter_data_sum_name:
      frames[self.inter_data_sum_name] = self.inter_data_sum_values
    
    # Note from @JimHokanson: This is correct for coiled events, 
    #                         not sure about others ...
    feature_dict['frames']    = frames
    feature_dict['frequency'] = self.frequency
    
    # Notes from @JimHokanson:
    # ??? - why the difference, how to know ????
    # ratio struct is present if worm can travel during event
    #
    #  - this might correspond to data_sum being defined 
    #
    # -  for motion codes - data and interdata
    # ratio.time
    # ratio.distance
    #
    # - for coils - just interdata
    # timeRatio - no ratio field
    #
    # Do we also need a check for inter_data as well?

    if not self.data_sum_name:
      feature_dict['timeRatio'] = self.time_ratio
    else:
      feature_dict['ratio'] = {'time': self.time_ratio,
                               'distance': self.data_ratio}
 
    return feature_dict
  
  # TODO: find out if anyone actually uses this method.
  #@staticmethod
  #def get_null_struct(data_sum_name, inter_data_sum_name):
#    """
#    Factory method that returns a blank event instance
#
#    Notes
#    ---------------------------------------
#    Formerly getNullStruct(fps,data_sum_name,inter_data_sum_name)
#    Formerly seg_worm.feature.event.getNullStruct
#    
#    """
  #  event_list = EventList(None)
    # TODO: get this code to work below:
    #obj = seg_worm.feature.event(event_ss,[],data_sum_name,inter_data_sum_name);
    #s = obj.getFeatureStruct();
    #return s

