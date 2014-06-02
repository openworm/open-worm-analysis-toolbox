# -*- coding: utf-8 -*-
"""
  EventFinder.py
  
  A module for finding the motion state of each frame of a worm video.

  Contents
  ---------------------------------------
  This module contains definitions for the following:

  Classes:
    EventSimpleStructure
      __init__
      num_events @property
      get_event_mask
      
    EventFinder
      __init__
      get_events

    EventOutputStructure
  
  Helper functions (should not be needed outside this module), called 
    in order by EventFinder.get_events():
      get_possible_events_by_threshold
      get_start_stop_indices
      unify_events
      remove_gaps
      remove_events_by_data_sum
      remove_too_small_or_large_events

      h__getPossibleEventsByThreshold
      h__getStartStopIndices  
      h__unifyEvents
      h__removeGaps
      h__removeEventsByDataSum
      h__removeTooSmallOrLargeEvents

  Usage
  ---------------------------------------
  LocomotionFeatures.get_motion_codes() calculates the motion codes for the
  worm, and to do so, for each of the possible motion states (forward, 
  backward, paused) it creates an instance of the EventFinder class,
  sets up appropriate parameters, and then calls EventFinder.get_events()
  to obtain an instance of EventSimpleStructure, the "result" class.
  
  Then to format the result appropriately, the EventOutputStructure class is 
  instantiated with our "result" and then get_feature_struct is called.
  
  So the flow from within LocomotionFeatures.get_motion_codes() is:
    # (approximately):
    ef = EventFinder()
    ess = ef.get_events()
    me = EventOutputStructure(ess)
    return me.get_feature_struct
  
  Notes
  ---------------------------------------
  See https://github.com/openworm/movement_validation/blob/master/
  documentation/Yemini%20Supplemental%20Data/Locomotion.md#2-motion-states
  for a plain-English description of a motion state.

  The code for this module came from several files in the 
  @event_finder and @EventSimpleStructure folders from:
  https://github.com/JimHokanson/SegwormMatlabClasses/blob/
    master/%2Bseg_worm/%2Bfeature/

"""

import numpy as np
from itertools import groupby


class EventSimpleStructure:
  """
  Class EventSimpleStructure
  
  Previous name:
  seg_worm.feature.event_ss ("ss" stands for "simple structure")
  
  encapsulation of the raw substructure, or output from finding the event
  
  @JimHokanson: I was going to leave this class as just a Matlab 
  structure but there is at least one function that would be better 
  as a method of this class
  
  """    
  def __init__(self, start_Is=None, end_Is=None):
    # @MichaelCurrie: in @JimHokanson's original code there were lines
    # to change start_Is and end_Is from column to row vectors, if
    # necessary.  Because here we use numpy arrays, they are not 
    # treated as matrices so we don't need to care.
    #if(np.shape(start_Is)[0] > 1):
    #    start_Is = np.transpose(start_Is)
    
    #if(np.shape(end_Is)[0] > 1):
    #    end_Is = np.transpose(end_Is)
    
    if(start_Is == None):
      self.start_Is = np.array([])
    else:
      self.start_Is = start_Is

    if(end_Is == None):
      self.end_Is = np.array([])
    else:
      self.end_Is = end_Is
  
  @property
  def num_events(self):
    return len(self.start_Is)

  def get_event_mask(self, n_frames):
    """
    Parameters
    ---------------------------------------
    n_frames: int
    
    Returns
    ---------------------------------------
    boolean numpy array of size n_frames with True entries
    only between start_Is[i] and end_Is[i], where 0 <= i < n_frames
    
    """
    # @JimHokanson TODO
    # seg_worm.events.events2stats - move here
    # fromStruct - from the old struct version ...
    
    # Create empty array of all False
    mask = np.zeros(n_frames, dtype='bool')

    for i_frame in range(n_frames):
      mask[self.start_Is[i_frame]:self.end_Is[i_frame]] = True
    
    return mask

  @classmethod
  def merge(cls, obj1, obj2):
    """
    Merges two instances of the EventSimpleStructure class together via concatenation
    
    Acts as a factory, producing a third instance of the EventSimpleStructure class
    that is the concatenation of the first two, with the start indices 
    blended and properly in order.

    Parameters
    ---------------------------------------
    cls     : The static class parameter, associated with @classmethod
    obj1    : EventSimpleStructure instance
    obj2    : EventSimpleStructure instance
    
    Returns
    ---------------------------------------
    EventSimpleStructure : A new EventSimpleStructure instance
    
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
    
    return EventSimpleStructure(new_starts, new_ends)


class EventFinder:
  """
  class EventFinder

  To use this, create an instance, then specify the options.  Default
  options are initialized in __init__.
  
  Then call get_events() to obtain an EventSimpleStructure instance 
  containing the desired events from a given block of data.
  
  """
  
  def __init__(self):
    # Options
    self.include_at_threshold = False
    
    # Temporal thresholds
    self.min_frames_threshold = None #(scalar or [1 x n_frames])
    self.max_frames_threshold = None #(scalar or [1 x n_frames])
    self.include_at_frames_threshold = False    
    self.min_inter_frames_threshold = None #(scalar or [1 x n_frames])
    self.include_at_inter_frames_threshold = False

    
    # Space (distance) and space&time (speed) thresholds
    self.min_distance_threshold = None #(scalar or [1 x n_frames])
    self.max_distance_threshold = None #(scalar or [1 x n_frames])
    self.include_at_distance_threshold = False
    self.min_speed_threshold = None #(scalar or [1 x n_frames])
    self.max_speed_threshold = None #(scalar or [1 x n_frames])
    self.include_at_speed_threshold = False


    # The data for thresholding based on the sum, if empty the event
    # data is used
    # I don't understand the use of the term "sum" here, aren't we just
    # thresholding based on "distance"?  I'm going to omit this
    # functionality - @MichaelCurrie
    #self.data_for_sum_threshold = [] 
       
    """
    #DEBUG: get rid of soon
    # @JimHokanson: This won't work, old code didn't support it
    # Data-based
    self.min_inter_sum_threshold = None
    self.max_inter_sum_threshold = None
    self.include_at_inter_sum_threshold = False
    """
  
  def get_events(self, event_data):
    """
    Obtain the events implied by event_data, given how this instance
    of EventFinder has been configured.
    
    Parameters
    ---------------------------------------
    event_data   : 1-d numpy array of length n
      The events to be extracted
    
    Returns
    ---------------------------------------
    An instance of class EventSimpleStructure

    Notes:
    ---------------------------------------
    If the first/last event are solely preceded/followed by NaN
    frames, these frames are swallowed into the respective event.

    Formerly getEvents.m.  Originally it was findEvent.m.  

    """
    
    # Fix the data.
    # @MichaelCurrie: I don't understand what is going on here.  Why do
    # we need to copy the arrays passed as parameters?  Here I am 
    # directly copying @JimHokanson's MATLAB code without understanding.
    #event_data = np.copy(event_data)    
    #data_for_sum_threshold = np.copy(self.data_for_sum_threshold)
    
    #if len(data_for_sum_threshold) == 0:
    #  self.data_for_sum_threshold = np.copy(event_data)

    # For each frame, determine if it matches our speed threshold criteria
    speed_mask = self.get_speed_threshold_mask(event_data)
    
    # Get indices for runs of data matching criteria
    starts_and_stops = self.get_start_stop_indices(event_data, speed_mask)

    # Possible short circuit: if we have absolutely no qualifying events 
    # in event_data, just exit early.
    if(not starts_and_stops):
      return EventSimpleStructure()

    """
    # In this function we remove gaps between events if the gaps are too small
    #(min_inter_frames_threshold) or too large (max_inter_frames_threshold)
    [start_frames, end_frames] = self.unify_events(start_frames, end_frames)
    
    # @JimHokanson: Is this really the same thing twice with 
    #               different values ???? I'm  99% sure this 
    #               isn't done right
    if(len(self.min_inter_sum_threshold) > 0 or 
       len(self.max_inter_sum_threshold) > 0):
      raise Exception("I don't think this was coded right to start; " + 
                      "... check code - @JimHokanson")
    
    # @JimHokanson NOTE: this should use data, but it doesn't
    #Perhaps there exists a correct version?
    #[start_frames,end_frames] = self.h__unifyEvents(start_frames,end_frames,...
    #    obj.min_inter_frames_threshold,...
    #    obj.max_inter_frames_threshold,...
    #    obj.include_at_inter_frames_threshold);
    
    # Filter events based on length
    [start_frames,end_frames] = \
      self.remove_too_small_or_large_events(start_frames, end_frames)
      
    # Filter events based on data sums during event
    [start_frames,end_frames] = \
      self.remove_events_by_data_sum(start_frames, end_frames,
                               data_for_sum_threshold)
    
    return EventSimpleStructure(start_frames, end_frames)
    """
    return EventSimpleStructure()

  
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

      0 1 2 3 4 5   <- true indices
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
    starts_and_stops: 2-d int numpy array
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
    starts_and_stops = [(i[0][1], i[-1][1]) for i in x_grouped]

    # Early exit if we have no starts and stops at all
    if not starts_and_stops:
      return starts_and_stops
    
    # If a run of NaNs precedes the first start index, all the way back to 
    # the first element, then revise our first (start, stop) entry to include
    # all those NaNs.
    if np.all(np.isnan(event_data[:starts_and_stops[0][0]])):
      starts_and_stops[0] = (0, starts_and_stops[0][1])
    
    # Same but with NaNs succeeding the final end index.    
    if np.all(np.isnan(event_data[starts_and_stops[-1][1]:])):
      starts_and_stops[-1] = (starts_and_stops[-1][0], event_data.size-1)
      
    return starts_and_stops

  
  def h__unifyEvents(self, starts_and_stops):
    """
    Combine events where the # of frames between them is less than
    self.min_inter_frames_threshold (or if 
    self.include_at_inter_frames_threshold == True, less than OR EQUAL to
    self.min_inter_frames_threshold).
    
    Parameters
    ---------------------------------------
    starts_and_stops: an array of (start, stop) duples
    
    Returns
    ---------------------------------------
    new_starts_and_stops: an array of (start, stop) duples satisfying 
    the above.
    
    """
    
    
    #
    #
    #   These functions are run on the time between frames
    #
    """
    #NOTE: This function could also exist for:
    #- min_inter_sum_threshold
    #- max_inter_sum_threshold
    #
    #   but the old code did not include any event_data in:
    #   h__removeGaps
    
    # Unify small time gaps.
    #Translation: if the gap between events is small, merge the events
    if len(min_inter_frames_threshold) > 0: #~isempty(min_inter_frames_threshold):
        if include_at_inter_frames_threshold:
            [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,min_inter_frames_threshold,@le); #  <=
        else: # the threshold is exclusive
            [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,min_inter_frames_threshold,@lt); #  <
    
    #????? - when would this one ever be useful??????
    # Unify large time gaps.
    #Translation: if the gap between events is large, merge the events
    if ~isempty(max_inter_frames_threshold):
        if include_at_inter_frames_threshold:
            [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,max_inter_frames_threshold,@ge); #  >=
        else:
            # the threshold is exclusive
            [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,max_inter_frames_threshold,@gt); #  >=
    """
    pass
  
  
  def h__removeGaps(self, start_frames, end_frames, right_comparison_value, fh):
    """
    
    Parameters
    ---------------------------------------
      
    
    Returns
    ---------------------------------------
    [start_frames,end_frames]
  
    
    """
    
    # Find small gaps.
    i = 1
  
    while i < len(start_frames):
      # Swallow the gaps.
      # NOTE: fh is either: <, <=, >, >=
      # NOTE: This implicitly uses a sample difference (time based) approach
      """
      while i < length(start_frames) && fh(start_frames(i + 1) - end_frames(i) - 1,right_comparison_value):
        #This little bit removes the gap between two events
        end_frames(i)       = end_frames(i + 1) #Set end of this event to
        
        #the next event
        start_frames(i + 1) = [] #delete the next event, it is redundant
        end_frames(i + 1)   = [] #delete the next event
      """
      # Advance.
      i += 1
  
  
  
  def h__removeTooSmallOrLargeEvents(self, start_frames, end_frames,
                                     min_frames_threshold, max_frames_threshold,
                                     include_at_frames_threshold):
    """
    This function filters events based on time (really sample count)
    
    Parameters
    ---------------------------------------
    
    Returns
    ---------------------------------------
    [start_frames,end_frames]
  
    
    """
    pass
    
    """
    # Check the event frames.
    if ~(isempty(min_frames_threshold) and isempty(max_frames_threshold)):
        # Compute the event frames.
        n_frames_per_event = end_frames - start_frames + 1
        
        # Remove small events.
        removeEvents = false(size(n_frames_per_event))
        if ~isempty(min_frames_threshold):
            if include_at_frames_threshold:
                removeEvents = n_frames_per_event <= min_frames_threshold
            else:
                removeEvents = n_frames_per_event < min_frames_threshold
        
        # Remove large events.
        if ~isempty(max_frames_threshold):
            if include_at_frames_threshold:
                removeEvents = removeEvents | n_frames_per_event >= max_frames_threshold
            else:
                removeEvents = removeEvents | n_frames_per_event > max_frames_threshold
        
        # Remove the events.
        start_frames(removeEvents) = []
        end_frames(removeEvents)   = []
        
    """  
    pass
    
    
  def h__removeEventsByDataSum(self, start_frames, end_frames,
                               min_distance_threshold, max_distance_threshold,
                               include_at_distance_threshold, data_for_sum_threshold):
    """
    
    Parameters
    ---------------------------------------
    
    Returns
    ---------------------------------------
    [start_frames,end_frames]
  
    
    """
  
    if len(min_distance_threshold) == 0 and len(max_distance_threshold) == 0:
       return 
    
    """    
    #????? - why do we do a sum in one location and a mean in the other????
    #------------------------------------------------------------------
    # Compute the event sums.
    eventSums = nan(length(start_frames), 1);
    for i = 1:length(eventSums)
        eventSums(i) = nansum(data_for_sum_threshold((start_frames(i)):(end_frames(i))));
    end
    
    # Compute the event sum thresholds.
    if length(min_distance_threshold) > 1 #i.e. if not a scaler
        newMinSumThr = nan(size(eventSums));
        for i = 1:length(newMinSumThr)
            newMinSumThr(i) = nanmean(min_distance_threshold((start_frames(i)):(end_frames(i))));
        end
        min_distance_threshold = newMinSumThr;
    end
    
    if length(max_distance_threshold) > 1
        newMaxSumThr = nan(size(eventSums));
        for i = 1:length(newMaxSumThr)
            newMaxSumThr(i) = nanmean(max_distance_threshold((start_frames(i)):(end_frames(i))));
        end
        max_distance_threshold = newMaxSumThr;
    end
        
    #Actual filtering of the data
    #------------------------------------------------------------------
    # Remove small events.
    removeEvents = false(size(eventSums));
    if ~isempty(min_distance_threshold)
        if include_at_distance_threshold
            removeEvents = eventSums <= min_distance_threshold;
        else
            removeEvents = eventSums < min_distance_threshold;
        end
    end
    
    # Remove large events.
    if ~isempty(max_distance_threshold)
        if include_at_distance_threshold
            removeEvents =  removeEvents | eventSums >= max_distance_threshold;
        else
            removeEvents =  removeEvents | eventSums > max_distance_threshold;
        end
    end
    
    # Remove the events.
    start_frames(removeEvents) = [];
    end_frames(removeEvents)   = [];
    """
    pass
  
















class EventOutputStructure:
  """
  EventOutputStructure
  
  formerly seg_worm.feature.event
  
    #   See Also:
    %   seg_worm.events.events2stats
    %
    %
    %   General Notes:
    %   -------------------------------------------------------------------
    %   - This is still very much a work in progress. The events are quite
    %   complicated and messy
    %   - An event can optionally contain
    
    %Known Uses:
    %----------------------------------------------------------------------
    %posture.coils - seg_worm.feature_helpers.posture.getCoils
    %
    %locomotion.turns.omegas
    %locomotion.turns.upsilons
    %
    %
    %Uses findEvent ...
    %----------------------------------------------------------------------
    %  in seg_worm.feature_helpers.locomotion.getWormMotionCodes
    %locomotion.motion.forward  
    %locomotion.motion.backward
    %locomotion.motion.paused
    
    %.frames    - event_stats (from event2stats)
    %.frequency -
    
    %Final outputs
    %{
    
    .frames
        .start
        .end
        .time
        .interTime
        .(data_sum_name) - if specified
        .(inter_data_sum_name) - if specified
    .frequency
    %
    %}
  """
  def __init__(self, frames_temp, distance_per_frame):
    pass

  def get_feature_struct(self):
    pass
  
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
        
        %Outputs - see events2stats
        %------------------------------------------------------------------
        event_durations %[1 n_events]
        inter_event_durations %[1 n_events], last value is NaN
        
        %These two properties are missing if the input names
        %are empty
        data_sum_values
        inter_data_sum_values
        
        total_time
        frequency
        time_ratio
        data_ratio %[1 1] might not exist if .data_sum_name is not specified
        
    end
    
    properties (Dependent)
        n_events
        n_events_for_stats
    end
    
    methods
        function value = get.n_events(obj)
            value = length(obj.start_Is);
        end
        function value = get.n_events_for_stats(obj)
            % Compute the number of events, excluding the partially recorded ones.
            value = obj.n_events;
            if value > 1
                if obj.start_Is(1) == 1
                    value = value - 1;
                end
                if obj.end_Is(end) == obj.n_video_frames
                    value = value - 1;
                end
            end
        end
    end
    
    methods (Static)
       %This is temporary, I'll probably create an event finder class ...
       %
       %    Used by:
       %    locomotion.motion.forward  
       %    locomotion.motion.backward
       %    locomotion.motion.paused
       %
       %    in seg_worm.feature_helpers.locomotion.getWormMotionCodes
       frames = findEvent(data, minThr, maxThr, varargin); 
       function s = getNullStruct(fps,data_sum_name,inter_data_sum_name)
           %
           %
           %    s = seg_worm.feature.event.getNullStruct(fps,data_sum_name,inter_data_sum_name)
           %
           
          event_ss = seg_worm.feature.event_ss([],[]); 
          obj = seg_worm.feature.event(event_ss,fps,[],data_sum_name,inter_data_sum_name);
          s = obj.getFeatureStruct();
       end
    end
    
    methods
        function obj = event(event_ss,fps,data,data_sum_name,inter_data_sum_name)
            %
            %   obj = seg_worm.feature.event(event_ss,fps,data,data_sum_name,inter_data_sum_name)
            %
            %
            %   Inputs
            %   ===========================================================
            %   event_ss : seg_worm.feature.event_ss
            %          .start_Is - frame numbers in which events start
            %          .end_Is   - frame numbers in which events end
            %
            %   fps      : (scalar) frames per second
            %   data     : This data is used for computations, it is
            %               either:
            %             1) distance
            %
            %               From: worm.locomotion.velocity.midbody.speed
            %               distance = abs(speed / fps);
            %
            %               Known users:
            %               seg_worm.feature_helpers.posture.getCoils
            %               seg_worm.feature_helpers.locomotion.getWormMotionCodes
            %
            %             2) ????
            %
            %   data_sum_name : (char) When retrieving the final structure
            %         this is the name given to the field that contains the
            %         sum of the input data during the event
            %   inter_data_sum_name : (char) "          " sum of the input
            %         data between events
            %
            %Some of this code is based on event2stats
            
            obj.fps            = fps;
            obj.n_video_frames = length(data);
            
            if isobject(event_ss)
                obj.start_Is  = event_ss.start_Is;
                obj.end_Is    = event_ss.end_Is;
            else
                obj.start_Is  = [event_ss.start];
                obj.end_Is    = [event_ss.end];                
            end
            obj.data_sum_name       = data_sum_name;
            obj.inter_data_sum_name = inter_data_sum_name;
            
            %Now populate the outputs ...
            %--------------------------------------------------------------
            if obj.n_events == 0
                return
            end

            %---------------------------
            obj.event_durations       = (obj.end_Is - obj.start_Is + 1)./obj.fps;
            obj.inter_event_durations = [obj.start_Is(2:end) - obj.end_Is(1:end-1) - 1 NaN]./fps;
            
            %---------------------------
            if ~isempty(obj.data_sum_name)
                temp = zeros(1,obj.n_events);
                for iEvent = 1:obj.n_events
                    temp(iEvent) = nansum(data(obj.start_Is(iEvent):obj.end_Is(iEvent)));
                end
                obj.data_sum_values = temp;
            end
            
            %---------------------------
            if ~isempty(inter_data_sum_name)
                temp = NaN(1,obj.n_events);
                for iEvent = 1:(obj.n_events-1)
                    start_frame  = obj.end_Is(iEvent)+1;
                    end_frame    = obj.start_Is(iEvent+1)-1;
                    temp(iEvent) = nansum(data(start_frame:end_frame));
                end
                obj.inter_data_sum_values = temp;
            end
            
            %----------------------------
            obj.total_time = obj.n_video_frames/obj.fps;
            obj.frequency  = obj.n_events_for_stats/obj.total_time;
            
            obj.time_ratio = nansum(obj.event_durations) / obj.total_time;
            if ~isempty(obj.data_sum_name)
               obj.data_ratio = nansum(obj.data_sum_values)/nansum(data); 
            end
        end
        function s = getFeatureStruct(obj)
            %   
            %   This function returns the structure that matches the form
            %   seen in the feature files
            %
            %   JAH TODO: Describe format of structure ...
            %
            %   
            
            %This bit of code is meant to replace all of the little
            %extra stuff that was previously being done after getting
            %the event frames and converting them to stats
            
            s = struct;
            
            if obj.n_events == 0
               if isempty(obj.data_sum_name)
                   s = struct('frames',[],'frequency',[],'timeRatio',[]);
               else
                   %ratio = struct('time',[],'distance',[]);
                   s = struct('frames',[],'frequency',[],'ratio',[]);
               end
               return
            end
            
            %--------------------------------------------------------------
            f = struct(...
                'start',        num2cell(obj.start_Is),...
                'end',          num2cell(obj.end_Is), ...
                'time',         num2cell(obj.event_durations),...
                'interTime',    num2cell(obj.inter_event_durations));
            f = f'; %f is apparently a column vector
            
            
            if ~isempty(obj.data_sum_name)
               temp = num2cell(obj.data_sum_values);
               [f.(obj.data_sum_name)] = deal(temp{:});
            end
            
            if ~isempty(obj.inter_data_sum_name)
               temp = num2cell(obj.inter_data_sum_values);
               [f.(obj.inter_data_sum_name)] = deal(temp{:});
            end
            
            %This is correct for coiled events, not sure about others ...
            %--------------------------------------------------------------
            s.frames    = f;
            s.frequency = obj.frequency;
            
            
            %??? - why the difference, how to know ????
            %------------------------------------------------
            %ratio struct is present if worm can travel during event
            %
            %  - this might correspond to data_sum being defined 
            %
            %- for motion codes - data and interdata
            %ratio.time
            %ratio.distance
            %
            %- for coils - just interdata
            %timeRatio - no ratio field
            
            %?? Do we also need a check for inter_data as well?
            if isempty(obj.data_sum_name)
                s.timeRatio = obj.time_ratio;
            else
                s.ratio.time     = obj.time_ratio;
                s.ratio.distance = obj.data_ratio;
            end
        end
        function new_obj = mergeEvents(obj1,obj2)
            
        end
    end    
  """











  
# HELPER FUNCTIONS


