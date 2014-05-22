# -*- coding: utf-8 -*-
"""
  Created on Sun May  4 22:33:36 2014
  
  EventFinder.py:  
  
  A module for finding the events in the frames of worm videos.
  
  This module contains definitions for the following:

  Classes:
    EventSimpleStructure
      __init__
      num_events @property
      get_event_mask
      
    EventFinder
      __init__
      get_events

    MotionEvent
  
  Helper functions (should not be needed outside this module):
    h__unifyEvents
    h__removeGaps
    h__removeEventsByDataSum
    h__removeTooSmallOrLargeEvents
    h__getPossibleEventsByThreshold
    h__getStartStopIndices  
  
  (The code for this module came from several files in the 
  @event_finder and @EventSimpleStructure folders from:
  https://github.com/JimHokanson/SegwormMatlabClasses/blob/
    master/%2Bseg_worm/%2Bfeature/ )
    
    
  How to use this module:
  
  TODO: ???
  
  TODO: describe what the above classes and functions are for
  
"""

import numpy as np

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
    
    #Sample-based
    self.min_frames_threshold = []
    self.max_frames_threshold = []
    self.include_at_frames_threshold = False    
    
    #Data-based    
    self.min_sum_threshold = [] #(scalar or [1 x n_frames])
    self.max_sum_threshold = [] #(scalar or [1 x n_frames])
    self.include_at_sum_threshold = False
       
    # The data for thresholding based on the sum, if empty the event
    # data is used
    self.data_for_sum_threshold = [] 
       
    # Sample-based
    self.min_inter_frames_threshold = []
    # @JimHokanson: ??? When would anyone want to join events if 
    # the time between them is too long???
    self.max_inter_frames_threshold = [] 
    self.include_at_inter_frames_threshold = False
       
    # @JimHokanson: This won't work, old code didn't support it
    # Data-based
    self.min_inter_sum_threshold = []
    self.max_inter_sum_threshold = []
    self.include_at_inter_sum_threshold = False
  
  def get_events(self, data, min_threshold, max_threshold):
    """
    Old Name: findEvent.m
    
    Parameters
    ---------------------------------------
    data    : [1 x n_frames]
    min_threshold : [1 x n_frames]
    max_threshold : [1 x n_frames]
    
    Returns
    ---------------------------------------
    An instance of class EventSimpleStructure

    Implementation Notes:
    ---------------------------------------
    If the first/last event are solely preceded/followed by NaN
    frames, these frames are swallowed into the respective event.
    
    """
    
    # Fix the data.
    # @MichaelCurrie: I don't understand what is going on here.  Why do
    # we need to copy the arrays passed as parameters?  Here I am 
    # directly copying @JimHokanson's MATLAB code without understanding.
    data = np.copy(data)    
    data_for_sum_threshold = np.copy(self.data_for_sum_threshold)
    
    if len(data_for_sum_threshold) == 0:
      self.data_for_sum_threshold = np.copy(data)

    min_threshold = np.copy(min_threshold)
    max_threshold = np.copy(max_threshold)


    # For each frame, determine if it matches our threshold criteria
    event_mask = h__getPossibleEventsByThreshold(data, 
                                                 min_threshold, 
                                                 max_threshold,
                                                 self.include_at_threshold)
    """
    # Get indices for runs of data matching criteria
    [start_frames, end_frames] = h__getStartStopIndices(data, event_mask)
    
    #Possible short circuit ...
    if(len(start_frames)==0):
      return EventSimpleStructure()
    
    #In this function we remove gaps between events if the gaps are too small
    #(min_inter_frames_threshold) or too large (max_inter_frames_threshold)
    [start_frames, end_frames] = h__unifyEvents(start_frames, end_frames, 
                                       self.min_inter_frames_threshold,
                                       self.max_inter_frames_threshold,
                                       self.include_at_inter_frames_threshold)
    
    # @JimHokanson: Is this really the same thing twice with 
    #               different values ???? I'm  99% sure this 
    #               isn't done right
    if(len(self.min_inter_sum_threshold) > 0 or 
       len(self.max_inter_sum_threshold) > 0):
      raise Exception("I don't think this was coded right to start; " + 
                      "... check code - @JimHokanson")
    
    # @JimHokanson NOTE: this should use data, but it doesn't
    #Perhaps there exists a correct version?
    #[start_frames,end_frames] = h__unifyEvents(start_frames,end_frames,...
    #    obj.min_inter_frames_threshold,...
    #    obj.max_inter_frames_threshold,...
    #    obj.include_at_inter_frames_threshold);
    
    #Filter events based on length
    [start_frames,end_frames] = \
      h__removeTooSmallOrLargeEvents(start_frames, end_frames,
                                     self.min_frames_threshold, 
                                     self.max_frames_threshold,
                                     self.include_at_frames_threshold)
    
    #Filter events based on data sums during event
    [start_frames,end_frames] = \
      h__removeEventsByDataSum(start_frames, end_frames,
                               self.min_sum_threshold, 
                               self.max_sum_threshold,
                               self.include_at_sum_threshold, 
                               data_for_sum_threshold)
    
    return EventSimpleStructure(start_frames, end_frames)
    """
    return EventSimpleStructure()



class MotionEvent:
  """
  MotionEvent
  
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

def h__unifyEvents(start_frames, end_frames, 
                   min_inter_frames_threshold, max_inter_frames_threshold, 
                   include_at_inter_frames_threshold):
  """
  
  Parameters
  ---------------------------------------
  start_frames:
  end_frames:
  
  Returns
  ---------------------------------------
  [start_frames,end_frames]

  
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
  #   but the old code did not include any data in:
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


def h__removeGaps(start_frames, end_frames, right_comparison_value, fh):
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


def h__removeEventsByDataSum(start_frames, end_frames,
                             min_sum_threshold, max_sum_threshold,
                             include_at_sum_threshold, data_for_sum_threshold):
  """
  
  Parameters
  ---------------------------------------
  
  Returns
  ---------------------------------------
  [start_frames,end_frames]

  
  """

  if len(min_sum_threshold) == 0 and len(max_sum_threshold) == 0:
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
  if length(min_sum_threshold) > 1 #i.e. if not a scaler
      newMinSumThr = nan(size(eventSums));
      for i = 1:length(newMinSumThr)
          newMinSumThr(i) = nanmean(min_sum_threshold((start_frames(i)):(end_frames(i))));
      end
      min_sum_threshold = newMinSumThr;
  end
  
  if length(max_sum_threshold) > 1
      newMaxSumThr = nan(size(eventSums));
      for i = 1:length(newMaxSumThr)
          newMaxSumThr(i) = nanmean(max_sum_threshold((start_frames(i)):(end_frames(i))));
      end
      max_sum_threshold = newMaxSumThr;
  end
      
  #Actual filtering of the data
  #------------------------------------------------------------------
  # Remove small events.
  removeEvents = false(size(eventSums));
  if ~isempty(min_sum_threshold)
      if include_at_sum_threshold
          removeEvents = eventSums <= min_sum_threshold;
      else
          removeEvents = eventSums < min_sum_threshold;
      end
  end
  
  # Remove large events.
  if ~isempty(max_sum_threshold)
      if include_at_sum_threshold
          removeEvents =  removeEvents | eventSums >= max_sum_threshold;
      else
          removeEvents =  removeEvents | eventSums > max_sum_threshold;
      end
  end
  
  # Remove the events.
  start_frames(removeEvents) = [];
  end_frames(removeEvents)   = [];
  """
  pass

def h__removeTooSmallOrLargeEvents(start_frames, end_frames,
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
  
  
def h__getPossibleEventsByThreshold(data, min_threshold, max_threshold, include_at_threshold):
  """
  
  Parameters
  ---------------------------------------
  
  Returns
  ---------------------------------------
  event_mask

  
  """
  pass
  """
  event_mask = true(length(data),1)
  if ~isempty(min_threshold):
      if include_at_threshold
          event_mask = data >= min_threshold
      else:
          event_mask = data > min_threshold
  
  if ~isempty(max_threshold) :
      if include_at_threshold:
          event_mask = event_mask & data <= max_threshold
      else:
          event_mask = event_mask & data < max_threshold

  """
  pass


def h__getStartStopIndices(data, event_mask):
  """
  
  Parameters
  ---------------------------------------
  
  Returns
  ---------------------------------------
  [starts,stops]

  
  """
  pass
  """
  # We concatenate falses to ensure event starts and stops at the edges
  # are caught i.e. allow edge detection if 
  dEvent = diff([false; event_mask; false]);
  
  # 0 1 2 3  4 5 6 <- true indices
  # x n n y  y n n <- event
  # 0 0 1 0 -1 0  <- diffs, 1 indicates start, -1 indicates end
  # 1 2 3 4  5 6  <- indices of diffs
  #     s    e    <- start and end
  # start matches its index
  # end is off by 1
  # 
  
  starts = find(dEvent == 1)
  stops  = find(dEvent == -1) - 1
  
  if isempty(starts):
      return
  
  # Include NaNs at the start and end.
  if all(isnan(data(1:(starts(1)-1)))):
      starts(1) = 1
  
  if all(isnan(data((stops(end)+1):end))):
      stops(end) = length(data)
  """
  pass



