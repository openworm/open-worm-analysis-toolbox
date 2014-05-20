# -*- coding: utf-8 -*-
"""
  Created on Sun May  4 22:33:36 2014
  
  EventFinder.py:  
  
  A module for finding the events in the frames of worm videos.
  
  This module contains definitions for the following:

  Classes:
    event_ss
    EventFinder
  
  Helper functions (should not be needed outside this module):
    h__unifyEvents
    h__removeGaps
    h__removeEventsByDataSum
    h__removeTooSmallOrLargeEvents
    h__getPossibleEventsByThreshold
    h__getStartStopIndices  
  
  (The code for this module came from several files in the 
  @event_finder and @event_ss folders from:
  https://github.com/JimHokanson/SegwormMatlabClasses/blob/
    master/%2Bseg_worm/%2Bfeature/ )
    
    
  How to use this module:
  
  ???
  
"""

import numpy as np

class event_ss:
  """
  Class:
  seg_worm.feature.event_ss

  "ss" stands for "simple structure"
  
  encapsulation of the raw substructure, or output from finding the event
  
  @JimHokanson: I was going to leave this class as just a Matlab 
  structure but there is at least one function that would be better 
  as a method of this class

  See Also:
  seg_worm.feature.event_finder
  seg_worm.feature.event
  
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
      self.start_Is   = start_Is

    if(end_Is == None):
      self.end_Is = np.array([])
    else:
      self.end_Is   = end_Is
  
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
    only between startIs[i] and endIs[i], where 0 <= i < n_frames
    
    """
    # @JimHokanson TODO
    # seg_worm.events.events2stats - move here
    # fromStruct - from the old struct version ...
    
    # Create empty array of all False
    mask = np.zeros(n_frames, dtype='bool')

    for iFrame in range(n_frames):
      mask[self.startIs[iFrame]:self.end_Is[iFrame]] = True
    
    return mask

  @classmethod
  def merge(cls, obj1, obj2):
    """
    Merges two instances of the event_ss class together via concatenation
    
    Acts as a factory, producing a third instance of the event_ss class
    that is the concatenation of the first two.

    Parameters
    ---------------------------------------
    cls     : The static class parameter, associated with @classmethod
    obj1    : event_ss instance
    obj2    : event_ss instance
    
    Returns
    ---------------------------------------
    event_ss : A new event_ss instance
    
    """
    all_starts = np.concatenate(obj1.start_Is, obj2.start_Is)
    all_ends   = np.concatenate(obj1.end_Is,   obj2.end_Is)
    
    # @JimHokanson TODO: Would be good to check that events don't overlap ...
    
    new_starts = np.sort(all_starts)
    order_I = np.argsort(all_starts)
        
    new_ends   = all_ends[order_I]
    
    # since we have sorted and intermingled the two sets of events, we
    # have lost information about which instance the events are a part of
    # at some point we could choose to alter this method and return this
    # variable since it stores information about which instance the 
    # element is a part of:
    #is_first   = np.concatenate(np.ones(obj1.num_events, dtype='bool'),
    #                            np.ones(obj2.num_events, dtype='bool'))
    #is_first_object = is_first[order_I]
    
    return event_ss(new_starts, new_ends)


class EventFinder:
  """
  class EventFinder
  
  """
  
  def __init__(self):
    # Options
    self.include_at_thr = False
    
    #Sample-based
    self.min_frames_thr = []
    self.max_frames_thr = []
    self.include_at_frames_thr = False    
    
    #Data-based    
    self.min_sum_thr = [] #(scalar or [1 x n_frames])
    self.max_sum_thr = [] #(scalar or [1 x n_frames])
    self.include_at_sum_thr = False
       
    # The data for thresholding based on the sum, if empty the event
    # data is used
    self.data_for_sum_thr = [] 
       
    # Sample-based
    self.min_inter_frames_thr = []
    # @JimHokanson: ??? When would anyone want to join events if 
    # the time between them is too long???
    self.max_inter_frames_thr = [] 
    self.include_at_inter_frames_thr = False
       
    # @JimHokanson: This won't work, old code didn't support it
    # Data-based
    self.min_inter_sum_thr = []
    self.max_inter_sum_thr = []
    self.include_at_inter_sum_thr = False
  
  def getEvents(self, data, min_thr, max_thr):
    """
    Old Name: findEvent.m
    
    Parameters
    ---------------------------------------
    data    : [1 x n_frames]
    min_thr : [1 x n_frames]
    max_thr : [1 x n_frames]
    
    Returns
    ---------------------------------------
    event_ss : Class seg_worm.feature.event_ss

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
    data_for_sum_thr = np.copy(self.data_for_sum_thr)
    
    if len(data_for_sum_thr) == 0:
      self.data_for_sum_thr = np.copy(data)

    min_thr = np.copy(min_thr)
    max_thr = np.copy(max_thr)


    # For each frame, determine if it matches our threshold criteria
    event_mask = h__getPossibleEventsByThreshold(data, min_thr, max_thr,
                                                 self.include_at_thr)
    
    # Get indices for runs of data matching criteria
    [start_frames, end_frames] = h__getStartStopIndices(data, event_mask)
    
    #Possible short circuit ...
    if(len(start_frames)==0):
      return event_ss()
    
    #In this function we remove gaps between events if the gaps are too small
    #(min_inter_frames_thr) or too large (max_inter_frames_thr)
    [start_frames, end_frames] = h__unifyEvents(start_frames, end_frames, 
                                             self.min_inter_frames_thr,
                                             self.max_inter_frames_thr,
                                             self.include_at_inter_frames_thr)
    
    # @JimHokanson: Is this really the same thing twice with 
    #               different values ???? I'm  99% sure this 
    #               isn't done right
    if len(self.min_inter_sum_thr) > 0 or len(self.max_inter_sum_thr) > 0:
      raise Exception("I don't think this was coded right to start; " + 
                      "... check code - @JimHokanson")
    
    # @JimHokanson NOTE: this should use data, but it doesn't
    #Perhaps there exists a correct version?
    #[start_frames,end_frames] = h__unifyEvents(start_frames,end_frames,...
    #    obj.min_inter_frames_thr,...
    #    obj.max_inter_frames_thr,...
    #    obj.include_at_inter_frames_thr);
    
    #Filter events based on length
    [start_frames,end_frames] = \
      h__removeTooSmallOrLargeEvents(start_frames, end_frames,
                                     self.min_frames_thr, self.max_frames_thr,
                                     self.include_at_frames_thr)
    
    #Filter events based on data sums during event
    [start_frames,end_frames] = \
      h__removeEventsByDataSum(start_frames, end_frames,
                               self.min_sum_thr, self.max_sum_thr,
                               self.include_at_sum_thr, 
                               data_for_sum_thr)
    
    return event_ss(start_frames, end_frames)


  
# HELPER FUNCTIONS

def h__unifyEvents(start_frames, end_frames, 
                   min_inter_frames_thr, max_inter_frames_thr, 
                   include_at_inter_frames_thr):
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
  #- min_inter_sum_thr
  #- max_inter_sum_thr
  #
  #   but the old code did not include any data in:
  #   h__removeGaps
  
  # Unify small time gaps.
  #Translation: if the gap between events is small, merge the events
  if len(min_inter_frames_thr) > 0: #~isempty(min_inter_frames_thr):
      if include_at_inter_frames_thr:
          [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,min_inter_frames_thr,@le); #  <=
      else: # the threshold is exclusive
          [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,min_inter_frames_thr,@lt); #  <
  
  #????? - when would this one ever be useful??????
  # Unify large time gaps.
  #Translation: if the gap between events is large, merge the events
  if ~isempty(max_inter_frames_thr):
      if include_at_inter_frames_thr:
          [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,max_inter_frames_thr,@ge); #  >=
      else:
          # the threshold is exclusive
          [start_frames,end_frames] = h__removeGaps(start_frames,end_frames,max_inter_frames_thr,@gt); #  >=
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
                             min_sum_thr, max_sum_thr,
                             include_at_sum_thr, data_for_sum_thr):
  """
  
  Parameters
  ---------------------------------------
  
  Returns
  ---------------------------------------
  [start_frames,end_frames]

  
  """

  if len(min_sum_thr) == 0 and len(max_sum_thr) == 0:
     return 
  
  """    
  #????? - why do we do a sum in one location and a mean in the other????
  #------------------------------------------------------------------
  # Compute the event sums.
  eventSums = nan(length(start_frames), 1);
  for i = 1:length(eventSums)
      eventSums(i) = nansum(data_for_sum_thr((start_frames(i)):(end_frames(i))));
  end
  
  # Compute the event sum thresholds.
  if length(min_sum_thr) > 1 #i.e. if not a scaler
      newMinSumThr = nan(size(eventSums));
      for i = 1:length(newMinSumThr)
          newMinSumThr(i) = nanmean(min_sum_thr((start_frames(i)):(end_frames(i))));
      end
      min_sum_thr = newMinSumThr;
  end
  
  if length(max_sum_thr) > 1
      newMaxSumThr = nan(size(eventSums));
      for i = 1:length(newMaxSumThr)
          newMaxSumThr(i) = nanmean(max_sum_thr((start_frames(i)):(end_frames(i))));
      end
      max_sum_thr = newMaxSumThr;
  end
      
  #Actual filtering of the data
  #------------------------------------------------------------------
  # Remove small events.
  removeEvents = false(size(eventSums));
  if ~isempty(min_sum_thr)
      if include_at_sum_thr
          removeEvents = eventSums <= min_sum_thr;
      else
          removeEvents = eventSums < min_sum_thr;
      end
  end
  
  # Remove large events.
  if ~isempty(max_sum_thr)
      if include_at_sum_thr
          removeEvents =  removeEvents | eventSums >= max_sum_thr;
      else
          removeEvents =  removeEvents | eventSums > max_sum_thr;
      end
  end
  
  # Remove the events.
  start_frames(removeEvents) = [];
  end_frames(removeEvents)   = [];
  """
  pass

def h__removeTooSmallOrLargeEvents(start_frames, end_frames,
                                   min_frames_thr, max_frames_thr,
                                   include_at_frames_thr):
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
  if ~(isempty(min_frames_thr) and isempty(max_frames_thr)):
      # Compute the event frames.
      n_frames_per_event = end_frames - start_frames + 1
      
      # Remove small events.
      removeEvents = false(size(n_frames_per_event))
      if ~isempty(min_frames_thr):
          if include_at_frames_thr:
              removeEvents = n_frames_per_event <= min_frames_thr
          else:
              removeEvents = n_frames_per_event < min_frames_thr
      
      # Remove large events.
      if ~isempty(max_frames_thr):
          if include_at_frames_thr:
              removeEvents = removeEvents | n_frames_per_event >= max_frames_thr
          else:
              removeEvents = removeEvents | n_frames_per_event > max_frames_thr
      
      # Remove the events.
      start_frames(removeEvents) = []
      end_frames(removeEvents)   = []
      
  """  
  pass
  
  
def h__getPossibleEventsByThreshold(data, min_thr, max_thr, include_at_thr):
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
  if ~isempty(min_thr):
      if include_at_thr
          event_mask = data >= min_thr
      else:
          event_mask = data > min_thr
  
  if ~isempty(max_thr) :
      if include_at_thr:
          event_mask = event_mask & data <= max_thr
      else:
          event_mask = event_mask & data < max_thr

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
