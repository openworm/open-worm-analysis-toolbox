# -*- coding: utf-8 -*-
"""
Created on Sun May  4 22:33:36 2014

EventFinder

Finds the events in the frames of worm videos

(From event_finder from the matlab code)
https://github.com/JimHokanson/SegwormMatlabClasses/blob/
  master/%2Bseg_worm/%2Bfeature/%40event_finder/event_finder.m

@author: mcurrie
"""

class EventFinder:
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
    #
    #   event_ss = seg_worm.feature.event_finder.getEvents(obj,data,min_thr,max_thr)
    #
    #   Old Name: findEvent.m
    #
    #   Inputs
    #   =======================================================================
    #   data    : [1 x n_frames]
    #   min_thr : [1 x n_frames]
    #   max_thr : [1 x n_frames]
    #
    #   Outputs
    #   =======================================================================
    #   event_ss : Class seg_worm.feature.event_ss
    #
    #   Implementation Notes:
    #   =======================================================================
    #   - If the first/last event are solely preceded/followed by NaN
    #   frames, these frames are swallowed into the respective event.
    """
    # Fix the data.
    #--------------------------------------------------------------------------
    data    = data(:);
    data_for_sum_thr = obj.data_for_sum_thr(:);
    if isempty(data_for_sum_thr)
       data_for_sum_thr = data; 
    end
    min_thr = min_thr(:);
    max_thr = max_thr(:);
    
    # For each frame, determine if it matches our threshold criteria
    #--------------------------------------------------------------------------
    event_mask = h__getPossibleEventsByThreshold(data,min_thr,max_thr,obj.include_at_thr);
    
    # Get indices for runs of data matching criteria
    #--------------------------------------------------------------------------
    [startFrames,endFrames] = h__getStartStopIndices(data,event_mask);
    
    #Possible short circuit ...
    #--------------------------------------------------------------------------
    if isempty(startFrames)
        event_ss = seg_worm.feature.event_ss([],[]);
        return
    end
    
    
    #In this function we remove gaps between events if the gaps are too small
    #(min_inter_frames_thr) or too large (max_inter_frames_thr)
    #--------------------------------------------------------------------------
    [startFrames,endFrames] = h__unifyEvents(startFrames,endFrames,...
        obj.min_inter_frames_thr,...
        obj.max_inter_frames_thr,...
        obj.include_at_inter_frames_thr);
    
    
    #--------------------------------------------------------------------------
    #Is this really the same thing twice with different values ????
    #I'm  99% sure this isn't done right
    if ~isempty(obj.min_inter_sum_thr) || ~isempty(obj.max_inter_sum_thr)
        error('I don''t think this was coded right to start ..., check code')
    end
    
    #{
    #NOTE: this should use data, but it doesn't
    #Perhaps there exists a correct version?
    [startFrames,endFrames] = h__unifyEvents(startFrames,endFrames,...
        obj.min_inter_frames_thr,...
        obj.max_inter_frames_thr,...
        obj.include_at_inter_frames_thr);
    #}
    
    #Filter events based on length
    #--------------------------------------------------------------------------
    [startFrames,endFrames] = h__removeTooSmallOrLargeEvents(startFrames,endFrames,...
        obj.min_frames_thr,...
        obj.max_frames_thr,...
        obj.include_at_frames_thr);
    
    
    #Filter events based on data sums during event
    #--------------------------------------------------------------------------
    [startFrames,endFrames] = h__removeEventsByDataSum(startFrames,endFrames,...
        obj.min_sum_thr,...
        obj.max_sum_thr,...
        obj.include_at_sum_thr,...
        data_for_sum_thr);
    
    event_ss = seg_worm.feature.event_ss(startFrames,endFrames);
    
    return event_ss
    """
    pass



  
"""  
HELPER FUNCTIONS
ORIGINAL SOURCE: 

function [startFrames,endFrames] = h__unifyEvents(startFrames,endFrames,...
    min_inter_frames_thr,max_inter_frames_thr,include_at_inter_frames_thr)
#
#
#   These functions are run on the time between frames
#

#NOTE: This function could also exist for:
#- min_inter_sum_thr
#- max_inter_sum_thr
#
#   but the old code did not include any data in:
#   h__removeGaps

# Unify small time gaps.
#Translation: if the gap between events is small, merge the events
if ~isempty(min_inter_frames_thr)
    if include_at_inter_frames_thr
        [startFrames,endFrames] = h__removeGaps(startFrames,endFrames,min_inter_frames_thr,@le); #  <=
    else # the threshold is exclusive
        [startFrames,endFrames] = h__removeGaps(startFrames,endFrames,min_inter_frames_thr,@lt); #  <
    end
end

#????? - when would this one ever be useful??????
# Unify large time gaps.
#Translation: if the gap between events is large, merge the events
if ~isempty(max_inter_frames_thr)
    if include_at_inter_frames_thr
        [startFrames,endFrames] = h__removeGaps(startFrames,endFrames,max_inter_frames_thr,@ge); #  >=
    else
        # the threshold is exclusive
        [startFrames,endFrames] = h__removeGaps(startFrames,endFrames,max_inter_frames_thr,@gt); #  >=
    end
end

end


function [startFrames,endFrames] = h__removeGaps(startFrames,endFrames,right_comparison_value,fh)

# Find small gaps.
i = 1;
while i < length(startFrames)
    
    # Swallow the gaps.
    #NOTE: fh is either: <, <=, >, >=
    #NOTE: This implicitly uses a sample difference (time based) approach
    while i < length(startFrames) && fh(startFrames(i + 1) - endFrames(i) - 1,right_comparison_value)
        
        #This little bit removes the gap between two events
        endFrames(i)       = endFrames(i + 1); #Set end of this event to
        #the next event
        startFrames(i + 1) = []; #delete the next event, it is redundant
        endFrames(i + 1)   = []; #delete the next event
    end
    
    # Advance.
    i = i + 1;
end
end

function [startFrames,endFrames] = h__removeEventsByDataSum(startFrames,endFrames,...
    min_sum_thr,max_sum_thr,include_at_sum_thr,data_for_sum_thr)

if isempty(min_sum_thr) && isempty(max_sum_thr)
   return 
end
    
#????? - why do we do a sum in one location and a mean in the other????
#------------------------------------------------------------------
# Compute the event sums.
eventSums = nan(length(startFrames), 1);
for i = 1:length(eventSums)
    eventSums(i) = nansum(data_for_sum_thr((startFrames(i)):(endFrames(i))));
end

# Compute the event sum thresholds.
if length(min_sum_thr) > 1 #i.e. if not a scaler
    newMinSumThr = nan(size(eventSums));
    for i = 1:length(newMinSumThr)
        newMinSumThr(i) = nanmean(min_sum_thr((startFrames(i)):(endFrames(i))));
    end
    min_sum_thr = newMinSumThr;
end

if length(max_sum_thr) > 1
    newMaxSumThr = nan(size(eventSums));
    for i = 1:length(newMaxSumThr)
        newMaxSumThr(i) = nanmean(max_sum_thr((startFrames(i)):(endFrames(i))));
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
startFrames(removeEvents) = [];
endFrames(removeEvents)   = [];



end

function [startFrames,endFrames] = h__removeTooSmallOrLargeEvents(startFrames,endFrames,...
    min_frames_thr,max_frames_thr,include_at_frames_thr)
#
#
#   This function filters events based on time (really sample count)
#


# Check the event frames.
#--------------------------------------------------------------------------
if ~(isempty(min_frames_thr) && isempty(max_frames_thr))
    
    # Compute the event frames.
    n_frames_per_event = endFrames - startFrames + 1;
    
    # Remove small events.
    removeEvents = false(size(n_frames_per_event));
    if ~isempty(min_frames_thr)
        if include_at_frames_thr
            removeEvents = n_frames_per_event <= min_frames_thr;
        else
            removeEvents = n_frames_per_event < min_frames_thr;
        end
    end
    
    # Remove large events.
    if ~isempty(max_frames_thr)
        if include_at_frames_thr
            removeEvents = removeEvents | n_frames_per_event >= max_frames_thr;
        else
            removeEvents = removeEvents | n_frames_per_event > max_frames_thr;
        end
    end
    
    # Remove the events.
    startFrames(removeEvents) = [];
    endFrames(removeEvents)   = [];
end

end

function event_mask = h__getPossibleEventsByThreshold(data,min_thr,max_thr,include_at_thr)

event_mask = true(length(data),1);
if ~isempty(min_thr)
    if include_at_thr
        event_mask = data >= min_thr;
    else
        event_mask = data > min_thr;
    end
end

if ~isempty(max_thr)
    if include_at_thr
        event_mask = event_mask & data <= max_thr;
    else
        event_mask = event_mask & data < max_thr;
    end
end


end

function [starts,stops] = h__getStartStopIndices(data,event_mask)

#We concatenate falses to ensure event starts and stops at the edges
#are caught 9i.e. allow edge detection if 
dEvent      = diff([false; event_mask; false]);

#0 1 2 3  4 5 6 <- true indices
#x n n y  y n n <- event
#0 0 1 0 -1 0  <- diffs, 1 indicates start, -1 indicates end
#1 2 3 4  5 6  <- indices of diffs
#    s    e    <- start and end
#start matches its index
#end is off by 1
#

starts = find(dEvent == 1);
stops  = find(dEvent == -1) - 1;

if isempty(starts)
    return
end

# Include NaNs at the start and end.
#----------------------------------------------------------------------
if all(isnan(data(1:(starts(1)-1))))
    starts(1) = 1;
end

if all(isnan(data((stops(end)+1):end)))
    stops(end) = length(data);
end

end
"""