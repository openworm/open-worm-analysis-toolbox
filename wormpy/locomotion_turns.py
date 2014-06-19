# -*- coding: utf-8 -*-
"""
locomotion_turns.py

Calculate the "Turns" locomotion feature

There are two kinds of turn, an omega and an upsilon.

The only external-facing item is LocomotionTurns.  The rest are internal 
to this module.


Classes
---------------------------------------    
LocomotionTurns
UpsilonTurns
OmegaTurns


Standalone Functions
---------------------------------------    
getTurnEventsFromSignedFrames


Notes
---------------------------------------    

For the Nature Methods description see 
/documentation/Yemini Supplemental Data/Locomotion.md#5-turns


Formerly this code was contained in four MATLAB files:
  seg_worm.feature_calculator.getOmegaAndUpsilonTurns, which called these 3:
    seg_worm.feature_helpers.locomotion.getOmegaEvents
    seg_worm.feature_helpers.locomotion.getUpsilonEvents
    seg_worm.feature_helpers.locomotion.getTurnEventsFromSignedFrames


TODO: OmegaTurns and UpsilonTurns should inherit from LocomotionTurns or something

"""

import utils
import operator
import re
from . import EventFinder
from . import feature_helpers
from . import config
import numpy as np
import collections


class LocomotionTurns(object):
  """
  LocomotionTurns

  Properties
  ---------------------------------------    
  omegas
  upsilons

  Methods
  ---------------------------------------    
  __init__
  

  Notes
  ---------------------------------------    
  Formerly this was not implemented as a class.

  """

  def __init__(self, nw, bend_angles, is_stage_movement, midbody_distance, sx, sy):
    """
    Constructor for the LocomotionTurns class

    Notes
    ---------------------------------------    
    Formerly getOmegaAndUpsilonTurns(obj, bend_angles,is_stage_movement,midbody_distance,sx,sy,FPS)
  
    Old Name: 
    - featureProcess.m
    - omegaUpsilonDetectCurvature.m
    
    """
    
    MAX_INTERPOLATION_GAP_ALLOWED = 9;

    n_frames = bend_angles.shape[1]
    
    a = collections.namedtuple('angles',['head_angles','body_angles',\
      'tail_angles','body_angles_with_long_nans','is_stage_movement'])

    first_third  = nw.get_subset_partition_mask('first_third')
    second_third = nw.get_subset_partition_mask('second_third')
    last_third   = nw.get_subset_partition_mask('last_third')
    
    #NOTE: For some reason the first and last few angles are NaN, so we use
    #nanmean instead of mean, could probably avoid this for the body ...
    a.head_angles = np.nanmean(bend_angles[first_third,:],axis=0)
    a.body_angles = np.nanmean(bend_angles[second_third,:],axis=0)
    a.tail_angles = np.nanmean(bend_angles[last_third,:],axis=0)
    a.is_stage_movement = is_stage_movement

    #Does this do a deep copy??? I want a deep copy
    body_angles_for_ht_change = np.copy(a.body_angles)

    n_head = np.sum(~np.isnan(a.head_angles))
    n_body = np.sum(~np.isnan(a.body_angles))
    n_tail = np.sum(~np.isnan(a.tail_angles))

    

    #only proceed if there are at least two non-NaN value in each angle vector
    if n_head < 2 or n_body < 2 or n_tail < 2:
       self.omegas   = EventFinder.EventListForOutput([],[],make_null=True)
       self.upsilons = EventFinder.EventListForOutput([],[],make_null=True)
       return 

    
    
    #Interpolation
    #--------------------------------------------------------------------------
    self.h__interpolateAngles(a,MAX_INTERPOLATION_GAP_ALLOWED)    
    
    #Get frames for each turn type
    #--------------------------------------------------------------------------
    #This doesn't match was is written in the supplemental material ...
    #Am I working off of old code??????

    c = collections.namedtuple('consts',['head_angle_start_const',\
      'tail_angle_start_const','head_angle_end_const',\
      'tail_angle_end_const','body_angle_const'])
      
            
    yuck = [[20,-20,15,-15],
            [30,  30, 30,  30],
            [40,  40, 30,  30],
            [20, -20, 15, -15],
            [20, -20, 15, -15]]
    """
    c = struct(...
        'head_angle_start_const',{20 -20 15 -15}, ...
        'tail_angle_start_const',{30  30 30  30}, ...
        'head_angle_end_const',  {40  40 30  30}, ...
        'tail_angle_end_const',  {20 -20 15 -15}, ...
        'body_angle_const'   ,   {20 -20 15 -15});
    """
        
    #NOTE: We need to run omegas first (false values) since upsilons are more
    #inclusive, but can not occur if an omega event occurs
    is_upsilon  = [False, False, True, True]
    
    #NOTE: We assign different values based on the sign of the angles
    values_to_assign = [1, -1,  1, -1];
    
    f = collections.namedtuple('frames',['omega_frames','upsilon_frames'])    
    
    f.omega_frames   = np.zeros(n_frames);
    f.upsilon_frames = np.zeros(n_frames);  
    
    for i in range(4):
        c.head_angle_start_const = yuck[0][i]
        c.tail_angle_start_const = yuck[1][i]
        c.head_angle_end_const   = yuck[2][i]
        c.tail_angle_end_const   = yuck[3][i]
        c.body_angle_const       = yuck[4][i]
        s = self.h__getConditionIndices(a,c)
        self.h__populateFrames(a,s,f,is_upsilon[i],values_to_assign[i])
        
    #Calculate the events from the frame values
    #--------------------------------------------------------------------------
    #self, nw, bend_angles, is_stage_movement, midbody_distance, sx, sy):
    self.omegas   = OmegaTurns(f.omega_frames,sx,sy,body_angles_for_ht_change,midbody_distance,config.FPS)    
    self.upsilons = UpsilonTurns(f.upsilon_frames,midbody_distance,config.FPS)    
    
    #obj.getOmegaEvents(f.omegaFrames,sx,sy,body_angles_for_ht_change,midbody_distance,FPS);
    #obj.getUpsilonEvents(f.upsilonFrames,midbody_distance,FPS);
    
    import pdb
    pdb.set_trace()    
    
    a = 1

    """
    %IMPORTANT: My events use 1 based indexing, the old code used 0 based
    %indexing

    

    

    

    

    
    """
    
  def h__interpolateAngles(self, a, MAX_INTERPOLATION_GAP_ALLOWED):
    """
  
    Formerly a = h__interpolateAngles(a,MAX_INTERPOLATION_GAP_ALLOWED)
    %
    %   Inputs
    %   =======================================================================
    %   a.head_angles
    %   a.body_angles
    %   a.tail_angles
    %
    %   Outputs
    %   =======================================================================
    
    %   TODO: Incorporate into 
    %   seg_worm.feature_helpers.interpolateNanData
    """
    
    feature_helpers.interpolate_with_threshold(a.head_angles,make_copy=False)
    #This might not actually have been applied - SEGWORM_MC used BodyAngles
    #feature_helpers.interpolate_with_threshold(a.body_angles,MAX_INTERPOLATION_GAP_ALLOWED+1)
    a.body_angles_with_long_nans = feature_helpers.interpolate_with_threshold(
        a.body_angles,MAX_INTERPOLATION_GAP_ALLOWED+1)
    feature_helpers.interpolate_with_threshold(a.body_angles,make_copy=False)
    feature_helpers.interpolate_with_threshold(a.tail_angles,make_copy=False)
      
    """
    
    #This code works, but the interpolation wasn't implemented.
    #Then I discovered Michael's code ...
    
    n = np.isnan(a.body_angles)

    regex_str = 'B{%d,}' %  (MAX_INTERPOLATION_GAP_ALLOWED + 1)    
    
    str_to_search = "".join(['A' if x else 'B' for x in n])
    
    temp = re.finditer(regex_str,str_to_search)
    
    starts_ends = [(x.start(),x.end()) for x in temp]    
    
    # interpolate arrays over NaN values (where there were stage
    # movements, touching, or some other segmentation problem)
    # ***This is of course only an approximate solution to the problem of
    # not segmenting coiled shapes***
    self.h__interp_NaN(a.head_angles)
    self.h__interp_NaN(a.body_angles)
    self.h__interp_NaN(a.tail_angles)
    
    # return long NaN stretches back to NaN- only for the body angles ...
    for start_I,end_I in starts_ends:
      a.body_angles[start_I:end_I+1] = np.NaN
    """
  
  def h__getConditionIndices(self, a, c):
  
    """
    %
    Formerly s = h__getConditionIndices(a, c)
    %
    %   This function implements a filter on the frames for the different
    %   conditions that we are looking for in order to get a particular turn.
    %   
    %   It does not however provide any logic on their relative order, i.e.
    %   that one condition occurs before another. This is done in a later
    %   function, h__populateFrames.
    """
    
    #Determine comparison function
    #----------------------------------------------------------
    is_positive = c.head_angle_start_const > 0    
    if is_positive:
      fh = operator.gt
    else:
      fh = operator.lt
      
    #start: when the head exceeds its angle but the tail does not
    #end  : when the tail exceeds its angle but the head does not
    
    #TODO: Rename to convention ...
    s = collections.namedtuple('stuffs',['startCond','startInds','midCond','midStarts','midEnds','endCond','endInds'])
    
    def find_diff(array,value):
      #diff on logical array doesn't work the same as it does in Matlab
      return np.flatnonzero(np.diff(array.astype(int)) == value)
 
    with np.errstate(invalid='ignore'):
       s.startCond = fh(a.head_angles, c.head_angle_start_const) & (np.abs(a.tail_angles) < c.tail_angle_start_const)
    
    s.startInds = find_diff(s.startCond,1) + 1 #add 1 for shift due to diff

    #NOTE: This is NaN check is a bit suspicious, as it implies that the
    #head and tail are parsed, but the body is not. The original code puts
    #NaN back in for long gaps in the body angle, so it is possible that
    #the body angle is NaN but the others are not.
    with np.errstate(invalid='ignore'):
      s.midCond   = fh(a.body_angles, c.body_angle_const) | \
        np.isnan(a.body_angles_with_long_nans)
      
    s.midStarts = find_diff(s.midCond,1) + 1 #add 1 for shift due to diff
    s.midEnds   = find_diff(s.midCond,-1)
    
    with np.errstate(invalid='ignore'):
      s.endCond   = np.logical_and(fh(a.tail_angles, c.tail_angle_end_const),np.abs(a.head_angles) < c.head_angle_end_const)
      
    s.endInds   = find_diff(s.endCond,-1)  
    
    return s
    """
    
    is_positive = c.head_angle_start_const > 0;
  
    if is_positive
        fh = @gt;
    else
        fh = @lt;
    end
    
    %start: when the head exceeds its angle but the tail does not
    %end  : when the tail exceeds its angle but the head does not
    
    s.startCond = fh(a.head_angles, c.head_angle_start_const) & abs(a.tail_angles) < c.tail_angle_start_const;
    s.startInds = find(diff(s.startCond) == 1) + 1; %add 1 for shift due to diff
    
    %NOTE: This is NaN check is a bit suspicious, as it implies that the
    %head and tail are parsed, but the body is not. The original code puts
    %NaN back in for long gaps in the body angle, so it is possible that
    %the body angle is NaN but the others are not.
    s.midCond   = fh(a.body_angles, c.body_angle_const) | isnan(a.bodyAngle);
    s.midStarts = find(diff(s.midCond) == 1) + 1; %add 1 for shift due to diff
    s.midEnds   = find(diff(s.midCond) == -1);
    
    s.endCond   = fh(a.tail_angles, c.tail_angle_end_const) & abs(a.head_angles) < c.head_angle_end_const;
    s.endInds   = find(diff(s.endCond) == -1);
  
    """
  
  
  
  def h__populateFrames(self, a, s, f, get_upsilon_flag, value_to_assign):
    """
    %
    Formerly f = h__populateFrames(a,s,f,get_upsilon_flag,value_to_assign)
    %
    %   Inputs
    %   =======================================================================
    %    a: (structure)
    %           head_angles: [1x4642 double]
    %           body_angles: [1x4642 double]
    %           tail_angles: [1x4642 double]
    %     is_stage_movement: [1x4642 logical]
    %             bodyAngle: [1x4642 double]
    %    s: (structure)
    %     startCond: [1x4642 logical]
    %     startInds: [1x81 double]
    %       midCond: [1x4642 logical]
    %     midStarts: [268 649 881 996 1101 1148 1202 1963 3190 3241 4144 4189 4246 4346 4390 4457 4572 4626]
    %       midEnds: [301 657 925 1009 1103 1158 1209 1964 3196 3266 4148 4200 4258 4350 4399 4461 4579]
    %       endCond: [1x4642 logical]
    %       endInds: [1x47 double]
    %   f: (structure)
    %       omegaFrames: [4642x1 double]
    %     upsilonFrames: [4642x1 double]
    %   get_upsilon_flag : toggled based on whether or not we are getting
    %               upsilon events or omega events
    %   sign_value : 
    %
    %   Outputs
    %   =======================================================================
    %
    
    %Algorithm:
    %-----------------------------------------------------------
    %- For the middle angle range, ensure one frame is valid and that
    %  the frame proceeding the start and following the end are valid
    %- Find start indices and end indices that bound this range
    %- For upsilons, exclude if they overlap with an omega bend ...
    """
    
    #import pdb
    #pdb.set_trace()    
    
    for cur_mid_start_I in s.midStarts:
      
      #JAH NOTE: This type of searching is inefficient in Matlab since 
      #the data is already sorted. It could be improved ...
      temp = np.flatnonzero(s.midEnds > cur_mid_start_I)
         
      #cur_mid_end_I   = s.midEnds[find(s.midEnds > cur_mid_start_I, 1))
    
      if temp.size != 0:
        cur_mid_end_I = s.midEnds[temp[0]]
        
        if ~np.all(a.is_stage_movement[cur_mid_start_I:cur_mid_end_I+1]) and \
          s.startCond[cur_mid_start_I - 1] and \
          s.endCond[cur_mid_end_I + 1]:
          
          temp2 = np.flatnonzero(s.startInds < cur_mid_start_I)
          temp3 = np.flatnonzero(s.endInds     > cur_mid_end_I)
          
          if temp2.size != 0 and temp3.size != 0:
                
            cur_start_I = s.startInds[temp2[-1]]
            cur_end_I   = s.endInds[temp3[0]]     
            
            if get_upsilon_flag:
                  #Don't populate upsilon if the data spans an omega
                  if ~np.any(np.abs(f.omega_frames[cur_start_I:cur_end_I+1])):
                      f.upsilon_frames[cur_start_I:cur_end_I+1] = value_to_assign
            else:
              f.omega_frames[cur_start_I:cur_end_I+1] = value_to_assign

    return None
    """
    
    for iMid = 1:length(s.midStarts)
        cur_mid_start_I = s.midStarts(iMid);
        
        %JAH NOTE: This type of searching is inefficient in Matlab since 
        %the data is already sorted. It could be improved ...
        cur_mid_end_I   = s.midEnds(find(s.midEnds > cur_mid_start_I, 1));
        
        if ~isempty(cur_mid_end_I)              && ...
            ~all(a.is_stage_movement(cur_mid_start_I:cur_mid_end_I)) && ...
            s.startCond(cur_mid_start_I - 1)    && ...
            s.endCond(cur_mid_end_I + 1)
 
            cur_start_I = s.startInds(find(s.startInds < cur_mid_start_I,   1, 'last'));
            cur_end_I   = s.endInds(find(s.endInds     > cur_mid_end_I,     1));

            if get_upsilon_flag
                %Don't populate upsilon if the data spans an omega
                if ~any(abs(f.omegaFrames(cur_start_I:cur_end_I)))
                    f.upsilonFrames(cur_start_I:cur_end_I) = value_to_assign;
                end
            else
                f.omegaFrames(cur_start_I:cur_end_I) = value_to_assign;
            end
        end
    end
    """
    pass


"""
===============================================================================
===============================================================================
"""


class UpsilonTurns(object):
  """
  Represents the Omega turn events


  Notes
  ---------------------------------------    
  Formerly this was not implemented as a class.
  
  """
  def __init__(self,upsilon_frames,midbody_distance,FPS):
    
    #How to reference??????
    self.value = getTurnEventsFromSignedFrames(upsilon_frames, midbody_distance, FPS)  
    
    
    """
    
    Formerly, in the SegWormMatlabClasses repo, this was not the constructor 
    of a class, but a locomotion method of called 
    getUpsilonEvents(obj,upsilon_frames,midbody_distance,FPS)
    """
    pass
    """    
    self.upsilons = obj.getTurnEventsFromSignedFrames(upsilon_frames,midbody_distance,FPS);
    """


"""
===============================================================================
===============================================================================
"""


class OmegaTurns(object):
  """
  Represents the Omega turn events

  Properties
  ---------------------------------------    
  omegas
  
  Methods
  ---------------------------------------    
  __init__  
  h_getHeadTailDirectionChange
  h__filterAndSignFrames
  h__interp_NaN
  
  """
  
  def __init__(self,omega_frames_from_angles,sx,sy,body_angles,midbody_distance,FPS):
    """
    
    Formerly, in the SegWormMatlabClasses repo, this was not the constructor 
    of a class, but a locomotion method of called 
    getOmegaEvents(obj,omega_frames_from_angles,sx,sy,
                   body_angles,midbody_distance,fps)

    %   Inputs
    %   =======================================================================
    %   sx :
    %   sy :
    %   fps
    %   body_angles : average bend angle of the middle third of the worm
    %   midbody_distance :
    %   omega_frames_from_angles : [1 x n_frames], each frame has the value 0,
    %       1, or -1, 
    %
    %   Outputs
    %   =======================================================================
    %   omega_events : event structure 
    %
    %   Called By:
    %   
    %
    %   See Also:
    %   seg_worm.features.locomotion.getOmegaAndUpsilonTurns
    %   seg_worm.features.locomotion.getTurnEventsFromSignedFrames

    
    """
    
    import pdb
    pdb.set_trace()
    
    MIN_OMEGA_EVENT_LENGTH = round(FPS/4)
    
    body_angles_i = feature_helpers.interpolate_with_threshold(body_angles, \
      extrapolate=True) #_i - interpolated
      
    #body_angles_i = h__interp_NaN(body_angles,true) 
    
        
    
    omega_frames_from_th_change = h_getHeadTailDirectionChange(fps,sx,sy);
    
    #Filter:
    #This is to be consistent with the old code. We filter then merge, then
    #filter again :/
    omega_frames_from_th_change = h__filterAndSignFrames(...
        body_angles_i,omega_frames_from_th_change,MIN_OMEGA_EVENT_LENGTH);
    
    is_omega_frame = omega_frames_from_angles | omega_frames_from_th_change;
    
    #Refilter and sign
    signed_omega_frames = h__filterAndSignFrames(body_angles_i,is_omega_frame,MIN_OMEGA_EVENT_LENGTH);
    
    #Convert frames to events ...
    self.values = getTurnEventsFromSignedFrames(signed_omega_frames, midbody_distance, FPS)    
    
    #obj.turns.omegas = obj.getTurnEventsFromSignedFrames(signed_omega_frames,midbody_distance,fps);

     
    
    """
    
    MIN_OMEGA_EVENT_LENGTH = round(fps/4);
    
    body_angles_i = h__interp_NaN(body_angles,true); %_i - interpolated
    
    omega_frames_from_th_change = h_getHeadTailDirectionChange(fps,sx,sy);
    
    %Filter:
    %This is to be consistent with the old code. We filter then merge, then
    %filter again :/
    omega_frames_from_th_change = h__filterAndSignFrames(...
        body_angles_i,omega_frames_from_th_change,MIN_OMEGA_EVENT_LENGTH);
    
    is_omega_frame = omega_frames_from_angles | omega_frames_from_th_change;
    
    %Refilter and sign
    signed_omega_frames = h__filterAndSignFrames(body_angles_i,is_omega_frame,MIN_OMEGA_EVENT_LENGTH);
    
    %Convert frames to events ...
    obj.turns.omegas = obj.getTurnEventsFromSignedFrames(signed_omega_frames,midbody_distance,fps);
    
    """
  
  


  def h_getHeadTailDirectionChange(self, FPS, sx, sy):
    """
    %
    Formerly is_omega_angle_change = h_getHeadTailDirectionChange(FPS,sx,sy)
    %
    %   NOTE: This change in direction of the head and tail indicates that
    %   either a turn occurred OR that an error in the parsing occurred.
    %   Basically we look for the angle from the head to the tail to all of a
    %   sudden change by 180 degrees. 
    %
    """
    pass
  
    """
    MAX_FRAME_JUMP_FOR_ANGLE_DIFF = round(FPS/2);
    
    %We compute a smoothed estimate of the angle change by using angles at
    %indices that are +/- this value ...
    HALF_WINDOW_SIZE = round(FPS/4);
    
    %NOTE: It would be better to have this be based on time, not samples
    MAX_INTERP_GAP_SIZE = 119;
    
    %????!!!!?? - why is this a per frame value instead of an average angular
    %velocity ????
    PER_FRAME_DEGREE_CHANGE_CUTOFF = 3;
    
    
    % Compute tail direction
    %----------------------------------------------------
    SI = seg_worm.skeleton_indices;
    
    head_x = mean(sx(SI.HEAD_INDICES,:),1);
    head_y = mean(sy(SI.HEAD_INDICES,:),1);
    tail_x = mean(sx(SI.TAIL_INDICES,:),1);
    tail_y = mean(sy(SI.TAIL_INDICES,:),1);
    
    th_angle  = atan2(head_y - tail_y, head_x - tail_x)*(180/pi);
    
    n_frames = length(th_angle);
    
    %Changed angles to being relative to the previous frame
    %--------------------------------------------------------------------------
    %Compute the angle change between subsequent frames. If a frame is not
    %valid, we'll use the last valid frame to define the difference, unless the
    %gap is too large.
    
    is_good_th_direction_value = ~isnan(th_angle);
    
    lastAngle  = th_angle(1);
    gapCounter = 0;
    
    th_angle_diff_temp = NaN(size(th_angle));
    for iFrame = 2:n_frames 
        if is_good_th_direction_value(iFrame)
            th_angle_diff_temp(iFrame) = th_angle(iFrame) - lastAngle;
            gapCounter = 0;
            lastAngle  = th_angle(iFrame);
        else
            gapCounter = gapCounter + 1;
        end
        if gapCounter > MAX_FRAME_JUMP_FOR_ANGLE_DIFF
            lastAngle = NaN;
        end
    end
    
    %???? - what does this really mean ??????
    %I think this basically says, instead of looking for gaps in the original
    %th_angle, we need to take into account how much time has passed between
    %successive differences
    %
    %i.e. instead of doing a difference in angles between all valid frames, we
    %only do a difference if the gap is short enough
    positiveJumps = find(th_angle_diff_temp > 180);
    negativeJumps = find(th_angle_diff_temp < -180);
    
    %For example data, these are the indices I get ...
    %P - 4625
    %N - 3634, 4521 
    
    
    
    
    %Fix the th_angles by unwrapping
    %--------------------------------------------------------------------------
    %NOTE: We are using the identified jumps from the fixed angles to unwrap
    %the original angle vector
    % subtract 2pi from remainging data after positive jumps
    for j = 1:length(positiveJumps)
        th_angle(positiveJumps(j):end) = th_angle(positiveJumps(j):end) - 2*180;
    end
    
    % add 2pi to remaining data after negative jumps
    for j = 1:length(negativeJumps)
        th_angle(negativeJumps(j):end) = th_angle(negativeJumps(j):end) + 2*180;
    end
    
    
    
    %Fix the th_angles through interpolation
    %--------------------------------------------------------------------------
    % get long NaN stretches
    n = isnan(th_angle);
    % save start and end indices for the stretches
    
    gap_str = sprintf('B{%d,}',MAX_INTERP_GAP_SIZE+1); %Add 1 so that we allow the max gap
    %but not anything greater
    [start1, end1] = regexp( char(n+'A'), gap_str, 'start', 'end' );
    
    % interpolate missing data
    th_angle = h__interp_NaN(th_angle,false);
    
    % return long NaN stretches back to NaN
    for iEvent=1:length(start1)
        th_angle(start1(iEvent):end1(iEvent)) = NaN;
    end
    
    %Determine frames that might be omega events (we'll filter later based on
    %length)
    %--------------------------------------------------------------------------
    % Compute angle difference
    th_angle_diff = NaN(length(th_angle),1);
    
    left_indices = (1:n_frames) - HALF_WINDOW_SIZE;
    right_indics = (1:n_frames) + HALF_WINDOW_SIZE;
    
    mask = left_indices > 1 & right_indics < n_frames;
    
    th_angle_diff(mask) = th_angle(right_indics(mask)) - th_angle(left_indices(mask));
    
    avg_angle_change_per_frame = abs(th_angle_diff/(HALF_WINDOW_SIZE*2));
    is_omega_angle_change      = avg_angle_change_per_frame > PER_FRAME_DEGREE_CHANGE_CUTOFF;
    
    end
    """
  
  def h__filterAndSignFrames(self, body_angles_i, is_omega_frame, MIN_OMEGA_EVENT_LENGTH):
    """
    Formerly signed_omega_frames = h__filterAndSignFrames(body_angles_i,is_omega_frame,MIN_OMEGA_EVENT_LENGTH)
    """
    pass
    """
    gap_str = sprintf('B{%d,}',MIN_OMEGA_EVENT_LENGTH);
    [start1, end1] = regexp( char(is_omega_frame+'A')', gap_str, 'start', 'end');
    
    signed_omega_frames = zeros(size(is_omega_frame));
    
    %NOTE: Here we keep the long gaps instead of removing them
    
    for iEvent = 1:length(start1)
        if mean(body_angles_i(start1(iEvent):end1(iEvent))) > 0
            signed_omega_frames(start1(iEvent):end1(iEvent)) = 1;
        else
            signed_omega_frames(start1(iEvent):end1(iEvent)) = -1;
        end
    end
    
    """
  
  
  def h__interp_NaN(self, x, use_extrap):
    """
    Formerly fixed_x = h__interp_NaN(x,use_extrap)
    """
    pass
    """
    fixed_x  = x;
    nan_mask = isnan(x);
    
    if use_extrap
        fixed_x(nan_mask) = interp1(find(~nan_mask),x(~nan_mask), find(nan_mask),'linear','extrap');
    else
        fixed_x(nan_mask) = interp1(find(~nan_mask),x(~nan_mask), find(nan_mask),'linear');    
    end
    
    end
    """


"""
===============================================================================
===============================================================================
"""

def getTurnEventsFromSignedFrames(signed_frames, midbody_distance, FPS):
  """
  Formerly function turn_events = getTurnEventsFromSignedFrames(obj,signed_frames,midbody_distance,FPS)
  %
  %   seg_worm.features.locomotion.getTurnEventsFromSignedFrames
  %
  %   Inputs
  %   =======================================================================
  %   obj : Class: seg_worm.features.locomotion
  %   signed_frames : ??? - I believe the values are -1 or 1, based on
  %   whether something is dorsal or ventral ....
  %
  %   This code is common to omega and upsilon turns.
  %
  %   Called by:
  %   seg_worm.features.locomotion.getUpsilonEvents
  %   seg_worm.features.locomotion.getOmegaEvents  
  %
  """
  
  ef = EventFinder.EventFinder()
  
  
  #import pdb
  #pdb.set_trace()
  
  ef.include_at_frames_threshold = True
  
  #get_events(self, speed_data, distance_data=None):  
  
  #JAH: This interface doesn't make as much sense anymore ...  
  
  #seg_worm.feature.event_finder.getEvents
  ef.min_speed_threshold = 1
  
  frames_dorsal  = ef.get_events(signed_frames)
  
  ef.min_speed_threshold =  None
  ef.max_speed_threshold = -1
  frames_ventral = ef.get_events(signed_frames)
  
  #import pdb
  #pdb.set_trace() 
  
  # Unify the ventral and dorsal turns.
  #--------------------------------------------------------------------------
  [frames_merged,is_ventral] = EventFinder.EventList.merge(frames_ventral,frames_dorsal)
    
  temp = EventFinder.EventListForOutput(frames_merged,midbody_distance) 
  
  temp.is_ventral = is_ventral  
  
  return temp

  """
  INTER_DATA_SUM_NAME = 'interDistance';
  DATA_SUM_NAME       = '';
  
  ef = seg_worm.feature.event_finder;
  ef.include_at_thr = true;
  
  %seg_worm.feature.event_finder.getEvents
  frames_dorsal  = ef.getEvents(signed_frames,1,[]);
  frames_ventral = ef.getEvents(signed_frames,[],-1);
  
  % Unify the ventral and dorsal turns.
  %--------------------------------------------------------------------------
  [frames_merged,is_ventral] = frames_ventral.merge(frames_dorsal);
  
  temp = seg_worm.feature.event(frames_merged,FPS,midbody_distance,DATA_SUM_NAME,INTER_DATA_SUM_NAME); 
  
  %seg_worm.feature.event.getFeatureStruct
  turn_events = temp.getFeatureStruct;
  
  %Add extra field, isVentral ...
  %---------------------------------------------------------------
  n_events = length(turn_events.frames);
  for iEvent = 1:n_events:
     turn_events.frames(iEvent).isVentral = is_ventral(iEvent); 
  
  """
  
  
  
  
