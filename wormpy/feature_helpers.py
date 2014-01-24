# -*- coding: utf-8 -*-
"""
  feature_helpers.py
  
  @authors: @MichaelCurrie, @JimHokanson
  
  Some helper functions that assist in the calculation of the attributes of
  WormFeatures
  
  
  © Medical Research Council 2012
  You will not remove any copyright or other notices from the Software; 
  you must reproduce all copyright notices and other proprietary 
  notices on any copies of the Software.

"""
import numpy as np
import collections
from wormpy import config

__ALL__ = ['get_motion_codes',                  # for locomotion
           'get_worm_velocity',                 # for locomotion
           'get_bends',                         # for posture
           'get_amplitude_and_wavelength',      # for posture
           'get_eccentricity_and_orientation']  # for posture

"""----------------------------------------------------
    motion codes:
    
    
"""

def get_motion_codes(midbody_speed, skeleton_lengths):
  """
    get_motion_codes: calculate motion codes (a locomotion feature)
      INPUT:
        midbody speed:  
        skeleton_lengths:
        
      OUTPUT
        the locomotion events; a dict with event fields:
          forward  - (event) forward locomotion
          paused   - (event) no locomotion (the worm is paused)
          backward - (event) backward locomotion
          mode     = [1 x num_frames] the locomotion mode:
                     -1 = backward locomotion
                      0 = no locomotion (the worm is paused)
                      1 = forward locomotion

  """

  # Compute the locomotion events.
  #--------------------------------------------------------------------------
  
  # Initialize the worm speed and video frames.
  num_frames = len(midbody_speed)
  
  # Compute the distance.
  distance = abs(midbody_speed / config.FPS)

  return 0  # debug
  """
  # Interpolate the missing lengths.
  isNotData = isnan(skeleton_lengths)
  isData    = ~isNotData
  dataI     = find(isData)
  interpI   = find(isNotData)
  if(~isempty(interpI) and length(dataI) > 1):
      skeleton_lengths(interpI) = interp1(dataI, skeleton_lengths(dataI), interpI, 'linear')
  end
  
  #==========================================================================
  worm_speed_threshold       = skeleton_lengths * SPEED_THRESHOLD_PCT
  worm_distance_threshold    = skeleton_lengths * DISTANCE_THRSHOLD_PCT 
  
  #Forward stuffs
  #--------------------------------------------------------------------------
  min_forward_speed    = worm_speed_threshold
  min_forward_distance = worm_distance_threshold
  
  #Backward stuffs
  #--------------------------------------------------------------------------
  max_backward_speed    = -worm_speed_threshold
  min_backward_distance = worm_distance_threshold
  
  #Paused stuffs
  #--------------------------------------------------------------------------
  worm_pause_threshold = skeleton_lengths * PAUSE_THRESHOLD_PCT # 2.5 percent of its length
  min_paused_speed     = -worm_pause_threshold
  max_paused_speed     = worm_pause_threshold

  min_speeds   = {min_forward_speed       []                      min_paused_speed}
  max_speeds   = {[]                      max_backward_speed      max_paused_speed}
  min_distance = {min_forward_distance    min_backward_distance   []}
  
  #--------------------------------------------------------------------------
  worm_event_frames_threshold          = fps * EVENT_FRAMES_THRESHOLD
  worm_event_min_interframes_threshold = fps * EVENT_MIN_INTER_FRAMES_THRESHOLD
  
  all_events_struct = struct
  
  FIELD_NAMES  = {'forward' 'backward' 'paused'}
  FRAME_VALUES = [1 -1 0]
  motion_mode = NaN(1,num_frames)
  
  for iType = 1:3
     
      #Determine when the event type occurred
      #----------------------------------------------------------------------
      ef = seg_worm.feature.event_finder
      
      ef.include_at_thr       = true
      ef.minum_frames_thr       = worm_event_frames_threshold
      ef.min_sum_thr          = min_distance{iType}
      ef.include_at_sum_thr   = true
      ef.data_for_sum_thr     = distance
      ef.min_inter_frames_thr = worm_event_min_interframes_threshold
      
      #seg_worm.feature.event_finder.getEvents
      frames_temp = ef.getEvents(midbody_speed,min_speeds{iType},max_speeds{iType})
      #frames_temp - class - seg_worm.feature.event_ss
      
      #Assign event type to relevant frames
      #----------------------------------------------------------------------
      mask = frames_temp.getEventMask(num_frames)
      motion_mode(mask) = FRAME_VALUES(iType)
  
      #Take the start and stop indices and convert them to the structure
      #used in the feature files ...
      #----------------------------------------------------------------------
      cur_field_name = FIELD_NAMES{iType}
  
      temp = seg_worm.feature.event(frames_temp,fps,distance,DATA_SUM_NAME,INTER_DATA_SUM_NAME)    
      all_events_struct.(cur_field_name) = temp.getFeatureStruct
      
  end
  
  all_events_struct.mode = motion_mode
  
  obj.motion = all_events_struct
  """






"""----------------------------------------------------
    velocity:
"""

def get_angles(segment_x, segment_y, head_to_tail=False):
  """
    get_angles: obtain the "angle" of a subset of the 49 points
                 of a worm, for each frame.
      INPUT: 
        segment_x, segment_y: numpy arrays of shape (p,n) where 
                              p is the size of the partition of the 49 points
                              n is the number of frames in the video
        head_to_tail: True means the worm points are order head to tail.
    
      OUTPUT: A numpy array of shape (n) and stores the worm body's "angle" 
              (in degrees) for each frame of video

  """
  
  if(not head_to_tail):
    # reverse the worm points so we go from tail to head
    segment_x = segment_x[::-1,:]
    segment_y = segment_y[::-1,:]

  # Diff calculates each point's difference between the segment's points
  # then we take the mean of these differences for each frame
  average_diff_x = np.nanmean(np.diff(segment_x, n=1, axis=0), 
                              axis=0) # shape (n)
  average_diff_y = np.nanmean(np.diff(segment_y, n=1, axis=0), 
                              axis=0) # shape (n)
  
  # angles has shape (n) and stores the worm body's "angle"
  # for each frame of video
  angles = np.degrees(np.arctan2(average_diff_y, average_diff_x))

  return angles


def get_partition_angles(nw, partition_key, data_key='skeletons',
                         head_to_tail=False):
  """
    get_partition_angles: obtain the "angle" of a subset of the 49 points 
                          of a worm for each frame
      INPUT: head_to_tail=True means the worm points are order head to tail.
    
      OUTPUT: A numpy array of shape (n) and stores the worm body's "angle" 
              (in degrees) for each frame of video
    
  """
  # the shape of both segment_x and segment_y is (partition length, n)
  segment_x, segment_y = nw.get_partition(partition_key, data_key, True)  

  return get_angles(segment_x, segment_y, head_to_tail)


def h__computeAngularSpeed(segment_x, segment_y, 
                           left_I, right_I, ventral_mode):
  """
    h__computeAngularSpeed
      INPUT: 
        segment_x: the x's of the partition being considered. shape (p,n)
        segment_y: the y's of the partition being considered. shape (p,n)
        left_I: the angle's first point
        right_I: the angle's second point
        ventral_mode: 0, 1, or 2, specifying that the ventral side is...
                        0 = unknown
                        1 = clockwise
                        2 = anticlockwise
                        
      OUTPUT: a numpy array of shape n, in units of degrees per second
      
  """
  # Compute the body part direction for each frame
  point_angle_d = get_angles(segment_x, segment_y, head_to_tail=False)

  angular_speed = point_angle_d(right_I) - point_angle_d(left_I)
  
  # Correct any jumps that result during the subtraction process
  # i.e. 1 - 359 ~= -358
  # by forcing -180 <= angular_speed[i] <= 180
  angular_speed = (angular_speed + 180) % (360) - 180

  # Change units from degrees per frame to degrees per second
  angular_speed = angular_speed * (1/config.FPS)
  
  # Sign the direction for dorsal/ventral locomotion.
  # if ventral_mode is anything but anticlockwise, then negate angular_speed:
  if(ventral_mode < 2):
    angular_speed = -angular_speed

  return angular_speed


def h__getVelocityIndices(frames_per_sample, good_frames_mask):
  """
    h__getVelocityIndices:
      Compute the speed using back/front nearest neighbours, avoiding NaNs,
      bounded at twice the scale.

      INPUTS:
        frames_per_sample: our sample scale, in frames.  must be an odd integer
        good_frames_mask: shape (num_frames), false if underlying angle is NaN
  
       OUTPUTS:
         keep_mask : shape (num_frames), this is used to indicate 
                     which original frames have valid velocity values, 
                     and which don't. 
                     NOTE: sum(keep_mask) == n_valid_velocity_values
         left_I    : shape (n_valid_velocity_values), for a given sample, this
                     indicates the index to the left of (less than) the sample
                     that should be used to calculate the velocity
         right_I   : shape (n_valid_velocity_values)
  
  """
  # Require that frames_per_sample be an odd integer
  assert(type(frames_per_sample)==int)
  assert(frames_per_sample%2==1)
  
  num_frames = len(good_frames_mask)
  
  # Create a "half" scale
  # NOTE: Since the scale is odd, the half
  #       scale will be even, because we subtract 1
  scale_minus_1 = frames_per_sample - 1
  half_scale    = int(scale_minus_1 / 2)
  
  # First frame for which we can assign a valid velocity:
  start_index = half_scale 
  # Final frame for which we can assign a valid velocity, plus one:
  end_index   = num_frames - half_scale
  
  # These are the indices we will use to compute the velocity. We add
  # a half scale here to avoid boundary issues. We'll subtract it out later.
  # See below for more details
  middle_I = np.array(np.arange(start_index, end_index, 1) + half_scale,
                      dtype='int32')
  # @MichaelCurrie: Wouldn't this make more sense?
  #middle_I = np.arange(start_index, end_index + half_scale, 1) + half_scale  
  
  """
     Our start_index frame can only have one valid start frame (frame 0)
     afterwards it is invalid. In other words, if frame 0 is not good, we
     can't check frame -1, or -2.
  
     However, in general I'd prefer to avoid doing some check on the bounds
     of the frames, i.e. for looking at starting frames, I'd like to avoid
     checking if the frame value is 0 or greater.
  
     To get around this we'll pad with bad values (which we'll never match)
     then shift the final indices. In this way, we can check these "frames",
     as they will have positive indices.
  
     e.g.
       scale = 5
       half_scale = 2
  
     This means the first frame in which we could possibly get a valid
     velocity is frame 2, computed using frames 0 and 4
  
  
     F   F   F T T  <- example good_frames_mask_padded values
             0 1 2  <- true indices (frame numbers)
     0   1   2 3 4  <- temporary indices
  
     NOTE: Now if frame 0 is bad, we can go left by half_scale + 1 to temp
     index 1 (frame -1) or even further to temp_index 0 (frame -2). we'll
     never use those values however because below we state that the values
     at those indices are bad (see good_frames_mask_padded)
  
  """
  
  # This tells us whether each value is useable or not for velocity
  # Better to do this out of the loop.
  # For real indices (frames 1:num_frames), we rely on whether or not the
  # mean position is NaN, for fake padding frames they can never be good so we
  # set them to be false
  stub_mask = np.zeros(half_scale, dtype=bool)
  good_frames_mask_padded = \
    np.concatenate((stub_mask, good_frames_mask, stub_mask))
  
  # These will be the final indices from which we estimate the velocity.
  # i.e. delta_position(I) = position(right_indices(I)) - 
  #                          position(left_indices(I))
  left_I  = np.empty(len(middle_I), dtype='int32')
  right_I = np.empty(len(middle_I), dtype='int32')
  left_I.fill(np.NaN)
  right_I.fill(np.NaN)
  
  # Track which ends we haven't yet matched, for each of the middle_I's.
  # since we are loopering over each possible shift, we need to track 
  # whether valid ends have been found for each of the middle_I's.
  unmatched_left_mask  = np.ones(len(middle_I), dtype=bool)
  unmatched_right_mask = np.ones(len(middle_I), dtype=bool)
  
  # Instead of looping over each centered velocity, we loop over each possible
  # shift. A shift is valid if the frame of the shift is good, and we have yet
  # to encounter a match for that centered index
  for shift_size in range(half_scale, scale_minus_1):
      # We grab indices that are the appropriate distance from the current
      # value. If we have not yet found a bound on the given side, and the
      # index is valid, we keep it.
      left_indices_temp  = middle_I - shift_size
      right_indices_temp = middle_I + shift_size
      
      is_good_left_mask  = good_frames_mask_padded[left_indices_temp]
      is_good_right_mask = good_frames_mask_padded[right_indices_temp]
      
      use_left_mask      = unmatched_left_mask  & is_good_left_mask
      use_right_mask     = unmatched_right_mask & is_good_right_mask
      
      # Change only those left_I's to our newly shifted outwards 
      # left_indices_temp, that the use_left_mask says should be used.
      left_I[use_left_mask]   = left_indices_temp[use_left_mask]
      right_I[use_right_mask] = right_indices_temp[use_right_mask]
      
      # Flag the matched items as being matched
      unmatched_left_mask[use_left_mask]   = False
      unmatched_right_mask[use_right_mask] = False
  
  # Remove the offset used to pad the numbers (discussed above)
  left_I   -= half_scale
  right_I  -= half_scale
  middle_I -= half_scale
  
  # Filter down to usable values, in which both left and right are defined
  valid_indices_mask = ~np.isnan(left_I) & ~np.isnan(right_I)
  left_I    = left_I[valid_indices_mask]
  right_I   = right_I[valid_indices_mask]
  middle_I  = middle_I[valid_indices_mask]
  
  keep_mask = np.zeros(num_frames, dtype=bool)
  keep_mask[middle_I] = True

  # sum(keep_mask) should equal the number of valid velocity values
  # left_I and right_I should store just these valid velocity values
  assert sum(keep_mask) == np.shape(left_I)[0] == np.shape(right_I)[0]
  
  return keep_mask, left_I, right_I


def get_frames_per_sample(sample_time):
  """
    get_window_width:
      We require sampling_scale to be an odd integer
      We calculate the scale as a scalar multiple of FPS.  We require the 
      scalar multiple of FPS to be an ODD INTEGER.

      INPUT: sample_time: number of seconds to sample.
  """

  ostensive_sampling_scale = sample_time * config.FPS
  
  # We need sampling_scale to be an odd integer, so 
  # first we check if we already have an integer.
  if((ostensive_sampling_scale).is_integer()):
    # In this case ostensive_sampling_scale is an integer.  
    # But is it odd or even?
    if(ostensive_sampling_scale % 2 == 0): # EVEN so add one
      sampling_scale = ostensive_sampling_scale + 1
    else:                                  # ODD
      sampling_scale = ostensive_sampling_scale
  else:
    # Otherwise, ostensive_sampling_scale is not an integer, 
    # so take the nearest odd integerw
    sampling_scale_low  = np.floor(ostensive_sampling_scale)
    sampling_scale_high = np.ceil(ostensive_sampling_scale)
    if(sampling_scale_high % 2 == 0): 
      sampling_scale = sampling_scale_low
    else:
      sampling_scale = sampling_scale_high

  assert(sampling_scale.is_integer())
  return int(sampling_scale)


def compute_velocity(sx, sy, avg_body_angle, sample_time, ventral_mode=0):
  """
    compute_velocity:
      The velocity is computed not using the nearest values but values
      that are separated by a sufficient time (sample_time). 
      If the encountered values are not valid (i.e. NaNs), the width of 
      time is expanded up to a maximum of 2*sample_time (or technically, 
      1 sample_time in either direction)

      INPUT:
        sx, sy: Two numpy arrays of shape (p, n) where p is the size of the 
              partition of worm's 49 points, and n is the number of frames 
              in the video
              
        avg_body_angle: The angles between the mean of the first-order 
                        differences.  Should have shape (n).
        
        sample_time: Time over which to compute velocity, in seconds.
        
        ventral_mode: 0, 1, or 2, specifying that the ventral side is...
                        0 = unknown
                        1 = clockwise
                        2 = anticlockwise
      OUTPUT:
        Two numpy arrays of shape (n), for 
        speed and direction, respectively.
        
  """
  num_frames = np.shape(sx)[1]
  
  # We need to go from a time over which to compute the velocity 
  # to a # of samples. The # of samples should be odd.
  frames_per_sample = get_frames_per_sample(sample_time)
  
  # If we don't have enough frames to satisfy our sampling scale,
  # return with nothing.
  if(frames_per_sample > num_frames):
    # Create numpy arrays filled with NaNs
    speed = np.empty((1, num_frames))
    speed.fill(np.NaN)
    direction = np.empty((1, num_frames))
    direction.fill(np.NaN)
    return speed, direction

  # Compute the indices that we will use for computing the velocity. We
  # calculate the velocity roughly centered on each sample, but with a
  # considerable width between frames that smooths the velocity.
  good_frames_mask = ~np.isnan(avg_body_angle)
  keep_mask, left_I, right_I = h__getVelocityIndices(frames_per_sample, 
                                                     good_frames_mask)


  # Compute speed
  # --------------------------------------------------------

  # Centroid of the current skeletal segment, frame-by-frame:
  x_mean = np.mean(sx, 1)
  y_mean = np.mean(sy, 1)
  
  dX  = x_mean[right_I] - x_mean[left_I]
  dY  = y_mean[right_I] - y_mean[left_I]
  
  distance = np.sqrt(dX**2 + dY**2)
  time     = (right_I - left_I) / config.FPS
  
  speed    = np.empty((1, num_frames))
  speed.fill(np.NaN)
  speed[keep_mask] = distance / time

  
  # Compute angular speed (Formally known as direction :/)
  # --------------------------------------------------------
  angular_speed = np.empty((1, num_frames))
  angular_speed.fill(np.NaN)
  angular_speed[keep_mask] = h__computeAngularSpeed(sx, sy,
                                                    left_I, right_I,
                                                    ventral_mode)
  
  # Sign the speed
  #   We want to know how the worm's movement direction compares 
  #   to the average angle it had (apparently at the start)
  motion_direction = np.empty((1, num_frames))
  motion_direction.fill(np.NaN)
  motion_direction[keep_mask] = np.degrees(np.arctan2(dY, dX))
  
  # This recentres the definition, as we are really just concerned
  # with the change, not with the actual value
  body_direction = np.empty((1, num_frames))
  body_direction.fill(np.NaN)
  body_direction[keep_mask] = motion_direction[keep_mask] \
                              - avg_body_angle[left_I]
  
  # Force all angles to be within -pi and pi
  body_direction = (body_direction + 180) % (360) - 180
  
  # Sign speed[i] as negative if the angle 
  # body_direction[i] lies in Q2 or Q3
  speed[abs(body_direction) > 90] = -speed[abs(body_direction) > 90]
  
  # (Added for wormPathCurvature)
  # Sign motion_direction[i] as negative if the angle 
  # body_direction[i] lies in Q3 or Q4
  motion_direction[body_direction < 0] = -motion_direction[body_direction < 0]
  if(ventral_mode == 2): # i.e. if ventral side is anticlockwise:
     motion_direction = -motion_direction 
  
  # Organize the velocity.
  #-----------------------------------------------------------
  velocity = {'speed': speed, 'angular_speed': angular_speed}
    
  return velocity

  # @MichaelCurrie: shouldn't we also return these?  Otherwise, why
  # did we both to calculate them?
  #            'body_direction': body_direction,
  #            'motion_direction': motion_direction




def get_worm_velocity(nw, ventral_mode=0):
  """
    get_worm_velocity:
      Compute the worm velocity (speed & direction) at the
      head-tip/head/midbody/tail/tail-tip
   
    INPUTS: nw: a NormalizedWorm instance
            ventral_mode: the ventral side mode:
              0 = unknown
              1 = clockwise
              2 = anticlockwise
    OUTPUT: a two-tiered dictionary containing
              the partitions in the first tier:
                headTip = the tip of the head (1/12 the worm at 0.25s)
                head    = the head (1/6 the worm at 0.5s)
                midbody = the midbody (2/6 the worm at 0.5s)
                tail    = the tail (1/6 the worm at 0.5s)
                tailTip = the tip of the tail (1/12 the worm at 0.25s)
              and 'speed' and 'direction' in the second tier.

  """

  # Let's use some partitions.  
  # NOTE: head_tip and tail_tip overlap head and tail, respectively, and
  #       this set of partitions does not cover the neck and hips
  partition_keys = ['head_tip', 'head', 'midbody', 'tail', 'tail_tip']
  
  avg_body_angle = get_partition_angles(nw, partition_key='body',
                                        data_key='skeletons', 
                                        head_to_tail=True)  # reverse
  
  sample_time_values = \
    {
      'head_tip': config.TIP_DIFF,
      'head':     config.BODY_DIFF,
      'midbody':  config.BODY_DIFF,
      'tail':     config.BODY_DIFF,
      'tail_tip': config.TIP_DIFF
    }  

  # Set up a dictionary to store the velocity for each partition
  velocity = {}
  
  for partition_key in partition_keys:
    x, y = nw.get_partition(partition_key, 'skeletons', True)
    speed, direction = compute_velocity(x, y, 
                                        avg_body_angle, 
                                        sample_time_values[partition_key], 
                                        ventral_mode)
    velocity[partition_key] = {'speed': speed, 'direction': direction}
  
  return velocity




























"""
###############################################################################
###############################################################################
###############################################################################
###############################################################################
"""


def get_bends(nw):
  """
    INPUT: A NormalizedWorm object
    OUTPUT: A dictionary containing bends data
  
  """
  # We care only about the head, neck, midbody, hips and tail 
  # (i.e. the 'normal' way to partition the worm)
  p = nw.get_partition_subset('normal')
  
  bends = {}
  
  for partition_key in p.keys():
    # retrieve the part of the worm we are currently looking at:
    bend_angles = nw.get_partition(partition_key, 'angles')
    
    bend_metrics_dict = {}
    # shape = (n):
    bend_metrics_dict['mean'] = np.nanmean(a=bend_angles, axis = 0) 
    bend_metrics_dict['std_dev'] = np.nanstd(a=bend_angles, axis = 0)
    
    # Sign the standard deviation (to provide the bend's 
    # dorsal/ventral orientation):
    
    # First find all entries where the mean is negative
    mask = np.ma.masked_where(condition=bend_metrics_dict['mean'] < 0,
                              a=bend_metrics_dict['mean']).mask
    # Now create a numpy array of -1 where the mask is True and 1 otherwise
    sign_array = -np.ones(np.shape(mask)) * mask + \
                 np.ones(np.shape(mask)) * (~mask)
    # Finally, multiply the std_dev array by our sign_array
    bend_metrics_dict['std_dev'] = bend_metrics_dict['std_dev'] * sign_array
    
    bends[partition_key] = bend_metrics_dict
    
  # The final bends dictionary now contains a mean and standard 
  # deviation for the head, neck, midbody, hips and tail.
  return bends


def get_amplitude_and_wavelength(theta_d, sx, sy, worm_lengths):
  """
  
  
     Inputs
     =======================================================================
     theta_d      : worm orientation based on fitting to an ellipse, in
                     degrees
     sx           : [49 x num_frames]
     sy           : [49 x num_frames]
     worm_lengths : [1 x num_frames], total length of each worm
  
  
     Output: A dictionary with three elements:
     =======================================================================
     amplitude    :
         .max       - [1 x num_frames] max y deviation after rotating major axis to x-axis
         .ratio     - [1 x num_frames] ratio of y-deviations (+y and -y) with worm centered
                      on the y-axis, ratio is computed to be less than 1
     wavelength   :
         .primary   - [1 x num_frames]
         .secondary - [1 x num_frames] this might not always be valid, even 
                       when the primary wavelength is defined
     track_length  : [1 x num_frames]
  
     
     Old Name: getAmpWavelength.m
     TODO: This function was missing from some of the original code
     distributions. I need to make sure I upload it.
  
  
     Nature Methods Description
     =======================================================================
     Amplitude. 
     ------------------
     Worm amplitude is expressed in two forms: a) the maximum
     amplitude found along the worm body and, b) the ratio of the maximum
     amplitudes found on opposing sides of the worm body (wherein the smaller of
     these two amplitudes is used as the numerator). The formula and code originate
     from the publication “An automated system for measuring parameters of
     nematode sinusoidal movement”6.
     The worm skeleton is rotated to the horizontal axis using the orientation of the
     equivalent ellipse and the skeleton’s centroid is positioned at the origin. The
     maximum amplitude is defined as the maximum y coordinate minus the minimum
     y coordinate. The amplitude ratio is defined as the maximum positive y coordinate
     divided by the absolute value of the minimum negative y coordinate. If the
     amplitude ratio is greater than 1, we use its reciprocal.
  
     Wavelength
     ------------------------
     Wavelength. The worm’s primary and secondary wavelength are computed by
     treating the worm’s skeleton as a periodic signal. The formula and code
     originate from the publication “An automated system for measuring
     parameters of nematode sinusoidal movement”6. The worm’s skeleton is
     rotated as described above for the amplitude. If there are any
     overlapping skeleton points (the skeleton’s x coordinates are not
     monotonically increasing or decreasing in sequence -- e.g., the worm is
     in an S shape) then the shape is rejected, otherwise the Fourier
     transform computed. The primary wavelength is the wavelength associated
     with the largest peak in the transformed data. The secondary wavelength
     is computed as the wavelength associated with the second largest
     amplitude (as long as it exceeds half the amplitude of the primary
     wavelength). The wavelength is capped at twice the value of the worm’s
     length. In other words, a worm can never achieve a wavelength more than
     double its size.
  
     Tracklength
     -----------------------------
     Track Length. The worm’s track length is the range of the skeleton’s
     horizontal projection (as opposed to the skeleton’s arc length) after
     rotating the worm to align it with the horizontal axis. The formula and
     code originate from the publication “An automated system for measuring
     parameters of nematode sinusoidal movement”.
  
  
     Code based on:
     ------------------------------------------------
     BMC Genetics, 2005
     C.J. Cronin, J.E. Mendel, S. Mukhtar, Young-Mee Kim, R.C. Stirb, J. Bruck,
     P.W. Sternberg
     "An automated system for measuring parameters of nematode
     sinusoidal movement" BMC Genetics 2005, 6:5
  
  """
  amp_wave_track = \
    collections.namedtuple('amp_wave_track', 
                           ['amplitude', 'wavelength', 'track_length'])
  amp_wave_track.amplitude = 'yay1'
  amp_wave_track.wavelength = 'yay2'
  amp_wave_track.track_length = 'yay3'

  onw = nw.re_orient_and_centre()  


  return amp_wave_track


def get_eccentricity_and_orientation(contour_x, contour_y):
  """
    get_eccentricity   
   
      [eccentricity, orientation] = seg_worm.feature_helpers.posture.getEccentricity(xOutline, yOutline, gridSize)
   
      Given x and y coordinates of the outline of a region of interest, fill
      the outline with a grid of evenly spaced points and use these points in
      a center of mass calculation to calculate the eccentricity and
      orientation of the equivalent ellipse.
   
      Placing points in the contour is a well known computer science problem
      known as the Point-in-Polygon problem.
   
      http://en.wikipedia.org/wiki/Point_in_polygon
   
      This function became a lot more complicated in an attempt to make it 
      go much faster. The complication comes from the simplication that can
      be made when the worm doesn't bend back on itself at all.
   
   
      OldName: getEccentricity.m
    
   
      Inputs:
      =======================================================================
      xOutline : [96 x num_frames] The x coordinates of the contour. In particular the contour
                  starts at the head and goes to the tail and then back to
                  the head (although no points are redundant)
      yOutline : [96 x num_frames]  The y coordinates of the contour "  "
      
      N_ECCENTRICITY (a constant from config.py):
                 (scalar) The # of points to place in the long dimension. More points
                 gives a more accurate estimate of the ellipse but increases
                 the calculation time.
   
      Outputs: a namedtuple containing:
      =======================================================================
      eccentricity - [1 x num_frames] The eccentricity of the equivalent ellipse
      orientation  - [1 x num_frames] The orientation angle of the equivalent ellipse
   
      Nature Methods Description
      =======================================================================
      Eccentricity. 
      ------------------
      The eccentricity of the worm’s posture is measured using
      the eccentricity of an equivalent ellipse to the worm’s filled contour.
      The orientation of the major axis for the equivalent ellipse is used in
      computing the amplitude, wavelength, and track length (described
      below).
   
      Status
      =======================================================================
      The code below is finished although I want to break it up into smaller
      functions. I also need to submit a bug report for the inpoly FEX code.

  Translation of: SegwormMatlabClasses / 
  +seg_worm / +feature_helpers / +posture / getEccentricity.m
  """
  # TODO: translate this function from Jim's code
  EccentricityAndOrientation = \
    collections.namedtuple('EccentricityAndOrientation', 
                           ['eccentricity', 'orientation'])
                           
  EccentricityAndOrientation.eccentricity = 'eccentricity example'
  EccentricityAndOrientation.wavelength = 'wavelength example'

  return EccentricityAndOrientation
  
  
  
  
  
  
  
