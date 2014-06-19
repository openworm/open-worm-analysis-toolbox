# -*- coding: utf-8 -*-
"""
  feature_helpers.py
  
  @authors: @MichaelCurrie, @JimHokanson
  
  Some helper functions that assist in the calculation of the attributes of
  WormFeatures
  
"""

from __future__ import division #distance/time compute_velocity

import warnings
import numpy as np
from itertools import groupby

#np.seterr(all='raise')           # DEBUG

import csv
from . import utils
import collections
from wormpy import config
from .EventFinder import EventFinder
from .EventFinder import EventListForOutput

import matplotlib.pyplot as plt

import pdb

__ALL__ = ['get_motion_codes',                  # for locomotion
           'get_worm_velocity',                 # for locomotion
           'get_bends',                         # for posture
           'get_amplitude_and_wavelength',      # for posture
           'get_eccentricity_and_orientation']  # for posture

def write_to_CSV(data_dict, filename):
  """
  Writes data to a CSV file, by saving it to the directory os.getcwd()
  
  Parameters
  ---------------------------------------
  data_dict: a dictionary of 1-dim ndarrays of dtype=float
    What is to be written to the file.  data.keys() provide the headers,
    and each column in turn is provided by the value for that key
  filename: string
    Name of file to be saved (not including the '.csv' part of the name)
  
  """
  csv_file = open(filename+'.csv', 'w')
  writer = csv.writer(csv_file, lineterminator='\n')

  # The first row of the file is the keys
  writer.writerow(list(data_dict.keys()))

  # Find the maximum number of rows across all our columns:
  max_rows = max([len(x) for x in list(data_dict.values())])

  # Combine all the dictionary entries so we can write them
  # row-by-row.
  columns_to_write = []
  for column_key in data_dict.keys():
    column = list(data_dict[column_key])
    # Create a mask that shows True for any unused "rows"
    m = np.concatenate([np.zeros(len(column), dtype=bool), 
                        np.ones(max_rows-len(column), dtype=bool)])
    # Create a masked array of size max_rows with unused entries masked
    column_masked = np.ma.array(np.resize(column, max_rows), mask=m)
    # Convert the masked array to an ndarray with the masked values
    # changed to NaNs
    column_masked = column_masked.filled(np.NaN)
    # Append this ndarray to our list
    columns_to_write.append(column_masked)
  
  # Combine each column's entries into an ndarray
  data_ndarray = np.vstack(columns_to_write)

  # We need the transpose so the individual data lists become transposed
  # to columns
  data_ndarray = data_ndarray.transpose()
  
  # We need in the form of nested sequences to satisfy csv.writer
  rows_to_write = data_ndarray.tolist()

  for row in rows_to_write:
    writer.writerow(list(row))
    
  csv_file.close()


"""----------------------------------------------------
    motion codes:
    
    
"""

"""  
  # example array  
  a = np.array([10, 12, 15, np.NaN, 17, \
                np.NaN, np.NaN, np.NaN, -5], dtype='float')
  
  a2 = interpolate_with_threshold(a, 5)
  
  print(a)
  print(a2)
  
"""

def interpolate_with_threshold(array, threshold=None, make_copy=True, extrapolate=False):
  """
  Linearly interpolate a numpy array along one dimension but only 
  for missing data n frames from a valid data point.  That is, 
  if there are too many contiguous missing data points, none of 
  those points get interpolated.

  Parameters
  ---------------------------------------
  array: 1-dimensional numpy array
    The array to be interpolated
  threshold: int
    The maximum size of a contiguous set of missing data points
    that gets interpolated.  Sets larger than this are left as NaNs.
    If threshold is set to NaN then all points are interpolated.
  
  Returns
  ---------------------------------------
  numpy array with the values interpolated
  
  """

  #EXTRAPOLATION:
  #http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
  #

  """
  # (SKIP THIS, THIS IS FOR THE N-DIMENSIONAL CASE WHICH WE
  # HAVE NOT IMPLEMENTED YET)
    # Check that any frames with NaN in at least one dimension must
    # have it in all:
    frames_with_at_least_one_NaN = np.all(np.isnan(array), frame_dimension)
    frames_with_no_NaNs          = np.all(~np.isnan(array), frame_dimension)
    # check that each frame is either True for one of these arrays or 
    # the other but not both.
    assert(np.logical_xor(frames_with_at_least_one_NaN, frames_with_no_NaNs))
    frame_dropped = frames_with_at_least_one_NaN

  """
  
  assert(threshold == None or threshold >= 0)
  
  if(threshold == 0):  # everything gets left as NaN
    return array
  
  # Say array = [10, 12, 15, nan, 17, nan, nan, nan, -5]
  # Then np.isnan(array) = 
  # [False, False, False, True, False True, True, True, False]
  # Let's obtain the "x-coordinates" of the NaN entries.
  # e.g. [3, 5, 6, 7]
  x = np.flatnonzero(np.isnan(array))
  
  # (If we weren't using a threshold and just interpolating all NaNs, 
  # we could skip the next four lines.)
  if(threshold != None):
    # Group these together using a fancy trick from 
    # http://stackoverflow.com/questions/2154249/, since
    # the lambda function x:x[0]-x[1] on an enumerated list will
    # group consecutive integers together
    # e.g. [[(0, 3)], [(1, 5), (2, 6), (3, 7)]]
    x_grouped = [list(group) for key, group in groupby(enumerate(x), 
                                                       lambda i:i[0]-i[1])]
    
    # We want to know the first element from each "run", and the run's length
    # e.g. [(3, 1), (5, 3)]
    x_runs = [(i[0][1], len(i)) for i in x_grouped]
    
    # We need only interpolate on runs of length <= threshold
    # e.g. if threshold = 2, then we have only [(3, 1)]
    x_runs = [i for i in x_runs if i[1] <= threshold]
  
    # now expand the remaining runs
    # e.g. if threshold was 5, then x_runs would be [(3,1), (5,3)] so
    #      x would be [3, 5, 6, 7]
    # this give us the x-coordinates of the values to be interpolated:
    x = np.concatenate([(i[0] + list(range(i[1]))) for i in x_runs])
  
  # The x-coordinates of the data points, must be increasing.
  xp = np.flatnonzero(~np.isnan(array))
  # The y-coordinates of the data points, same length as xp
  yp = array[~np.isnan(array)]

  if make_copy:
    # Use a new array so we don't modify the original array passed to us
    new_array = np.copy(array)
  else:
    new_array = array
  
  if extrapolate:
    pass # :/  Might need to use scipy
  
  # Place the interpolated values into the array
  new_array[x] = np.interp(x, xp, yp)  
  
  return new_array






def get_motion_codes(midbody_speed, skeleton_lengths):
  """ 
  Calculate motion codes of the locomotion events

  A locomotion feature.

  See feature description at 
    /documentation/Yemini%20Supplemental%20Data/Locomotion.md

  Parameters
  ---------------------------------------
  midbody_speed: numpy array 1 x n_frames
    from locomotion.velocity.midbody.speed / config.FPS
  skeleton_lengths: numpy array 1 x n_frames

  Returns
  ---------------------------------------
  The locomotion events; a dict (called locally all_events_dict) 
  with event fields:
    forward  - (event) forward locomotion
    paused   - (event) no locomotion (the worm is paused)
    backward - (event) backward locomotion
    mode     = [1 x num_frames] the locomotion mode:
               -1 = backward locomotion
                0 = no locomotion (the worm is paused)
                1 = forward locomotion

  Notes
  ---------------------------------------
  Formerly +seg_worm / +features / @locomotion / getWormMotionCodes.m

  """
  # Initialize the worm speed and video frames.
  num_frames = len(midbody_speed)
  
  # Compute the midbody's "instantaneous" distance travelled at each frame, 
  # distance per second / (frames per second) = distance per frame
  distance_per_frame = abs(midbody_speed / config.FPS)


  #  Interpolate the missing lengths.
  skeleton_lengths = interpolate_with_threshold(skeleton_lengths, 
                 config.MOTION_CODES_LONGEST_NAN_RUN_TO_INTERPOLATE)

  #===================================
  # SPACE CONSTRAINTS
  # Make the speed and distance thresholds a fixed proportion of the 
  # worm's length at the given frame:
  worm_speed_threshold    = skeleton_lengths * config.SPEED_THRESHOLD_PCT
  worm_distance_threshold = skeleton_lengths * config.DISTANCE_THRSHOLD_PCT 
  worm_pause_threshold    = skeleton_lengths * config.PAUSE_THRESHOLD_PCT 
  
  # Minimum speed and distance required for movement to 
  # be considered "forward"
  min_forward_speed    = worm_speed_threshold
  min_forward_distance = worm_distance_threshold
  
  # Minimum speed and distance required for movement to 
  # be considered "backward"
  max_backward_speed    = -worm_speed_threshold
  min_backward_distance = worm_distance_threshold
  
  # Boundaries between which the movement would be considered "paused"
  min_paused_speed     = -worm_pause_threshold
  max_paused_speed     = worm_pause_threshold

  # Note that there is no maximum forward speed nor minimum backward speed.
  frame_values = {'forward': 1, 'backward': -1, 'paused': 0}
  min_speeds   = {'forward': min_forward_speed, 
                  'backward': None, 
                  'paused': min_paused_speed}
  max_speeds   = {'forward': None, 
                  'backward': max_backward_speed, 
                  'paused': max_paused_speed}
  min_distance = {'forward': min_forward_distance, 
                  'backward': min_backward_distance, 
                  'paused': None}

  #===================================
  # TIME CONSTRAINTS
  # The minimum number of frames an event had to be taking place for
  # to be considered a legitimate event
  min_frames_threshold = \
    config.FPS * config.EVENT_MIN_FRAMES_THRESHOLD
  # Maximum number of contiguous contradicting frames within the event
  # before the event is considered to be over.
  max_interframes_threshold = \
    config.FPS * config.EVENT_MAX_INTER_FRAMES_THRESHOLD
  
  # This is the dictionary this function will return.  Keys will be:
  # forward
  # backward
  # paused
  # mode
  all_events_dict = {}

  # Start with a blank numpy array, full of NaNs: 
  all_events_dict['mode'] = np.empty(num_frames, dtype='float') * np.NaN

  

  for motion_type in frame_values:
    # We will use EventFinder to determine when the 
    # event type "motion_type" occurred
    ef = EventFinder()

    # "Space and time" constraints
    ef.min_distance_threshold        = min_distance[motion_type]
    ef.max_distance_threshold        = None # we are not constraining max dist
    ef.min_speed_threshold           = min_speeds[motion_type]
    ef.max_speed_threshold           = max_speeds[motion_type]

    # "Time" constraints
    ef.min_frames_threshold          = min_frames_threshold
    ef.max_inter_frames_threshold    = max_interframes_threshold
    
    event_list = ef.get_events(midbody_speed, distance_per_frame)

    #Start at 1, not 7 for forward
    #What happens from 1900 to 1907

    # Obtain only events entirely before the num_frames intervals
    event_mask = event_list.get_event_mask(num_frames)

    # Take the start and stop indices and convert them to the structure
    # used in the feature files
    m_event = EventListForOutput(event_list, distance_per_frame, True)

    # Record this motion_type to our all_events_dict!
    # Assign event type to relevant frames of all_events_dict['mode']
    all_events_dict['mode'][event_mask] = frame_values[motion_type]
    all_events_dict[motion_type] = m_event
  
  return all_events_dict




"""----------------------------------------------------
    velocity:
"""

def get_angles(segment_x, segment_y, head_to_tail=False):
  """ Obtain the "angle" of a subset of the 49 points
      of a worm, for each frame.
      
  Parameters
  ---------------------------------------
  segment_x, segment_y: numpy arrays of shape (p,n) where 
    p is the size of the partition of the 49 points
    n is the number of frames in the video
  head_to_tail: bool
  True means the worm points are order head to tail.
    
  Returns
  ---------------------------------------
  A numpy array of shape (n) and stores the worm body's "angle" 
  (in degrees) for each frame of video

  """
  
  if(not head_to_tail):
    # reverse the worm points so we go from tail to head
    segment_x = segment_x[::-1,:]
    segment_y = segment_y[::-1,:]



  # Diff calculates each point's difference between the segment's points
  # then we take the mean of these differences for each frame
  with warnings.catch_warnings():  #ignore mean of empty slice from np.nanmean
    #This warning arises when all values are NaN in an array
    #This occurs in not for all values but only for some rows, other rows
    #may be a mix of valid and NaN values
    warnings.simplefilter("ignore")
    #with np.errstate(invalid='ignore'):  #doesn't work, numpy warning
    #is not of the invalid type, just says "mean of empty slice"
    average_diff_x = np.nanmean(np.diff(segment_x, n=1, axis=0), axis=0) # shape (n)
    average_diff_y = np.nanmean(np.diff(segment_y, n=1, axis=0), axis=0) # shape (n)
    
  # angles has shape (n) and stores the worm body's "angle"
  # for each frame of video
  angles = np.degrees(np.arctan2(average_diff_y, average_diff_x))

  return angles


def get_partition_angles(nw, partition_key, data_key='skeletons',
                         head_to_tail=False):
  """ Obtain the "angle" of a subset of the 49 points of a worm for each 
      frame
  
      INPUT: head_to_tail=True means the worm points are order head to tail.
    
      OUTPUT: A numpy array of shape (n) and stores the worm body's "angle" 
              (in degrees) for each frame of video
    
  """
  # the shape of both segment_x and segment_y is (partition length, n)
  segment_x, segment_y = nw.get_partition(partition_key, data_key, True)  

  return get_angles(segment_x, segment_y, head_to_tail)


def h__computeAngularSpeed(segment_x, segment_y, 
                           left_I, right_I, ventral_mode):
  """ INPUT: 
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

  angular_speed = point_angle_d[right_I] - point_angle_d[left_I]
  
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
  
  For each point, we calculate the velocity using frames prior to and following
  a frame. Given that some frames are not valid (have NaN), we move the index
  backward (prior frame) or forward (following frame), essentially slightly
  widening the time frame over which the velocity is computed.
  
  This function determines what the indices are that each frame will use to 
  calculate the velocity at that frame. For example, at frame 5 we might decide
  to use frames 2 and 8.

  Parameters:
  -----------
  frames_per_sample : int
    Our sample scale, in frames. The integer must be odd.
    
  good_frames_mask : 
    Shape (num_frames), false if underlying angle is NaN
  
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
  
  """
  Approach, rather than interating over each frame, we iterate over the
  possible shifts. Since this tends to be significantly less than the # of
  frames, we save a bit of time in the execution.

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
  # numpy integer arrays cannot accept NaN, which is a float concept, but
  # filling them with NaN fills them with the largest negative number
  # possible, -2**31.  We can easily filter for this later.
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
  for shift_size in range(half_scale, frames_per_sample):
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
  # We have to avoid decrementing negative numbers because our negative
  # number is our NaN proxy and it's already as negative as it can be
  # without wrapping back up to positive again
  left_I[left_I>0]   -= half_scale
  right_I[right_I>0] -= half_scale
  middle_I           -= half_scale
  
  # Filter down to usable values, in which both left and right are defined
  # Remember than np.NaN is not valid number for integer numpy arrays
  # so instead of checking for which entries are NaN, we check for 
  # which entries are negative, since no array indices can be 
  # negative!
  valid_indices_mask = (left_I>=0) & (right_I>=0)
  left_I    = left_I[valid_indices_mask]
  right_I   = right_I[valid_indices_mask]
  middle_I  = middle_I[valid_indices_mask]
  
  keep_mask = np.zeros(num_frames, dtype=bool)
  keep_mask[middle_I] = True

  # sum(keep_mask) should equal the number of valid velocity values
  # left_I and right_I should store just these valid velocity values
  assert sum(keep_mask) == len(left_I) == len(right_I)
  assert all(left_I>=0) and all(left_I<num_frames)
  assert all(right_I>=0) and all(right_I<num_frames)  
  return keep_mask, left_I, right_I


def get_frames_per_sample(sample_time):
  """
  
  Matlab code: getWindowWidthAsInteger
    
  
    get_window_width:
      We require sampling_scale to be an odd integer
      We calculate the scale as a scalar multiple of FPS.  We require the 
      scalar multiple of FPS to be an ODD INTEGER.

      INPUT: sample_time: number of seconds to sample.
  """

  ostensive_sampling_scale = sample_time * config.FPS
  
  #Code would be better as: (Matlab code shown)
  #------------------------------------------------
  #half_scale = round(window_width_as_samples/2);
  #window_width_integer = 2*half_scale + 1;
  
  
  
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
  The velocity is computed not using the nearest values but values
  that are separated by a sufficient time (sample_time). 
  If the encountered values are not valid (i.e. NaNs), the width of 
  time is expanded up to a maximum of 2*sample_time (or technically, 
  1 sample_time in either direction)

  Parameters
  ----------
  sx, sy: Two numpy arrays of shape (p, n) where p is the size of the 
        partition of worm's 49 points, and n is the number of frames 
        in the video
    The worm skeleton's x and y coordinates, respectively.
              
  avg_body_angle: 1-dimensional numpy array of floats, of size n.
    The angles between the mean of the first-order differences.
  
  sample_time: int
    Time over which to compute velocity, in seconds.
  
  ventral_mode: int
    0, 1, or 2, specifying that the ventral side is...
      0 = unknown
      1 = clockwise
      2 = anticlockwise

  Returns
  -------
  Three numpy arrays of shape (n), speed, angular_speed, motion_direction
        
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


  #NOTE: In Matlab this is done only over a certain range of the body
  #TODO: Who calls this, can we just hard code mimic old behavior here???

  x_mean = np.mean(sx, 0)
  y_mean = np.mean(sy, 0)
  
  dX  = x_mean[right_I] - x_mean[left_I]
  dY  = y_mean[right_I] - y_mean[left_I]
    
  distance = np.sqrt(dX**2 + dY**2)
  time     = (right_I - left_I) / config.FPS
  
  speed    = np.empty((num_frames))
  speed.fill(np.NaN)
  speed[keep_mask] = distance / time
  
  # Compute angular speed (Formally known as direction :/)
  # --------------------------------------------------------
  angular_speed = np.empty((num_frames))
  angular_speed.fill(np.NaN)
  angular_speed[keep_mask] = h__computeAngularSpeed(sx, sy,left_I, right_I,
                                                    ventral_mode)

  # Sign the speed.
  #   We want to know how the worm's movement direction compares 
  #   to the average angle it had (apparently at the start)
  motion_direction = np.empty((num_frames))
  motion_direction.fill(np.NaN)
  motion_direction[keep_mask] = np.degrees(np.arctan2(dY, dX))
  
  # This recentres the definition, as we are really just concerned
  # with the change, not with the actual value
  body_direction = np.empty((num_frames))
  body_direction.fill(np.NaN)
  body_direction[keep_mask] = motion_direction[keep_mask] - \
                              avg_body_angle[left_I]
  
  # Force all angles to be within -pi and pi
  with np.errstate(invalid='ignore'):
    body_direction = (body_direction + 180) % (360) - 180
  
  # Sign speed[i] as negative if the angle
  # body_direction[i] lies in Q2 or Q3
  with np.errstate(invalid='ignore'):
    speed[abs(body_direction) > 90] = -speed[abs(body_direction) > 90]
    
  # (Added for wormPathCurvature)
  # Sign motion_direction[i] as negative if the angle 
  # body_direction[i] lies in Q3 or Q4
    
  with np.errstate(invalid='ignore'):
    motion_direction[body_direction < 0] = -motion_direction[body_direction<0]
    
  if(ventral_mode == 2): # i.e. if ventral side is anticlockwise:
     motion_direction = -motion_direction 
    
  return speed, angular_speed, motion_direction

  # @MichaelCurrie: shouldn't we also return these?  Otherwise, why
  # did we both to calculate them?
  #            'body_direction': body_direction,




def get_worm_velocity(nw, ventral_mode=0):
  """
  
  This is for the 'velocity' locomotion feature. The helper function,
  'compute_velocity' is used elsewhere  
  
  Compute the worm velocity (speed & direction) at the
  head-tip/head/midbody/tail/tail-tip

  Parameters
  ----------------------------
  nw: a NormalizedWorm instance

  ventral_mode: int
    The ventral side mode:
      0 = unknown
      1 = clockwise
      2 = anticlockwise

  Returns
  ----------------------------
  A two-tiered dictionary containing the partitions 
  in the first tier:
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
                                        head_to_tail=False)  # reverse
  
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
    
   #import pdb
    #pdb.set_trace()      
    
    speed, direction = compute_velocity(x, y, 
                                        avg_body_angle, 
                                        sample_time_values[partition_key], 
                                        ventral_mode)[0:2]
    velocity[partition_key] = {'speed': speed, 'direction': direction}
  
  return velocity

  

