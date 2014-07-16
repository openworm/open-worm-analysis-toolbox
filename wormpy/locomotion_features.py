# -*- coding: utf-8 -*-
"""
Insert description here
"""

import numpy as np

from . import Events
from . import feature_helpers as fh
from . import config

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
    # TODO: Rename partition and data keys to have better names
    partition_keys = ['head_tip', 'head', 'midbody', 'tail', 'tail_tip']

    data_keys = list(partition_keys)

    if(config.MIMIC_OLD_BEHAVIOUR):
        data_keys[2] = 'old_midbody_velocity'

    avg_body_angle = fh.get_partition_angles(nw, partition_key='body',
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

    for partition_key, data_key in zip(partition_keys, data_keys):
        x, y = nw.get_partition(data_key, 'skeletons', True)

        speed, direction = fh.compute_velocity(x, y,
                                            avg_body_angle,
                                            sample_time_values[partition_key],
                                            ventral_mode)[0:2]
        velocity[partition_key] = {'speed': speed, 'direction': direction}

    return velocity
    
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
    skeleton_lengths = fh.interpolate_with_threshold(skeleton_lengths,
                                                     config.MOTION_CODES_LONGEST_NAN_RUN_TO_INTERPOLATE)

    #===================================
    # SPACE CONSTRAINTS
    # Make the speed and distance thresholds a fixed proportion of the
    # worm's length at the given frame:
    worm_speed_threshold = skeleton_lengths * config.SPEED_THRESHOLD_PCT
    worm_distance_threshold = skeleton_lengths * config.DISTANCE_THRSHOLD_PCT
    worm_pause_threshold = skeleton_lengths * config.PAUSE_THRESHOLD_PCT

    # Minimum speed and distance required for movement to
    # be considered "forward"
    min_forward_speed = worm_speed_threshold
    min_forward_distance = worm_distance_threshold

    # Minimum speed and distance required for movement to
    # be considered "backward"
    max_backward_speed = -worm_speed_threshold
    min_backward_distance = worm_distance_threshold

    # Boundaries between which the movement would be considered "paused"
    min_paused_speed = -worm_pause_threshold
    max_paused_speed = worm_pause_threshold

    # Note that there is no maximum forward speed nor minimum backward speed.
    frame_values = {'forward': 1, 'backward': -1, 'paused': 0}
    min_speeds = {'forward': min_forward_speed,
                  'backward': None,
                  'paused': min_paused_speed}
    max_speeds = {'forward': None,
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
        ef = Events.EventFinder()

        # "Space and time" constraints
        ef.min_distance_threshold = min_distance[motion_type]
        ef.max_distance_threshold = None  # we are not constraining max dist
        ef.min_speed_threshold = min_speeds[motion_type]
        ef.max_speed_threshold = max_speeds[motion_type]

        # "Time" constraints
        ef.min_frames_threshold = min_frames_threshold
        ef.max_inter_frames_threshold = max_interframes_threshold

        event_list = ef.get_events(midbody_speed, distance_per_frame)

        # Start at 1, not 7 for forward
        # What happens from 1900 to 1907

        # Obtain only events entirely before the num_frames intervals
        event_mask = event_list.get_event_mask(num_frames)

        # Take the start and stop indices and convert them to the structure
        # used in the feature files
        m_event = Events.EventListWithFeatures(event_list,
                                               distance_per_frame,
                                               compute_distance_during_event=True)

        # Record this motion_type to our all_events_dict!
        # Assign event type to relevant frames of all_events_dict['mode']
        all_events_dict['mode'][event_mask] = frame_values[motion_type]
        all_events_dict[motion_type] = m_event

    return all_events_dict