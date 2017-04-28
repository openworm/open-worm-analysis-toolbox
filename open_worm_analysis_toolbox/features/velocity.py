# -*- coding: utf-8 -*-
"""
Velocity calculation methods: used in locomotion and in path features

"""
from __future__ import division

import warnings
import numpy as np

__ALL__ = ['get_angles',
           'get_partition_angles',
           'h__computeAngularSpeed',
           'compute_speed',
           'get_frames_per_sample']


def get_angles(segment_x, segment_y, head_to_tail=False):
    """

    TODO: This might be better named as:
    get_average_angle_per_frame

    Obtain the average angle (see definition in output) for a subset of the
    worm for each frame.

    Parameters
    ----------
    segment_x, segment_y: numpy arrays of shape (p,n) where
          - 'p' is the size of the partition of the 49 points
          - 'n' is the number of frames in the video
          Values are typically (always) the skeleton x & y positions.
    head_to_tail: bool
        - if True
             - closer to tail are tips (where vector is tip - tail)
             - closer to head are tails
             1 2 3 4
             ======>   #This is supposed to be the vector
        - if False
             - closer to tail are tails
             - closer to head are tips
             1 2 3 4
             <======
        True means the worm points are ordered head to tail.

    Returns
    -------
    Angles in degrees. A numpy array of shape (n) and stores the worm body's
    "angle" (in degrees) for each frame of video. All angles for a given
    frame are averaged together. Angles are computed between sequential points
    and not smoothed.

    Implementation Notes
    --------------------
    JAH: I'm surprised this works given that there is relatively little control
    over the density of sampling of the skeleton (or segment_x and segment_y).
    As the spatial sampling changes, I would expect the results of this function
    to change drastically.

    """

    if not head_to_tail:
        # reverse the worm points so we go from tail to head
        segment_x = segment_x[::-1, :]
        segment_y = segment_y[::-1, :]

    # Diff calculates each point's difference between the segment's points
    # then we take the mean of these differences for each frame
    # ignore mean of empty slice from np.nanmean
    with warnings.catch_warnings():

        # TODO: Why do we need to filter if we are catching warnings?????
        #
        # This warning arises when all values are NaN in an array
        # This occurs in not for all values but only for some rows, other rows
        # may be a mix of valid and NaN values
        warnings.simplefilter("ignore")
        # with np.errstate(invalid='ignore'):  #doesn't work, numpy warning
        # is not of the invalid type, just says "mean of empty slice"
        average_diff_x = np.nanmean(
            np.diff(segment_x, n=1, axis=0), axis=0)  # shape (n)
        average_diff_y = np.nanmean(
            np.diff(segment_y, n=1, axis=0), axis=0)  # shape (n)

    # angles has shape (n) and stores the worm body's "angle"
    # for each frame of video
    angles = np.degrees(np.arctan2(average_diff_y, average_diff_x))

    return angles


def get_partition_angles(nw, partition_key, data_key='skeleton',
                         head_to_tail=False):
    """

    Obtain the "angle" of a subset of the 49 points of a worm for each frame.

    This is a wrapper around get_angles(). This function resolves a body
    partition into actual x,y data for that function.

    Parameters
    ----------
    nw: NormalizedWorm

    partition_key : str
        Examples include head, tail, or body
    data_key : str
        The attribute of normalized worm to use. Examples include:
            - skeleton
            - anything else?
    head_to_tail : bool
            - True means the worm points are ordered head to tail.
            TODO: Explain this more

    Returns
    -------
    numpy array of shape (n)
        Stores the worm body's "angle" (in degrees) for each frame of video.
        See the get_angles() function for how the angles are calculated.

    See Also
    --------
    get_angles

    """

    # the shape of both segment_x and segment_y is (partition length, n)
    segment_x, segment_y = nw.get_partition(partition_key, data_key, True)

    return get_angles(segment_x, segment_y, head_to_tail)


def h__computeAngularSpeed(time, segment_x, segment_y,
                           left_I, right_I, ventral_mode):
    """

    This function is called by compute_speeed().

    TODO: These units are wrong ...
    TODO: This is actually angular velocity

    Parameters
    ----------
    time :
    segment_x : numpy array, shape (p,n)
        The x's of the partition being considered.
    segment_y : numpy array, shape (p,n)
        The y's of the partition being considered.
    left_I : numpy array
        For each frame, an index (earlier in time) from which to compute
        the desired value. These values only exist for cases in which
        both left_I and right_I were computed to be valid.
    right_I : numpy array
        For each frame, an index (later in time) from which to compute the
        desired vlaue.
    ventral_mode :
        0, 1, or 2, specifying that the ventral side is...
          0 = unknown
          1 = clockwise
          2 = anticlockwise

    Returns
    -------
    a numpy array of shape n, in units of degrees per second

    See Also
    --------
    compute_speed
    get_angles

    """
    # Compute the body part direction for each frame
    point_angle_d = get_angles(segment_x, segment_y, head_to_tail=False)

    angular_speed = point_angle_d[right_I] - point_angle_d[left_I]

    # Correct any jumps that result during the subtraction process
    # i.e. 1 - 359 ~= -358
    # by forcing -180 <= angular_speed[i] <= 180
    angular_speed = (angular_speed + 180) % (360) - 180

    # Change units from degrees per frame to degrees per second
    angular_speed = angular_speed  / time

    # Sign the direction for dorsal/ventral locomotion.
    # if ventral_mode is anything but anticlockwise, then negate angular_speed:
    if(ventral_mode < 2):
        angular_speed = -angular_speed

    return angular_speed


def h__getSpeedIndices(frames_per_sample, good_frames_mask):
    """

    For each point, we calculate the speed using frames prior to and following
    a frame. Given that some frames are not valid (have NaN), we move the index
    backward (prior frame) or forward (following frame), essentially slightly
    widening the time frame over which the speed is computed.

    This function determines what the indices are that each frame will use to
    calculate the velocity at that frame. For example, at frame 5 we might decide
    to use frames 2 and 8.

    Parameters
    ----------
    frames_per_sample : int
      Our sample scale, in frames. The integer must be odd.

    good_frames_mask :
      Shape (num_frames), false if underlying angle is NaN

    Returns
    -------
    (keep_mask,left_I,right_I)
           keep_mask : shape (num_frames), this is used to indicate
                       which original frames have valid velocity values,
                       and which don't.
                       NOTE: sum(keep_mask) == n_valid_velocity_values
           left_I    : shape (n_valid_velocity_values), for a given sample, this
                       indicates the index to the left of (less than) the sample
                       that should be used to calculate the velocity
           right_I   : shape (n_valid_velocity_values)

    Explanation
    -----------
    Assume we have the following data ........
    half_scale = 1 # minimum shift for computing value at a frame
    full_scale = 2 # maximum shift for computing value at a frame
    frame   : 0 1 2 3 4 5 6 7 8 9
    is_good : T F T T T F T F T F

    Our first possibly valid frame is 1, because we require 1 on each side
    (i.e. half_scale = 1).

    frame 0:
        can't go left, left is invalid
        check 1, it is invalid
        check 2, it is valid, right index is 2

    So for frame 1:
        check 0, it is valid, left index is set
        check 2, also valid, right index is set

    Now onto 2:
        check 1, invalid
        check 0, valid, left index at 0
        check 3, valid, right index is 3

    So far we have:
        left_I  = [NaN, 0, 0]
        right_I = [2,   2, 3]

    Interpretation:
    frame 0: left index is invalid, so we don't compute the value
    frame 1: compute value taking values at frame 0 (left) and frame 2 (right)

    This is a frame by frame approach, but below we actually loop over
    the shifts since the shifts tends to be much less than in size than the
    # of frames

    i.e. (by shift approach explained)
    For all frames, check 1 to the left. For all valid frames, set these
    as the left_I values for the current frames.
    Do the same thing as well for frames that are 1 to the right.
    Now we are done with a shift size of 1.

    Now repeat this process for a shift size of 2, however only override left_I
    and right_I values if they were not set for a shift size of 1.


    """

    # Require that frames_per_sample be an odd integer
    assert(isinstance(frames_per_sample, int))
    assert(frames_per_sample % 2 == 1)

    num_frames = len(good_frames_mask)

    # Create a "half" scale
    # NOTE: Since the scale is odd, the half
    #       scale will be even, because we subtract 1
    scale_minus_1 = frames_per_sample - 1
    half_scale = int(scale_minus_1 / 2)

    # First frame for which we can assign a valid velocity:
    start_index = half_scale
    # Final frame for which we can assign a valid velocity, plus one:
    end_index = num_frames - half_scale

    # These are the indices we will use to compute the speed. We add
    # a half scale here to avoid boundary issues (i.e. because we pad the. We'll subtract it out later.
    # See below for more details
    middle_I = np.array(np.arange(start_index, end_index, 1) + half_scale,
                        dtype='int32')

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
    left_I = np.empty(len(middle_I), dtype='int32')
    right_I = np.empty(len(middle_I), dtype='int32')
    # numpy integer arrays cannot accept NaN, which is a float concept, but
    # filling them with NaN fills them with the largest negative number
    # possible, -2**31.  We can easily filter for this later.
    left_I.fill(np.NaN)
    right_I.fill(np.NaN)

    # Track which ends we haven't yet matched, for each of the middle_I's.
    # since we are loopering over each possible shift, we need to track
    # whether valid ends have been found for each of the middle_I's.
    unmatched_left_mask = np.ones(len(middle_I), dtype=bool)
    unmatched_right_mask = np.ones(len(middle_I), dtype=bool)

    # Instead of looping over each centered velocity, we loop over each possible
    # shift. A shift is valid if the frame of the shift is good, and we have yet
    # to encounter a match for that centered index
    for shift_size in range(half_scale, frames_per_sample):
        # We grab indices that are the appropriate distance from the current
        # value. If we have not yet found a bound on the given side, and the
        # index is valid, we keep it.
        left_indices_temp = middle_I - shift_size
        right_indices_temp = middle_I + shift_size

        is_good_left_mask = good_frames_mask_padded[left_indices_temp]
        is_good_right_mask = good_frames_mask_padded[right_indices_temp]

        use_left_mask = unmatched_left_mask & is_good_left_mask
        use_right_mask = unmatched_right_mask & is_good_right_mask

        # Change only those left_I's to our newly shifted outwards
        # left_indices_temp, that the use_left_mask says should be used.
        left_I[use_left_mask] = left_indices_temp[use_left_mask]
        right_I[use_right_mask] = right_indices_temp[use_right_mask]

        # Flag the matched items as being matched
        unmatched_left_mask[use_left_mask] = False
        unmatched_right_mask[use_right_mask] = False

    # Remove the offset used to pad the numbers (discussed above)
    # We have to avoid decrementing negative numbers because our negative
    # number is our NaN proxy and it's already as negative as it can be
    # without wrapping back up to positive again
    left_I[left_I > 0] -= half_scale
    right_I[right_I > 0] -= half_scale
    middle_I -= half_scale

    # Filter down to usable values, in which both left and right are defined
    # Remember than np.NaN is not valid number for integer numpy arrays
    # so instead of checking for which entries are NaN, we check for
    # which entries are negative, since no array indices can be
    # negative!
    valid_indices_mask = (left_I >= 0) & (right_I >= 0)
    left_I = left_I[valid_indices_mask]
    right_I = right_I[valid_indices_mask]
    middle_I = middle_I[valid_indices_mask]

    keep_mask = np.zeros(num_frames, dtype=bool)
    keep_mask[middle_I] = True

    # sum(keep_mask) should equal the number of valid velocity values
    # left_I and right_I should store just these valid velocity values
    assert sum(keep_mask) == len(left_I) == len(right_I)
    assert all(left_I >= 0) and all(left_I < num_frames)
    assert all(right_I >= 0) and all(right_I < num_frames)
    return keep_mask, left_I, right_I


def compute_speed(fps, sx, sy, avg_body_angle, sample_time, ventral_mode=0):
    """

    Previous Name: compute_velocity

    The speed is computed not using the nearest values but values
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

    ventral_mode: int (0,1,2)
        Options for specification of the ventral side:
        - 0 = unknown (default)
        - 1 = clockwise
        - 2 = anticlockwise

    Returns
    -------
    (speed, angular_speed, motion_direction)
    Three numpy arrays of shape (n), speed, angular_speed, motion_direction
        speed:
        angular_speed:
        motion_direction:

    Known Callers
    -------------
    LocomotionVelocity

    """

    num_frames = np.shape(sx)[1]
    speed = np.full((num_frames), np.nan)
    angular_speed = np.full((num_frames), np.nan)
    motion_direction = np.full((num_frames), np.nan)

    # We need to go from a time over which to compute the velocity
    # to a # of samples. The # of samples should be odd.
    frames_per_sample = get_frames_per_sample(fps, sample_time)

    # If we don't have enough frames to satisfy our sampling scale,
    # return with nothing.
    if(frames_per_sample > num_frames):
        return speed, angular_speed, motion_direction

    # Compute the indices that we will use for computing the velocity. We
    # calculate the velocity roughly centered on each sample, but with a
    # considerable width between frames that smooths the velocity estimate.
    good_frames_mask = ~np.isnan(avg_body_angle)
    keep_mask, left_I, right_I = h__getSpeedIndices(frames_per_sample,
                                                    good_frames_mask)

    # 1) Compute speed
    # --------------------------------------------------------
    # Centroid of the current skeletal segment, frame-by-frame:
    x_mean = np.mean(sx, 0)
    y_mean = np.mean(sy, 0)

    dX = x_mean[right_I] - x_mean[left_I]
    dY = y_mean[right_I] - y_mean[left_I]

    distance = np.sqrt(dX ** 2 + dY ** 2)
    time = (right_I - left_I) / fps

    #import pdb
    # pdb.set_trace()

    speed[keep_mask] = distance / time

    # 2) Compute angular speed (Formally known as direction :/)
    # --------------------------------------------------------
    angular_speed[keep_mask] = h__computeAngularSpeed(time, sx, sy,
                                                      left_I, right_I,
                                                      ventral_mode)

    # 3) Sign the speed.
    # ------------------------------------------------------------
    #   We want to know how the worm's movement direction compares
    #   to the average angle it had (apparently at the start)
    motion_direction[keep_mask] = np.degrees(np.arctan2(dY, dX))

    # This recentres the definition, as we are really just concerned
    # with the change, not with the actual value
    body_direction = np.full((num_frames), np.nan)
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
        motion_direction[body_direction < 0] = - \
            motion_direction[body_direction < 0]

    if(ventral_mode == 2):  # i.e. if ventral side is anticlockwise:
        motion_direction = -motion_direction

    return speed, angular_speed, motion_direction


def get_frames_per_sample(fps, sample_time):
    """
    Converts a specified sample_time from seconds to # of samples.

    Parameters
    ----------
    fps: float
        Video sampling rate
    sample_time: float
        Duration (in seconds) to sample

    Returns
    -------
    int
        The # of samples that corresponds to the specified sample_time. This
        value will always be odd. Values are always increased to obtain the
        odd value (e.g. an exact sampling of 4 samples becomes 5). Odd values
        are meant to specify grabbing a set of points that is inclusive of the
        end point so as to occupy the specified duration.
    """

    ostensive_sampling_scale = sample_time * fps

    half_scale = int(round(ostensive_sampling_scale / 2))
    sampling_scale = 2 * half_scale + 1

    assert(isinstance(sampling_scale, int) or sampling_scale.is_integer())
    return int(sampling_scale)
