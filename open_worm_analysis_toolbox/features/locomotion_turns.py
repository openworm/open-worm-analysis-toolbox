# -*- coding: utf-8 -*-
"""
Calculate the "Turns" locomotion feature

There are two kinds of turns:
    - omega
    - upsilon.

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


Formerly this code was contained in four Matlab files:
  seg_worm.feature_calculator.getOmegaAndUpsilonTurns, which called these 3:
    seg_worm.feature_helpers.locomotion.getOmegaEvents
    seg_worm.feature_helpers.locomotion.getUpsilonEvents
    seg_worm.feature_helpers.locomotion.getTurnEventsFromSignedFrames


TODO: OmegaTurns and UpsilonTurns should inherit from LocomotionTurns or something


IMPORTANT: My events use 1 based indexing, the old code used 0 based
indexing - @JimHokanson

"""

import numpy as np

import collections
import warnings
import operator
import re

from .generic_features import Feature

from .. import utils

from . import events

#%%


class LocomotionTurns(object):

    """

    LocomotionTurns

    Attributes
    ----------
    omegas : OmegaTurns
    upsilons : UpsilonTurns

    Methods
    -------
    __init__

    """

    def __init__(
            self,
            features_ref,
            bend_angles,
            is_stage_movement,
            midbody_distance,
            sx,
            sy):
        """
        Initialiser for the LocomotionTurns class

        Parameters
        ----------
        features_ref :
        bend_angles :
        is_stage_movement :
        midbody_distance :
        sx :
        sy :

        Notes
        ---------------------------------------
        Formerly getOmegaAndUpsilonTurns

        Old Name:
        - featureProcess.m
        - omegaUpsilonDetectCurvature.m

        """

        nw = features_ref.nw

        if not features_ref.options.should_compute_feature(
                'locomotion.turns', features_ref):
            self.omegas = None
            self.upsilons = None
            return

        options = features_ref.options.locomotion.locomotion_turns

        fps = features_ref.video_info.fps

        timer = features_ref.timer
        timer.tic()

        n_frames = bend_angles.shape[1]

        angles = collections.namedtuple('angles',
                                        ['head_angles',
                                         'body_angles',
                                         'tail_angles',
                                         'body_angles_with_long_nans',
                                         'is_stage_movement'])

        first_third = nw.get_subset_partition_mask('first_third')
        second_third = nw.get_subset_partition_mask('second_third')
        last_third = nw.get_subset_partition_mask('last_third')

        # NOTE: For some reason the first and last few angles are NaN, so we use
        # nanmean instead of mean.  We could probably avoid this for the body.
        # Suppress RuntimeWarning: Mean of empty slice for those frames
        # that are ALL NaN.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            angles.head_angles = np.nanmean(
                bend_angles[first_third, :], axis=0)
            angles.body_angles = np.nanmean(
                bend_angles[second_third, :], axis=0)
            angles.tail_angles = np.nanmean(bend_angles[last_third, :], axis=0)
            angles.is_stage_movement = is_stage_movement

        # Deep copy.
        # To @JimHokanson from @MichaelCurrie: what does "ht" stand for?
        #             consider expanding this variable name so it's clear
        body_angles_for_ht_change = np.copy(angles.body_angles)

        n_head = np.sum(~np.isnan(angles.head_angles))
        n_body = np.sum(~np.isnan(angles.body_angles))
        n_tail = np.sum(~np.isnan(angles.tail_angles))

        # Only proceed if there are at least two non-NaN
        # value in each angle vector
        if n_head < 2 or n_body < 2 or n_tail < 2:
            # Make omegas and upsilons into blank events lists and return
            self.omegas = events.EventListWithFeatures(fps, make_null=True)
            self.upsilons = events.EventListWithFeatures(fps, make_null=True)
            return

        # Interpolate the angles.  angles is modified.
        self.h__interpolateAngles(
            angles, options.max_interpolation_gap_allowed)

        # Get frames for each turn type
        #----------------------------------------------------------------------
        # This doesn't match was is written in the supplemental material ...
        # Am I working off of old code??????

        # TODO: Move this all to options ...
        consts = collections.namedtuple('consts',
                                        ['head_angle_start_const',
                                         'tail_angle_start_const',
                                         'head_angle_end_const',
                                         'tail_angle_end_const',
                                         'body_angle_const'])

        yuck = [[20, -20, 15, -15],
                [30, 30, 30, 30],
                [40, 40, 30, 30],
                [20, -20, 15, -15],
                [20, -20, 15, -15]]
        """
        OLD Matlab CODE:

        consts = struct(...
            'head_angle_start_const',{20 -20 15 -15}, ...
            'tail_angle_start_const',{30  30 30  30}, ...
            'head_angle_end_const',  {40  40 30  30}, ...
            'tail_angle_end_const',  {20 -20 15 -15}, ...
            'body_angle_const'   ,   {20 -20 15 -15})
        """

        # NOTE: We need to run omegas first (false values) since upsilons are
        # more inclusive, but can not occur if an omega event occurs
        is_upsilon = [False, False, True, True]

        # NOTE: We assign different values based on the sign of the angles
        values_to_assign = [1, -1, 1, -1]

        frames = collections.namedtuple('frames',
                                        ['omega_frames', 'upsilon_frames'])

        frames.omega_frames = np.zeros(n_frames)
        frames.upsilon_frames = np.zeros(n_frames)

        for i in range(4):
            consts.head_angle_start_const = yuck[0][i]
            consts.tail_angle_start_const = yuck[1][i]
            consts.head_angle_end_const = yuck[2][i]
            consts.tail_angle_end_const = yuck[3][i]
            consts.body_angle_const = yuck[4][i]
            condition_indices = self.h__getConditionIndices(angles, consts)
            self.h__populateFrames(angles,
                                   condition_indices,
                                   frames,
                                   is_upsilon[i],
                                   values_to_assign[i])

        # Calculate the events from the frame values
        self.omegas = OmegaTurns.create(options,
                                        frames.omega_frames,
                                        nw,
                                        body_angles_for_ht_change,
                                        midbody_distance,
                                        fps)

        self.upsilons = UpsilonTurns.create(frames.upsilon_frames,
                                            midbody_distance,
                                            fps)

        timer.toc('locomotion.turns')

    def __repr__(self):
        return utils.print_object(self)

    @classmethod
    def from_disk(cls, turns_ref):

        self = cls.__new__(cls)

        self.omegas = OmegaTurns.from_disk(turns_ref)
        self.upsilons = UpsilonTurns.from_disk(turns_ref)

        return self

    def __eq__(self, other):
        return self.upsilons.test_equality(
            other.upsilons, 'locomotion.turns.upsilons') and self.omegas.test_equality(
            other.omegas, 'locomotion.turns.omegas')

    def h__interpolateAngles(self, angles, MAX_INTERPOLATION_GAP_ALLOWED):
        """
        Interpolate the angles in the head, body, and tail.
        For the body, also interpolate with a threshold, and assign this
        to body_angles_with_long_nans

        Parameters
        ---------------------------------------
        angles: a named tuple
          angles = collections.namedtuple('angles',
                                          ['head_angles',
                                           'body_angles',
                                           'tail_angles',
                                           'body_angles_with_long_nans',
                                           'is_stage_movement'])


        Returns
        ---------------------------------------
        None; instead the angles parameter has been modified in place

        Notes
        ---------------------------------------
        Formerly a = h__interpolateAngles(a, MAX_INTERPOLATION_GAP_ALLOWED)

        TODO: Incorporate into the former
        seg_worm.feature_helpers.interpolateNanData

        """
        # Let's use a shorter expression for clarity
        interp = utils.interpolate_with_threshold

        # This might not actually have been applied - SEGWORM_MC used BodyAngles
        # - @JimHokanson
        angles.body_angles_with_long_nans = interp(
            angles.body_angles, MAX_INTERPOLATION_GAP_ALLOWED + 1, make_copy=True)

        interp(angles.head_angles, make_copy=False)
        interp(angles.body_angles, make_copy=False)
        interp(angles.tail_angles, make_copy=False)

    def h__getConditionIndices(self, a, c):
        """
        This function implements a filter on the frames for the different
        conditions that we are looking for in order to get a particular turn.

        It does not however provide any logic on their relative order, i.e.
        that one condition occurs before another. This is done in a later
        function, h__populateFrames.

        Parameters
        ---------------------------------------
        a:

        c:


        Notes
        ---------------------------------------
        Formerly s = h__getConditionIndices(a, c)

        """

        # Determine comparison function
        #----------------------------------------------------------
        is_positive = c.head_angle_start_const > 0
        if is_positive:
            fh = operator.gt
        else:
            fh = operator.lt

        # start: when the head exceeds its angle but the tail does not
        # end  : when the tail exceeds its angle but the head does not

        # TODO: Rename to convention ...
        s = collections.namedtuple('stuffs',
                                   ['startCond',
                                    'startInds',
                                    'midCond',
                                    'midStarts',
                                    'midEnds',
                                    'endCond',
                                    'endInds'])

        def find_diff(array, value):
            # diff on logical array doesn't work the same as it does in Matlab
            return np.flatnonzero(np.diff(array.astype(int)) == value)

        with np.errstate(invalid='ignore'):
            s.startCond = fh(a.head_angles, c.head_angle_start_const) & \
                (np.abs(a.tail_angles) < c.tail_angle_start_const)

        # add 1 for shift due to diff
        s.startInds = find_diff(s.startCond, 1) + 1

        # NOTE: This is NaN check is a bit suspicious, as it implies that the
        # head and tail are parsed, but the body is not. The original code puts
        # NaN back in for long gaps in the body angle, so it is possible that
        # the body angle is NaN but the others are not.
        with np.errstate(invalid='ignore'):
            s.midCond   = fh(a.body_angles, c.body_angle_const) | \
                np.isnan(a.body_angles_with_long_nans)

        # add 1 for shift due to diff
        s.midStarts = find_diff(s.midCond, 1) + 1
        s.midEnds = find_diff(s.midCond, -1)

        with np.errstate(invalid='ignore'):
            s.endCond = np.logical_and(fh(a.tail_angles, c.tail_angle_end_const),
                                       np.abs(a.head_angles) <
                                       c.head_angle_end_const)

        s.endInds = find_diff(s.endCond, -1)

        return s

    def h__populateFrames(self, a, s, f, get_upsilon_flag, value_to_assign):
        """

        Algorithm
        ---------------------------------------
        - For the middle angle range, ensure one frame is valid and that
          the frame proceeding the start and following the end are valid
        - Find start indices and end indices that bound this range
        - For upsilons, exclude if they overlap with an omega bend ...


        Parameters
        ---------------------------------------
        a: named tuple
               head_angles: [1x4642 double]
               body_angles: [1x4642 double]
               tail_angles: [1x4642 double]
         is_stage_movement: [1x4642 logical]
                 bodyAngle: [1x4642 double]

        s: named tuple
         startCond: [1x4642 logical]
         startInds: [1x81 double]
           midCond: [1x4642 logical]
         midStarts: [268 649 881 996 1101 1148 1202 1963 3190 3241 4144 4189 4246 4346 4390 4457 4572 4626]
           midEnds: [301 657 925 1009 1103 1158 1209 1964 3196 3266 4148 4200 4258 4350 4399 4461 4579]
           endCond: [1x4642 logical]
           endInds: [1x47 double]

        f: named tuple
           omegaFrames: [4642x1 double]
         upsilonFrames: [4642x1 double]

        get_upsilon_flag: bool
          Toggled based on whether or not we are getting upsilon events
          or omega events

        value_to_assign:


        Returns
        ---------------------------------------
        None; modifies parameters in place.


        Notes
        ---------------------------------------
        Formerly f = h__populateFrames(a,s,f,get_upsilon_flag,value_to_assign)

        """

        for cur_mid_start_I in s.midStarts:

            # JAH NOTE: This type of searching is inefficient in Matlab since
            # the data is already sorted. It could be improved ...
            temp = np.flatnonzero(s.midEnds > cur_mid_start_I)

            # cur_mid_end_I   = s.midEnds[find(s.midEnds > cur_mid_start_I, 1))

            if temp.size != 0:
                cur_mid_end_I = s.midEnds[temp[0]]

                if ~np.all(
                    a.is_stage_movement[
                        cur_mid_start_I:cur_mid_end_I +
                        1]) and s.startCond[
                    cur_mid_start_I -
                    1] and s.endCond[
                    cur_mid_end_I +
                        1]:

                    temp2 = np.flatnonzero(s.startInds < cur_mid_start_I)
                    temp3 = np.flatnonzero(s.endInds > cur_mid_end_I)

                    if temp2.size != 0 and temp3.size != 0:
                        cur_start_I = s.startInds[temp2[-1]]
                        cur_end_I = s.endInds[temp3[0]]

                        if get_upsilon_flag:
                            # Don't populate upsilon if the data spans an omega
                            if ~np.any(
                                np.abs(
                                    f.omega_frames[
                                        cur_start_I:cur_end_I +
                                        1])):
                                f.upsilon_frames[
                                    cur_start_I:cur_end_I + 1] = value_to_assign
                        else:
                            f.omega_frames[
                                cur_start_I:cur_end_I + 1] = value_to_assign

        # Nothing needs to be returned since we have modified our parameters
        # in place
        return None


"""
===============================================================================
===============================================================================
"""

#%%


class UpsilonTurns(object):

    """
    Represents the Omega turn events


    Notes
    ---------------------------------------
    Formerly this was not implemented as a class.

    """

    def __init__(self, upsilon_frames, midbody_distance, fps):
        """
        Initialiser for the UpsilonTurns class.

        Parameters
        ----------
        upsilon_frames :
        midbody_distance :
        fps :

        Notes
        ---------------------------------------
        Formerly, in the SegWormMatlabClasses repo, this was not the constructor
        of a class, but a locomotion method of called
        getUpsilonEvents(obj,upsilon_frames,midbody_distance,FPS)

        """

        self.value = getTurnEventsFromSignedFrames(upsilon_frames,
                                                   midbody_distance,
                                                   fps)

        self.no_events = self.value.is_null

    @staticmethod
    def create(upsilon_frames, midbody_distance, fps):

        temp = UpsilonTurns(upsilon_frames, midbody_distance, fps)

        return temp.value

    @classmethod
    def from_disk(cls, turns_ref):

        return events.EventListWithFeatures.from_disk(
            turns_ref['upsilons'], 'MRC')


"""
===============================================================================
===============================================================================
"""

#%%


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

    """

    def __init__(self, options, omega_frames_from_angles, nw, body_angles,
                 midbody_distance, fps):
        """
        Initialiser for the OmegaTurns class.

        Parameters
        ----------
        omega_frames_from_angles: [1 x n_frames]
          Each frame has the value 0, 1, or -1,

        nw: NormalizedWorm instance
          We only use it for its skeleton.

        body_angles
          average bend angle of the middle third of the worm

        midbody_distance:

        FPS: float
          Frames per second


        Returns
        ---------------------------------------
        None


        Notes
        ---------------------------------------
        Formerly, in the SegWormMatlabClasses repo, this was not the initialiser
        of a class, but a locomotion method of called
        omega_events = getOmegaEvents(obj,omega_frames_from_angles,sx,sy,
                       body_angles,midbody_distance,fps)

        omega_events was an event structure.  now self.omegas just contains
        the omega turns.

        See also:
        seg_worm.features.locomotion.getOmegaAndUpsilonTurns
        seg_worm.features.locomotion.getTurnEventsFromSignedFrames

        """

        body_angles_i = \
            utils.interpolate_with_threshold(body_angles, extrapolate=True)

        self.omegas = None  # DEBUG: remove once the below code is ready

        omega_frames_from_th_change = self.h_getHeadTailDirectionChange(
            nw, fps)

        # Filter:
        # This is to be consistent with the old code. We filter then merge,
        # then filter again :/
        omega_frames_from_th_change = \
            self.h__filterAndSignFrames(body_angles_i,
                                        omega_frames_from_th_change,
                                        options.min_omega_event_length(fps))

        is_omega_frame = (omega_frames_from_angles != 0) | \
                         (omega_frames_from_th_change != 0)

        # Refilter and sign
        signed_omega_frames = \
            self.h__filterAndSignFrames(body_angles_i,
                                        is_omega_frame,
                                        options.min_omega_event_length(fps))

        # Convert frames to events ...
        self.value = getTurnEventsFromSignedFrames(signed_omega_frames,
                                                   midbody_distance,
                                                   fps)

        self.no_events = self.value.is_null

    @staticmethod
    def create(
            options,
            omega_frames_from_angles,
            nw,
            body_angles,
            midbody_distance,
            fps):

        temp = OmegaTurns(options, omega_frames_from_angles, nw, body_angles,
                          midbody_distance, fps)

        return temp.value

    @classmethod
    def from_disk(cls, turns_ref):

        return events.EventListWithFeatures.from_disk(
            turns_ref['omegas'], 'MRC')

    def h_getHeadTailDirectionChange(self, nw, FPS):
        """


        Parameters
        ---------------------------------------
        nw: A NormalizedWorm instance

        FPS: int
          Frames Per Second


        Returns
        ---------------------------------------
        A boolean numpy array indicating in each frame whether or not
        it's an omega angle change


        Notes
        ---------------------------------------
        Formerly is_omega_angle_change = h_getHeadTailDirectionChange(FPS,sx,sy)

        NOTE: This change in direction of the head and tail indicates that
        either a turn occurred OR that an error in the parsing occurred.
        Basically we look for the angle from the head to the tail to all of a
        sudden change by 180 degrees.

        """
        MAX_FRAME_JUMP_FOR_ANGLE_DIFF = round(FPS / 2)

        # We compute a smoothed estimate of the angle change by using angles at
        # indices that are +/- this value ...
        HALF_WINDOW_SIZE = round(FPS / 4)

        # NOTE: It would be better to have this be based on time, not samples
        MAX_INTERP_GAP_SIZE = 119

        #????!!!!?? - why is this a per frame value instead of an average angular
        # velocity ????
        PER_FRAME_DEGREE_CHANGE_CUTOFF = 3

        # Compute tail direction
        #----------------------------------------------------
        head_x, head_y = nw.get_partition('head',
                                          data_key='skeleton',
                                          split_spatial_dimensions=True)
        tail_x, tail_y = nw.get_partition('tail',
                                          data_key='skeleton',
                                          split_spatial_dimensions=True)

        # Take the mean across the partition, so that we are left with a single
        # value for each frame (i.e. 1-d an array of length n_frames)
        # Suppress RuntimeWarning: Mean of empty slice for those frames
        # that are ALL NaN.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            head_x = np.nanmean(head_x, axis=0)
            head_y = np.nanmean(head_y, axis=0)
            tail_x = np.nanmean(tail_x, axis=0)
            tail_y = np.nanmean(tail_y, axis=0)

        th_angle = np.arctan2(head_y - tail_y, head_x - tail_x) * (180 / np.pi)

        n_frames = len(th_angle)

        # Changed angles to being relative to the previous frame
        #----------------------------------------------------
        # Compute the angle change between subsequent frames. If a frame is not
        # valid, we'll use the last valid frame to define the difference, unless the
        # gap is too large.

        is_good_th_direction_value = ~np.isnan(th_angle)

        lastAngle = th_angle[0]
        gapCounter = 0

        th_angle_diff_temp = np.empty(th_angle.size) * np.NAN
        for iFrame in range(n_frames)[1:]:   # formerly 2:n_frames
            if is_good_th_direction_value[iFrame]:
                th_angle_diff_temp[iFrame] = th_angle[iFrame] - lastAngle
                gapCounter = 0
                lastAngle = th_angle[iFrame]
            else:
                gapCounter += 1

            if gapCounter > MAX_FRAME_JUMP_FOR_ANGLE_DIFF:
                lastAngle = np.NaN

        #???? - what does this really mean ??????
        # I think this basically says, instead of looking for gaps in the original
        # th_angle, we need to take into account how much time has passed between
        # successive differences
        #
        # i.e. instead of doing a difference in angles between all valid frames, we
        # only do a difference if the gap is short enough

        # We go through some heroics to avoid the "RuntimeWarning: invalid
        # value encountered" warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            positiveJumps = np.flatnonzero(th_angle_diff_temp > 180)
            negativeJumps = np.flatnonzero(th_angle_diff_temp < -180)

        # For example data, these are the indices I get ...
        #P - 4625
        #N - 3634, 4521

        # Fix the th_angles by unwrapping
        #----------------------------------------------------
        # NOTE: We are using the identified jumps from the fixed angles to unwrap
        # the original angle vector
        # subtract 2pi from remainging data after positive jumps
        for j in range(len(positiveJumps)):
            th_angle[positiveJumps[j]:] = th_angle[positiveJumps[j]:] - 2 * 180

        # add 2pi to remaining data after negative jumps
        for j in range(len(negativeJumps)):
            th_angle[negativeJumps[j]:] = th_angle[negativeJumps[j]:] + 2 * 180

        # Fix the th_angles through interpolation
        #----------------------------------------------------
        th_angle = \
            utils.interpolate_with_threshold(th_angle,
                                             MAX_INTERP_GAP_SIZE + 1,
                                             make_copy=False,
                                             extrapolate=False)

        # Determine frames that might be omega events (we'll filter later based on
        # length)
        #----------------------------------------------------
        # Compute angle difference
        th_angle_diff = np.empty(len(th_angle)) * np.NaN

        left_indices = np.array(range(n_frames)) - HALF_WINDOW_SIZE
        right_indices = np.array(range(n_frames)) + HALF_WINDOW_SIZE

        mask = (left_indices > 1) & (right_indices < n_frames)

        th_angle_diff[mask] = th_angle[right_indices[mask].astype(int)] - \
            th_angle[left_indices[mask].astype(int)]

        avg_angle_change_per_frame = abs(
            th_angle_diff / (HALF_WINDOW_SIZE * 2))

        # Now return whether or not it's an omega angle change
        # Again we go through some heroics to avoid the "RuntimeWarning: invalid
        # value encountered" warning

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return avg_angle_change_per_frame > PER_FRAME_DEGREE_CHANGE_CUTOFF

    def h__filterAndSignFrames(self, body_angles_i, is_omega_frame,
                               min_omega_event_length):
        """
        Filter and sign frames.

        Notes
        ---------------------------------------
        Formerly signed_omega_frames =
          h__filterAndSignFrames(body_angles_i, is_omega_frame,
                                 MIN_OMEGA_EVENT_LENGTH)

        """
        # Let's take a boolean numpy array and change it to a string where
        # A is false and B is true: e.g.
        # [True, True, False] turns into 'BBA'
        # (Note: this is all a translation of this Matlab line:
        # [start1, end1] = \
        #   regexp(is_omega_frame_as_string, gap_str, 'start', 'end')
        is_omega_frame_as_ascii_codes = is_omega_frame.astype(int) + ord('A')
        is_omega_frame_as_list = [chr(x)
                                  for x in is_omega_frame_as_ascii_codes]
        is_omega_frame_as_string = ''.join(is_omega_frame_as_list)
        gap_re = re.compile(r'B{%d,}' % min_omega_event_length)
        # Obtain a iterator of the results that match our regex, gap_re.
        re_result = list(gap_re.finditer(is_omega_frame_as_string))
        start1 = [m.start(0) for m in re_result]
        end1 = [m.end(0) for m in re_result]

        signed_omega_frames = np.zeros(is_omega_frame.size)

        # Note: Here we keep the long gaps instead of removing them
        for iEvent in range(len(start1)):
            if np.mean(body_angles_i[start1[iEvent]:end1[iEvent]]) > 0:
                signed_omega_frames[start1[iEvent]:end1[iEvent]] = 1
            else:
                signed_omega_frames[start1[iEvent]:end1[iEvent]] = -1

        return signed_omega_frames


"""
===============================================================================
===============================================================================
"""

#%%


def getTurnEventsFromSignedFrames(signed_frames, midbody_distance, FPS):
    """
    Get turn events from signed frames

    Parameters
    ---------------------------------------
    signed_frames:
      ??? - I believe the values are -1 or 1, based on
      whether something is dorsal or ventral ....


    Notes
    ---------------------------------------
    This code is common to omega and upsilon turns.

    Formerly function turn_events = \
      seg_worm.features.locomotion.getTurnEventsFromSignedFrames(
                            obj,signed_frames,midbody_distance,FPS)

    Called by:
    seg_worm.features.locomotion.getUpsilonEvents
    seg_worm.features.locomotion.getOmegaEvents

    """

    ef = events.EventFinder()
    ef.include_at_frames_threshold = True

    # get_events(self, speed_data, distance_data=None):

    # JAH: This interface doesn't make as much sense anymore ...

    # seg_worm.feature.event_finder.getEvents
    ef.min_speed_threshold = 1

    frames_dorsal = ef.get_events(signed_frames)

    ef = events.EventFinder()
    ef.include_at_frames_threshold = True
    ef.min_speed_threshold = None
    ef.max_speed_threshold = -1
    frames_ventral = ef.get_events(signed_frames)

    # Unify the ventral and dorsal turns.
    [frames_merged, is_ventral] = events.EventList.merge(frames_ventral,
                                                         frames_dorsal)

    turn_event_output = events.EventListWithFeatures(FPS,
                                                     frames_merged,
                                                     midbody_distance)

    turn_event_output.is_ventral = is_ventral

    """
    Note that in the past, the former (Matlab) code for this function
    added an is_ventral to each FRAME.  EventListForOutput does not have a
    frames variable, so instead we simply have an is_ventral numpy array.
    - @MichaelCurrie

    Here is the former code, using correct variable names and Python syntax:

    # Add extra field, isVentral ...
    for iEvent = range(len(turn_event_output.frames)):
        turn_event_output.frames[iEvent].isVentral = is_ventral[iEvent]

    """

    return turn_event_output


class TurnProcessor(Feature):

    """
    Feature:   'locomotion.turn_processor'

    LocomotionTurns

    Attributes
    ----------
    omegas : OmegaTurns
    upsilons : UpsilonTurns

    Methods
    -------
    __init__

    """

    def __init__(self, wf, feature_name):
        """
        Initialiser for the LocomotionTurns class

        Parameters
        ----------
        features_ref :
        bend_angles :
        is_stage_movement :
        midbody_distance :
        sx :
        sy :

        Notes
        ---------------------------------------
        Formerly getOmegaAndUpsilonTurns

        Old Name:
        - featureProcess.m
        - omegaUpsilonDetectCurvature.m

        """

        # features_ref, , , midbody_distance, sx, sy

        self.name = feature_name

        options = wf.options.locomotion.locomotion_turns

        video_info = wf.video_info
        fps = video_info.fps
        is_stage_movement = video_info.is_stage_movement

        nw = wf.nw
        bend_angles = nw.angles

        #sx = nw.skeleton_x
        #sy = nw.skeleton_y

        midbody_distance = self.get_feature(
            wf, 'locomotion.velocity.mibdody.distance').value

        timer = wf.timer
        timer.tic()

        n_frames = bend_angles.shape[1]

        angles = collections.namedtuple('angles',
                                        ['head_angles',
                                         'body_angles',
                                         'tail_angles',
                                         'body_angles_with_long_nans',
                                         'is_stage_movement'])

        first_third = nw.get_subset_partition_mask('first_third')
        second_third = nw.get_subset_partition_mask('second_third')
        last_third = nw.get_subset_partition_mask('last_third')

        # NOTE: For some reason the first and last few angles are NaN, so we use
        # nanmean instead of mean.  We could probably avoid this for the body.
        # Suppress RuntimeWarning: Mean of empty slice for those frames
        # that are ALL NaN.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            angles.head_angles = np.nanmean(
                bend_angles[first_third, :], axis=0)
            angles.body_angles = np.nanmean(
                bend_angles[second_third, :], axis=0)
            angles.tail_angles = np.nanmean(bend_angles[last_third, :], axis=0)
            angles.is_stage_movement = is_stage_movement

        n_head = np.sum(~np.isnan(angles.head_angles))
        n_body = np.sum(~np.isnan(angles.body_angles))
        n_tail = np.sum(~np.isnan(angles.tail_angles))

        # Only proceed if there are at least two non-NaN
        # value in each angle vector
        if n_head < 2 or n_body < 2 or n_tail < 2:
            # Make omegas and upsilons into blank events lists and return
            self.omegas = events.EventListWithFeatures(fps, make_null=True)
            self.upsilons = events.EventListWithFeatures(fps, make_null=True)
            return

        # Deep copy.
        body_angles_for_head_tail_change = np.copy(angles.body_angles)

        # Interpolate the angles.  angles is modified.
        self.h__interpolateAngles(
            angles, options.max_interpolation_gap_allowed)

        # Get frames for each turn type
        #----------------------------------------------------------------------
        # This doesn't match was is written in the supplemental material ...
        # Am I working off of old code??????

        # TODO: Move this all to options ...
        consts = collections.namedtuple('consts',
                                        ['head_angle_start_const',
                                         'tail_angle_start_const',
                                         'head_angle_end_const',
                                         'tail_angle_end_const',
                                         'body_angle_const'])

        yuck = [[20, -20, 15, -15],
                [30, 30, 30, 30],
                [40, 40, 30, 30],
                [20, -20, 15, -15],
                [20, -20, 15, -15]]
        """
        OLD Matlab CODE:

        consts = struct(...
            'head_angle_start_const',{20 -20 15 -15}, ...
            'tail_angle_start_const',{30  30 30  30}, ...
            'head_angle_end_const',  {40  40 30  30}, ...
            'tail_angle_end_const',  {20 -20 15 -15}, ...
            'body_angle_const'   ,   {20 -20 15 -15})
        """

        # NOTE: We need to run omegas first (false values) since upsilons are
        # more inclusive, but can not occur if an omega event occurs
        is_upsilon = [False, False, True, True]

        # NOTE: We assign different values based on the sign of the angles
        values_to_assign = [1, -1, 1, -1]

        frames = collections.namedtuple('frames',
                                        ['omega_frames', 'upsilon_frames'])

        frames.omega_frames = np.zeros(n_frames)
        frames.upsilon_frames = np.zeros(n_frames)

        for i in range(4):
            consts.head_angle_start_const = yuck[0][i]
            consts.tail_angle_start_const = yuck[1][i]
            consts.head_angle_end_const = yuck[2][i]
            consts.tail_angle_end_const = yuck[3][i]
            consts.body_angle_const = yuck[4][i]
            condition_indices = self.h__getConditionIndices(angles, consts)
            self.h__populateFrames(angles,
                                   condition_indices,
                                   frames,
                                   is_upsilon[i],
                                   values_to_assign[i])

        # Calculate the events from the frame values
        self.omegas = OmegaTurns.create(options,
                                        frames.omega_frames,
                                        nw,
                                        body_angles_for_head_tail_change,
                                        midbody_distance,
                                        fps)

        self.upsilons = UpsilonTurns.create(frames.upsilon_frames,
                                            midbody_distance,
                                            fps)

        timer.toc('locomotion.turns')

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name

        # TODO: This should be a method, somewhere ...
        ref = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'turns', 'omegas'], resolve_value=False)
        self.omegas = events.EventListWithFeatures.from_disk(ref, 'MRC')

        ref = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'turns', 'upsilons'], resolve_value=False)
        self.upsilons = events.EventListWithFeatures.from_disk(ref, 'MRC')

        return self

#    @classmethod
#    def from_disk(cls, turns_ref):
#
#        self = cls.__new__(cls)
#
#        self.omegas   = OmegaTurns.from_disk(turns_ref)
#        self.upsilons = UpsilonTurns.from_disk(turns_ref)
#
#        return self
#
#    def __eq__(self, other):
#        return \
#            self.upsilons.test_equality(other.upsilons,'locomotion.turns.upsilons') and \
#            self.omegas.test_equality(other.omegas,'locomotion.turns.omegas')

    def h__interpolateAngles(self, angles, MAX_INTERPOLATION_GAP_ALLOWED):
        """
        Interpolate the angles in the head, body, and tail.
        For the body, also interpolate with a threshold, and assign this
        to body_angles_with_long_nans

        Parameters
        ---------------------------------------
        angles: a named tuple
          angles = collections.namedtuple('angles',
                                          ['head_angles',
                                           'body_angles',
                                           'tail_angles',
                                           'body_angles_with_long_nans',
                                           'is_stage_movement'])


        Returns
        ---------------------------------------
        None; instead the angles parameter has been modified in place

        Notes
        ---------------------------------------
        Formerly a = h__interpolateAngles(a, MAX_INTERPOLATION_GAP_ALLOWED)

        TODO: Incorporate into the former
        seg_worm.feature_helpers.interpolateNanData

        """
        # Let's use a shorter expression for clarity
        interp = utils.interpolate_with_threshold

        # This might not actually have been applied - SEGWORM_MC used BodyAngles
        # - @JimHokanson
        angles.body_angles_with_long_nans = interp(
            angles.body_angles, MAX_INTERPOLATION_GAP_ALLOWED + 1, make_copy=True)

        interp(angles.head_angles, make_copy=False)
        interp(angles.body_angles, make_copy=False)
        interp(angles.tail_angles, make_copy=False)

    def h__getConditionIndices(self, a, c):
        """
        This function implements a filter on the frames for the different
        conditions that we are looking for in order to get a particular turn.

        It does not however provide any logic on their relative order, i.e.
        that one condition occurs before another. This is done in a later
        function, h__populateFrames.

        Parameters
        ---------------------------------------
        a:

        c:


        Notes
        ---------------------------------------
        Formerly s = h__getConditionIndices(a, c)

        """

        # Determine comparison function
        #----------------------------------------------------------
        is_positive = c.head_angle_start_const > 0
        if is_positive:
            fh = operator.gt
        else:
            fh = operator.lt

        # start: when the head exceeds its angle but the tail does not
        # end  : when the tail exceeds its angle but the head does not

        # TODO: Rename to convention ...
        s = collections.namedtuple('stuffs',
                                   ['startCond',
                                    'startInds',
                                    'midCond',
                                    'midStarts',
                                    'midEnds',
                                    'endCond',
                                    'endInds'])

        def find_diff(array, value):
            # diff on logical array doesn't work the same as it does in Matlab
            return np.flatnonzero(np.diff(array.astype(int)) == value)

        with np.errstate(invalid='ignore'):
            s.startCond = fh(a.head_angles, c.head_angle_start_const) & \
                (np.abs(a.tail_angles) < c.tail_angle_start_const)

        # add 1 for shift due to diff
        s.startInds = find_diff(s.startCond, 1) + 1

        # NOTE: This is NaN check is a bit suspicious, as it implies that the
        # head and tail are parsed, but the body is not. The original code puts
        # NaN back in for long gaps in the body angle, so it is possible that
        # the body angle is NaN but the others are not.
        with np.errstate(invalid='ignore'):
            s.midCond   = fh(a.body_angles, c.body_angle_const) | \
                np.isnan(a.body_angles_with_long_nans)

        # add 1 for shift due to diff
        s.midStarts = find_diff(s.midCond, 1) + 1
        s.midEnds = find_diff(s.midCond, -1)

        with np.errstate(invalid='ignore'):
            s.endCond = np.logical_and(fh(a.tail_angles, c.tail_angle_end_const),
                                       np.abs(a.head_angles) <
                                       c.head_angle_end_const)

        s.endInds = find_diff(s.endCond, -1)

        return s

    def h__populateFrames(self, a, s, f, get_upsilon_flag, value_to_assign):
        """

        Algorithm
        ---------------------------------------
        - For the middle angle range, ensure one frame is valid and that
          the frame proceeding the start and following the end are valid
        - Find start indices and end indices that bound this range
        - For upsilons, exclude if they overlap with an omega bend ...


        Parameters
        ---------------------------------------
        a: named tuple
               head_angles: [1x4642 double]
               body_angles: [1x4642 double]
               tail_angles: [1x4642 double]
         is_stage_movement: [1x4642 logical]
                 bodyAngle: [1x4642 double]

        s: named tuple
         startCond: [1x4642 logical]
         startInds: [1x81 double]
           midCond: [1x4642 logical]
         midStarts: [268 649 881 996 1101 1148 1202 1963 3190 3241 4144 4189 4246 4346 4390 4457 4572 4626]
           midEnds: [301 657 925 1009 1103 1158 1209 1964 3196 3266 4148 4200 4258 4350 4399 4461 4579]
           endCond: [1x4642 logical]
           endInds: [1x47 double]

        f: named tuple
           omegaFrames: [4642x1 double]
         upsilonFrames: [4642x1 double]

        get_upsilon_flag: bool
          Toggled based on whether or not we are getting upsilon events
          or omega events

        value_to_assign:


        Returns
        ---------------------------------------
        None; modifies parameters in place.


        Notes
        ---------------------------------------
        Formerly f = h__populateFrames(a,s,f,get_upsilon_flag,value_to_assign)

        """

        for cur_mid_start_I in s.midStarts:

            # JAH NOTE: This type of searching is inefficient since the data
            # are sorted
            temp = np.flatnonzero(s.midEnds > cur_mid_start_I)

            if temp.size != 0:
                cur_mid_end_I = s.midEnds[temp[0]]

                if ~np.all(
                    a.is_stage_movement[
                        cur_mid_start_I:cur_mid_end_I +
                        1]) and s.startCond[
                    cur_mid_start_I -
                    1] and s.endCond[
                    cur_mid_end_I +
                        1]:

                    temp2 = np.flatnonzero(s.startInds < cur_mid_start_I)
                    temp3 = np.flatnonzero(s.endInds > cur_mid_end_I)

                    if temp2.size != 0 and temp3.size != 0:
                        cur_start_I = s.startInds[temp2[-1]]
                        cur_end_I = s.endInds[temp3[0]]

                        if get_upsilon_flag:
                            # Don't populate upsilon if the data spans an omega
                            if ~np.any(
                                np.abs(
                                    f.omega_frames[
                                        cur_start_I:cur_end_I +
                                        1])):
                                f.upsilon_frames[
                                    cur_start_I:cur_end_I + 1] = value_to_assign
                        else:
                            f.omega_frames[
                                cur_start_I:cur_end_I + 1] = value_to_assign

        # Nothing needs to be returned since we have modified our parameters
        # in place
        return None


class NewUpsilonTurns(Feature):
    """
    Feature: locomotion.upsilon_turns
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(wf, 'locomotion.turn_processor').upsilons
        self.no_events = self.value.is_null

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class NewOmegaTurns(Feature):
    """
    Feature: 'locomotion.omega_turns'
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(wf, 'locomotion.turn_processor').omegas
        self.no_events = self.value.is_null

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)
