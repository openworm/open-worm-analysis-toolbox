# -*- coding: utf-8 -*-
"""
A module for finding and describing frames-spanning "events" given a
worm video.

Use EventFinder() to get an EventList()

Contents
---------------------------------------
This module contains definitions for the following:

Classes:
  EventFinder
    __init__
    get_events

  EventList
    __init__
    num_events @property
    get_event_mask
    merge

  EventListWithFeatures
    __init__
    from_disk
    test_equality


Usage
-----
Places used (Not exhaustive):
- locomotion_features.MotionEvents
- posture_features.get_worm_coils
- locomotion_turns.LocomotionTurns

One usage is in locomotion features.

LocomotionFeatures.get_motion_codes() calculates the motion codes for the
worm, and to do so, for each of the possible motion states (forward,
backward, paused) it creates an instance of the EventFinder class,
sets up appropriate parameters, and then calls EventFinder.get_events()
to obtain an instance of EventList, the "result" class.

Then to format the result appropriately, the EventListWithFeatures class is
instantiated with our "result" and then get_feature_dict is called.

So the flow from within LocomotionFeatures.get_motion_codes() is:
    # (approximately):
    ef = EventFinder()
    event_list = ef.get_events()
    me = EventListWithFeatures(event_list, features_per_frame)

EventListWithFeatures is used by not just get_motion_codes but also ...
DEBUG (add the other uses (e.g. upsilons))

Notes
---------------------------------------
See https://github.com/openworm/open-worm-analysis-toolbox/blob/master/
documentation/Yemini%20Supplemental%20Data/Locomotion.md#2-motion-states
for a plain-English description of a motion state.

The code for this module came from several files in the
@event_finder, @event, and @event_ss folders from:
https://github.com/JimHokanson/SegwormMatlabClasses/blob/
    master/%2Bseg_worm/%2Bfeature/

"""

import numpy as np
import operator
import h5py
import warnings

from itertools import groupby
from .. import utils


class EventFinder:
    """
    To use this, create an instance, then specify the options.  Default
    options are initialized in __init__.

    Then call get_events() to obtain an EventList instance
    containing the desired events from a given block of data.

    """

    def __init__(self):
        # Temporal thresholds
        self.min_frames_threshold = None  # (scalar or [1 x n_frames])
        self.include_at_frames_threshold = False

        self.max_inter_frames_threshold = None  # (scalar or [1 x n_frames])
        self.include_at_inter_frames_threshold = False

        # Space (distance) and space&time (speed) thresholds
        self.min_distance_threshold = None  # (scalar or [1 x n_frames])
        self.max_distance_threshold = None  # (scalar or [1 x n_frames])
        self.include_at_distance_threshold = True

        self.min_speed_threshold = None  # (scalar or [1 x n_frames])
        self.max_speed_threshold = None  # (scalar or [1 x n_frames])
        self.include_at_speed_threshold = True

    def get_events(self, speed_data, distance_data=None):
        """
        Obtain the events implied by event_data, given how this instance
        of EventFinder has been configured.

        Parameters
        ----------
        speed_data : 1-d numpy array of length n
            The per-frame instantaneous speed as % of skeleton length
        distance_data   : 1-d numpy array of length n (optional)
            The per-frame distance travelled as % of skeleton length
            If not specified, speed_data will be used to derive distance_data,
            since speed = distance x time.

        Returns
        -------
        EventList

        Notes:
        ---------------------------------------
        If the first/last event are solely preceded/followed by NaN
        frames, these frames are swallowed into the respective event.

        Formerly getEvents.m.  Originally it was findEvent.m.

        """
        # Override distance_data with speed_data if it was not provided
        if distance_data is None:
            distance_data = speed_data

        # For each frame, determine if it matches our speed threshold criteria
        speed_mask = self.get_speed_threshold_mask(speed_data)

        # Convert our mask into the indices of the "runs" of True, that is
        # of the data matching our above speed criteria
        event_candidates = self.get_start_stop_indices(speed_data, speed_mask)

        # ERROR: start is not at 0
        #??? Starts might all be off by 1 ...

        # Possible short circuit: if we have absolutely no qualifying events
        # in event_data, just exit early.
        if not event_candidates.size:
            return EventList()

        if self.max_inter_frames_threshold:
            # Decide if we are removing gaps AT the threshold or
            # just strictly smaller than the threshold.
            if self.include_at_inter_frames_threshold:
                inter_frames_comparison_operator = operator.le
            else:
                inter_frames_comparison_operator = operator.lt

            # In this function we remove time gaps between events if the gaps
            # are too small (max_inter_frames_threshold)
            event_candidates = self.remove_gaps(
                event_candidates,
                self.max_inter_frames_threshold,
                inter_frames_comparison_operator)

        if self.min_frames_threshold:
            # Remove events that aren't at least
            # self.min_frames_threshold in length
            event_candidates = self.remove_too_small_events(event_candidates)

        # For each candidate event, sum the instantaneous speed at all
        # frames in the event, and decide if the worm moved enough distance
        # for the event to qualify as genuine.
        # i.e. Filter events based on data sums during event
        event_candidates = \
            self.remove_events_by_data_sum(event_candidates, distance_data)

        return EventList(event_candidates)

    def __repr__(self):
        return utils.print_object(self)

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

        if self.min_speed_threshold is not None:
            # suppress runtime warning of comparison to None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if self.include_at_speed_threshold:
                    event_mask = event_data >= self.min_speed_threshold
                else:
                    event_mask = event_data > self.min_speed_threshold

        if self.max_speed_threshold is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if self.include_at_speed_threshold:
                    event_mask = event_mask & (
                        event_data <= self.max_speed_threshold)
                else:
                    event_mask = event_mask & (
                        event_data < self.max_speed_threshold)

        return event_mask

    def get_start_stop_indices(self, event_data, event_mask):
        """
        From a numpy event mask, get the start and stop indices.  For
        example:

          0 1 2 3 4 5   <- indices
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
        event_candidates: 2-d int numpy array
          An array of tuples giving the start and stop, respectively,
          of each run of Trues in the event_mask. IMPORTANTLY, these are indices,
          NOT slice values

        Notes
        ---------------------------------------
        Formerly h__getStartStopIndices, in
        seg_worm/feature/event_finder/getEvents.m

        """
        # NO
        # Go from shape == (1, 4642) to (4642,) (where 4642 is the number of
        # frames, for instance)
        #event_data = event_data.flatten()
        #event_mask = event_mask.flatten()

        # Make sure our parameters are of the correct type and dimension
        assert(isinstance(event_data, np.ndarray))
        assert(isinstance(event_mask, np.ndarray))
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
        x = np.flatnonzero(bracketed_event_mask) - 1

        # Group these together using a fancy trick from
        # http://stackoverflow.com/questions/2154249/, since
        # the lambda function x:x[0]-x[1] on an enumerated list will
        # group consecutive integers together
        # e.g. [[(0, 3)], [(1, 5), (2, 6), (3, 7)]]
        # list(group)
        x_grouped = [list(group) for key, group in
                     groupby(enumerate(x), lambda i:i[0] - i[1])]

        # We want to know the first element from each "run", and the
        # last element e.g. [[3, 4], [5, 7]]
        event_candidates = [(i[0][1], i[-1][1]) for i in x_grouped]

        # Early exit if we have no starts and stops at all
        if not event_candidates:
            return np.array(event_candidates)

        # If a run of NaNs precedes the first start index, all the way back to
        # the first element, then revise our first (start, stop) entry to
        # include all those NaNs.
        if np.all(np.isnan(event_data[:event_candidates[0][0]])):
            event_candidates[0] = (0, event_candidates[0][1])

        # Same but with NaNs succeeding the final end index.
        if np.all(np.isnan(event_data[event_candidates[-1][1] + 1:])):
            event_candidates[-1] = (event_candidates[-1]
                                    [0], event_data.size - 1)

        return np.array(event_candidates)

    def remove_gaps(self, event_candidates, threshold,
                    comparison_operator):
        """
        Remove time gaps in the events that are smaller/larger than a given
        threshold value.

        That is, apply a greedy right-concatenation to any (start, stop) duples
        within threshold of each other.

        Parameters
        ---------------------------------------
        event_candidates: a list of (start, stop) duples
          The start and stop indexes of the events
        threshold: int
          Number of frames to do the comparison on
        comparison_operator: a comparison function
          One of operator.lt, le, ge, gt

        Returns
        ---------------------------------------
        A new numpy array of (start, stop) duples giving the indexes
        of the events with the gaps removed.

        """
        assert(comparison_operator == operator.lt or
               comparison_operator == operator.le or
               comparison_operator == operator.gt or
               comparison_operator == operator.ge)

        new_event_candidates = []
        num_groups = np.shape(event_candidates)[0]

        i = 0
        while(i < num_groups):
            # Now advance through groups to the right,
            # continuing as long as they satisfy our comparison operator
            ii = i
            while(ii + 1 < num_groups and comparison_operator(
                    event_candidates[ii + 1][0] - event_candidates[ii][1] - 1,
                    threshold)):
                ii += 1

            # Add this largest possible start/stop duple to our NEW revised
            # list
            new_event_candidates.append((event_candidates[i][0],
                                         event_candidates[ii][1]))
            i = ii + 1

        return np.array(new_event_candidates)

    def remove_too_small_events(self, event_candidates):
        """
        This function filters events based on time (really sample count)

        Parameters
        ---------------------------------------
        event_candidates: numpy array of (start, stop) duples

        Returns
        ---------------------------------------
        numpy array of (start, stop) duples

        """
        if not self.min_frames_threshold:
            return event_candidates

        event_num_frames = event_candidates[:, 1] - event_candidates[:, 0] + 1

        events_to_remove = np.zeros(len(event_num_frames), dtype=bool)

        if self.include_at_frames_threshold:
            events_to_remove = event_num_frames <= self.min_frames_threshold
        else:
            events_to_remove = event_num_frames < self.min_frames_threshold

        return event_candidates[np.flatnonzero(~events_to_remove)]

    def remove_events_by_data_sum(self, event_candidates, distance_data):
        """
        This function removes events by data sum.  An event is only valid
        if the worm has moved a certain minimum proportion of its mean length
        over the course of the event.

        For example, a forward motion state is only a forward event if the
        worm has moved at least 5% of its mean length over the entire period.

        For a given event candidate, to calculate the worm's movement we
        SUM the worm's distance travelled per frame (distance_data) over the
        frames in the event.

        Parameters
        ---------------------------------------
        event_candidates: numpy array of (start, stop) duples

        Returns
        ---------------------------------------
        numpy array of (start, stop) duples
          A subset of event_candidates with only the qualifying events.

        Notes
        ---------------------------------------
        Formerly h____RemoveEventsByDataSum

        """
        # If we've established no distance thresholds, we have nothing to
        # remove from event_candidates
        if self.min_distance_threshold is None and \
           self.max_distance_threshold is None:
            return event_candidates

        # Sum over the event and threshold data so we know exactly how far
        # the worm did travel in each candidate event, and also the min/max
        # distance it MUST travel for the event to be considered valid
        # --------------------------------------------------------

        num_runs = np.shape(event_candidates)[0]

        # Sum the actual distance travelled by the worm during each candidate
        # event
        event_sums = np.empty(num_runs, dtype=float)
        for i in range(num_runs):
            event_sums[i] = np.nansum(
                distance_data[
                    event_candidates[i][0]:(
                        event_candidates[i][1] + 1)])

        # self.min_distance_threshold contains a 1-d n-element array of
        # skeleton lengths * 5% or whatever proportion we've decided the
        # worm must move for our event to be valid.  So to figure out the
        # threshold for a given we event, we must take the MEAN of this
        # threshold array.
        #
        # Note that we test if min_distance_threshold event contains any
        # elements, since we may have opted to simply not include this
        # threshold at all.
        min_threshold_sums = np.empty(num_runs, dtype=float)
        if self.min_distance_threshold is not None:
            for i in range(num_runs):
                min_threshold_sums[i] = np.nanmean(
                    self.min_distance_threshold[
                        event_candidates[i][0]:(
                            event_candidates[i][1] + 1)])

        # Same procedure as above, but for the maximum distance threshold.
        max_threshold_sums = np.empty(num_runs, dtype=float)
        if self.max_distance_threshold is not None:
            for i in range(num_runs):
                max_threshold_sums[i] = np.nanmean(
                    self.max_distance_threshold[
                        event_candidates[i][0]:(
                            event_candidates[i][1] + 1)])

        # Actual filtering of the candidate events
        # --------------------------------------------------------

        events_to_remove = np.zeros(num_runs, dtype=bool)

        # Remove events where the worm travelled too little
        if self.min_distance_threshold is not None:
            if self.include_at_distance_threshold:
                events_to_remove = (event_sums <= min_threshold_sums)
            else:
                events_to_remove = (event_sums < min_threshold_sums)

        # Remove events where the worm travelled too much
        if self.max_distance_threshold is not None:
            if self.include_at_distance_threshold:
                events_to_remove = events_to_remove | \
                    (event_sums >= max_threshold_sums)
            else:
                events_to_remove = events_to_remove | \
                    (event_sums > max_threshold_sums)

        return event_candidates[np.flatnonzero(~events_to_remove)]


class EventList(object):
    """
    The EventList class is a relatively straightforward class specifying
    when "events" start and stop.

    The EventListWithFeatures class, on the other hand, computes other
    statistics on the data over which the event occurs.

    An event is a contiguous subset of frame indices.

    You can ask for a representation of the event list as
    1) a sequence of (start, stop) tuples
    2) a boolean array of length num_frames with True for all event frames

    Attributes
    ----------
    start_frames : numpy.array 1-d
        Frames when each event starts.
    end_frames : numpy.array 1-d
        Frames when each event ends (is inclusive, i.e. the last frame is
        a part of the event)
    starts_and_stops : 2-d numpy.array
    num_events : int
    num_events_for_stats : int
    last_frame : int

    Methods
    -------
    get_event_mask: returns 1-d boolean numpy array
    merge: returns an EventList instance

    Notes
    -----
    Previous name:
    seg_worm.feature.event_ss ("ss" stands for "simple structure")


    @JimHokanson: This class is the encapsulation of the raw
    substructure, or output from finding the event.

    """

    def __init__(self, event_starts_and_stops=None):
        """

        Parameters
        ----------

        """
        # self.start_frames and self.end_frames will be the internal representation
        # of the events within this class.
        self.start_frames = None
        self.end_frames = None

        # Check if our events array exists and there is at least one event
        if (event_starts_and_stops is not None and
                event_starts_and_stops.size != 0):
            self.start_frames = event_starts_and_stops[:, 0]
            self.end_frames = event_starts_and_stops[:, 1]

        if(self.start_frames is None):
            self.start_frames = np.array([], dtype=int)

        if(self.end_frames is None):
            self.end_frames = np.array([], dtype=int)

    def __repr__(self):
        return utils.print_object(self)

    @property
    def starts_and_stops(self):
        """
        Returns the start_frames and end_frames as a single numpy array
        """
        s_and_s = np.array([self.start_frames, self.end_frames])

        # check that we didn't have s_and_s = [None, None] or something
        if len(np.shape(s_and_s)) == 2:
            # We need the first dimension to be the events, and the second
            # to be the start / end, not the other way around.
            # i.e. we want it to be n x 2, not 2 x n
            s_and_s = np.rollaxis(s_and_s, 1)
            return s_and_s
        else:
            return np.array([])

    @property
    def __len__(self):
        """
        Return the number of events stored by a given instance of this class.

        Notes
        ---------------------------------------
        Formerly n_events

        """
        # TODO: I think we are mixing lists and numpy arrays - let's remove the lists
        # TypeError: object of type 'numpy.float64' has no len()
        try:
            return len(self.start_frames)
        except TypeError:
            return self.start_frames.size

    @property
    def last_event_frame(self):
        """
        Return the frame # of end of the final event

        Notes
        ---------------------------------------
        Note that the events represented by a given instance
        must have come from a video of at least this many
        frames.

        """
        # Check if the end_frames have any entries at all
        if self.end_frames is not None and self.end_frames.size != 0:
            return self.end_frames[-1]
        else:
            return 0

    @property
    def num_events_for_stats(self):
        """
        Compute the number of events, excluding the partially recorded ones.
        Partially recorded ones are:
        1) An event that has already started at the first frame
        2) An event that is still going at the last frame
        """
        value = self.__len__
        if value > 1:
            if self.start_frames[0] == 0:
                value = value - 1
            if self.end_frames[-1] == self.num_video_frames - 1:
                value = value - 1

        return value

    def get_event_mask(self, num_frames=None):
        """

        This was developed for locomotion_features.MotionEvents in which
        the goal was to go from events back to the original timeline to
        calculate for every frame which event that frame is a part of.

        Returns an array with True entries only between
        start_frames[i] and end_frames[i], for all i such that
        0 <= end_frames[i] < num_frames

        Parameters
        ----------
        num_frames: int (optional)
          The number of frames to use in the mask
          If num_frames is not given, a mask just large enough to accomodate
          all the events is returned (i.e. of length self.last_event_frame+1)

        Returns
        -------
        1-d boolean numpy array of length num_frames

        """
        # Create empty array of all False, as large as
        # it might possibly need to be
        if num_frames is None:
            num_frames = self.last_event_frame + 1

        mask = np.zeros(max(self.last_event_frame +
                            1, num_frames), dtype='bool')

        for i_event in range(self.__len__):
            mask[self.start_frames[i_event]:self.end_frames[i_event] + 1] = True

        #??? Why are we slicing the output?
        # This appears to be because last_event_frame+1 could be larger
        # than num_frames
        # TODO: This should be fixed, let's make this truncation more explicit
        #in the documentation. We should also consider embedding the #
        # of frames into the event
        return mask[0:num_frames]

    @classmethod
    def merge(cls, obj1, obj2):
        """
        Merge two EventList instances into a single, larger, EventList

        Acts as a factory, producing a third instance of the EventList class
        that is the concatenation of the first two, with the start indices
        blended and properly in order.

        Parameters
        ----------
        cls: The static class parameter, associated with @classmethod
        obj1: EventList instance
        obj2: EventList instance

        Returns
        ---------------------------------------
        Tuple (EventList, is_from_first_object)
          EventList: A new EventList instance
          is_from_first_object: A mask in case you care which indices are
                                from the first object.

        """
        all_starts = np.concatenate((obj1.start_frames, obj2.start_frames))
        all_ends = np.concatenate((obj1.end_frames, obj2.end_frames))

        # TODO: It would be good to check that events don't overlap

        new_starts = np.sort(all_starts)
        order_I = np.argsort(all_starts)

        new_ends = all_ends[order_I]

        is_from_first_object = order_I < obj1.start_frames.size

        starts_stops = np.transpose(np.vstack((new_starts, new_ends)))

        return (EventList(starts_stops), is_from_first_object)


class EventListWithFeatures(EventList):

    """
    An list of events, but also with a set of features calculated for those
    events.  e.g. time_between_events, etc.

    The list of events can also be embued with another dimension of data,
    called "distance" (i.e. the distance the worm has travelled during the
    given frame) but which can be generalized to anything that can happen
    over time to the worm.

    With this extra dimension other features can be calculated, such as
    distance_during_events.

    Properties
    ---------------------------------------
    num_video_frames
    start_frames
    end_frames
    event_durations
    time_between_events
    distance_during_events
    distance_between_events
    total_time
    frequency
    time_ratio
    data_ratio
    num_events_for_stats

    Notes
    ---------------------------------------
    Formerly seg_worm.feature.event
    See also seg_worm.events.events2stats

    Known Uses
    ---------------------------------------
    posture.coils
    locomotion.turns.omegas
    locomotion.turns.upsilons
    locomotion.motion.forward
    locomotion.motion.backward
    locomotion.motion.paused

      %.frames    - event_stats (from event2stats)
      %.frequency -

      THSI IS ALL OUTDATED


      properties
          fps
          n_video_frames

          %INPUTS
          %------------------------------------------------------------------
      #Old Names: start and end
      #NOTE: These are the exact frames, the end is NOT setup for slicing
          start_frames %[1 n_events]
          end_frames   %[1 n_events]
          data_sum_name %[1 n_events]
          inter_data_sum_name %[1 n_events], last value is NaN


      end


    """

    def __init__(self, fps, event_list=None, distance_per_frame=None,
                 compute_distance_during_event=False, make_null=False):
        """
        Initialize an instance of EventListWithFeatures

        Parameters:
        -----------
        event_list : EventList (default None)
            A list of all events
        distance_per_frame : numpy.array (default None)
            Distance moved per frame.  In fact, as discussed in the class
            definition for EventListWithFeatures, this parameter can be used
            for any quantifiable behaviour the worm is engaged in over time,
            not just distance travelled.  Perhaps therefore could be renamed
            to something more general.
        compute_distance_during_event: boolean (default False)
            Whether or not to compute the distance during the even.
        make_null: boolean (default False)
            Whether or not the caller wants simply a blank instance
            to be returned. This is for cases in which there are no events
            for a particular feature.

            This is different than if the event has not been computed, in which
            case the object itself should be None


        Parameters:
        -----------
        Original Code: +seg_worm/+feature/+event/event.m

        Used by:
        get_motion_codes  - computes data and interdata
        get_coils         - computes only interdata
        omega and upsilon - computes only interdata

        """
        if event_list is None:
            # If no event_list instance is supplied, instantiate
            EventList.__init__(self, None)
        else:
            EventList.__init__(self, event_list.starts_and_stops)

        self.distance_per_frame = distance_per_frame

        # If a blank instance has been requested, or if a blank event_list
        # has been provided, flag the self.is_null variable as such
        self.is_null = make_null or (self.__len__ == 0)

        # Only calculate the extra features if this is not a "null" instance
        if not self.is_null:
            # Calculate the features
            self.calculate_features(fps, compute_distance_during_event)
        else:
            # Otherwise, populate with blanks
            self.event_durations = np.array([], dtype=float)
            self.distance_during_events = np.array([], dtype=float)
            self.time_between_events = np.array([], dtype=float)
            self.distance_between_events = np.array([], dtype=float)
            self.frequency = np.NaN
            self.time_ratio = np.NaN
            self.data_ratio = np.NaN

    def calculate_features(self, fps, compute_distance_during_event):
        """
        num_video_frames
        start_frames
        end_frames
        event_durations
        time_between_events
        distance_during_events
        distance_between_events
        total_time
        frequency
        time_ratio
        data_ratio
        num_events_for_stats

        """

        self.num_video_frames = len(self.distance_per_frame)

        # Old Name: time
        self.event_durations = (self.end_frames - self.start_frames + 1) / fps

        # Old Name: interTime
        self.time_between_events = (
            self.start_frames[1:] - self.end_frames[:-1] - 1) / fps

        # Old Name: interDistance
        # Distance moved during events
        if compute_distance_during_event:
            self.distance_during_events = np.zeros(self.__len__)
            for i in range(self.__len__):
                self.distance_during_events[i] = np.nansum(
                    self.distance_per_frame[
                        self.start_frames[i]:self.end_frames[i] + 1])
            self.data_ratio = np.nansum(self.distance_during_events) \
                / np.nansum(self.distance_per_frame)
        else:
            self.distance_during_events = np.array([])
            self.data_ratio = np.NaN

        # Old Name: distance
        # Distance moved between events
        self.distance_between_events = np.zeros(self.__len__ - 1)
        for i in range(self.__len__ - 1):
            # Suppress "FutureWarning: In Numpy 1.9 the sum along empty
            # slices will be zero."
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                self.distance_between_events[i] = \
                    np.nansum(
                        self.distance_per_frame[self.end_frames[i] + 1:
                                                self.start_frames[i + 1]])

        #self.distance_between_events[-1] = np.NaN

        self.total_time = self.num_video_frames / fps

        # How frequently an event occurs - add to documentation
        self.frequency = self.num_events_for_stats / self.total_time

        self.time_ratio = np.nansum(self.event_durations) / self.total_time

    def get_event_mask(self):
        """
        Return a numpy boolean array corresponding to the events

        Returns
        ---------------------------------------
        A 1-d numpy boolean array

        Notes
        ---------------------------------------

        EventListWithFeatures has overridden its superclass, EventList's,
        get_event_mask with its own here, since self.distance_per_frame
        gives it the precise number of frames so it no longer needs to
        accept it as a parameter.

        """
        return EventList.get_event_mask(self, self.num_video_frames)

    @classmethod
    def from_disk(cls, event_ref, ref_format):
        """
        Class factory method to return an instance of the class, loaded from
        disk.

        Returns
        ---------------------------------------
        ref_format : {'MRC'}
          The format used.  Currently 'MRC' is the only option.

        """
        # Construct the class
        self = cls.__new__(cls)
        # Initialize the superclass
        EventList.__init__(self, None)

        """
        num_video_frames
        start_frames
        end_frames
        event_durations
        time_between_events
        distance_during_events
        distance_between_events
        total_time
        frequency
        time_ratio
        data_ratio
        num_events_for_stats
        """

        if ref_format is 'MRC':
            frames = event_ref['frames']

            if isinstance(frames, h5py._hl.dataset.Dataset):
                self.is_null = True
                return self
            else:
                self.is_null = False

            # In Matlab this is a structure array
            # Our goal is to go from an array of structures to a
            # single "structure" with arrays of values
            #
            # If only a single element is present, the data are saved
            # differently. In this case the values are saved directly without
            # a reference to dereference.
            frame_values = {}
            file_ref = frames.file
            for key in frames:
                ref_array = frames[key]
                try:
                    # Yikes, getting the indexing right here was a PITA

                    # (1,5) -> second option - <HDF5 dataset "start": shape (1, 5), type "|O8">
                    # Seriously Matlab :/
                    if ref_array.shape[0] > 1:
                        frame_values[key] = np.array(
                            [file_ref[x[0]][0][0] for x in ref_array])
                    else:
                        # This is correct for omegas ...
                        frame_values[key] = np.array(
                            [file_ref[x][0][0] for x in ref_array[0]])

                except AttributeError:
                    # AttributeError: 'numpy.float64' object has no attribute
                    #                 'encode'
                    ref_element = ref_array
                    frame_values[key] = [ref_element[0][0]]

            self.start_frames = np.array(frame_values['start'], dtype=int)
            self.end_frames = np.array(frame_values['end'], dtype=int)
            self.event_durations = np.array(frame_values['time'])
            if('isVentral' in frame_values.keys()):
                # For isVentral to even exist we must be at a signed event,
                # such as where
                # frames.name == '/worm/locomotion/turns/omegas/frames
                self.is_ventral = np.array(frame_values['isVentral'],
                                           dtype=bool)

            # Remove NaN value at end
            n_events = self.start_frames.size
            if n_events < 2:
                self.time_between_events = np.zeros(0)
                self.distance_between_events = np.zeros(0)
            else:
                self.time_between_events = np.array(
                    frame_values['interTime'][:-1])
                self.distance_between_events = np.array(
                    frame_values['interDistance'][:-1])

            # JAH: I found float was needed as the calculated frequency was also
            # of this type. I'm not sure why we lost the numpy array entry
            # for the calculated frequency ...
            self.frequency = event_ref['frequency'].value[0][0]

            if 'ratio' in event_ref.keys():
                ratio = event_ref['ratio']
                self.distance_during_events = np.array(
                    frame_values['distance'])
                self.time_ratio = ratio['time'][0][0]
                self.data_ratio = ratio['distance'][0][0]
            else:
                self.time_ratio = event_ref['timeRatio'].value[0][0]
                self.data_ratio = np.NaN
                self.distance_during_events = np.zeros(0)
        else:
            raise Exception('Other formats not yet supported :/')

        # Num_video_frames - CRAP: :/   - @JimHokanson
        # Look away ...
        temp_length = file_ref['worm/morphology/length']
        self.num_video_frames = len(temp_length)

        # Total_time - CRAP :/   - @JimHokanson
        self.total_time = self.num_events_for_stats / self.frequency

        return self

    def __repr__(self):
        return utils.print_object(self)

    def test_equality(self, other, event_name):

        try:
            if self.is_null and other.is_null:
                return True
            elif self.is_null != other.is_null:
                print('Event mismatch %s' % event_name)
                return False
        except:
            raise Exception("Problem while testing inequality")

        # TODO: Add an integer equality comparison with name printing
        return utils.compare_is_equal(
            self.num_video_frames,
            other.num_video_frames,
            event_name +
            '.num_video_frames') and utils.correlation(
            self.start_frames,
            other.start_frames,
            event_name +
            '.start_frames') and utils.correlation(
            self.end_frames,
            other.end_frames,
            event_name +
            '.end_frames') and utils.correlation(
                self.event_durations,
                other.event_durations,
                event_name +
                '.event_durations') and utils.correlation(
                    self.distance_during_events,
                    other.distance_during_events,
                    event_name +
                    '.distance_during_events') and utils.compare_is_equal(
                        self.total_time,
                        other.total_time,
                        event_name +
                        '.total_time',
                        0.01) and utils.compare_is_equal(
                            self.frequency,
                            other.frequency,
                            event_name +
                            '.frequency',
                            0.01) and utils.compare_is_equal(
                                self.time_ratio,
                                other.time_ratio,
                                event_name +
                                '.time_ratio',
                                0.01) and utils.compare_is_equal(
                                    self.data_ratio,
                                    other.data_ratio,
                                    event_name +
                                    '.data_ratio',
                                    0.01) and utils.compare_is_equal(
                                        self.num_events_for_stats,
                                        other.num_events_for_stats,
                                        event_name +
            '.total_time')
