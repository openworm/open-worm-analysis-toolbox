# -*- coding: utf-8 -*-
"""
Locomotion features

Contains Processing code for:
-----------------------------
locomotion.velocity
locomotion.motion_events

"""

import numpy as np

from .. import utils

from .generic_features import Feature, get_parent_feature_name
from . import events
# To avoid conflicting with variables named 'velocity', we
# import this as 'velocity_module':
from . import velocity as velocity_module


class LocomotionVelocityElement(object):
    """

    This can be deleted when we move over to the new feature organization

    This class is a simple container class for a velocity element.

    Attributes
    ----------
    name : string
    speed : numpy array
    direction : numpy array

    See Also
    --------
    LocomotionVelocity

    """

    def __init__(self, name, speed, direction):
        self.name = name
        self.speed = speed
        self.direction = direction

    def __eq__(self, other):
        return utils.correlation(
            self.speed,
            other.speed,
            'locomotion.velocity.' +
            self.name +
            '.speed') and utils.correlation(
            self.direction,
            other.direction,
            'locomotion.velocity.' +
            self.name +
            '.direction')

    def __repr__(self):
        return utils.print_object(self)

    @classmethod
    def from_disk(cls, parent_ref, name):

        self = cls.__new__(cls)

        self.name = name
        self.speed = utils._extract_time_from_disk(parent_ref, 'speed')
        self.direction = utils._extract_time_from_disk(parent_ref, 'direction')

        return self


class LocomotionVelocity(object):

    """

    This can be deleted when we move over to the new feature organization

    This is for the 'velocity' locomotion feature. The helper function,
    'compute_speed' is used elsewhere.

    Computes the worm velocity at various parts of the worm body.

    Attributes
    ----------
    head_tip : LocomotionVelocityElement
    head : LocomotionVelocityElement
    midbody : LocomotionVelocityElement
    tail : LocomotionVelocityElement
    tail_tip : LocomotionVelocityElement

    fps : float

    """

    # NOTE: head_tip and tail_tip overlap head and tail, respectively, and
    #       this set of partitions does not cover the neck and hips
    attribute_keys = ['head_tip', 'head', 'midbody', 'tail', 'tail_tip']

    def __init__(self, features_ref):
        """
            nw : a NormalizedWorm instance

        """
        timer = features_ref.timer
        timer.tic()

        nw = features_ref.nw
        ventral_mode = nw.video_info.ventral_mode
        fps = nw.video_info.fps

        # We need this for the midbody_distance calculation
        self.fps = fps

        # Make a copy:
        # - Data keys will correspond to sections of the skeleton.
        # - Attribute keys correspond to attributes of this class.
        data_keys = list(self.attribute_keys)

        if(features_ref.options.mimic_old_behaviour):
            data_keys[2] = 'old_midbody_velocity'

        # Step 1: Compute the average body angle
        # ---------------------------------------------------------
        avg_body_angle = velocity_module.get_partition_angles(
            nw, partition_key='body', data_key='skeleton', head_to_tail=False)

        locomotion_options = features_ref.options.locomotion
        sample_time_values = {
            'head_tip': locomotion_options.velocity_tip_diff,
            'head': locomotion_options.velocity_body_diff,
            'midbody': locomotion_options.velocity_body_diff,
            'tail': locomotion_options.velocity_body_diff,
            'tail_tip': locomotion_options.velocity_tip_diff
        }

        # Step 2: Run the compute_speed function on the different
        # parts of the body,
        for attribute_key, data_key in zip(self.attribute_keys, data_keys):
            x, y = nw.get_partition(data_key, 'skeleton', True)

            speed, direction = velocity_module.compute_speed(fps, x, y,
                                                             avg_body_angle,
                                                             sample_time_values[attribute_key],
                                                             ventral_mode)[0:2]

            # Go from shape (1, 4642) to shape (4642,), for
            # num_frames == 4642, say.
            #speed = speed.flatten()
            #direction = direction.flatten()

            setattr(self,
                    attribute_key,
                    LocomotionVelocityElement(attribute_key, speed, direction))

        timer.toc('locomotion.velocity')

    def get_midbody_distance(self):
        """
        The midbody distance is used by a lot of other functions.

        Returns
        -------
        numpy.array

        """

        return abs(self.midbody.speed / self.fps)

    def __eq__(self, other):
        is_same = True
        for attribute_key in self.attribute_keys:
            is_same = \
                getattr(self, attribute_key) == getattr(other, attribute_key)
            if not is_same:
                break

        return is_same

    def __repr__(self):
        return utils.print_object(self)

    @classmethod
    def from_disk(cls, parent_ref):

        self = cls.__new__(cls)

        velocity_ref = parent_ref['velocity']

        for key in self.attribute_keys:

            # NOTE: We'll eventually need to check for old or new data ...
            # These changes are only valid for MRC features
            if key is 'head_tip':
                old_key = 'headTip'
            elif key is 'tail_tip':
                old_key = 'tailTip'
            else:
                old_key = key
            local_ref = velocity_ref[old_key]
            setattr(self, key,
                    LocomotionVelocityElement.from_disk(local_ref, key))
        return self


class MotionEvents(object):

    """


    This can be deleted when we move our new features into place.
    Replaced by MotionEvent

    Attributes
    ----------
    forward : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    paused : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    backward : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    mode : numpy.array
        - shape num_frames
        - Values are:
            -1, backward locomotion
            0, no locomotion (the worm is paused)
            1, forward locomotion

    """

    attribute_keys = ['forward', 'backward', 'paused']

    def __init__(self, features_ref, midbody_speed, skeleton_lengths):
        """
        Calculate motion codes of the locomotion events

        A locomotion feature.

        See feature description at
          /documentation/Yemini%20Supplemental%20Data/Locomotion.md

        Parameters
        ----------
        midbody_speed: numpy.array
            [n_frames]
        skeleton_lengths: numpy.array
            [n_frames]

        Notes
        -----
        Formerly +seg_worm/+features/@locomotion/getWormMotionCodes.m

        """

        timer = features_ref.timer
        timer.tic()

        fps = features_ref.video_info.fps

        locomotion_options = features_ref.options.locomotion

        # Initialize the worm speed and video frames.
        num_frames = len(midbody_speed)

        # Compute the midbody's "instantaneous" distance travelled at each frame,
        # distance per second / (frames per second) = distance per frame
        distance_per_frame = abs(midbody_speed / fps)

        #  Interpolate the missing lengths.
        skeleton_lengths = utils.interpolate_with_threshold(
            skeleton_lengths,
            locomotion_options.motion_codes_longest_nan_run_to_interpolate)

        #===================================
        # SPACE CONSTRAINTS
        # Make the speed and distance thresholds a fixed proportion of the
        # worm's length at the given frame:
        worm_speed_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_speed_threshold_pct
        worm_distance_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_distance_threshold_pct
        worm_pause_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_pause_threshold_pct

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
        # TODO: This might be better as a class attribute
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
            fps * locomotion_options.motion_codes_min_frames_threshold
        # Maximum number of contiguous contradicting frames within the event
        # before the event is considered to be over.
        max_interframes_threshold = \
            fps * locomotion_options.motion_codes_max_interframes_threshold

        # Start with a blank numpy array, full of NaNs:
        self._mode = np.empty(num_frames, dtype='float') * np.NaN

        for motion_type in frame_values:
            # We will use EventFinder to determine when the
            # event type "motion_type" occurred
            ef = events.EventFinder()

            # "Space and time" constraints
            ef.min_distance_threshold = min_distance[motion_type]
            ef.max_distance_threshold = None  # We are not constraining max dist
            ef.min_speed_threshold = min_speeds[motion_type]
            ef.max_speed_threshold = max_speeds[motion_type]

            # "Time" constraints
            ef.min_frames_threshold = min_frames_threshold
            ef.max_inter_frames_threshold = max_interframes_threshold

            event_list = ef.get_events(midbody_speed, distance_per_frame)

            # Take the start and stop indices and convert them to the structure
            # used in the feature files
            m_event = events.EventListWithFeatures(
                fps, event_list, distance_per_frame, compute_distance_during_event=True)

            setattr(self, motion_type, m_event)

            # Assign motion modes
            #------------------------------------------------------------------
            event_mask = event_list.get_event_mask(num_frames)
            self._mode[event_mask] = frame_values[motion_type]

        timer.toc('locomotion.motion_events')

    def get_motion_mode(self):
        return self._mode

    @classmethod
    def from_disk(cls, parent_ref):

        self = cls.__new__(cls)

        motion_ref = parent_ref['motion']

        for key in self.attribute_keys:
            setattr(
                self,
                key,
                events.EventListWithFeatures.from_disk(
                    motion_ref[key],
                    'MRC'))

        self._mode = utils._extract_time_from_disk(motion_ref, 'mode')

        return self

    def __eq__(self, other):
        motion_events_same = [self.__dict__[x].test_equality(
            other.__dict__[x], 'locomotion.motion_events.' + x)
            for x in self.attribute_keys]
        return all(motion_events_same)

    def __repr__(self):
        return utils.print_object(self)

    @property
    def is_paused(self):
        return self._mode == 0

#==============================================================================
#                               NEW CODE
#==============================================================================


class AverageBodyAngle(Feature):

    """
    Temporary Feature: locomotion.velocity.avg_body_angle

    This is a temporary feature that is needed for
    locomotion.velocity features

    Description
    -----------
    For the "body" parition, compute angles between each point
    on the skeleton and average them together to compute a single value for
    each frame.

    See Also
    --------
    LocomotionVelocitySection
    velocity_module.get_partition_angles

    """

    def __init__(self, wf, feature_name):
        nw = wf.nw
        self.name = feature_name
        self.value = velocity_module.get_partition_angles(nw,
                                                          partition_key='body',
                                                          data_key='skeleton',
                                                          head_to_tail=False)

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        # TODO: Move this into a function in generic_features
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = None
        self.missing_from_disk = True
        return self


class LocomotionVelocitySection(Feature):

    """
    Temporary Feature:

    This is a generic temporary class that implements:
    locomotion.velocity.head_tip,
    locomotion.velocity.head, etc.

    This is the parent feature which computes and then temporarily holds
    attributes for more specific child features.

    Attributes
    ----------
    speed : VelocitySpeed
    direction : VelocityDirection


    """

    def __init__(self, wf, feature_name, segment):
        """
        Parameters
        ----------
        segment : string
            Options include:
            - head_tip
            - head
            - midbody
            - tail
            - tail_tip

        Feature Dependencies
        --------------------
        - locomotion.velocity.avg_body_angle

        See Also
        --------
        - velocity_module.compute_speed      #  This is the function that
                                             #  does all the work

        """

        self.name = feature_name

        # Unpacking
        #-------------------------
        nw = wf.nw
        ventral_mode = nw.video_info.ventral_mode
        fps = nw.video_info.fps

        # TODO: I'd like this to be inside the class
        locomotion_options = wf.options.locomotion

        avg_body_angle = self.get_feature(
            wf, 'locomotion.velocity.avg_body_angle').value

        # Options by segment
        #--------------------------------------------------
        if segment == 'head_tip' or segment == 'tail_tip':
            sample_time = locomotion_options.velocity_tip_diff
        else:
            sample_time = locomotion_options.velocity_body_diff

        data_key = segment
        if segment == 'midbody' and wf.options.mimic_old_behaviour:
            data_key = 'old_midbody_velocity'

        # The actual computation
        #----------------------
        # If we ever move nw features into the self.get_feature approach, this
        # would be tougher to replicate
        #i.e. x = self.get_feature(nw,'skeleton_x')
        x, y = nw.get_partition(data_key, 'skeleton', True)
        # The real work ...
        speed, direction = velocity_module.compute_speed(fps, x, y,
                                                         avg_body_angle,
                                                         sample_time,
                                                         ventral_mode)[0:2]

        self.speed = speed
        self.direction = direction

    @classmethod
    def from_schafer_file(cls, wf, feature_name, segment):
        self = cls.__new__(cls)
        self.name = feature_name

        # These particular segments were renamed internally to follow naming
        # conventions. The other ones were fine
        if segment == 'head_tip':
            old_key = 'headTip'
        elif segment == 'tail_tip':
            old_key = 'tailTip'
        else:
            old_key = segment

        temp = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'velocity', old_key], resolve_value=False)

        self.speed = utils.get_nested_h5_field(temp, 'speed')
        self.direction = utils.get_nested_h5_field(temp, 'direction')

        return self

    def __eq__(self, other):
        # I'm not sure what we want to do for these temporary features ...
        return True


class VelocitySpeed(Feature):
    """
    Feature: locomotion.velocity.[segment].speed

    This feature is actually calculated via LocomotionVelocitySection
    """

    def __init__(self, wf, feature_name, segment):
        self.name = feature_name
        parent_feature_name = get_parent_feature_name(feature_name)
        self.value = self.get_feature(wf, parent_feature_name).speed

    @classmethod
    def from_schafer_file(cls, wf, feature_name, segment):
        return cls(wf, feature_name, segment)


class VelocityDirection(Feature):
    """
    Feature: locomotion.velocity.[segment].speed
    """

    def __init__(self, wf, feature_name, segment):
        self.name = feature_name
        parent_feature_name = get_parent_feature_name(feature_name)
        self.value = self.get_feature(wf, parent_feature_name).direction

    @classmethod
    def from_schafer_file(cls, wf, feature_name, segment):
        return cls(wf, feature_name, segment)


# New motion events code
#=====================================================

"""
    forward : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    paused : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    backward : open-worm-analysis-toolbox.features.events.EventListWithFeatures
    mode : numpy.array
        - shape num_frames
        - Values are:
            -1, backward locomotion
            0, no locomotion (the worm is paused)
            1, forward locomotion
 """


class MidbodyVelocityDistance(Feature):
    """
    Temporary Feature: 'locomotion.velocity.mibdody.distance'

    Used for turns
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name

        fps = wf.video_info.fps

        midbody_speed = self.get_feature(
            wf, 'locomotion.velocity.midbody.speed').value
        self.value = abs(midbody_speed / fps)

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        # We could calculate this but the features that need it are already
        # calculated
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = None
        self.missing_from_disk = True
        return self


class MotionEvent(Feature):

    """
    Implements:
    locomotion.motion_events.forward
    locomotion.motion_events.backward
    locomotion.motion_events.paused
    """

    def __init__(self, wf, feature_name, motion_type):

        self.name = feature_name

        fps = wf.video_info.fps
        locomotion_options = wf.options.locomotion

        midbody_speed = self.get_feature(
            wf, 'locomotion.velocity.midbody.speed').value

        skeleton_lengths = self.get_feature(wf, 'morphology.length').value

        # TODO: Move this to the video_info object
        num_frames = len(midbody_speed)

        # Compute the midbody's "instantaneous" distance travelled at each frame,
        # distance per second / (frames per second) = distance per frame
        distance_per_frame = abs(midbody_speed / fps)

        # Interpolate the missing lengths.
        #------------------------------------
        # TODO: This process should be saved as an intermediate feature

        skeleton_lengths = utils.interpolate_with_threshold(
            skeleton_lengths,
            locomotion_options.motion_codes_longest_nan_run_to_interpolate)

        # Set Event filter parameters
        #--------------------------------
        # Make the speed and distance thresholds a fixed proportion of the
        # worm's length at the given frame:
        worm_speed_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_speed_threshold_pct
        worm_distance_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_distance_threshold_pct
        worm_pause_threshold = skeleton_lengths * \
            locomotion_options.motion_codes_pause_threshold_pct

        #   Event Constraints -------
        # The minimum number of frames an event had to be taking place for
        # to be considered a legitimate event
        min_frames_threshold = \
            fps * locomotion_options.motion_codes_min_frames_threshold
        # Maximum number of contiguous contradicting frames within the event
        # before the event is considered to be over.
        max_interframes_threshold = \
            fps * locomotion_options.motion_codes_max_interframes_threshold

        if motion_type == 'forward':
            min_speed_threshold = worm_speed_threshold
            max_speed_threshold = None
            min_distance_threshold = worm_distance_threshold
        elif motion_type == 'backward':
            min_speed_threshold = None
            max_speed_threshold = -worm_speed_threshold
            min_distance_threshold = worm_distance_threshold
        else:  # paused
            min_speed_threshold = -worm_pause_threshold
            max_speed_threshold = worm_pause_threshold
            min_distance_threshold = None

        # We will use EventFinder to determine when the
        # event type "motion_type" occurred
        ef = events.EventFinder()

        # "Space and time" constraints
        ef.min_distance_threshold = min_distance_threshold
        ef.max_distance_threshold = None  # we are not constraining max dist
        ef.min_speed_threshold = min_speed_threshold
        ef.max_speed_threshold = max_speed_threshold

        # "Time" constraints
        ef.min_frames_threshold = min_frames_threshold
        ef.max_inter_frames_threshold = max_interframes_threshold

        event_list = ef.get_events(midbody_speed, distance_per_frame)

        # Take the start and stop indices and convert them to the structure
        # used in the feature files
        m_event = events.EventListWithFeatures(
            fps, event_list, distance_per_frame, compute_distance_during_event=True)

        # This is temporary until a bug is fixed, at which point in time
        # it will likeely need to move into the method directly above
        m_event.num_video_frames = num_frames

        self.value = m_event

        # I think this is an equivalence
        self.no_events = m_event.is_null

    @classmethod
    def from_schafer_file(cls, wf, feature_name, motion_type):

        self = cls.__new__(cls)
        self.name = feature_name

        ref = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'motion', motion_type], resolve_value=False)

        self.value = events.EventListWithFeatures.from_disk(ref, 'MRC')
        self.no_events = self.value.is_null

        return self

    def __eq__(self, other):
        # temp feature ...
        return True


class MotionMode(Feature):

    """
    Temporary Feature:

    This is a temporary feature. For each frame it indicates whether that
    frame is part of a forward, backward, or paused event.

    Some frames
    may not be part of any event type, as is indicated by a NaN value.

    forward: 1
    backward: -1
    paused: 0
    """

    frame_values = {'forward': 1, 'backward': -1, 'paused': 0}

    def __init__(self, wf, feature_name):

        self.name = feature_name

        # Hack to get num_frames
        skeleton_lengths = self.get_feature(wf, 'morphology.length').value

        # TODO: Get this from video_info
        num_frames = len(skeleton_lengths)

        # TODO: Can't we initialize NaN directly?
        self.value = np.empty(num_frames, dtype='float') * np.NaN

        for key, value in self.frame_values.items():
            motion_type = 'locomotion.motion_events.' + key
            event_feature = self.get_feature(wf, motion_type).value

            event_mask = event_feature.get_event_mask()
            self.value[event_mask] = value

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)

        self.name = feature_name

        self.value = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'motion', 'mode'])

        return self


class IsPaused(Feature):

    """
    Temporary Feature: locomotion.motion_events.is_paused
    """

    def __init__(self, wf, feature_name):

        # TODO: We could eventually only compute the paused event
        # rather than checking the mode. We would need to add on support for
        # checking if a feature had been computed
        self.name = feature_name
        mode = self.get_feature(wf, 'locomotion.motion_mode').value
        self.value = mode == 0

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)
